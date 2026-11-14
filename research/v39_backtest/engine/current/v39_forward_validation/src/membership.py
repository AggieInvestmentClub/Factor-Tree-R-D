from common import *
import pandas as pd
sys.path.insert(0,str(SOURCE/'src'))
from historical_sources import canonical_ric,membership_names

def main():
    raw=HERE/'data/raw';probe=HERE/'data/source_probe/host';records=[];corrections=[]
    transitions=[]
    for row in read(HERE/'data/sector_transition_evidence.json'):
        for date in ('2026-07-30','2026-07-31','2026-08-03'):
            for sector in (row['from_sector'],row['to_sector']):
                f=raw/(sector+'_checkpoint_'+date+'.pkl');members=set(pd.read_pickle(f)['Constituent RIC'].map(canonical_ric));expected=sector==(row['from_sector'] if date<'2026-08-03' else row['to_sector']);assert (row['ric'] in members)==expected,(row,date,sector)
        transitions.append(dict(row,effective_date='2026-08-03',status='Announcement effective July31 close; independently captured sector-index snapshots change first at August3 session',dated_checkpoint_sha256={f.name:sha(f) for f in raw.glob('*_checkpoint_2026-0*.pkl') if f.name.startswith((row['from_sector'],row['to_sector']))}))
    write(HERE/'data/verified_sector_transitions.json',transitions)
    oldb=membership_names(pd.read_csv(SOURCE/'data/membership_candidate_snapshots.csv.gz'))
    olds=membership_names(pd.read_csv(SOURCE/'data/sector_candidate_snapshots.csv.gz'))
    def reconstruct(label,old,changes,latest):
        changes=changes.copy();changes['Date']=pd.to_datetime(changes.Date).dt.strftime('%Y-%m-%d');changes['Constituent RIC']=changes['Constituent RIC'].map(canonical_ric)
        # Sector indexes are subsets of SPX. Independently captured broad deletions
        # resolve two omitted sector-index deletion messages.
        for sector,ric in [('XLE','HES.N^G25'),('XLV','CTLT.N^L24')]:
            if label!=sector:continue
            source=pd.read_csv(probe/'changes.csv');row=source[source['Constituent RIC'].eq(ric)&source.Change.eq('Leaver')].iloc[0]
            if not (changes['Constituent RIC'].eq(ric)&changes.Change.eq('Leaver')).any():
                correction=dict(Date=row.Date,**{'Constituent RIC':ric,'Change':'Leaver'});changes=pd.concat([changes,pd.DataFrame([correction])],ignore_index=True)
                corrections.append(dict(universe=label,**correction,reason='Missing sector deletion; exact dated broad-index deletion and latest sector anchor reconcile',source_sha256=sha(probe/'changes.csv')))
        transitions=HERE/'data/verified_sector_transitions.json'
        if transitions.exists():
            for r in read(transitions):
                if label not in (r['from_sector'],r['to_sector']):continue
                correction=dict(Date=r['effective_date'],**{'Constituent RIC':r['ric'],'Change':'Leaver' if label==r['from_sector'] else 'Joiner'})
                changes=pd.concat([changes,pd.DataFrame([correction])],ignore_index=True);corrections.append(dict(universe=label,**correction,verification=r))
        correction=dict(Date='2026-11-04',**{'Constituent RIC':'KVUE.N^K26','Change':'Leaver'})
        if label in ('SPX','XLP') and not ((changes.Date.eq(correction['Date']))&changes['Constituent RIC'].eq('KVUE.N^K26')&changes.Change.eq('Leaver')).any():
            changes=pd.concat([changes,pd.DataFrame([correction])],ignore_index=True)
            corrections.append(dict(universe=label,**correction,announced='2026-10-30',effective='before open 2026-11-04',source='https://press.spglobal.com/2026-10-30-Okta-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600',reason='Vendor joins/leavers omitted independently documented deletion; latest anchor confirms'))
        members=set(old.loc[old.date.eq(old.date.max()),'ric']);rows=[];issues=[]
        for date,g in changes.groupby('Date',sort=True):
            joined=set(g.loc[g.Change.eq('Joiner'),'Constituent RIC'].dropna());left=set(g.loc[g.Change.eq('Leaver'),'Constituent RIC'].dropna());both=joined&left;joined-=both;left-=both
            issues.extend(dict(date=date,ric=n,issue='leaver_absent') for n in left-members);issues.extend(dict(date=date,ric=n,issue='joiner_present') for n in joined&members)
            members=(members-left)|joined;rows.extend(dict(date=date,ric=n) for n in sorted(members))
        expected=set(latest['Constituent RIC'].map(canonical_ric).dropna())
        records.append(dict(universe=label,issues=issues,latest_exact=members==expected,only_reconstructed=sorted(members-expected),only_latest=sorted(expected-members)))
        return pd.concat([old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates()
    broad=reconstruct('SPX',oldb,pd.read_csv(probe/'changes.csv'),pd.read_csv(probe/'anchor_latest.csv'))
    for date in ('2024-12-31','2025-12-31'):
        current=broad[broad.date.le(date)];actual=set(current[current.date.eq(current.date.max())].ric);expected=set(pd.read_pickle(raw/('broad_'+date+'.pkl'))['Constituent RIC'].map(canonical_ric))
        records.append(dict(universe='SPX_'+date,latest_exact=actual==expected,only_reconstructed=sorted(actual-expected),only_latest=sorted(expected-actual),issues=[]))
    sectors=[]
    for sector in INDICES:
        old=olds[olds.sector.eq(sector)].drop(columns='sector')
        out=reconstruct(sector,old,pd.read_pickle(raw/(sector+'_changes.pkl')),pd.read_pickle(raw/(sector+'_latest.pkl')));out['sector']=sector;sectors.append(out)
        for file in sorted(raw.glob(sector+'_checkpoint_*.pkl')):
            date=file.stem.rsplit('_',1)[1];current=out[out.date.le(date)];actual=set(current[current.date.eq(current.date.max())].ric);expected=set(pd.read_pickle(file)['Constituent RIC'].map(canonical_ric));records.append(dict(universe=sector+'_'+date,latest_exact=actual==expected,only_reconstructed=sorted(actual-expected),only_latest=sorted(expected-actual),issues=[]))
    good=all(r['latest_exact'] and not r['issues'] for r in records)
    write(HERE/'data/membership_qualification.json',dict(passed=good,reconciliation=records,explicit_corrections=corrections,design_sha256=sha(HERE/'design.json')))
    assert good,records
    broad.to_csv(HERE/'data/membership_snapshots.csv.gz',index=False);pd.concat(sectors,ignore_index=True).to_csv(HERE/'data/sector_snapshots.csv.gz',index=False)
    print('MEMBERSHIP_QUALIFIED',len(broad),len(sectors),flush=True)
if __name__=='__main__':main()
