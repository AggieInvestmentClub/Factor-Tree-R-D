from common import *
import pandas as pd
def main():
    p=HERE/'design.json'
    if not p.exists():
        files=[V31/'protocol.json',V31/'src/learner.py',V31/'src/allocation.py',V29/'src/policy.py',V34/'protocol.json',V34/'src/risk_learning.py',V36/'protocol.json',V36/'account_inputs/fixed_corr_long50_desired.pkl',V38/'protocol.json',V38/'src/exits.py',V38/'src/exit_account.py',SOURCE/'data/assembled/approval.json']
        write(p,dict(created_utc=now(),user_authorized='Test the benchmark and volatility-scaled stop from2024 to present; expands prior pre2024 research boundary only in this isolated study',start=START,end=END,exclusive_boundary=BOUNDARY,
            accounts=['benchmark','vol2sigma'],benchmark='Exact v38 none / v36 fixed_corr_long50: v31 log-growth RFF learner, unchanged58features32randomfeatures; three correlation-factor constraints on alpha; market-bias cap.50,12% shared forecast risk',
            training='Preserve annual10year rolling refits,100Adam epochs,regularization.1/.01 chosen on prior2years with original repaired daily-engine log growth; strict mature inner/outer labels. Refit2024 on pre2024 only,2025 on mature pre2025,2026 on mature pre2026. No selection of architectures,stops,risk limits or policy based on forward results.',
            gate='Preserve original H5 Huber-ridge efficacy gate from strictly matured whole-date OOS forecasts; original21observed-session forecast cadence continues from2007 anchor',
            account_start='Both accounts start in cash at first2024 execution session. Preserve pre2024 forecast/feature history and original cadence. Stop entry basis begins with actual new-account fills. No optimization of initial holdings or entry timing.',
            stop='Frozen v38 vol2sigma:2*entry-time63session daily sigma*sqrt21,clipped3%-20%;adjusted close trigger,nextclose actual execution;exclude until next original21session forecast;existing risk-preserving reprojection;no profit target',
            constraints='Same2bps/side,gross3 scaled active/500,name.15,buffer.2,dailybeta/net.05 guards,grossguard3.3,no cash borrowing,zero borrow/funding/cashyield. Last session marked with inherited terminal liquidation convention; report it.',
            source='Extend historical membership and sectors via dated anchors plus joiner/leaver events; separately dated prices,raw cash-conversion quotes,distribution/terminal audit; preserve original pre2024 sources. No survivor-only substitution,no missing-return zero imputation,no future-label selection of prediction universe.',
            holdout='Later-period walk-forward validation of this frozen design,not pristine holdout:2024+ was examined in older110stock project. Current v31-v38 fitting/results used only pre2024. No performance viewed in v39 before this design declaration.',
            inference='One paired stop-versus-benchmark comparison;annual loggrowth primary;2000paired circular block bootstraps20/60sessions;CAGR,Sharpe,drawdown,turnover,exposure,2024/2025/2026partial descriptive. No automatic promotion.',
            source_hashes={str(f):sha(f) for f in files}));p.with_suffix('.sha256').write_text(sha(p))
    assert sha(p)==p.with_suffix('.sha256').read_text().strip()
    sys.path.insert(0,str(SOURCE/'src'));from historical_sources import canonical_ric
    raw=HERE/'data/source_probe/host';anchor=pd.read_csv(raw/'anchor_2023.csv');changes=pd.read_csv(raw/'changes.csv');latest=pd.read_csv(raw/'anchor_latest.csv')
    for frame in (anchor,changes,latest):frame['Constituent RIC']=frame['Constituent RIC'].map(canonical_ric)
    members=set(anchor['Constituent RIC'].dropna());initial=members.copy();allnames=set(members);snapshots=[dict(date='2023-12-29',ric=n) for n in sorted(members)];issues=[]
    for date,g in changes.groupby('Date',sort=True):
        joined=set(g.loc[g.Change.eq('Joiner'),'Constituent RIC'].dropna());left=set(g.loc[g.Change.eq('Leaver'),'Constituent RIC'].dropna());both=joined&left;joined-=both;left-=both
        issues.extend(dict(date=date,ric=n,issue='leaver_absent') for n in left-members);issues.extend(dict(date=date,ric=n,issue='joiner_present') for n in joined&members);members=(members-left)|joined;allnames|=members;snapshots.extend(dict(date=date,ric=n) for n in sorted(members))
    expected=set(latest['Constituent RIC'].dropna());allnames|=expected;pd.DataFrame(snapshots).to_csv(HERE/'data/membership_raw_snapshots.csv.gz',index=False)
    write(HERE/'data/membership_precheck.json',dict(passed=not issues and members==expected,anchor=len(initial),latest=len(members),unique_issues=len(allnames),events=len(changes),latest_anchor_exact=members==expected,issues=issues,only_reconstructed=sorted(members-expected),only_latest=sorted(expected-members),original_anchor_sha256=sha(raw/'anchor_2023.csv'),scope='Acquisition roster only; reconciliation required before accounts'))
    requests=[];identity_fields=['TR.IndexConstituentRIC','TR.IndexConstituentName'];change_fields=['TR.IndexJLConstituentChangeDate','TR.IndexJLConstituentRIC','TR.IndexJLConstituentName','TR.IndexJLConstituentituentChange']
    for date in ('2024-12-31','2025-12-31'):requests.append(dict(name='broad_'+date,kind='metadata',universe=['.SPX'],fields=identity_fields,parameters={'SDate':date}))
    for sector,index in INDICES.items():
        requests.append(dict(name=sector+'_latest',kind='metadata',universe=[index],fields=identity_fields,parameters={'SDate':END}));requests.append(dict(name=sector+'_changes',kind='metadata',universe=[index],fields=change_fields,parameters={'SDate':'2024-01-01','EDate':END,'IC':'B'}))
    names=sorted(allnames|set(INDICES)|{'IYZ','SPY'});fields=[item for f in ('TR.PriceOpen','TR.PriceHigh','TR.PriceLow','TR.PriceClose','TR.Volume','TR.TotalReturn1D') for item in (f+'.date',f)]+['TR.ClosePrice.date','TR.ClosePrice(Adjusted=0)','TR.ClosePrice(Adjusted=1)','TR.OpenPrice.date','TR.OpenPrice(Adjusted=0)','TR.OpenPrice(Adjusted=1)']
    for year in (2023,2024,2025,2026):
        for i in range(0,len(names),20):requests.append(dict(name=f'quotes_{year}_{i//20:03}',kind='quotes',universe=names[i:i+20],fields=fields,parameters={'SDate':f'{year}-01-01','EDate':f'{year}-12-31' if year<2026 else END,'Frq':'D'}))
    for i in range(0,len(names),30):requests.append(dict(name=f'distributions_{i//30:03}',kind='distributions',universe=names[i:i+30],fields=['TR.DivExDate','TR.DivAdjustedGross','TR.DivUnadjustedGross','TR.DivType'],parameters={'SDate':'2024-01-01','EDate':END,'Frq':'C'}))
    plan=HERE/'data/capture_plan.json'
    if not plan.exists():write(plan,dict(created_utc=now(),design_sha256=sha(p),requests=requests,issues=names,minimum_spacing_seconds=5,automatic_retries=0,transport='Explicit existing local LSEG desktop; no credentials/config reads or platform fallback'))
    else:assert read(plan)['requests']==requests
    print('FROZEN_DESIGN',sha(p),'NAMES',len(names),'REQUESTS',len(requests),flush=True)
if __name__=='__main__':main()
