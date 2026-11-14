from common import *
def main():
    names=read(HERE/'data/capture_plan.json')['issues'];jobs=[]
    # Industry code has its own timestamp, distinct from index membership.
    jobs.append(dict(name='gics_history',kind='metadata',universe=['APP.OQ','DD.N'],fields=['TR.GICSSector.date','TR.GICSSector','TR.GICSIndustry.date','TR.GICSIndustry'],parameters={'SDate':'2024-01-01','EDate':END,'Frq':'D'}))
    for sector in ('XLK','XLC','XLI','XLB'):
        for date in ('2025-12-31','2026-03-31','2026-06-30','2026-07-30','2026-07-31','2026-08-03','2026-09-30'):
            jobs.append(dict(name=sector+'_checkpoint_'+date,kind='metadata',universe=[INDICES[sector]],fields=['TR.IndexConstituentRIC','TR.IndexConstituentName'],parameters={'SDate':date}))
    for i in range(0,len(names),30):
        jobs.append(dict(name=f'capital_{i//30:03}',kind='metadata',universe=names[i:i+30],fields=['TR.CAEventID','TR.CACorpActEventType','TR.CACorpActDesc','TR.CAAnnouncementDate','TR.CAExDate','TR.CAEffectiveDate','TR.CAIsRescinded','TR.CATermsOldShares','TR.CATermsNewShares','TR.CAAdjustmentFactor','TR.CAAdjustmentType'],parameters={'SDate':'2023-12-01','EDate':END,'CAEventType':'All'}))
    fields=next(j['fields'] for j in read(HERE/'data/capture_plan.json')['requests'] if j['kind']=='quotes')
    jobs.append(dict(name='event_GRAL_2024',kind='quotes',universe=['GRAL.OQ'],fields=fields,parameters={'SDate':'2024-06-24','EDate':'2024-06-27','Frq':'D'}))
    p=HERE/'data/supplement_plan.json';value=dict(created_utc=now(),design_sha256=sha(HERE/'design.json'),requests=jobs,minimum_spacing_seconds=5,reason='Resolve source events and omitted sector transitions before features/accounts; no outcome-driven policy changes')
    if not p.exists():write(p,value)
    else:assert read(p)['requests']==jobs
    print('SUPPLEMENT_REQUESTS',len(jobs))
if __name__=='__main__':main()
