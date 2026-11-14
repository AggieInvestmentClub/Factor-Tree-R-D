from common import *
def main():
    primary=read(HERE/'data/capture_plan.json');supp=read(HERE/'data/supplement_plan.json');quote=next(j for j in primary['requests'] if j['kind']=='quotes');jobs=[]
    for year in (2023,2024,2025,2026):jobs.append(dict(name=f'XLRE_explicit_{year}',kind='quotes',universe=['XLRE.P'],fields=quote['fields'],parameters={'SDate':f'{year}-01-01','EDate':f'{year}-12-31' if year<2026 else END,'Frq':'D'}))
    for kind,original in [('distributions',next(j for j in primary['requests'] if j['kind']=='distributions')),('capital',next(j for j in supp['requests'] if j['name'].startswith('capital_')))]:jobs.append(dict(original,name='XLRE_explicit_'+kind,universe=['XLRE.P']))
    p=HERE/'data/xlre_repair_plan.json'
    if not p.exists():write(p,dict(created_utc=now(),design_sha256=sha(HERE/'design.json'),minimum_spacing_seconds=5,requests=jobs,reason='Bare XLRE did not resolve. Use already verified original-source alias XLRE.P, ISIN US81369Y8600; same ETF, no universe substitution.'))
    else:assert read(p)['requests']==jobs
if __name__=='__main__':main()
