from common import *
def main():
    fields=next(j['fields'] for j in read(HERE/'data/capture_plan.json')['requests'] if j['kind']=='quotes')
    jobs=[dict(name='event_NXT_2024',kind='quotes',universe=['NXT.OQ'],fields=fields,parameters={'SDate':'2024-01-02','EDate':'2024-01-04','Frq':'D'})]
    p=HERE/'data/event_quotes_plan.json'
    if not p.exists():write(p,dict(created_utc=now(),design_sha256=sha(HERE/'design.json'),minimum_spacing_seconds=5,requests=jobs,reason='Complete corporate-action census: Flex mandatory Nextracker distribution; no strategy outcomes inspected'))
    else:assert read(p)['requests']==jobs
if __name__=='__main__':main()
