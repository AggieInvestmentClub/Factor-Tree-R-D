from common import *
import contextlib,io,logging,time,importlib.metadata,argparse
from urllib.parse import urlparse
def main(maximum,planfile='capture_plan.json'):
    p=HERE/'data'/planfile;plan=read(p);assert sha(HERE/'design.json')==plan['design_sha256'];out=HERE/'data/raw';out.mkdir(exist_ok=True);sink=io.StringIO();logging.disable(logging.CRITICAL);session=None;count=0
    try:
        with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            assert importlib.metadata.version('lseg-data')=='2.1.1';import lseg.data as ld
            from lseg.data import _configure as cfg
            assert cfg._config is None;cfg._read_config_file=lambda path:{};c=cfg.get_config();c.set_param('http.auto-retry.number-of-retries',1);c.set_param('http.max-connections',1);c.set_param('http.max-keepalive-connections',0);c.set_param('http.request-timeout',60)
            session=ld.session.desktop.Definition(app_key=None).get_session();assert urlparse(str(session._get_base_url())).hostname in ('localhost','127.0.0.1','::1');assert session.open().name=='Opened';ld.session.set_default(session)
        for job in plan['requests']:
            f=out/(job['name']+'.pkl');m=f.with_suffix('.json')
            if m.exists():assert f.exists() and sha(f)==read(m)['response_sha256'];continue
            if count>=maximum:break
            rec=dict(job=job,started_utc=now(),capture_plan_sha256=sha(p));began=time.monotonic()
            try:
                with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):frame=ld.get_data(job['universe'],fields=job['fields'],parameters=job['parameters'],header_type=ld.HeaderType.NAME if job['kind']=='quotes' else ld.HeaderType.TITLE)
                temp=f.with_suffix('.pkl.partial');frame.to_pickle(temp);temp.replace(f);rec.update(rows=len(frame),columns=list(frame.columns),response_sha256=sha(f),seconds=time.monotonic()-began);write(m,rec);print('CAPTURED',job['name'],len(frame),round(rec['seconds'],1),flush=True)
            except Exception as exc:
                message=str(exc).lower();category='rate_limit' if '429' in message or 'limit' in message else 'entitlement' if any(x in message for x in ('401','403','entitled','forbidden')) else 'connection' if any(x in message for x in ('timeout','connect','proxy')) else 'other'
                code=getattr(exc,'code',None);code=code if isinstance(code,(int,float)) else None
                flags=[s for s in ('timeout','server','temporarily','unavailable','backend','permission','bad request','invalid','empty','session','service','socket','connection','limit','400','401','403','429','500','502','503','504') if s in message]
                rec.update(error_type=type(exc).__name__,error_category=category,numeric_code=code,message_flags=flags);write(out/(job['name']+'_failure.json'),rec);print('CAPTURE_FAILED',job['name'],category,type(exc).__name__,code,flags,flush=True);return
            count+=1;time.sleep(plan['minimum_spacing_seconds'])
        write(p.with_name(p.stem+'_progress.json'),dict(updated_utc=now(),captured=sum((out/(j['name']+'.json')).exists() for j in plan['requests']),requested=len(plan['requests']),new_calls=count))
    finally:
        if session:
            with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):session.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--maximum',type=int,default=1000);p.add_argument('--plan',default='capture_plan.json');a=p.parse_args();main(a.maximum,a.plan)
