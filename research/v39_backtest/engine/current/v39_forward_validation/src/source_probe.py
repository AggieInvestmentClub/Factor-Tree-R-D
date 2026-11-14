"""Bounded read-only desktop data check for the requested 2024+ validation."""
import sys,contextlib,datetime,hashlib,importlib.metadata,io,json,logging,time
from pathlib import Path
from urllib.parse import urlparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'data/source_probe';OUT.mkdir(parents=True,exist_ok=True)
def save(p,o):p.write_text(json.dumps(o,indent=2,default=str),encoding='utf-8')
def main():
    suffix='host' if '--host' in sys.argv else 'sandbox';out=OUT/suffix;out.mkdir(exist_ok=True);record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Requested forward validation; explicit local LSEG desktop only; no credentials/config reads, no platform fallback',results=[]);sink=io.StringIO();logging.disable(logging.CRITICAL);session=None
    plans=[('anchor_2023',['.SPX'],['TR.IndexConstituentRIC','TR.IndexConstituentName'],{'SDate':'2023-12-29'}),('anchor_latest',['.SPX'],['TR.IndexConstituentRIC','TR.IndexConstituentName'],{'SDate':'2026-11-11'}),('changes',['.SPX'],['TR.IndexJLConstituentChangeDate','TR.IndexJLConstituentRIC','TR.IndexJLConstituentName','TR.IndexJLConstituentituentChange'],{'SDate':'2024-01-01','EDate':'2026-11-11','IC':'B'}),('latest_spy',['SPY'],['TR.PriceClose.date','TR.PriceClose'],{'SDate':'2026-11-05','EDate':'2026-11-11','Frq':'D'})]
    try:
        with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            record['stage']='sdk_version';record['sdk_version']=importlib.metadata.version('lseg-data');assert record['sdk_version']=='2.1.1';import lseg.data as ld
            from lseg.data import _configure as cfg
            record['stage']='config_initialization';assert cfg._config is None;cfg._read_config_file=lambda path:{};c=cfg.get_config();c.set_param('http.auto-retry.number-of-retries',1);c.set_param('http.max-connections',1);c.set_param('http.max-keepalive-connections',0);c.set_param('http.request-timeout',20)
            record['stage']='desktop_definition';session=ld.session.desktop.Definition(app_key=None).get_session();assert urlparse(str(session._get_base_url())).hostname in ('localhost','127.0.0.1','::1');record['stage']='desktop_open';state=session.open();record['open_state']=state.name;assert state.name=='Opened';ld.session.set_default(session)
        record['opened']=True
        for name,universe,fields,parameters in plans:
            began=time.monotonic();rec=dict(name=name,fields=fields,parameters=parameters);path=out/f'{name}.csv'
            with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):frame=ld.get_data(universe,fields=fields,parameters=parameters)
            frame.to_csv(path,index=False);rec.update(rows=len(frame),columns=list(frame.columns),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-began);record['results'].append(rec);save(out/'status.json',record);print(json.dumps(rec),flush=True);time.sleep(2)
        record['status']='completed'
    except Exception as exc:
        message=str(exc).lower();category='connection' if any(x in message for x in ('connect','port','proxy','timeout')) else 'entitlement' if any(x in message for x in ('permission','forbidden','401','403','entitled')) else 'other';record.update(status='failed',error_type=type(exc).__name__,error_category=category,raw_exception_saved=False);print('PROBE_FAILED',category,type(exc).__name__,flush=True)
    finally:
        if session:
            with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):session.close()
        record['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(out/'status.json',record)
if __name__=='__main__':main()
