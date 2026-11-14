from common import *
import subprocess,argparse,time

STAGES=[('membership','membership.py',[]),('assemble','assemble.py',['assemble']),('census','census.py',[]),('events','qualify_events.py',[]),('source_qa','source_qa.py',[]),('features','features.py',[]),('labels','labels.py',[]),('gate','gate.py',[]),('arrays','build_arrays.py',[]),('fit','portfolio.py',['fit']),('targets','portfolio.py',['prepare']),('accounts','portfolio.py',['accounts']),('assess','assess.py',[])]

def main(first):
    logs=HERE/'logs';logs.mkdir(exist_ok=True);active=False
    for stage,file,args in STAGES:
        active=active or stage==first
        if not active:continue
        begin=time.monotonic();print('STAGE_BEGIN',stage,now(),flush=True);path=logs/(stage+'.log')
        with path.open('w') as stream:result=subprocess.run([sys.executable,'-u',str(HERE/'src'/file),*args],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            print('STAGE_FAILED',stage,'exit',result.returncode,flush=True);print(path.read_text()[-9000:],flush=True);raise SystemExit(result.returncode)
        print('STAGE_DONE',stage,round(time.monotonic()-begin,1),flush=True)
    write(HERE/'reports/code_manifest.json',dict(created_utc=now(),design_sha256=sha(HERE/'design.json'),files={p.name:sha(p) for p in (HERE/'src').glob('*.py')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--from-stage',choices=[s[0] for s in STAGES],default='membership');a=p.parse_args();main(a.from_stage)
