"""Fit the frozen 2024 portfolio learner without accessing forward observations."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('v39_config',Path(__file__).with_name('common.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.sys.path.insert(0,str(c.V31/'src'))
from learner import train,predict
from settings import load,selected,np,pd,PENALTIES
from runner import inputs,validation
from policy import torch
import time,hashlib

def main():
    done=c.HERE/'models/log_growth_2024.json'
    if done.exists():
        assert c.read(done)['design_sha256']==c.sha(c.HERE/'design.json');return
    began=time.monotonic();d=load();cut=pd.Timestamp('2024-01-01');start=pd.Timestamp('2014-01-01');split=pd.Timestamp('2022-01-01')
    full=selected(d,cut,start);inner=selected(d,split,start);val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut))
    assert d['dates'].max()<cut and d['ends'][full].max()<cut and d['ends'][inner].max()<split
    loaded=inputs();choices=[]
    for penalty in PENALTIES:
        model,stats=train(d,inner,'log_growth',penalty)
        target=pd.DataFrame(predict(model,d,val),index=d['dates'][val],columns=d['names'])
        v=validation(loaded,target,loaded['clock'][-1],'log_growth')
        choices.append(dict(penalty=penalty,validation=v,training=stats));print('INNER_2024',penalty,flush=True)
    penalty=max(choices,key=lambda x:x['validation']['utility'])['penalty']
    model,stats=train(d,full,'log_growth',penalty)
    file=done.with_suffix('.pt');file.parent.mkdir(exist_ok=True);torch.save(model.state_dict(),file)
    c.write(done,dict(year=2024,objective='log_growth',design_sha256=c.sha(c.HERE/'design.json'),weights_sha256=c.sha(file),
        training_periods=len(full),training_query_sha256=hashlib.sha256(d['dates'][full].asi8.tobytes()).hexdigest(),
        max_training_label_end=str(d['ends'][full].max()),refit_boundary=str(cut),inner_max_training_label_end=str(d['ends'][inner].max()),
        validation_start=str(split),selection=choices,selected_penalty=penalty,fallback=False,training=stats,seconds=time.monotonic()-began,
        source_data_manifest_sha256=c.sha(c.V29/'data_manifest.json'),forward_data_accessed=False))
    print('FIT_2024_COMPLETE',round(time.monotonic()-began,1),flush=True)
if __name__=='__main__':main()
