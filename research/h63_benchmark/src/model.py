"""Frozen H63 ranker and the original long-only capital policy."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy.stats import rankdata
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT=Path(__file__).resolve().parents[1]

class Policy(torch.nn.Module):
    def __init__(self,kind='replay_random'):
        super().__init__();torch.manual_seed(20261111)
        self.linear=torch.nn.Linear(58,1,bias=False)
        torch.nn.init.normal_(self.linear.weight,std=.03)
        self.register_buffer('frequency',torch.randn(58,32)/np.sqrt(58))
        self.register_buffer('phase',torch.rand(32)*2*np.pi)
        self.nonlinear=torch.nn.Linear(32,1,bias=False)
        torch.nn.init.normal_(self.nonlinear.weight,std=.01)
    def forward(self,x):
        return (self.linear(x)+self.nonlinear(torch.cos(x@self.frequency+self.phase))).squeeze(-1)

# The preserved capital-training objective uses these original names.
import types
p=types.SimpleNamespace(Policy=Policy)
def limits(mode='long_only'):
    if mode!='long_only':raise ValueError('This release supports H63 long-only only')
    return dict(long_only=True,min_net=0.,max_net=.99,max_abs_beta=1.,guard_tolerance=.05)
def gross_cap(mode='long_only'):limits(mode);return 1.

def load(root):
    root=Path(root);d=json.loads((root/'arrays/metadata.json').read_text())
    d['dates']=pd.DatetimeIndex(d['dates']);d['ends']=pd.DatetimeIndex(d['label_ends'])
    for k in ('x','centered','diag','mask','beta','returns','complete','query_context','returns_h63','market_h63','ends_h63','common_complete'):
        d[k]=np.load(root/f'arrays/{k}.npy',mmap_mode='r')
    return d

def design(d,ix):
    x=np.array(d['x'][ix],dtype=np.float32,copy=True);z=d['query_context'][ix]
    return np.concatenate([x,np.broadcast_to(z[:,None,:],(len(ix),len(d['names']),8))],axis=2)

def contract(w,d,j,mask=None,beta=None):
    mask=d['mask'][j] if mask is None else mask;beta=d['beta'][j] if beta is None else beta
    w=np.maximum(np.where(mask,w,0.),0.);c=np.array(d['centered'][j])
    variance=252*(.5*np.sum((c@w)**2)/62+.5*np.sum(d['diag'][j]*w*w))
    net=float(w.sum());b=float(w@beta)
    scale=min(1.,.12/max(np.sqrt(variance),1e-15),min(mask.sum()/500,1)/max(abs(w).sum(),1e-15),.15/max(abs(w).max(initial=0),1e-15),.99/max(net,1e-15),1/max(-net,1e-15),1/max(abs(b),1e-15))
    return w*scale*(1-1e-12)

def transported(rank,capital,mask):
    out=np.zeros_like(capital);active=np.flatnonzero(mask)
    order=active[np.argsort(rank[active],kind='stable')];out[order]=np.sort(capital[active]);return out

def targets(input_root,out):
    d=load(input_root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];dates=[];errors=[]
    for year in range(2007,2027):
        ix=np.flatnonzero(d['dates'].year==year);m=Policy();m.load_state_dict(torch.load(ROOT/f'checkpoints/capital/{year}.pt',weights_only=True))
        with torch.no_grad():capital=m(torch.from_numpy(np.array(d['x'][ix]))).numpy()
        ranker=lgb.Booster(model_file=str(ROOT/f'checkpoints/ranker/{year}.txt'));x=design(d,ix)
        rank=ranker.predict(x.reshape(-1,66),num_threads=1).reshape(x.shape[:2])
        for key,value in [('capital',capital),('ranker',rank)]:
            ref=np.load(Path(input_root)/f'reference/{key}_{year}.npy');error=float(abs(value-ref).max());assert error<1e-10,(key,year,error)
            errors.append(dict(year=year,component=key,max_score_error=error))
        for j,r,c in zip(ix,rank,capital):
            rows.append(contract(3*transported(r,c,d['mask'][j])/max(1,d['mask'][j].sum()),d,j));dates.append(d['dates'][j])
        if year==2024:
            j=np.flatnonzero(d['dates']<pd.Timestamp('2024-01-01'))[-1];x=design(d,np.array([j]))
            first_rank=ranker.predict(x.reshape(-1,66),num_threads=1)
            with torch.no_grad():first_capital=m(torch.from_numpy(np.array(d['x'][[j]]))).numpy()[0]
            f=contract(3*transported(first_rank,first_capital,d['mask'][j])/max(1,d['mask'][j].sum()),d,j)
            beta=pd.read_pickle(Path(input_root)/'account/beta.pkl').loc['2023-12-29',d['names']].to_numpy()
            eligible=pd.read_pickle(Path(input_root)/'account/eligible.pkl').loc['2023-12-29',d['names']].to_numpy(bool)
            mask=d['mask'][j]&eligible&np.isfinite(beta)
            first=contract(f,d,j,mask,np.where(mask,beta,0.))
    table=pd.DataFrame(rows,index=pd.DatetimeIndex(dates),columns=d['names'])
    h=table[table.index<'2024-01-01'];f=pd.concat([pd.DataFrame([first],index=pd.DatetimeIndex(['2023-12-29']),columns=d['names']),table[table.index>='2024-01-01']])
    for part,frame in [('historical',h),('forward',f)]:
        ref=pd.read_pickle(Path(input_root)/f'reference/{part}_desired.pkl')
        assert frame.index.equals(ref.index) and frame.columns.equals(ref.columns)
        error=float(abs(frame.to_numpy()-ref.to_numpy()).max());assert error<1e-10,(part,error)
        assert np.isfinite(frame.to_numpy()).all() and frame.to_numpy().min()>=0 and frame.sum(1).max()<=.99+1e-10
        frame.to_pickle(out/f'{part}_desired.pkl');errors.append(dict(part=part,max_target_error=error,rows=len(frame)))
    return dict(passed=True,checks=errors)

def fit_ranker(input_root,year,out):
    d=load(input_root);cut=pd.Timestamp(year,1,1)
    ix=np.flatnonzero(d['common_complete']&(d['dates']<cut)&(d['dates']>=cut-pd.DateOffset(years=10))&(d['ends_h63']<cut.to_datetime64()))
    mask=d['mask'][ix];values=d['returns_h63'][ix]-d['beta'][ix]*d['market_h63'][ix,None];ys=[]
    for row,valid in zip(values,mask):
        r=rankdata(row[valid],method='average');ys.extend(np.minimum(4,np.floor(5*(r-.5)/len(r))).astype(np.int32))
    meta=json.loads((ROOT/f'checkpoints/ranker/{year}.json').read_text());model=lgb.LGBMRanker(**meta['parameters'])
    model.fit(design(d,ix)[mask],np.asarray(ys),group=mask.sum(1))
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.booster_.save_model(str(out/f'{year}.txt'))
    query=np.flatnonzero(d['dates'].year==year);x=design(d,query);scores=model.booster_.predict(x.reshape(-1,66),num_threads=1).reshape(x.shape[:2])
    error=float(abs(scores-np.load(Path(input_root)/f'reference/ranker_{year}.npy')).max());assert error<1e-10
    return dict(year=year,queries=len(ix),rows=int(mask.sum()),max_label_end=str(d['ends_h63'][ix].max()),max_refit_score_error=error)
