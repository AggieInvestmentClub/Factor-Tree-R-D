"""Direct characteristic-portfolio learning with a convex economic objective."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,datetime
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
V23=ROOT.parent/'v23_horizon_targets'
SOURCE=Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(V23/'src'))
CHALLENGER='h21_projected_scaled_matched'
MODELS=('direct_risk_only','direct_risk_cost','direct_risk_cost_persist')
ARMS=(*MODELS,'direct_risk_cost_persist_target12','sample_aligned_h21','h21_replay')
GAMMA=5.
FEE=.0002
LAMBDAS=(.1,.01,.001)
EPS=1e-6

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');tmp.replace(p)
def protocol():
    p=ROOT/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip()
    spec=read(p)
    for n,h in spec['code_hashes'].items():assert sha(ROOT/'src'/n)==h,n
    return sha(p)

def freeze():
    p=ROOT/'protocol.json'
    if p.exists():return protocol()
    columns=read(SOURCE/'data/candidate_features/columns.json')
    spec=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      purpose='Replace pointwise regression by direct economic portfolio learning, building from the price-only H21 market-projected challenger on its21-session schedule',
      status='Declared before v28 performance; adaptive next research study on reused history',
      challenger=CHALLENGER,models=list(MODELS),accounts=list(ARMS),features=columns,
      sources=dict(panel=sha(SOURCE/'data/candidate_features/panel.pkl'),labels=sha(V23/'data/horizon_labels.pkl'),
          approval=sha(SOURCE/'data/assembled/approval.json'),v23_protocol=sha(V23/'protocol.json'),
          v23_account_contract=sha(V23/'account_contract.json'),challenger_account=sha(V23/f'accounts/{CHALLENGER}_complete.json')),
      policy=dict(basis='Each of the same36 bounded source characteristics becomes a dollar/beta-neutral portfolio, sized with exact63-session diag50 risk and original12%/gross3/name.15 limits',
          coefficient_constraint='Signed coefficient L1 norm <=1; convex combinations of signed feasible basis portfolios preserve every risk/gross/name/neutrality constraint',
          fast_alpha=.25,persistence_alpha=float(1-2**(-1/21)),persistence='Full policy learns separate weights for fast and causal21-session-half-life characteristic histories; no future feature data',
          risk_budget='12% ceiling; learned portfolio may use less. A separately declared target12 account applies original continuous sizing to the same learned direction, controlling for exposure suppression.',
          annual_refit='2007..2023; prior10 calendar years; observed21-session schedule extended backward from exact2007 challenger anchor',
          training_queries='Whole-date common H5/H10/H21-mature census, restricted to actual21-session dates. No future-label filtering of live target universe.',
          equal_period_weights=True,matched_sample_huber='Additional H21 Huber fit on exactly the economic learners scheduled training dates; original daily EWMA and account allocator'),
      objective=dict(formula='annualized mean(B_t theta dot R_t) - gamma/2 mean(theta.T Q_t theta) - annualized c mean(sum smooth_abs((B_t - diag(1+R_prev) B_prev) theta)) - lambda||theta||2^2/2',
          gamma=GAMMA,cost_per_traded_side=FEE,annual_periods=12,lambdas=list(LAMBDAS),smooth_abs_epsilon=EPS,
          risk='Q_t is252 times daily portfolio-basis covariance known at decision close',
          persistence='Lagged characteristic states enter weights directly and their realized cross-date changes enter the cost loss',
          validation='Last2calendar years of each outer training window; inner training endpoints strictly before validation start; choose lambda by net economic utility, not OOS performance; tie chooses larger penalty. If fewer than20inner or12validation periods, use largest lambda .1 without tuning.',
          solver='Convex smooth objective on signed L1 ball via SLSQP positive/negative split; gradient and Frank-Wolfe gap verified',
          proxy_limits=['Training uses fixed-notional21-session holding returns and drifted-basket turnover, not the full path-dependent ledger',
            'Missing/incomplete predecessor labels or gaps reset turnover to a cash entry; no zero-return substitution',
            'No future holdings are input to policy; realized endpoint returns occur only in training rewards/costs',
            'Common efficacy gate, .2 buffer and daily risk guards are applied by unchanged account engine, not differentiated through training']),
      execution=dict(cadence=21,next_close=True,original_H5_gate=True,volatility_target=.12,gross_cap=3,name_cap=.15,
          buffer_l1=.2,gross_guard=3.3,cost_bps=2,borrow=False,financing=False,unchanged_source_engine=True),
      primary_pairs=[['direct_risk_cost_persist',CHALLENGER],['direct_risk_cost_persist_target12',CHALLENGER],
          ['direct_risk_cost','direct_risk_only'],['direct_risk_cost_persist','direct_risk_cost'],
          ['direct_risk_cost_persist_target12','sample_aligned_h21'],['direct_risk_cost_persist_target12','huber_ridge_10y']],
      inference=dict(primary='2012-2023',early='2007-2011',full='2007-2023',blocks=[20,60],replications=2000,
          correction='Holm6 per endpoint/block, only this declared family; no correction for earlier/adaptive research',promotion=False),
      boundary_exclusive='2024-01-01',code_hashes={q.name:sha(q) for q in sorted((ROOT/'src').glob('*.py')) if q.name!='report.py'},
      limitations=['Prior conditional valuations/source qualifications retained','Modern data reconstruction and reused development history',
          'Linear portfolio policy with two temporal feature banks, not a replication of the papers full nonlinear framework',
          'Persistence comparison adds capacity as well as causal history; no claim to isolate a unique structural persistence parameter'])
    write(p,spec);p.with_suffix('.sha256').write_text(sha(p));return sha(p)

def neutral_basis(values,beta,active,centered,indices):
    """All columns sized by exact source arithmetic; Q annual, full names."""
    n,k=values.shape;out=np.zeros((n,k));mask=np.asarray(active,bool).copy()
    valid=np.zeros(n,bool);valid[indices]=True;mask &= valid & np.isfinite(beta)
    ix=np.flatnonzero(mask)
    if len(ix)<3:return out,np.zeros((k,k)),mask
    rows=np.vstack([np.ones(len(ix)),beta[ix]])
    _,s,v=np.linalg.svd(rows,full_matrices=False)
    rank=np.count_nonzero(s>max(rows.shape)*np.finfo(float).eps*s[0])
    a=np.nan_to_num(values[ix],nan=0,posinf=0,neginf=0)
    d=a-v[:rank].T@(v[:rank]@a)
    # Reorthogonalize before leverage: nearly constant availability histories
    # can otherwise magnify first-pass floating-point neutrality residuals.
    d-=v[:rank].T@(v[:rank]@d)
    pos=np.searchsorted(indices,ix);x=centered[:,pos]
    var=np.sum(x*x,axis=0)/62
    annual=252*(.5*np.sum((x@d)**2,axis=0)/62+.5*np.sum(var[:,None]*d*d,axis=0))
    vol=np.sqrt(np.maximum(annual,0));gross=np.abs(d).sum(axis=0);largest=np.abs(d).max(axis=0)
    requested=np.divide(.12,vol,out=np.zeros(k),where=(vol>1e-10)&(gross>1e-12)&(largest>1e-12))
    g=np.divide(3*min(1,len(ix)/500),gross,out=np.full(k,np.inf),where=gross>1e-12)
    cap=np.divide(.15,largest,out=np.full(k,np.inf),where=largest>1e-12)
    scale=np.minimum(requested,np.minimum(g,cap));scale[scale<requested]*=1-1e-12
    out[ix]=d*scale
    a=out[ix];xb=x@a
    q=252*(.5*xb.T@xb/62+.5*a.T@(var[:,None]*a))
    assert np.max(abs(out.sum(axis=0)),initial=0)<1e-9
    assert np.max(abs(beta[ix]@out[ix]),initial=0)<1e-9
    return out,(q+q.T)/2,mask

def projected_l1(x,radius=1.):
    a=abs(x)
    if a.sum()<=radius:return x.copy()
    s=np.sort(a)[::-1];c=np.cumsum(s);j=np.flatnonzero(s>(c-radius)/(np.arange(len(s))+1))[-1]
    tau=(c[j]-radius)/(j+1)
    return np.sign(x)*np.maximum(a-tau,0)

def update_states(state,raw,indices,mask,alphas):
    """One causal observed-day step; an absent member loses its history."""
    for p,alpha in enumerate(alphas):
        prior=state[p,indices]
        state[p,indices]=(1-alpha)*np.where(np.isfinite(prior),prior,raw)+alpha*raw
        state[p,~mask]=np.nan
    return state

def objective(theta,mu,q,d,penalty,fee,periods):
    loss=-float(mu@theta)+GAMMA/2*float(theta@q@theta)+penalty/2*float(theta@theta)
    grad=-mu+GAMMA*q@theta+penalty*theta
    if fee:
        change=d@theta;s=np.sqrt(change*change+EPS*EPS)
        loss += 12*fee/periods*float(np.sum(s-EPS))
        grad += 12*fee/periods*(d.T@(change/s))
    return loss,grad

def solve(mu,q,d,penalty,fee,periods):
    k=len(mu)
    def f(z):
        value,g=objective(z[:k]-z[k:],mu,q,d,penalty,fee,periods)
        return value,np.r_[g,-g]
    result=minimize(f,np.zeros(2*k),jac=True,method='SLSQP',bounds=[(0,1)]*(2*k),
        constraints=[dict(type='ineq',fun=lambda z:1-z.sum(),jac=lambda z:-np.ones(2*k))],
        options=dict(maxiter=1500,ftol=1e-11))
    theta=result.x[:k]-result.x[k:]
    assert np.abs(theta).sum()<=1+1e-8
    theta=projected_l1(theta)
    value,g=objective(theta,mu,q,d,penalty,fee,periods)
    gap=float(g@theta+np.max(abs(g)))
    if not result.success or gap>2e-5:raise RuntimeError(f'Optimizer not qualified: {result.message}; gap={gap}')
    return theta,dict(success=bool(result.success),iterations=int(result.nit),objective=float(value),gap=gap,l1=float(abs(theta).sum()))

def loss_inputs(data,selected,k):
    """Chronological loss blocks; endpoints are training rewards only."""
    idx=np.asarray(selected,int);b=np.asarray(data['basis'][idx,:,:k],float)
    r=np.asarray(data['returns'][idx],float);active=data['known'][idx]
    assert data['complete'][idx].all() and np.isfinite(r[active]).all()
    r=np.where(active,r,0.)
    mu=12*np.einsum('tnk,tn->k',b,r)/len(idx)
    q=np.asarray(data['risk'][idx,:k,:k],float).mean(axis=0)
    delta=b.copy();resets=0
    for j in range(len(idx)):
        if j and idx[j]==idx[j-1]+1:
            delta[j] -= (1+r[j-1,:,None])*b[j-1]
        else:resets+=1
    d=delta.reshape(-1,k);d=d[np.any(d!=0,axis=1)]
    return mu,q,np.ascontiguousarray(d),dict(periods=len(idx),cash_entry_resets=resets)

if __name__=='__main__':print(freeze())
