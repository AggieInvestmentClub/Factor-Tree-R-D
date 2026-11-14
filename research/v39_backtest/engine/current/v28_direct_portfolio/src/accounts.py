"""Original accounting engine; exact unchanged challenger replay and new policies."""
from core import *
import pandas as pd,importlib.util,time,gc,argparse
import horizon_accounts as bridge
from lazy_covariances import LazyCovariances

def source_gate():return pd.read_csv(V23/'account_inputs/original_huber_gate_cadence21.csv',index_col=0,parse_dates=True)
def contract():
    return dict(protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'),
        source_status=bridge.legacy.sources(),gate_sha256=sha(V23/'account_inputs/original_huber_gate_cadence21.csv'))

def prepare():
    from fit_policy import load_data
    data=load_data();meta=contract();gate=source_gate();dates=data['dates'];ix=np.flatnonzero(dates>=pd.Timestamp(data['anchor']))
    assert dates[ix].equals(gate.index);names=data['names'];(ROOT/'account_inputs').mkdir(exist_ok=True)
    diagnostic=[]
    for name in MODELS:
        chunks=[];bindings={}
        for year in range(2007,2024):
            info=read(ROOT/f'models/{name}_{year}.json');assert info['protocol_sha256']==meta['protocol_sha256']
            file=ROOT/f'account_inputs/{name}_{year}.npy';assert sha(file)==info['targets_sha256']
            chunks.append(np.load(file));bindings[str(year)]=sha(ROOT/f'models/{name}_{year}.json')
        values=np.concatenate(chunks);assert values.shape==(len(ix),len(names))
        if name==MODELS[-1]:rescaled=values.copy()
        for row,j in enumerate(ix):
            theta=np.load(ROOT/f'models/{name}_{dates[j].year}.npy');k=len(theta)
            assert np.max(abs(data['basis'][j,:,:k]@theta-values[row]))<1e-12
            active=data['feasible'][j];beta=data['beta'][j]
            vol=float(np.sqrt(max(0,theta@data['risk'][j,:k,:k]@theta)))
            gross=float(abs(values[row]).sum());peak=float(abs(values[row]).max())
            limit=3*min(1,active.sum()/500)
            assert vol<=.12+1e-8 and gross<=limit+1e-8 and peak<=.15+1e-8
            assert abs(values[row].sum())<1e-9 and abs(values[row,active]@beta[active])<1e-9
            assert np.max(abs(values[row,~active]),initial=0)<1e-12
            diagnostic.append(dict(arm=name,date=str(dates[j].date()),pre_gate_vol=vol,pre_gate_gross=gross,gate=gate.multiplier.iloc[row]))
            if name==MODELS[-1] and vol>1e-10 and gross>1e-12:
                requested=.12/vol;scale=min(requested,limit/gross,.15/peak)
                if scale<requested:scale*=1-1e-12
                rescaled[row]*=scale
        target=pd.DataFrame(values,index=gate.index,columns=names).mul(gate.multiplier,axis=0)
        file=ROOT/f'account_inputs/{name}_desired.pkl';target.to_pickle(file)
        write(ROOT/f'account_inputs/{name}_prepared.json',dict(**meta,arm=name,target_sha256=sha(file),model_bindings=bindings,scheduled_dates=len(target)))
    name='direct_risk_cost_persist_target12';file=ROOT/f'account_inputs/{name}_desired.pkl'
    pd.DataFrame(rescaled,index=gate.index,columns=names).mul(gate.multiplier,axis=0).to_pickle(file)
    write(ROOT/f'account_inputs/{name}_prepared.json',dict(**meta,arm=name,target_sha256=sha(file),model_bindings=bindings,scheduled_dates=len(gate),risk_rescaling=True))
    name='h21_replay';file=ROOT/f'account_inputs/{name}_desired.pkl'
    baseline=V23/f'account_inputs/{CHALLENGER}_desired.pkl';reference=pd.read_pickle(baseline)
    assert reference.index.equals(gate.index) and list(reference.columns)==names
    reference.to_pickle(file)
    write(ROOT/f'account_inputs/{name}_prepared.json',dict(**meta,arm=name,target_sha256=sha(file),reference_target_sha256=sha(baseline),scheduled_dates=len(reference)))
    pd.DataFrame(diagnostic).to_csv(ROOT/'reports/target_diagnostics.csv',index=False)
    # Sample-aligned regression retains literal source forecast->rank->risk path.
    loaded=bridge.legacy.inputs();cov=LazyCovariances(V23/'account_inputs/covariances_cadence21')
    frames=[];bindings={}
    for year in range(2007,2024):
        file=ROOT/f'forecast_runs/sample_aligned_h21_{year}.csv.gz';info=read(ROOT/f'models/sample_aligned_h21_{year}.json')
        assert sha(file)==info['forecast_sha256'] and info['protocol_sha256']==meta['protocol_sha256']
        frames.append(pd.read_csv(file,parse_dates=['decision_date','refit_boundary','max_training_label_end']));bindings[str(year)]=sha(file)
    predictions=pd.concat(frames,ignore_index=True)
    targets,audit,smooth=bridge.build_targets(predictions,loaded['panel'],loaded['clock'],names,cov,anchor=loaded['anchor'],cadence=21)
    targets=targets.mul(gate.multiplier,axis=0);file=ROOT/'account_inputs/sample_aligned_h21_desired.pkl';targets.to_pickle(file)
    smooth.to_pickle(ROOT/'account_inputs/sample_aligned_h21_smoothed.pkl')
    audit.to_csv(ROOT/'reports/sample_aligned_h21_risk.csv.gz',index=False)
    write(ROOT/'account_inputs/sample_aligned_h21_prepared.json',dict(**meta,arm='sample_aligned_h21',target_sha256=sha(file),forecast_bindings=bindings,scheduled_dates=len(targets)))
    print('ALL_TARGETS_PREPARED',len(ARMS),flush=True)

def run(arms):
    meta=contract();loaded=bridge.legacy.inputs(need_prices=True);del loaded['panel'];gc.collect()
    out=ROOT/'accounts';out.mkdir(exist_ok=True)
    for arm in arms:
        done=out/f'{arm}_complete.json'
        if done.exists():
            old=read(done);assert old['protocol_sha256']==meta['protocol_sha256'] and old['ledger_sha256']==sha(out/f'{arm}_ledger.csv.gz');continue
        prepared=read(ROOT/f'account_inputs/{arm}_prepared.json');file=ROOT/f'account_inputs/{arm}_desired.pkl'
        for k,v in meta.items():assert prepared[k]==v,k
        assert sha(file)==prepared['target_sha256'];targets=pd.read_pickle(file)
        start=time.monotonic();print('ACCOUNT_BEGIN',arm,flush=True)
        noncash=bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));rows=0
        def consume(frame):
            nonlocal rows
            noncash.consume(frame);rows+=len(frame)
        ledger,positions,actual,decisions,audit=bridge.account_engine.simulate_stock_portfolio(
            loaded['prices'],targets,loaded['events'],start=loaded['clock'][loaded['clock'].get_loc(loaded['anchor'])+1],end=loaded['clock'][-1],
            cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],neutral=True,dollar_neutral=True,beta_neutral=True,
            eligibility=loaded['eligible'],position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff='2024-01-01',
            record_positions=True,position_sink=consume,distribution_claims=loaded['claims'])
        assert positions.empty and rows==audit['streamed_position_rows'] and audit['arithmetic_passed']
        audit['noncash_exposure_audit']=noncash.report()
        if noncash.rows:audit['issues'].extend(noncash.rows);audit['validated_performance']=False
        parity=None
        if arm=='h21_replay':
            ref=pd.read_csv(V23/f'accounts/{CHALLENGER}_ledger.csv.gz',index_col=0,parse_dates=True)
            numeric=ledger.select_dtypes(include='number').columns.intersection(ref.select_dtypes(include='number').columns)
            assert ledger.index.equals(ref.index)
            parity={c:float(np.nanmax(abs(ledger[c].to_numpy()-ref[c].to_numpy()),initial=0)) for c in numeric}
            assert max(parity.values())<1e-10
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        write(done,dict(**meta,arm=arm,target_sha256=prepared['target_sha256'],prepared_sha256=sha(ROOT/f'account_inputs/{arm}_prepared.json'),
            audit=audit,ledger_sha256=sha(out/f'{arm}_ledger.csv.gz'),reference_ledger_parity=parity,seconds=time.monotonic()-start,
            state='validated' if audit['validated_performance'] else 'conditional research performance; unresolved valuation sessions'))
        print('ACCOUNT_DONE',arm,round(time.monotonic()-start,1),flush=True)
        del ledger,positions,actual,decisions,targets;gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));a=p.parse_args()
    (ROOT/'reports').mkdir(exist_ok=True)
    if a.action=='prepare':prepare()
    else:run(a.arms)
