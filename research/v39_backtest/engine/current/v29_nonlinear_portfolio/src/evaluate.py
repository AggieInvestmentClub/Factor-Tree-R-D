from common import *
from policy import Policy,predict,tensors,weights,torch
from accounts import contract
from evaluate_horizons import paired_comparisons,PERIODS
from analyze_state_v20 import load_ledger,performance
from lazy_covariances import LazyCovariances
from risk_engine import allocate_target

def validate():
    seal=protocol();spec=read(ROOT/'protocol.json');dm=read(ROOT/'data_manifest.json')
    for p,h in spec['sources'].items():assert sha(p)==h,p
    for p,h in dm['files'].items():assert sha(ROOT/'data'/p)==h,p
    d=load();rows=[];mx=0.;gate=pd.read_csv(V23/'account_inputs/original_huber_gate_cadence21.csv',index_col=0,parse_dates=True)
    ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));assert d['dates'][ix].equals(gate.index)
    cov=LazyCovariances(V23/'account_inputs/covariances_cadence21');parity=[]
    for name in MODELS:
        target=pd.read_pickle(ROOT/f'account_inputs/{name}_desired.pkl');assert target.index.equals(gate.index) and list(target.columns)==d['names']
        w=target.to_numpy();mask=d['mask'][ix];b=d['beta'][ix]
        assert np.max(abs(w[~mask]),initial=0)<1e-10 and abs(w.sum(1)).max()<1e-9 and abs((w*b).sum(1)).max()<1e-9
        vol=np.sqrt(252*(.5*np.sum(np.einsum('tkn,tn->tk',d['centered'][ix],w)**2,axis=1)/62+.5*np.sum(d['diag'][ix]*w*w,axis=1)))
        assert np.max(vol-.12*d['gate'][ix])<1e-9 and np.max(abs(w).sum(1)-3*np.minimum(1,mask.sum(1)/500))<1e-9 and abs(w).max()<=.15+1e-9
        for year in range(2007,2024):
            m=read(ROOT/f'models/{name}_{year}.json');assert m['protocol_sha256']==seal and m['data_manifest_sha256']==sha(ROOT/'data_manifest.json')
            cut=pd.Timestamp(year,1,1);sel=selected(d,cut,cut-pd.DateOffset(years=10));query=np.flatnonzero(d['dates'].year==year)
            assert m['training_query_sha256']==hashlib.sha256(d['dates'][sel].asi8.tobytes()).hexdigest()
            assert pd.Timestamp(m['max_training_label_end'])<cut and pd.Timestamp(m['inner_max_label_end'])<pd.Timestamp(m['validation_start'])
            f=ROOT/f'models/{name}_{year}.pt';assert sha(f)==m['weights_sha256'];model=Policy(name);model.load_state_dict(torch.load(f,weights_only=True))
            expected=predict(model,d,query);saved=np.load(ROOT/f'account_inputs/{name}_{year}.npy')
            np.testing.assert_allclose(expected,saved,atol=1e-12);mx=max(mx,float(abs(expected-saved).max()))
            for c in m['selection']:
                v=c['validation'];assert pd.Timestamp(v['end'])<cut and v['actual_engine']
                shortened=v.get('unsettled_claim_boundary_shortened',False)
                if shortened:assert year==2012 and v['end'].startswith('2011-12-13')
                if year==2012:assert shortened
            if m['selection']:assert m['selected_penalty']==max(m['selection'],key=lambda c:c['validation']['utility'])['penalty']
            rows.append(dict(model=name,year=year,training_periods=m['training_periods'],penalty=m['selected_penalty'],parameters=m['parameter_count'],training_utility=m['training']['utility'],loss_start=m['training']['trace'][0]['loss'],loss_end=m['training']['trace'][-1]['loss'],validation_utility=max([c['validation']['utility'] for c in m['selection']],default=np.nan)))
            if year in (2007,2015,2023):
                j=query[len(query)//2];tensor=tensors(d,[j]);
                with torch.no_grad():score=model(tensor['x']).numpy()[0]
                old,_=allocate_target(pd.Series(score,index=d['names']),pd.Series(d['beta'][j],index=d['names']),cov[d['dates'][j]],pd.Series(d['mask'][j],index=d['names']),date=d['dates'][j],vol_target=.12,gross_cap=3,name_cap=.15)
                err=float(np.max(abs(old.to_numpy()*d['gate'][j]-predict(model,d,[j])[0])));assert err<1e-9
                parity.append(dict(model=name,date=str(d['dates'][j]),error=err))
                with torch.no_grad():
                    before=weights(model,tensor).numpy();tensor['returns'][:]=float('nan');after=weights(model,tensor).numpy()
                np.testing.assert_array_equal(before,after)
    out=ROOT/'reports';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'fit_diagnostics.csv',index=False)
    accounts={}
    for name in [*MODELS,'h21_replay']:
        m=read(ROOT/f'accounts/{name}_complete.json');assert m['protocol_sha256']==seal and m['ledger_sha256']==sha(ROOT/f'accounts/{name}_ledger.csv.gz') and m['target_sha256']==sha(ROOT/f'account_inputs/{name}_desired.pkl')
        assert m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed']
        accounts[name]={k:m['audit'][k] for k in ['provisional_sessions','validated_performance']}
    replay=read(ROOT/'accounts/h21_replay_complete.json')['baseline_parity'];assert max(replay.values())<1e-10
    result=dict(passed=True,annual_models=51,inner_exact_engine_validations=sum(len(read(ROOT/f'models/{n}_{y}.json')['selection']) for n in MODELS for y in range(2007,2024)),accounts=accounts,source_allocator_parity=parity,recomputed_targets_max_error=mx,benchmark_replay_max_error=max(replay.values()),future_label_poison_target_test=True,strict_inner_and_outer_purge=True,original_source_hashes_verified=True,gradient_objective_is_surrogate=True)
    write(out/'validation.json',result);return result

def main():
    validation=validate();spec=read(ROOT/'protocol.json');ledgers={n:load_ledger(ROOT/f'accounts/{n}_ledger.csv.gz',n) for n in MODELS}
    ledgers[CHALLENGER]=load_ledger(ROOT/'accounts/h21_replay_ledger.csv.gz',CHALLENGER)
    old='direct_price_persistence_target12';ledgers[old]=load_ledger(V28/f'accounts/{old}_ledger.csv.gz',old)
    returns=pd.DataFrame({n:f.daily_return for n,f in ledgers.items()});assert np.isfinite(returns.to_numpy()).all();rows=[];annual=[]
    for n,f in ledgers.items():
        prior=f.nav.shift(1,fill_value=1)
        for period,left,right in PERIODS:
            part=f.loc[left:right];rows.append(performance(n,part,prior.reindex(part.index),period,None))
        for y in range(2007,2024):
            part=f.loc[str(y)];a=performance(n,part,prior.reindex(part.index),f'year_{y}',None);a['year']=y;annual.append(a)
    perf=pd.DataFrame(rows);inf=pd.concat([paired_comparisons(returns.loc['2012':'2023'],spec['pairs'],b,family_size=5) for b in (20,60)],ignore_index=True)
    out=ROOT/'reports';perf.to_csv(out/'performance.csv',index=False);pd.DataFrame(annual).to_csv(out/'annual.csv',index=False);inf.to_csv(out/'paired_primary.csv',index=False);returns.to_csv(out/'daily_returns.csv.gz')
    positive=[]
    for a,b in spec['pairs']:
        q=inf.loc[inf.arm.eq(a)&inf.baseline.eq(b)]
        if q.delta_sharpe.gt(0).all() and q.p_holm.lt(.05).all():positive.append([a,b])
    write(out/'status.json',dict(completed=True,validation=validation,positive_holm_both_blocks=positive,promotion=False))
    print(perf.loc[perf.period.isin(['primary','early','full']),['arm','period','sharpe','cagr','volatility','max_drawdown']].to_string(index=False))
if __name__=='__main__':main()
