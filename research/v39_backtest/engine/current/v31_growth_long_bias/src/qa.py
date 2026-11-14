"""Independent final checks of frozen fitting, targets and account outputs."""
from settings import *
from allocation import annual_variance
from learner import predict,Policy,torch
from analyze_state_v20 import load_ledger

def main():
    seal=protocol();spec=read(ROOT/'protocol.json');d=load();rows=[];mx=0.;inner_count=0;selection_changes=[];same_penalty_errors=[]
    for p,h in spec['sources'].items():assert sha(p)==h,p
    dm=read(V29/'data_manifest.json')
    for p,h in dm['files'].items():assert sha(V29/'data'/p)==h,p
    for objective in OBJECTIVES:
        for year in range(2007,2024):
            m=read(ROOT/f'models/{objective}_{year}.json');cut=pd.Timestamp(year,1,1);split=cut-pd.DateOffset(years=2);start=cut-pd.DateOffset(years=10)
            assert m['protocol_sha256']==seal and pd.Timestamp(m['max_training_label_end'])<cut and pd.Timestamp(m['inner_max_training_label_end'])<split
            ix=selected(d,cut,start);assert m['training_query_sha256']==hashlib.sha256(d['dates'][ix].asi8.tobytes()).hexdigest()
            p=ROOT/f'models/{objective}_{year}.pt';assert sha(p)==m['weights_sha256'];model=Policy('replay_random');model.load_state_dict(torch.load(p,weights_only=True))
            query=np.flatnonzero(d['dates'].year==year);expected=predict(model,d,query);file=ROOT/f'account_inputs/{objective}_{year}.npy';assert sha(file)==m['targets_sha256'];saved=np.load(file)
            np.testing.assert_allclose(expected,saved,atol=1e-12);mx=max(mx,float(abs(expected-saved).max()))
            if objective=='utility5':
                old=read(V29/f'models/replay_random_{year}.json')
                if old['selected_penalty']==m['selected_penalty']:
                    old_target=np.load(V29/f'account_inputs/replay_random_{year}.npy');error=float(abs(saved-old_target).max());assert error<1e-9;same_penalty_errors.append(error)
                else:selection_changes.append(year)
            for c in m['selection']:
                v=c['validation'];assert pd.Timestamp(v['end'])<cut and v['exact_repaired_engine']
                if year==2012:assert v['claim_boundary_shortened'] and v['end'].startswith('2011-12-13')
                utility={'utility5':v['annual_mean']-2.5*v['annual_variance'],'utility1':v['annual_mean']-.5*v['annual_variance'],'net_return':v['annual_mean'],'log_growth':v['annual_log_growth']}[objective]
                assert abs(utility-v['utility'])<1e-12
            if m['selection']:assert m['selected_penalty']==max(m['selection'],key=lambda c:c['validation']['utility'])['penalty']
            else:assert m['fallback'] and m['selected_penalty']==.1
            inner_count+=len(m['selection']);rows.append(dict(objective=objective,year=year,penalty=m['selected_penalty'],training_periods=m['training_periods'],train_utility=m['training']['utility'],loss_start=m['training']['trace'][0]['loss'],loss_end=m['training']['trace'][-1]['loss']))
    target_errors=[];accounts={};ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));td=pd.read_csv(ROOT/'reports/target_diagnostics.csv',parse_dates=['date'])
    for name,a in arm_specs().items():
        f=ROOT/f'account_inputs/{name}_desired.pkl';w=pd.read_pickle(f);p=read(f.with_name(f'{name}.json'));assert p['target_sha256']==sha(f) and p['protocol_sha256']==seal
        assert w.index.equals(d['dates'][ix]) and list(w.columns)==d['names'];w=w.to_numpy();mask=d['mask'][ix];b=d['beta'][ix]
        t=td.loc[td.arm.eq(name)].set_index('date').loc[d['dates'][ix]];bias=t.effective_bias.to_numpy();assert np.max(abs(w[~mask]),initial=0)<1e-10
        net=abs(w.sum(1)-bias).max();beta=abs((w*b).sum(1)-bias).max();assert max(net,beta)<1e-9
        assert np.max(abs(w).sum(1)-3*np.minimum(1,mask.sum(1)/500))<1e-9 and abs(w).max()<=.15+1e-9
        vols=np.sqrt(252*(.5*np.sum(np.einsum('tkn,tn->tk',d['centered'][ix],w)**2,axis=1)/62+.5*np.sum(d['diag'][ix]*w*w,axis=1)))
        assert np.max(vols-a['vol'])<1e-9;np.testing.assert_allclose(vols,t.forecast_vol,atol=1e-12)
        target_errors.append(dict(arm=name,dates=len(w),net_error=net,beta_error=beta,max_forecast_vol=float(vols.max()),max_gross=float(abs(w).sum(1).max())))
        m=read(ROOT/f'accounts/{name}_complete.json');assert m['protocol_sha256']==seal and sha(ROOT/f'accounts/{name}_ledger.csv.gz')==m['ledger_sha256'] and m['target_sha256']==sha(f)
        assert m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed'];ledger=load_ledger(ROOT/f'accounts/{name}_ledger.csv.gz',name)
        accounts[name]={k:m['audit'][k] for k in ('provisional_sessions','validated_performance','stale_risk_repairs')}
        assert ledger.cash.min()>-1e-9
        accounts[name]['margin_calls']=int(ledger.margin_calls.sum())
    parity=read(ROOT/'accounts/h21_legacy_complete.json')['benchmark_parity'];assert max(parity.values())<1e-10
    p=pd.read_csv(ROOT/'reports/performance.csv');errors=[]
    for _,r in p.iterrows():
        folder=V29 if r.arm=='old_rff' else ROOT;name='replay_random' if r.arm=='old_rff' else r.arm
        f=load_ledger(folder/f'accounts/{name}_ledger.csv.gz',name).loc[r.start:r.end];x=f.daily_return.to_numpy();vol=np.std(x,ddof=1)*np.sqrt(252)
        e=max(abs(np.expm1(252*np.mean(np.log1p(x)))-r.cagr),abs(vol-r.volatility),abs(252*np.mean(x)/vol-r.sharpe));assert e<1e-10;errors.append(e)
    pd.DataFrame(rows).to_csv(ROOT/'reports/fit_diagnostics.csv',index=False)
    result=dict(passed=True,protocol_sha256=seal,models=68,inner_account_validations=inner_count,recomputed_target_max_error=mx,original_sources_unchanged=True,strict_endpoint_purge=True,
        baseline_replay_error=max(parity.values()),target_checks=target_errors,accounts=accounts,performance_rows=len(errors),performance_max_error=max(errors),source_performance_is_conditional=True,
        utility5_regularization_changes=selection_changes,utility5_same_penalty_target_error=max(same_penalty_errors,default=0.))
    write(ROOT/'reports/qa.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
