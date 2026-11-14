from compare import *
from analyze_state_v20 import load_ledger,performance,stats
s=importlib.util.spec_from_file_location('v35_metrics',V35/'src/assess.py');sys.path.insert(0,str(V35/'src'));shared=importlib.util.module_from_spec(s);s.loader.exec_module(shared)

def main():
    seal=protocol();out=HERE/'reports';q=read(out/'preparation_qa.json');assert q['passed'] and q['protocol_sha256']==seal;assert read(V34/'reports/qa.json')['passed'];d=learner.load();ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));loads=np.load(V34/'data/factor_loadings.npy');ledgers={};states={};riskerr=0.
    for a in ARMS:
        targetfile=HERE/f'account_inputs/{a}_desired.pkl';w=pd.read_pickle(targetfile);assert w.index.equals(d['dates'][ix]) and list(w.columns)==d['names'];values=w.to_numpy();b=d['beta'][ix];mask=d['mask'][ix];vol=np.sqrt(252*(.5*np.sum(np.einsum('tkn,tn->tk',d['centered'][ix],values)**2,axis=1)/62+.5*np.sum(d['diag'][ix]*values**2,axis=1)))
        assert vol.max()<=.12+1e-9 and abs(values).max()<=.15+1e-9 and np.max(abs(values).sum(1)-3*np.minimum(1,mask.sum(1)/500))<1e-9 and abs(values[~mask]).max(initial=0)<1e-10
        assert abs((values*b).sum(1)-values.sum(1)).max()<1e-9 and values.sum(1).min()>=0 and values.sum(1).max()<=.5+1e-9
        for row,j in enumerate(ix):
            alpha=values[row]-unit_market(b[row],mask[row])*values[row].sum();err=float(abs(loads[j]@alpha).max());assert err<1e-9;riskerr=max(riskerr,err)
        m=read(HERE/f'accounts/{a}_complete.json');assert m['protocol_sha256']==seal and m['target_sha256']==sha(targetfile) and m['ledger_sha256']==sha(HERE/f'accounts/{a}_ledger.csv.gz') and m['audit']['arithmetic_passed'] and m['audit']['noncash_exposure_audit']['passed'];ledgers[a]=load_ledger(HERE/f'accounts/{a}_ledger.csv.gz',a);assert ledgers[a].cash.min()>-1e-9
        dec=pd.read_csv(HERE/f'accounts/{a}_decisions.csv.gz',parse_dates=['decision_date','execution_date']);known=dec.execution_date.notna();assert dec.loc[known,'execution_date'].gt(dec.loc[known,'decision_date']).all();states[a]={k:m['audit'][k] for k in ('provisional_sessions','validated_performance')}
    q.update(accounts=states,independent_alpha_factor_error=riskerr,all_target_risk_constraints=True,strict_next_close=True,inherited_training_purge_qa=True,original_sources_unchanged=True);write(out/'qa.json',q)
    for a,file in [('monthly_leader','growth_long50_v12'),('h21_original','h21_legacy')]:ledgers[a]=load_ledger(V31/f'accounts/{file}_ledger.csv.gz',a)
    clock=next(iter(ledgers.values())).index;assert all(f.index.equals(clock) for f in ledgers.values());spy=shared.evaluation.market_returns();rows=[];annual=[];borrows=[]
    for a,f in ledgers.items():
        prev=f.nav.shift(1,fill_value=1)
        for period,left,right in shared.PERIODS:
            z=f.loc[left:right];r=performance(a,z,prev.loc[z.index],period,spy);r.update(shared.underwater(z.daily_return));r['annual_log_growth']=252*np.log1p(z.daily_return).mean();rows.append(r)
        for year in range(2007,2024):
            z=f.loc[str(year)];r=performance(a,z,prev.loc[z.index],str(year),spy);r['year']=year;annual.append(r)
        for bps in (0,100,300):
            adjusted=f.daily_return-f.short_notional.shift(1,fill_value=0)/prev*bps/10000/252
            for period,left,right in shared.PERIODS[:2]:borrows.append(dict(arm=a,period=period,annual_borrow_bps=bps,**stats(adjusted.loc[left:right].to_numpy())))
    returns=pd.DataFrame({a:f.daily_return for a,f in ledgers.items()});growth=pd.concat([shared.evaluation.paired_growth(returns.loc[left:right],PAIRS,block).assign(period=period) for period,left,right in shared.PERIODS[:2] for block in (20,60)],ignore_index=True);passing={}
    for period in ('primary','full'):
        passing[period]=[[a,b] for a,b in PAIRS if (lambda z:z.delta_annual_log_growth.gt(0).all() and z.p_holm.lt(.05).all())(growth.loc[growth.period.eq(period)&growth.arm.eq(a)&growth.baseline.eq(b)])]
    for name,f in [('performance.csv',pd.DataFrame(rows)),('annual.csv',pd.DataFrame(annual)),('paired_growth.csv',growth),('borrow_sensitivity.csv',pd.DataFrame(borrows))]:f.to_csv(out/name,index=False)
    returns.to_csv(out/'daily_returns.csv.gz');write(out/'status.json',dict(completed=True,protocol_sha256=seal,promotion=False,growth_holm_both_blocks=passing,primary='2012-2023',full='2007-2023',not_retrained_for_long_bias=True))
    result=pd.DataFrame(rows);print(result.loc[result.period.isin(['primary','full','stress_2022']),['arm','period','cagr','net_return','sharpe','volatility','max_drawdown']].to_string(index=False));print('GROWTH_HOLM',passing)
if __name__=='__main__':main()
