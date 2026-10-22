"""Predeclared, matched-date ETF ablations. All research inputs end before 2024.

No experiment is promoted from these inspected historical results. The immutable
v2 procedure supplies year-specific base specifications, not outer outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
import research as base
from etf_context import load_context

ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/'config/experiments_v3.json'
CFG=json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
VARIANTS=list(CFG['variants'])
OUT=ROOT/'runs/v3_1'
ALIASES={'ctx_equity_bond_corr63':'rel_market_correlation_63','ctx_property_equity_corr63':'rel_market_correlation_63'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pre_reserve_prices():
    # Date parsing and slicing precede every feature/return/label operation.
    return {t:p.loc[p.index<CFG['reserved_start']].copy() for t,p in base.load_prices().items()}

def fast_simulate(price,weights,start=None,end=None,cost_bps=5):
    """Array implementation of v2's exact self-financing account equations."""
    w=weights.copy();w.index=pd.to_datetime(w.index)
    if not w.index.is_unique:raise ValueError('Duplicate execution date')
    if ((w<0)|(w>1)|~np.isfinite(w)).any():raise ValueError('Weight outside long/cash bounds')
    p=price.loc[start:end];wm=w.to_dict();fee=cost_bps/10000
    cash=1.;shares=0.;previous=1.;rows=[]
    for i,(date,opening,close) in enumerate(zip(p.index,p.adj_open.to_numpy(),p.adj_close.to_numpy())):
        old=shares*opening;wealth=cash+old;trade=cost=turnover=0.
        if date in wm:
            target=float(wm[date])
            value=target*(wealth+fee*old)/(1+fee*target) if target*wealth>=old else target*(wealth-fee*old)/(1-fee*target)
            trade=value-old;cost=abs(trade)*fee;turnover=abs(trade)/wealth
            shares=value/opening;cash=wealth-value-cost
        nav=cash+shares*close;held=shares*close/nav;terminal=terminal_cost=0.
        if i==len(p)-1 and shares:
            terminal=shares*close;terminal_cost=terminal*fee;turnover+=terminal/nav
            cost+=terminal_cost;cash+=terminal-terminal_cost;shares=0.;nav=cash
        if cash < -1e-10 or not np.isfinite(nav) or nav<=0:raise ValueError('Account conservation/bounds failure')
        rows.append((date,nav,max(0,cash),shares,shares*close/nav,held,cost,turnover,trade,terminal,terminal_cost,wealth,nav/previous-1))
        previous=nav
    return pd.DataFrame(rows,columns=['Date','nav','cash','shares','weight','held_exposure','costs','turnover','trade_value','terminal_sale','terminal_cost','opening_nav','daily_return']).set_index('Date')

def feature_columns(events,recipe,context=False):
    cols=base.feature_columns(events,recipe)
    if context:
        ctx=[c for c in events if c.startswith('ctx_')]
        for name,alias in ALIASES.items():
            if alias in cols:
                ctx=[c for c in ctx if c not in [name,name+'_available']]
        cols+=ctx
    return cols

def prepare_events(prices,ticker):
    f=base.build_features(prices,ticker)
    f=f.join(load_context(ticker),how='left')
    if f.index.max()>=pd.Timestamp(CFG['reserved_start']):raise AssertionError('Reserved feature date')
    r=prices[ticker].adj_close.pct_change(fill_method=None)
    f['diagnostic_ewma_vol']=r.pow(2).ewm(alpha=1-CFG['risk']['ewma_lambda'],adjust=False,min_periods=20).mean().pow(.5)*np.sqrt(252)
    monthly=prices[ticker].adj_close.resample('ME').last()
    monthly_signal=(monthly>monthly.rolling(10,min_periods=10).mean()).where(monthly.rolling(10,min_periods=10).count().ge(10))
    # Calendar-month endpoints fall after mid-month observations. Forward-only
    # alignment therefore uses only a calendar-completed month, with a small
    # conservative delay when calendar month-end is a nontrading day.
    f['diagnostic_trend10m']=monthly_signal.reindex(f.index,method='ffill').astype(float)
    e=base.weekly_events(prices[ticker],f)
    e['weekly_label_end']=e.entry_date.shift(-1)
    op=prices[ticker].adj_open
    entries=e.entry_date.map(op)
    following=e.weekly_label_end.map(op)
    e['y_weekly']=following/entries-1-base.CFG['label_roundtrip_hurdle']
    return e

def matured_train(events,cutoff,cols,label):
    train=base.mature_rows(events,pd.Timestamp(cutoff),cols)
    return train.dropna(subset=[label])

def fit_predict(train,test,cols,target,label,weighted=False):
    y=train[label]
    classification=target=='classification'
    if classification:y=(y>0).astype(int)
    if classification and y.nunique()==1:
        pred=np.repeat(float(y.iloc[0]),len(test));return pred,None
    model=(DecisionTreeClassifier if classification else DecisionTreeRegressor)(**base.CFG['cart'])
    weights=None
    if weighted:
        age=(pd.Timestamp(test.index.min())-train.index).days.to_numpy()/365.25
        weights=np.power(.5,age/3.)
    model.fit(train[cols],y,sample_weight=weights)
    pred=model.predict_proba(test[cols])[:,list(model.classes_).index(1)] if classification else model.predict(test[cols])
    return pred,model

def prior_oof(events,cols,horizon,target,outer_year,context):
    cutoff=pd.Timestamp(outer_year,1,1);frames=[]
    for year in range(outer_year-CFG['calibration']['prior_oof_years'],outer_year):
        fitcut=pd.Timestamp(year,1,1)
        tr=matured_train(events,fitcut,cols,f'y_{horizon}')
        te=events.loc[(events.index>=fitcut)&(events.index<pd.Timestamp(year+1,1,1))&(events.label_end_20<cutoff)].dropna(subset=cols+[f'y_{horizon}','entry_date'])
        if len(tr)<CFG['learning']['minimum_matured_rows'] or te.empty:continue
        pred,_=fit_predict(tr,te,cols,target,f'y_{horizon}')
        frame=pd.DataFrame({'decision_date':te.index,'entry_date':te.entry_date.to_numpy(),
            'prediction':pred,'outcome':te[f'y_{horizon}'].to_numpy(),'label_end':te[f'label_end_{horizon}'].to_numpy(),
            'label_end20':te.label_end_20.to_numpy(),'fit_cutoff':fitcut,'calibration_cutoff':cutoff,
            'outer_year':outer_year,'context':context,'oof_train_rows':len(tr),
            'oof_train_latest_label':tr.label_end_20.max()})
        frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['prediction','outcome'])

def fit_payoff_calibrator(oof,target):
    """Bin prior OOF forecasts, shrink payoffs to the prior unconditional mean.

    It estimates conditional net payoff, not a probability-to-position mapping.
    Outcome dispersion is descriptive; overlapping labels are not independent.
    """
    d=oof[['prediction','outcome']].replace([np.inf,-np.inf],np.nan).dropna()
    if len(d)<CFG['calibration']['min_matured_oof_rows']:
        return {'eligible':False,'n':len(d),'reason':'insufficient_matured_OOF'}
    if target=='classification':edges=np.asarray(CFG['calibration']['classifier_edges'][1:-1],float)
    else:edges=np.unique(np.quantile(d.prediction,np.linspace(0,1,CFG['calibration']['regression_bins']+1)[1:-1]))
    bucket=np.searchsorted(edges,d.prediction.to_numpy(),side='right')
    global_mean=float(d.outcome.mean());global_std=float(d.outcome.std(ddof=1))
    pseudo=CFG['calibration']['pseudo_observations'];bins=[]
    for j in range(len(edges)+1):
        values=d.outcome.to_numpy()[bucket==j];n=len(values)
        mean=(float(values.sum())+pseudo*global_mean)/(n+pseudo)
        bins.append({'n':n,'expected_payoff':mean,'outcome_std':float(np.std(values,ddof=1)) if n>1 else global_std})
    return {'eligible':True,'n':len(d),'edges':edges.tolist(),'bins':bins,'global_mean':global_mean,
            'global_outcome_std':global_std,'pseudo_observations':pseudo,'target':target}

def apply_calibrator(cal,pred):
    n=len(pred)
    if not cal['eligible']:
        return {'expected_payoff':np.full(n,np.nan),'bin_n':np.zeros(n,int),'dispersion':np.full(n,np.nan)}
    bucket=np.searchsorted(np.asarray(cal['edges']),np.asarray(pred),side='right')
    return {'expected_payoff':np.asarray([cal['bins'][b]['expected_payoff'] for b in bucket]),
            'bin_n':np.asarray([cal['bins'][b]['n'] for b in bucket]),
            'dispersion':np.asarray([cal['bins'][b]['outcome_std'] for b in bucket])}

def basic_signal(pred,target):return np.asarray(pred)>(.5 if target=='classification' else 0)

def hysteresis(values,entry,exit,fallback=None):
    """Carry only the past signal; equality retains it. Initial account is cash."""
    state=False;signals=[]
    for i,value in enumerate(np.asarray(values)):
        if not np.isfinite(value):state=bool(fallback[i]) if fallback is not None else False
        elif not state and value>entry:state=True
        elif state and value<exit:state=False
        signals.append(state)
    return np.asarray(signals)

def sizing_scale(variant,forecasts):
    """Resolve the explicit EWMA arm before generic '_vol' rule suffixes."""
    if variant=='ewma_vol':
        return np.minimum(1,.1/forecasts.ewma_vol.clip(lower=.05).to_numpy())
    if variant.endswith('_fixed'):
        return np.ones(len(forecasts))
    volscale=np.minimum(1,.1/forecasts.vol63.clip(lower=.05).to_numpy())
    if variant.endswith('_vol') or variant=='fixed_recipe_cart':return volscale
    return np.where(forecasts.base_sizing.eq('vol'),volscale,1.)

def save_model(model,path,cols):
    data=pickle.dumps(model);path.write_bytes(data)
    if model is not None:
        path.with_suffix('.txt').write_text(export_text(model,feature_names=cols),encoding='utf-8')
    return hashlib.sha256(data).hexdigest()

def run_ticker(ticker,prices):
    folder=OUT/ticker;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():raise FileExistsError(f'Completed immutable v3 experiment exists for {ticker}')
    p=prices[ticker];events=prepare_events(prices,ticker)
    old=pd.read_csv(ROOT/f'runs/v2/{ticker}/forecasts.csv',parse_dates=['decision_date','entry_date','fit_cutoff'])
    old=old.loc[old.family.eq('CART')].sort_values('decision_date').set_index('decision_date')
    sel=pd.read_csv(ROOT/f'runs/v2/{ticker}/selections.csv')
    predictions=[];oof_all=[];calibrators=[];features=[]
    for spec in sel.itertuples():
        year=int(spec.outer_year);cut=pd.Timestamp(year,1,1);h=int(spec.horizon);target=spec.target
        reference=old.loc[old.outer_year.eq(year)]
        test=events.reindex(reference.index)
        cols=feature_columns(events,spec.recipe);ctxcols=feature_columns(events,spec.recipe,True)
        train=matured_train(events,cut,cols,f'y_{h}')
        ctxtrain=matured_train(events,cut,ctxcols,f'y_{h}')
        if len(train)!=int(spec.train_rows) or len(ctxtrain)!=len(train):raise AssertionError('Baseline training-row match failed')
        native,model=fit_predict(train,test,cols,target,f'y_{h}')
        if not np.allclose(native,reference.prediction,rtol=1e-10,atol=1e-12):raise AssertionError('Baseline model forecasts not reproduced')
        ctx,ctxmodel=fit_predict(ctxtrain,test,ctxcols,target,f'y_{h}')
        mh=save_model(model,folder/f'{year}_native.pkl',cols)
        ch=save_model(ctxmodel,folder/f'{year}_context.pkl',ctxcols)
        for kind,mm,cc in [('native',model,cols),('context',ctxmodel,ctxcols)]:
            if mm is not None:
                features.extend(dict(year=year,model=kind,feature=c,importance=float(v)) for c,v in zip(cc,mm.feature_importances_))
        maps={}
        for use_ctx,cc in [(False,cols),(True,ctxcols)]:
            oof=prior_oof(events,cc,h,target,year,use_ctx)
            if len(oof):oof_all.append(oof)
            cal=fit_payoff_calibrator(oof,target);maps[use_ctx]=apply_calibrator(cal,ctx if use_ctx else native)
            latest_label=oof.label_end20.max() if len(oof) else pd.NaT
            calibrators.append(dict(outer_year=year,context=use_ctx,calibration_cutoff=str(cut),
                latest_matured_label=str(latest_label),calibrator=cal))
            maps[use_ctx]['total_oof_n']=cal['n'];maps[use_ctx]['latest_label']=latest_label
        weekly_train=matured_train(events,cut,cols,'y_weekly')
        weekly,wm=fit_predict(weekly_train,test,cols,target,'y_weekly')
        # Scale target using only training decision-time volatility and horizon.
        scaled_events=events.copy()
        scaled_events['y_scaled']=events[f'y_{h}']/(events.tech_vol_63.clip(lower=.05)*np.sqrt(h/252))
        scaled_train=matured_train(scaled_events,cut,cols,'y_scaled')
        scaled,sm=fit_predict(scaled_train,test,cols,'regression','y_scaled')
        scaled=scaled*test.tech_vol_63.clip(lower=.05).to_numpy()*np.sqrt(h/252)
        recency,rm=fit_predict(train,test,cols,target,f'y_{h}',weighted=True)
        fixedcols=feature_columns(events,'technical')
        fixedtrain=matured_train(events,cut,fixedcols,'y_10')
        fixed,fm=fit_predict(fixedtrain,test,fixedcols,'classification','y_10')
        qpred=pd.Series(index=test.index,dtype=float);qcuts=pd.Series(index=test.index,dtype='datetime64[ns]');qn=pd.Series(index=test.index,dtype=int);qlast=pd.Series(index=test.index,dtype='datetime64[ns]');qhash=pd.Series(index=test.index,dtype=object)
        for quarter,qt in test.groupby(test.index.quarter):
            qcut=pd.Timestamp(year,1+(int(quarter)-1)*3,1)
            qtrain=matured_train(events,qcut,cols,f'y_{h}')
            qp,qm=fit_predict(qtrain,qt,cols,target,f'y_{h}')
            qpred.loc[qt.index]=qp;qcuts.loc[qt.index]=qcut;qn.loc[qt.index]=len(qtrain);qlast.loc[qt.index]=qtrain.label_end_20.max()
            qhash.loc[qt.index]=save_model(qm,folder/f'{year}_quarter{quarter}.pkl',cols)
        extra_hash={}
        for name,mm,cc in [('weekly_target',wm,cols),('risk_scaled_target',sm,cols),('recency_weighted',rm,cols),('fixed_recipe_cart',fm,fixedcols)]:
            extra_hash[name]=save_model(mm,folder/f'{year}_{name}.pkl',cc)
        raw={'baseline':native,'payoff_only':native,'context_only':ctx,'payoff_context':ctx,'payoff_buffer':native,
             'weekly_target':weekly,'risk_scaled_target':scaled,'recency_weighted':recency,'quarterly_refit':qpred.to_numpy(),
             'ewma_vol':native,'turnover_buffer':native,'trend_10m_fixed':test.diagnostic_trend10m.to_numpy(),
             'trend_10m_vol':test.diagnostic_trend10m.to_numpy(),'tsmom12m_fixed':test.tech_mom_252.to_numpy(),
             'tsmom12m_vol':test.tech_mom_252.to_numpy(),'fixed_recipe_cart':fixed}
        for variant in VARIANTS:
            ct=variant in ['context_only','payoff_context']
            uses_payoff=variant in ['payoff_only','payoff_context','payoff_buffer']
            fixed_rule=variant.startswith('trend_') or variant.startswith('tsmom')
            label_target='regression' if variant=='risk_scaled_target' else 'classification' if variant=='fixed_recipe_cart' else target
            for i,date in enumerate(test.index):
                m=maps[ct]
                predictions.append(dict(ticker=ticker,variant=variant,decision_date=date,entry_date=test.entry_date.iloc[i],
                    outer_year=year,fit_cutoff=pd.NaT if fixed_rule else qcuts.iloc[i] if variant=='quarterly_refit' else cut,
                    train_rows=0 if fixed_rule else int(qn.iloc[i]) if variant=='quarterly_refit' else len(train),
                    train_latest_label=pd.NaT if fixed_rule else qlast.iloc[i] if variant=='quarterly_refit' else train.label_end_20.max(),
                    uses_fitted_model=not fixed_rule,target='rule' if fixed_rule else label_target,
                    horizon=0 if fixed_rule else 5 if variant=='weekly_target' else 10 if variant=='fixed_recipe_cart' else h,
                    base_horizon=h,base_recipe=spec.recipe,base_sizing=spec.sizing,prediction=float(raw[variant][i]),
                    base_signal=bool(basic_signal([ctx[i] if ct else native[i]],target)[0]),
                    vol63=float(test.tech_vol_63.iloc[i]),ewma_vol=float(test.diagnostic_ewma_vol.iloc[i]),
                    expected_payoff=float(m['expected_payoff'][i]) if uses_payoff else np.nan,
                    payoff_bin_n=int(m['bin_n'][i]) if uses_payoff else 0,payoff_dispersion=float(m['dispersion'][i]) if uses_payoff else np.nan,
                    oof_rows=int(m['total_oof_n']) if uses_payoff else 0,
                    calibration_cutoff=cut if uses_payoff else pd.NaT,
                    calibration_latest_label=m['latest_label'] if uses_payoff else pd.NaT,
                    feature_count=len(ctxcols) if ct else len(fixedcols) if variant=='fixed_recipe_cart' else len(cols),
                    model_sha256='rule:'+hashlib.sha256(CFG['variants'][variant].encode()).hexdigest() if fixed_rule else qhash.iloc[i] if variant=='quarterly_refit' else ch if ct else extra_hash.get(variant,mh),
                    label_end=pd.NaT if fixed_rule else test.weekly_label_end.iloc[i] if variant=='weekly_target' else test.label_end_10.iloc[i] if variant=='fixed_recipe_cart' else test[f'label_end_{h}'].iloc[i]))
        print(f'v3 {ticker} {year}: matched baseline; {len(test)} weekly decisions; OOF {maps[False]["total_oof_n"]}/{maps[True]["total_oof_n"]}',flush=True)
    forecasts=pd.DataFrame(predictions)
    metrics=[]
    first=old.entry_date.min();last=p.index[-1]
    for variant,g in forecasts.groupby('variant',sort=False):
        g=g.sort_values('decision_date').copy()
        pred=g.prediction.to_numpy()
        default=np.where(g.target.eq('classification'),pred>.5,pred>0)
        if variant in ['payoff_only','payoff_context']:
            signal=np.where(g.expected_payoff.notna(),g.expected_payoff>0,g.base_signal)
        elif variant=='payoff_buffer':signal=hysteresis(g.expected_payoff,.001,-.001,g.base_signal.to_numpy())
        elif variant=='turnover_buffer':
            # Normalize the two target types to a common entry=1/exit=-1 scale.
            margin=np.where(g.target.eq('classification'),(pred-.5)/.05,pred/.001)
            signal=hysteresis(margin,1,-1)
        elif variant.startswith('trend_10m'):signal=pred>.5
        elif variant.startswith('tsmom'):signal=pred>0
        else:signal=default
        scale=sizing_scale(variant,g)
        g['signal']=signal.astype(int);g['target_weight']=signal*scale
        forecasts.loc[g.index,'signal']=g.signal
        forecasts.loc[g.index,'target_weight']=g.target_weight
        weights=pd.Series(g.target_weight.to_numpy(),index=pd.DatetimeIndex(g.entry_date))
        for cost in CFG['cost_bps']:
            ledger=fast_simulate(p,weights,first,last,cost)
            if variant=='baseline':
                native_ledger=pd.read_csv(ROOT/f'runs/v2/{ticker}/ledger_CART_selected_{cost}bps.csv',parse_dates=['Date']).set_index('Date')
                if not ledger.index.equals(native_ledger.index) or not np.allclose(ledger.nav,native_ledger.nav,rtol=1e-11,atol=1e-12):raise AssertionError('Baseline NAV replay mismatch')
            ledger.to_csv(folder/f'ledger_{variant}_{cost}bps.csv')
            metrics.append(dict(ticker=ticker,variant=variant,cost_bps=cost,start=str(first.date()),end=str(last.date()),**base.metrics(ledger)))
    forecasts.to_csv(folder/'forecasts.csv',index=False)
    pd.concat(oof_all,ignore_index=True).to_csv(folder/'calibration_oof.csv',index=False) if oof_all else None
    (folder/'calibrators.json').write_text(json.dumps(calibrators,default=str,indent=2),encoding='utf-8')
    pd.DataFrame(features).to_csv(folder/'feature_usage.csv',index=False)
    sel.to_csv(folder/'baseline_specifications.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    (folder/'completed.json').write_text(json.dumps({'ticker':ticker,'status':'complete','variants':len(VARIANTS),
        'reserved_scored':False,'forecast_sha256':digest(folder/'forecasts.csv'),'source_sha256':digest(__file__),
        'config_sha256':digest(CONFIG_PATH),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()},indent=2))
    return metrics

def summarize():
    parts=[pd.read_csv(f) for f in OUT.glob('*/metrics.csv')]
    if not parts:return
    m=pd.concat(parts,ignore_index=True);m.to_csv(ROOT/'reports/v3_metrics.csv',index=False)
    base_m=m[m.variant.eq('baseline')].set_index(['ticker','cost_bps'])
    compare=m.copy()
    compare['baseline_sharpe']=[base_m.loc[(r.ticker,r.cost_bps),'sharpe'] for r in compare.itertuples()]
    compare['delta_sharpe']=compare.sharpe-compare.baseline_sharpe
    old_metrics=pd.read_csv(ROOT/'reports/metrics.csv')
    bh=old_metrics[old_metrics.strategy.eq('buy_hold')].set_index(['ticker','cost_bps'])
    compare['buy_hold_sharpe']=[bh.loc[(r.ticker,r.cost_bps),'sharpe'] for r in compare.itertuples()]
    compare['delta_sharpe_vs_buy_hold']=compare.sharpe-compare.buy_hold_sharpe
    compare.to_csv(ROOT/'reports/v3_comparison.csv',index=False)
    declarations=[]
    for ticker in CFG['universe']:
        for name,description in CFG['variants'].items():
            declarations.append({'ticker':ticker,'variant':name,'description':description,'status':'completed' if ticker in set(m.ticker) else 'pending',
                'parameter_search_trials':1,'promotion_allowed':False,'comparison_metric':CFG['primary_metric'],
                'hypothesis_declared_at_utc':CFG['declared_at_utc'],'reserved_start':CFG['reserved_start']})
    pd.DataFrame(declarations).to_csv(ROOT/'reports/v3_hypothesis_ledger.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tickers',nargs='*');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'source_snapshot.py').exists():
        (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (OUT/'experiment_config.json').write_bytes(CONFIG_PATH.read_bytes())
    prices=pre_reserve_prices()
    for ticker in args.tickers or CFG['universe']:
        run_ticker(ticker,prices);summarize()
    (ROOT/'reports/v3_run_manifest.json').write_text(json.dumps({'status':'complete' if len(list(OUT.glob('*/completed.json')))==15 else 'partial',
        'source_sha256':digest(__file__),'config_sha256':digest(CONFIG_PATH),'reserved_scored':False,
        'completed_instruments':len(list(OUT.glob('*/completed.json'))),'declared_variants':len(VARIANTS),'run_directory':OUT.relative_to(ROOT).as_posix(),
        'generated_utc':pd.Timestamp.now(tz='UTC').isoformat()},indent=2))

if __name__=='__main__':main()
