"""Local, independent ETF long/cash research. No order routing or portfolio layer."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config/run_config.json').read_text())

def load_prices(root=ROOT):
    result = {}
    for path in (root/'data/normalized/prices').glob('*.csv'):
        d = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
        result[path.stem] = d
    return result

def build_features(prices: dict, ticker: str, macro: pd.DataFrame | None = None):
    """Rolling features use today's completed close and earlier values only."""
    p = prices[ticker]
    c, o, h, l = (p['adj_'+x] for x in ['close','open','high','low'])
    r = c.pct_change(fill_method=None)
    f = pd.DataFrame(index=p.index)
    for n in [5,10,20,63,126,252]:
        f[f'tech_mom_{n}'] = c.pct_change(n, fill_method=None)
    for n in [20,63]:
        f[f'tech_vol_{n}'] = r.rolling(n).std()*np.sqrt(252)
    f['tech_downside_20'] = r.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)*np.sqrt(252)
    f['tech_trend_50'] = c/c.rolling(50).mean()-1
    f['tech_trend_200'] = c/c.rolling(200).mean()-1
    f['tech_drawdown_252'] = c/c.rolling(252).max()-1
    f['tech_range_20'] = ((h-l)/c).rolling(20).mean()
    f['tech_gap_5'] = (o/c.shift(1)-1).rolling(5).mean()
    f['tech_reversal_5'] = -r.rolling(5).sum()
    f['tech_volume_ratio'] = p['Volume'].rolling(5).mean()/p['Volume'].rolling(63).mean()-1
    f['tech_shock'] = r/r.rolling(63).std().shift(1)
    f['tech_vol_jump'] = r.rolling(10).std()/r.rolling(63).std()-1
    f['tech_acceleration'] = f['tech_mom_20']-f['tech_mom_20'].shift(20)
    f['tech_fast_slow'] = c.ewm(span=10,adjust=False).mean()/c.ewm(span=63,adjust=False).mean()-1
    for peer in ['SPY','IEF','GLD','DBC']:
        if peer not in prices:
            continue
        q = prices[peer]['adj_close'].reindex(p.index)
        pr = q.pct_change(fill_method=None)
        for n in [20,63]:
            pm = q.pct_change(n,fill_method=None)
            # Explicit zero only for not-yet-listed peers; availability flag retained.
            f[f'rel_{peer}_available_{n}'] = pm.notna().astype(float)
            f[f'rel_{peer}_mom_{n}'] = pm.fillna(0)
        if peer == 'SPY':
            beta = r.rolling(63).cov(pr)/pr.rolling(63).var()
            f['rel_market_beta_63'] = beta
            f['rel_market_correlation_63'] = r.rolling(63).corr(pr)
            f['rel_market_residual_20'] = (r-beta.shift(1)*pr).rolling(20).sum()
            f['rel_strength_63'] = f['tech_mom_63']-f['rel_SPY_mom_63']
    if macro is not None and len(macro):
        m = macro.copy()
        if 'Date' in m:
            m = m.set_index('Date')
        m.index = pd.to_datetime(m.index)
        for col in m.select_dtypes(include='number'):
            if col.startswith('macro_'):
                f[col] = m[col].reindex(f.index)
    return f.replace([np.inf,-np.inf],np.nan)

def weekly_events(price: pd.DataFrame, features: pd.DataFrame):
    """End-of-week close decisions, next-open entry, open-to-open horizons."""
    dates = price.index
    # Do not treat Monday's latest downloaded bar as a completed week.
    complete = dates.to_period('W-FRI').end_time.normalize() <= dates[-1]
    decision = pd.Series(dates[complete],index=dates[complete]).groupby(dates[complete].to_period('W-FRI')).last()
    d = pd.DatetimeIndex(decision.values)
    ix = dates.get_indexer(d)
    events = features.reindex(d).copy()
    events.index.name = 'decision_date'
    next_i = ix+1
    events['entry_date'] = [dates[j] if j<len(dates) else pd.NaT for j in next_i]
    for horizon in [5,10,20]:
        events[f'label_end_{horizon}'] = [dates[j+horizon] if j+horizon<len(dates) else pd.NaT for j in next_i]
        op = price['adj_open'].to_numpy()
        events[f'y_{horizon}'] = [op[j+horizon]/op[j]-1-CFG['label_roundtrip_hurdle'] if j+horizon<len(op) else np.nan for j in next_i]
    return events

def simulate(price: pd.DataFrame, weights: pd.Series, start=None, end=None, cost_bps=5):
    """Self-financing adjusted-unit account. Cash yields zero; weights drift between decisions."""
    p = price.loc[start:end]
    w = weights.copy()
    w.index = pd.to_datetime(w.index)
    if not w.index.is_unique:
        raise ValueError('Duplicate execution date')
    if ((w<0)|(w>1)|~np.isfinite(w)).any():
        raise ValueError('Weight outside long/cash bounds')
    cash, shares, previous_nav = 1.0, 0.0, 1.0
    records=[]
    fee = cost_bps/10000
    for i,(date,row) in enumerate(p.iterrows()):
        opening, close = row.adj_open,row.adj_close
        old = shares*opening
        wealth = cash+old
        trade = cost = turnover = 0.0
        if date in w.index:
            target = float(w.loc[date])
            if target*wealth >= old:
                value = target*(wealth+fee*old)/(1+fee*target)
            else:
                value = target*(wealth-fee*old)/(1-fee*target)
            trade = value-old
            cost = abs(trade)*fee
            turnover = abs(trade)/wealth
            shares=value/opening
            cash=wealth-value-cost
        nav=cash+shares*close
        held_exposure=shares*close/nav
        terminal=terminal_cost=0.0
        # Comparable terminal liquidation charged on final close for every account.
        if i == len(p)-1 and shares:
            terminal=shares*close
            terminal_cost=terminal*fee
            turnover += terminal/nav
            cost += terminal_cost
            cash += terminal-terminal_cost
            shares=0.0
            nav=cash
        if cash < -1e-10 or not np.isfinite(nav) or nav <= 0:
            raise ValueError('Account conservation/bounds failure')
        records.append(dict(Date=date,nav=nav,cash=max(0,cash),shares=shares,weight=shares*close/nav,
                            held_exposure=held_exposure,costs=cost,turnover=turnover,trade_value=trade,
                            terminal_sale=terminal,terminal_cost=terminal_cost,opening_nav=wealth,
                            daily_return=nav/previous_nav-1))
        previous_nav=nav
    return pd.DataFrame(records).set_index('Date')

def metrics(ledger):
    r = ledger['daily_return']
    nav = ledger['nav']
    years = len(r)/252
    peak = nav.cummax().clip(lower=1)
    return dict(total_return=float(nav.iloc[-1]-1),cagr=float(nav.iloc[-1]**(1/years)-1),
        volatility=float(r.std()*np.sqrt(252)),sharpe=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0.,
        max_drawdown=float((nav/peak-1).min()),average_exposure=float(ledger.held_exposure.mean()),
        turnover_per_year=float(ledger.turnover.sum()/years),cost_normalized_currency=float(ledger.costs.sum()),
        trade_events=int((ledger.trade_value.abs()>1e-10).sum()+(ledger.terminal_sale>0).sum()),
        average_holding_sessions=holding_duration(ledger),days=len(r))

def holding_duration(ledger):
    active=(ledger.held_exposure>1e-10).astype(int)
    groups=(active!=active.shift()).cumsum()
    lengths=active[active.eq(1)].groupby(groups[active.eq(1)]).size()
    return float(lengths.mean()) if len(lengths) else 0.

def feature_columns(events, recipe):
    prefixes=['tech_']
    if recipe in ['relationships','all']: prefixes+=['rel_']
    if recipe in ['macro','all']: prefixes+=['macro_']
    return [c for c in events if any(c.startswith(p) for p in prefixes)]

def make_model(family,target):
    classification=target=='classification'
    if family=='CART':
        cls=DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(**CFG['cart'])
    if family=='RF':
        cls=RandomForestClassifier if classification else RandomForestRegressor
        return cls(**CFG['random_forest'])
    cls=XGBClassifier if classification else XGBRegressor
    return cls(**CFG['xgboost'])

def fit_predict(train, test, columns, horizon, target, family):
    model=make_model(family,target)
    y=train[f'y_{horizon}']
    y=(y>0).astype(int) if target=='classification' else y
    if target=='classification' and y.nunique()<2:
        return np.repeat(float(y.iloc[0]),len(test)), None
    model.fit(train[columns],y)
    if target=='classification':
        p=model.predict_proba(test[columns])
        pred=p[:,list(model.classes_).index(1)]
    else:
        pred=model.predict(test[columns])
    return pred,model

def weights_from_predictions(test,pred,target,sizing):
    signal=np.asarray(pred)>(.5 if target=='classification' else 0)
    scale=np.ones(len(test))
    if sizing=='vol':
        scale=np.minimum(1,CFG['volatility_target']/test['tech_vol_63'].clip(lower=CFG['volatility_floor']).to_numpy())
    return pd.Series(signal*scale,index=pd.DatetimeIndex(test.entry_date))

def mature_rows(events, cutoff, columns):
    # Common 20-session gate ensures identical training dates across horizons.
    return events.loc[(events.index<cutoff)&(events.label_end_20<cutoff)].dropna(subset=columns+['y_5','y_10','y_20','entry_date'])

def eval_inner(events,price,year,recipe,horizon,target,family='CART',sizing='fixed'):
    cols=feature_columns(events,recipe)
    weights_all=[]
    for valyear in range(year-CFG['inner_years'],year):
        cutoff=pd.Timestamp(valyear,1,1)
        tr=mature_rows(events,cutoff,cols)
        # Do not let the final inner evaluation consume even one outer-year price.
        te=events.loc[(events.index>=cutoff)&(events.index<pd.Timestamp(valyear+1,1,1))&(events.entry_date<pd.Timestamp(year,1,1))].dropna(subset=cols+['entry_date'])
        if len(tr)<CFG['minimum_matured_weekly_rows'] or len(te)<40:
            return -np.inf,[]
        pred,_=fit_predict(tr,te,cols,horizon,target,family)
        weights=weights_from_predictions(te,pred,target,sizing)
        weights_all.append(weights)
    weights=pd.concat(weights_all).sort_index()
    end=price.index[price.index<pd.Timestamp(year,1,1)][-1]
    ledger=simulate(price,weights,start=weights.index[0],end=end,cost_bps=5)
    rr=ledger.daily_return
    score=float(rr.mean()/rr.std()*np.sqrt(252)) if rr.std()>0 else 0.
    folds=[]
    for valyear,frame in ledger.groupby(ledger.index.year):
        frame=frame.copy();frame['nav']=(1+frame.daily_return).cumprod()
        folds.append(dict(year=int(valyear),**metrics(frame)))
    return score,folds

def bootstrap_difference(a,b,seed=42):
    diff=(a.daily_return-b.daily_return).dropna().to_numpy()
    rng=np.random.default_rng(seed)
    n=len(diff);length=20
    sample=[]
    for _ in range(500):
        starts=rng.integers(0,max(1,n-length+1),size=math.ceil(n/length))
        x=np.concatenate([diff[s:s+length] for s in starts])[:n]
        sample.append(x.mean()*252)
    return np.quantile(sample,[.025,.975]).tolist()

def run_instrument(ticker,prices,macro,out):
    # Reserved prices are removed BEFORE any feature, label, model, or selection step.
    prices={k:v.loc[v.index<CFG['reserved_start']] for k,v in prices.items()}
    price=prices[ticker]
    f=build_features(prices,ticker,macro)
    events=weekly_events(price,f)
    events.to_csv(out/'events.csv')
    ledger_trials=[];selections=[];predictions=[];models_meta=[]
    specs=list(itertools.product(CFG['horizons'],CFG['targets'],CFG['recipes']))
    for year in range(CFG['historical_outer_first_year'],CFG['historical_outer_last_year']+1):
        candidates=[]
        for h,target,recipe in specs:
            rid=f'{h:02}_{target}_{recipe}'
            if recipe not in CFG['primary_eligible_recipes']:
                ledger_trials.append(dict(ticker=ticker,outer_year=year,recipe_id=rid,status='ineligible_macro_PIT_gate',score=None))
                continue
            for sizing in ['fixed','vol']:
                score,folds=eval_inner(events,price,year,recipe,h,target,sizing=sizing)
                ledger_trials.append(dict(ticker=ticker,outer_year=year,recipe_id=rid,sizing=sizing,status='eligible' if np.isfinite(score) else 'insufficient_prior_history',score=score if np.isfinite(score) else None,inner_folds=json.dumps(folds)))
                if np.isfinite(score):
                    candidates.append((score,rid,sizing,h,target,recipe))
        if not candidates: continue
        # Explicit deterministic tie break. No outer results enter selection.
        chosen=sorted(candidates,key=lambda x:(-x[0],x[1],0 if x[2]=='fixed' else 1))[0]
        score,rid,sizing,h,target,recipe=chosen
        cols=feature_columns(events,recipe)
        cutoff=pd.Timestamp(year,1,1)
        train=mature_rows(events,cutoff,cols)
        test=events.loc[(events.index>=cutoff)&(events.index<pd.Timestamp(year+1,1,1))].dropna(subset=cols+['entry_date'])
        if len(test)<40: continue
        selections.append(dict(ticker=ticker,outer_year=year,recipe_id=rid,horizon=h,target=target,recipe=recipe,sizing=sizing,inner_score=score,train_rows=len(train),train_last_decision=str(train.index[-1]),train_latest_label=str(train.label_end_20.max()),cutoff=str(cutoff),feature_count=len(cols)))
        for family in ['CART','RF','XGB']:
            pred,model=fit_predict(train,test,cols,h,target,family)
            model_bytes=pickle.dumps(model)
            model_hash=hashlib.sha256(model_bytes).hexdigest()
            (out/f'model_{year}_{family}.pkl').write_bytes(model_bytes)
            if family=='CART' and model is not None:
                (out/f'rules_{year}.txt').write_text(export_text(model,feature_names=cols),encoding='utf-8')
            if model is not None:
                for col,importance in zip(cols,model.feature_importances_):
                    models_meta.append(dict(year=year,family=family,feature=col,importance=float(importance)))
            for decision,prediction in zip(test.index,pred):
                predictions.append(dict(ticker=ticker,family=family,outer_year=year,decision_date=decision,entry_date=test.loc[decision,'entry_date'],label_end=test.loc[decision,f'label_end_{h}'],matured_outcome=test.loc[decision,f'y_{h}'],prediction=float(prediction),target=target,horizon=h,recipe=recipe,sizing=sizing,vol63=test.loc[decision,'tech_vol_63'],model_sha256=model_hash,fit_cutoff=cutoff))
        print(f'{ticker} {year}: {rid} {sizing}, {len(train)} train / {len(test)} forecasts',flush=True)
    pd.DataFrame(ledger_trials).to_csv(out/'candidate_ledger.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selections.csv',index=False)
    pd.DataFrame(models_meta).to_csv(out/'feature_usage.csv',index=False)
    forecasts=pd.DataFrame(predictions)
    if forecasts.empty: raise ValueError(f'{ticker}: no qualified outer periods')
    forecasts.to_csv(out/'forecasts.csv',index=False)
    return score_instrument(ticker,price,forecasts,out)

def score_instrument(ticker,price,forecasts,out):
    summary=[]
    first=pd.Timestamp(forecasts.entry_date.min())
    last=price.index[-1]
    for family in ['CART','RF','XGB']:
        pp=forecasts.loc[forecasts.family==family].sort_values('entry_date')
        signals=np.where(pp.target.eq('classification'),pp.prediction>.5,pp.prediction>0).astype(float)
        selected_scale=np.where(pp.sizing.eq('vol'),np.minimum(1,.10/pp.vol63.clip(lower=.05)),1.)
        for size,scale in [('selected',selected_scale),('fixed',np.ones(len(pp))),('vol',np.minimum(1,.10/pp.vol63.clip(lower=.05)))]:
            weights=pd.Series(signals*np.asarray(scale),index=pd.DatetimeIndex(pp.entry_date))
            for bps in CFG['cost_bps_one_way']:
                ledger=simulate(price,weights,first,last,bps)
                tag=f'{family}_{size}_{bps}bps'
                ledger.to_csv(out/f'ledger_{tag}.csv')
                summary.append(dict(ticker=ticker,strategy=family,sizing=size,cost_bps=bps,start=str(first.date()),end=str(last.date()),**metrics(ledger)))
                if family=='CART' and size=='selected' and bps==5:
                    primary=ledger
    # Buy/hold and descriptive exposure-matched references are on identical dates.
    for name,weight in [('buy_hold',1.),('matched_exposure',float(primary.weight.mean()))]:
        for bps in CFG['cost_bps_one_way']:
            ledger=simulate(price,pd.Series([weight],index=[first]),first,last,bps)
            ledger.to_csv(out/f'ledger_{name}_{bps}bps.csv')
            summary.append(dict(ticker=ticker,strategy=name,sizing='static',cost_bps=bps,start=str(first.date()),end=str(last.date()),**metrics(ledger)))
            if name=='buy_hold' and bps==5: buy=ledger
    spy=load_prices()['SPY']
    spy=spy.loc[spy.index<CFG['reserved_start']]
    sl=simulate(spy,pd.Series([1.],index=[first]),first,last,5)
    sl.to_csv(out/'ledger_SPY_context_5bps.csv')
    summary.append(dict(ticker=ticker,strategy='SPY_context',sizing='static',cost_bps=5,start=str(first.date()),end=str(last.date()),**metrics(sl)))
    ci=bootstrap_difference(primary,buy)
    (out/'uncertainty.json').write_text(json.dumps(dict(annual_mean_return_difference_95pct=ci,method=CFG['uncertainty']),indent=2))
    annual=[]
    for year,frame in primary.groupby(primary.index.year):
        yy=frame.copy();yy['nav']=(1+yy.daily_return).cumprod()
        annual.append(dict(year=year,**metrics(yy)))
    pd.DataFrame(annual).to_csv(out/'annual_primary.csv',index=False)
    calibration=[]
    for fam,pp in forecasts.groupby('family'):
        cc=pp.loc[pp.target.eq('classification')&pp.matured_outcome.notna()]
        if len(cc):
            for bucket,g in cc.groupby(pd.cut(cc.prediction,[-.001,.2,.4,.6,.8,1.001]),observed=True):
                calibration.append(dict(family=fam,bucket=str(bucket),n=len(g),predicted=g.prediction.mean(),observed=(g.matured_outcome>0).mean(),brier=np.mean((g.prediction-(g.matured_outcome>0).astype(float))**2)))
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tickers',nargs='*');args=parser.parse_args()
    prices=load_prices()
    macro_path=ROOT/'data/normalized/macro_features.csv'
    macro=pd.read_csv(macro_path,parse_dates=['Date']).set_index('Date') if macro_path.exists() else None
    reports=ROOT/'reports';models=ROOT/'runs'/CFG.get('output_run_directory','v2');models.mkdir(parents=True,exist_ok=True)
    metric_path=reports/'metrics.csv'
    all_summary=pd.read_csv(metric_path).to_dict('records') if metric_path.exists() else []
    for ticker in args.tickers or CFG['universe']:
        out=models/ticker;out.mkdir(exist_ok=True)
        if (out/'forecasts.csv').exists():
            raise FileExistsError(f'Immutable completed forecasts exist at {out}; declare a new run version before rerunning')
        instrument_summary=run_instrument(ticker,prices,macro,out)
        pd.DataFrame(instrument_summary).to_csv(out/'metrics.csv',index=False)
        all_summary=[x for x in all_summary if x['ticker']!=ticker]+instrument_summary
        pd.DataFrame(all_summary).to_csv(reports/'metrics.csv',index=False)
    snapshot={'config_sha256':hashlib.sha256((ROOT/'config/run_config.json').read_bytes()).hexdigest(),
        'research_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fitted_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'status':'completed','reserved_start':CFG['reserved_start']}
    (reports/'run_manifest.json').write_text(json.dumps(snapshot,indent=2))

if __name__=='__main__':main()
