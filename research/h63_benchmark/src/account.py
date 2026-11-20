"""Explicit-boundary joint stock account, copied from sealed v9 economics.

Quotes are never filled for execution. Cash mergers settle at the effective
session OPEN before market orders; a verified last adjusted/raw factor converts
raw cash consideration. Accounting identities and validated market P&L are
reported separately. No historical stock model is fitted here.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
STOCK_SLOTS=500
MAX_HISTORY_NAMES=2000 # Storage capacity; never the risk-budget denominator
import numpy as np
import pandas as pd

from allocation import project_biased
import portfolio_account as _old
AccountInsolvencyError=_old.AccountInsolvencyError
DEFAULT_DATA_CUTOFF=pd.Timestamp('2024-01-01')


def _value(shares,marks):
    out=np.zeros_like(shares,dtype=float)
    np.multiply(shares,marks,out=out,where=shares!=0)
    return out


def _json(names,values):return json.dumps(dict(zip(names,map(float,values))),allow_nan=False,separators=(',',':'))


def _callback_weights(value,diagnostics,names,beta,known_quote_mask,position_cap,sectors=None,sector_neutral=False,
        gross_cap=1.,dollar_neutral=True,beta_neutral=True,declared_eligibility=None,stock_slots=None,asset_caps=None,trade_support=None):
    """Validate returned targets jointly, using decision-known inputs only."""
    if not isinstance(diagnostics,dict):raise ValueError('Callback diagnostics must be a dict')
    if len(names)>(MAX_HISTORY_NAMES+12):raise ValueError('At most110stockslots plus12 explicit hedge instruments')
    if isinstance(value,pd.Series):
        if not value.index.is_unique or set(value.index)!=set(names):raise ValueError('Callback target names must match account')
        weights=value.reindex(names).to_numpy(dtype=float)
    else:weights=np.asarray(value,dtype=float)
    if weights.shape!=(len(names),) or not np.isfinite(weights).all():raise ValueError('Finite full callback target required')
    b=np.full(len(names),np.nan) if beta is None else np.asarray(beta,dtype=float)
    eligible=np.asarray(known_quote_mask,dtype=bool)&np.isfinite(b)
    if declared_eligibility is not None:eligible &= np.asarray(declared_eligibility,dtype=bool)
    if 'eligible_mask' in diagnostics:
        supplied=diagnostics['eligible_mask']
        if isinstance(supplied,(dict,pd.Series)):
            s=pd.Series(supplied)
            if not s.index.is_unique or set(s.index)!=set(names):raise ValueError('Callback eligibility names must match account')
            supplied=s.reindex(names).to_numpy()
        supplied=np.asarray(supplied)
        if supplied.shape!=eligible.shape or not np.isin(supplied,[False,True,0,1]).all():raise ValueError('Literal full eligibility mask required')
        supplied=supplied.astype(bool)
        if np.any(supplied&~eligible):raise ValueError('Callback eligibility includes unavailable quote or risk')
        eligible=supplied
    gross_limit=float(gross_cap*min(1.,(eligible if stock_slots is None else eligible&stock_slots).sum()/STOCK_SLOTS))
    if (np.abs(weights[~eligible])>1e-12).any():raise ValueError('Callback target uses ineligible member')
    if (dollar_neutral and abs(weights.sum())>1e-9) or (beta_neutral and abs(weights[eligible]@b[eligible])>1e-9):raise ValueError('Callback target violates joint dollar/beta neutrality')
    if trade_support is not None and np.abs(weights[~trade_support]).max(initial=0)>1e-12:raise ValueError('Callback target violates selected support')
    caps=np.full(len(names),min(.15,position_cap)) if asset_caps is None else asset_caps
    if np.abs(weights).sum()>gross_limit+1e-9 or (np.abs(weights)>caps+1e-9).any():raise ValueError('Callback target violates cohort/name caps')
    if sector_neutral:
        labels=pd.Series(sectors).reindex(names)
        for label in labels.dropna().unique():
            if abs(weights[labels.eq(label)].sum())>1e-9:raise ValueError('Callback target violates declared sector neutrality')
    detail=dict(diagnostics)
    detail.update(account_checked_eligible=int(eligible.sum()),account_checked_gross_limit=gross_limit,
        account_checked_net=float(weights.sum()),account_checked_beta=float(weights[eligible]@b[eligible]),
        account_checked_target_gross=float(np.abs(weights).sum()))
    def encode(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,pd.Series):return item.to_dict()
        if isinstance(item,pd.Timestamp):return str(item)
        raise TypeError('Callback diagnostic is not JSON serializable: '+str(type(item)))
    serialized=json.dumps(detail,allow_nan=False,default=encode,separators=(',',':'))
    return weights,serialized


def _project_declared(weights,beta,eligible,gross_limit,position_cap,dollar_neutral,beta_neutral,
        names,sectors=None,sector_neutral=False,asset_caps=None):
    """Feasible contraction supporting explicit leverage; no implicit1x/.10 cap."""
    w=np.asarray(weights,dtype=float);b=np.asarray(beta,dtype=float);mask=np.asarray(eligible,dtype=bool)
    out=np.zeros_like(w)
    if not mask.any() or gross_limit<=0:return out
    rows=[]
    if dollar_neutral:rows.append(np.ones(int(mask.sum())))
    if beta_neutral:rows.append(b[mask])
    if sector_neutral:
        labels=pd.Series(sectors).reindex(names).to_numpy()
        for label in pd.unique(labels[pd.notna(labels)]):rows.append((labels[mask]==label).astype(float))
    projected=w[mask].copy()
    if rows:
        matrix=np.vstack(rows);_,singular,right=np.linalg.svd(matrix,full_matrices=False)
        rank=int(np.count_nonzero(singular>max(matrix.shape)*np.finfo(float).eps*singular[0])) if singular[0]>0 else 0
        projected-=right[:rank].T@(right[:rank]@projected)
    gross=float(np.abs(projected).sum());largest=float(np.abs(projected).max(initial=0.))
    name_bound=position_cap/largest if largest else 1.
    if asset_caps is not None:
        positive=np.abs(projected)>1e-15
        name_bound=float(np.min(np.asarray(asset_caps)[mask][positive]/np.abs(projected[positive]))) if positive.any() else 1.
    multiplier=min(1.,gross_limit/gross if gross else 1.,name_bound)
    out[mask]=projected*multiplier
    return out


def _inputs(prices,weights,start,end,data_cutoff):
    if 'SPY' not in prices:raise ValueError('Full SPY session clock required')
    if not isinstance(weights,pd.DataFrame) or not weights.columns.is_unique:raise ValueError('Full desired-weight DataFrame required')
    names=list(weights);w=weights.copy();w.index=pd.DatetimeIndex(w.index)
    w=w.loc[(w.index<data_cutoff)&(w.index<pd.Timestamp(end))]
    if w.index.hasnans or not w.index.is_unique or not w.index.is_monotonic_increasing:raise ValueError('Unique increasing decision dates required')
    if not np.isfinite(w.to_numpy(dtype=float)).all():raise ValueError('Finite desired weights required')
    clock=pd.DatetimeIndex(prices['SPY'].index);clock=clock[(clock<data_cutoff)&(clock<=pd.Timestamp(end))]
    if clock.hasnans or not clock.is_unique or not clock.is_monotonic_increasing:raise ValueError('Invalid SPY calendar')
    dates=clock[(clock>=pd.Timestamp(start))&(clock<=pd.Timestamp(end))]
    if dates.empty:raise ValueError('Empty account period')
    data={}
    for t in names:
        f=prices[t].copy();f.index=pd.DatetimeIndex(f.index);f=f.loc[(f.index<data_cutoff)&(f.index<=pd.Timestamp(end))]
        if not f.index.is_unique or not f.index.is_monotonic_increasing:raise ValueError('Invalid asset calendar')
        if not {'adj_open','adj_close'}.issubset(f):raise ValueError('Adjusted open/close required')
        # Normalize object-dtype numeric prefixes ONLY after reserved rows are
        # removed; a future poison string must never enter numeric conversion.
        for column in ('adj_open','adj_close','raw_close','distribution_reinvestment_turnover'):
            if column in f:f[column]=pd.to_numeric(f[column],errors='raise')
        if 'distribution_reinvestment_turnover' in f:
            if not np.isfinite(f.distribution_reinvestment_turnover).all() or f.distribution_reinvestment_turnover.lt(0).any():raise ValueError('Observed nonnegative distribution turnover required')
        f=f.reindex(clock)
        if 'q' in f:q=f.q.eq(1)
        elif {'source_available','tr_chain_available'}.issubset(f):q=f.source_available.eq(1)&f.tr_chain_available.eq(1)
        elif 'tr_qualified' in f:q=f.tr_qualified.eq(1)
        elif 'price_qualified' in f:q=f.price_qualified.eq(1)
        else:raise ValueError('Explicit quote qualification flags required')
        f['__q']=q
        f['__open_ok']=q&np.isfinite(f.adj_open)&f.adj_open.gt(0)
        f['__close_ok']=q&np.isfinite(f.adj_close)&f.adj_close.gt(0)
        if 'tr_segment_id' not in f:f['tr_segment_id']=1.
        f['__open_ok'] &= f.tr_segment_id.notna();f['__close_ok'] &= f.tr_segment_id.notna()
        data[t]=f
    if not w.index.isin(clock).all():raise ValueError('Decision outside SPY clock')
    return names,w,clock,dates,data


def _event_table(events,names,clock,end,data_cutoff):
    if events is None:return pd.DataFrame(columns=['ticker','effective_date'])
    e=pd.DataFrame(events).copy()
    required={'ticker','effective_date','raw_cash_price'}
    if not required.issubset(e):raise ValueError('Cash events need ticker/effective_date/raw_cash_price')
    e.effective_date=pd.to_datetime(e.effective_date)
    e=e.loc[e.effective_date.lt(data_cutoff)&e.effective_date.le(pd.Timestamp(end))&e.ticker.isin(names)]
    if e.duplicated('ticker').any():raise ValueError('At most one terminal cash event per asset')
    if not e.effective_date.isin(clock).all():raise ValueError('Cash event must name an observed session')
    if not np.isfinite(e.raw_cash_price.astype(float)).all() or not e.raw_cash_price.astype(float).gt(0).all():raise ValueError('Positive raw cash consideration required')
    if 'event_type' in e and not e.event_type.isin(['cash_merger','stock_merger_liquidation','otc_liquidation']).all():raise ValueError('Unsupported corporate action')
    return e



def _directional_contract(values,beta,eligible,gross_limit,position_cap,limits):
    w=np.where(eligible,np.asarray(values,dtype=float),0.)
    if limits['long_only']:w=np.maximum(w,0.)
    b=np.where(eligible,beta,0.);net=float(w.sum());market=float(w@b)
    scale=min(1.,gross_limit/max(float(abs(w).sum()),1e-15),position_cap/max(float(abs(w).max(initial=0)),1e-15))
    if net>limits['max_net']:scale=min(scale,limits['max_net']/net)
    if net<limits['min_net']:scale=min(scale,limits['min_net']/net)
    if abs(market)>limits['max_abs_beta']:scale=min(scale,limits['max_abs_beta']/abs(market))
    return w*scale

def simulate_stock_portfolio(prices,desired_weights,events=None,start='2020-01-03',end='2023-12-29',
        cost_bps=2,buffer_l1=.10,beta_guard=.05,gross_guard=1.10,risk_betas=None,neutral=True,
        position_cap=.10,maintenance_margin=.30,record_positions=True,execution_clock='next_close',
        sectors=None,sector_neutral=False,target_callback=None,gross_cap=1.,
        dollar_neutral=None,beta_neutral=None,eligibility=None,stock_slot_mask=None,asset_name_caps=None,trade_support=None,buffer_l1_by_date=None,data_cutoff='2024-01-01',spy_benchmark=False,
        position_sink=None,position_batch_rows=10000,distribution_claims=None,
        repair_stale_risk=False,controlled_long_bias=False,position_exit_policy=None,directional_limits=None,long_only_risk=None):
    """Return ledger,positions,execution_targets,decision_audit,account_audit.

    data_cutoff is EXCLUSIVE and explicit; the default retains pre2024 protection.
    Source numeric values are bounded through end BEFORE conversion or quote checks.
    The engine cannot choose a model, fit a gate, or inspect its future outcomes.
    Targets are fractions of post-fee NAV. Any unavailable required open quote
    cancels the WHOLE scheduled batch, without fills/fees. A missing held mark is
    provisional, never an executable quote, and invalidates the daily return
    path. Unresolved terminal holdings are not turned into fictitious cash.
    """
    if position_sink is not None and (not callable(position_sink) or not record_positions):raise ValueError('Position sink requires recorded positions and a callable')
    if not isinstance(position_batch_rows,int) or position_batch_rows<1:raise ValueError('Positive position output batch size required')
    streamed_position_rows=0
    if execution_clock!='next_close':raise ValueError('Next-session close execution only')
    data_cutoff=pd.Timestamp(data_cutoff);start_bound=pd.Timestamp(start);end_bound=pd.Timestamp(end)
    for label,bound in [('data_cutoff',data_cutoff),('start',start_bound),('end',end_bound)]:
        if pd.isna(bound) or bound.tz is not None or bound!=bound.normalize():raise ValueError('Explicit naive normalized '+label+' required')
    if start_bound>end_bound or end_bound>=data_cutoff:raise ValueError('Account start<=end<exclusive data_cutoff required')
    dollar_neutral=bool(neutral) if dollar_neutral is None else bool(dollar_neutral)
    beta_neutral=bool(neutral) if beta_neutral is None else bool(beta_neutral)
    if directional_limits is not None:
        if neutral or dollar_neutral or beta_neutral or controlled_long_bias or position_exit_policy is not None:raise ValueError('Conflicting directional mandate')
        if risk_betas is None or eligibility is None:raise ValueError('Directional mandate needs known beta and eligibility')
        if set(directional_limits)!={'long_only','min_net','max_net','max_abs_beta','guard_tolerance'}:raise ValueError('Explicit directional limits required')
        if directional_limits['min_net'] not in (-1.,0.) or directional_limits['max_net']!=.99 or directional_limits['max_abs_beta']!=1. or directional_limits['guard_tolerance']!=.05:raise ValueError('Unexpected mandate bounds')
        if directional_limits['long_only'] and directional_limits['min_net']!=0.:raise ValueError('Long-only net lower bound')
        if directional_limits['long_only'] and desired_weights.to_numpy().min() < -1e-12:raise ValueError('Long-only desired target has shorts')
    if spy_benchmark:
        if (list(desired_weights.columns)!=['SPY'] or bool(neutral) or dollar_neutral or beta_neutral or sector_neutral
                or gross_cap!=1. or position_cap!=1.):
            raise ValueError('SPY benchmark requires single SPY, unlevered gross1/cap1 and no neutrality constraints')
        if any(value is not None for value in (target_callback,risk_betas,eligibility,stock_slot_mask,asset_name_caps,trade_support,sectors)):
            raise ValueError('SPY benchmark uses source quotes only, without callbacks or cohort/risk constraints')
    if target_callback is not None and not callable(target_callback):raise ValueError('Callback must be callable')
    if execution_clock=='next_close':
        # A declared accounting adapter: all daily market price P&L accrues to
        # existing shares before queued close fills, and the intraday leg is0.
        bounded={}
        for ticker,source in prices.items():
            index=pd.DatetimeIndex(source.index)
            keep=(index<data_cutoff)&(index<=end_bound)
            frame=source.loc[keep].copy();frame.index=index[keep]
            bounded[ticker]=frame.assign(adj_open=frame.adj_close)
        prices=bounded
    position_limit=1. if spy_benchmark else .15
    if not np.isfinite([cost_bps,buffer_l1,beta_guard,gross_guard,position_cap,gross_cap,maintenance_margin]).all() or not 0<=cost_bps<10000 or buffer_l1<0 or beta_guard<0 or gross_guard<=0 or not 0<position_cap<=position_limit or not 0<gross_cap<=3 or not 0<=maintenance_margin<=1:raise ValueError('Invalid account settings')
    if cost_bps/10000*gross_cap>=1:raise ValueError('Fees times declared gross cap must be below1 for monotone post-fee NAV solve')
    names,desired,clock,dates,data=_inputs(prices,desired_weights,start,end,data_cutoff);n=len(names);fee=cost_bps/10000
    if controlled_long_bias:
        if dollar_neutral or beta_neutral or neutral:raise ValueError('Controlled bias requires explicit nonneutral mandate')
        if risk_betas is None or eligibility is None:raise ValueError('Controlled bias needs known risk and eligibility')
        if desired.sum(axis=1).lt(-1e-9).any() or desired.sum(axis=1).gt(.5+1e-9).any():raise ValueError('Net bias must lie in[0,.5]')
    from dated_quote_cache import DatedQuoteCache
    quotes=DatedQuoteCache(data,names,clock)
    if spy_benchmark and desired.lt(0).any().any():raise ValueError('SPY benchmark is long-only cash-to-100% exposure')
    buffer_schedule=None
    if buffer_l1_by_date is not None:
        buffer_schedule=pd.Series(buffer_l1_by_date).copy()
        buffer_schedule.index=pd.DatetimeIndex(buffer_schedule.index)
        buffer_schedule=buffer_schedule.loc[(buffer_schedule.index<data_cutoff)&(buffer_schedule.index<=end_bound)]
        buffer_schedule=pd.to_numeric(buffer_schedule,errors='raise')
        if buffer_schedule.index.has_duplicates or not np.isfinite(buffer_schedule).all() or (buffer_schedule<0).any():raise ValueError('Finite nonnegative bounded buffer schedule required')
    stock_slots=np.ones(n,dtype=bool) if stock_slot_mask is None else pd.Series(stock_slot_mask).reindex(names).to_numpy()
    if stock_slots.shape!=(n,) or not np.isin(stock_slots,[True,False,0,1]).all() or int(np.sum(stock_slots))>MAX_HISTORY_NAMES:raise ValueError('Explicit stock-slot mask required for added hedges')
    stock_slots=stock_slots.astype(bool)
    caps=np.full(n,position_cap) if asset_name_caps is None else pd.Series(asset_name_caps).reindex(names).to_numpy(dtype=float)
    if not np.isfinite(caps).all() or (caps<=0).any() or (caps[stock_slots]>position_limit).any() or (caps[~stock_slots]>.75).any():raise ValueError('Invalid stock/hedge name caps')
    if (desired.abs().sum(axis=1)>gross_cap+1e-10).any() or (desired.abs().to_numpy()>caps[None,:]+1e-10).any():raise ValueError('Desired target exceeds gross/name caps')
    support_frame=None
    if trade_support is not None:
        support_frame=trade_support.copy();support_frame.index=pd.DatetimeIndex(support_frame.index)
        support_frame=support_frame.loc[(support_frame.index<data_cutoff)&(support_frame.index<=end_bound)].reindex(columns=names)
        if support_frame.index.has_duplicates or not support_frame.isin([False,True,0,1]).all().all():raise ValueError('Complete known support flags required')
    if dollar_neutral and (desired.sum(axis=1).abs()>1e-10).any():raise ValueError('Desired target is not dollar neutral')
    if sector_neutral:
        if sectors is None:raise ValueError('Explicit sector map required')
        sector_labels=pd.Series(sectors).reindex(names)
        for label in sector_labels.dropna().unique():
            if (desired.loc[:,sector_labels.eq(label)].sum(axis=1).abs()>1e-10).any():raise ValueError('Desired target not sector neutral')
    if beta_neutral and risk_betas is None:raise ValueError('Strict beta-neutral account requires known-close risk_betas')
    risk=None if risk_betas is None else risk_betas.copy()
    if risk is not None:
        risk.index=pd.DatetimeIndex(risk.index);risk=risk.loc[(risk.index<data_cutoff)&(risk.index<=end_bound)].reindex(columns=names)
        if not risk.index.is_unique:raise ValueError('Duplicate risk dates')
    eligibility_frame=None
    if eligibility is not None:
        eligibility_frame=eligibility.copy();eligibility_frame.index=pd.DatetimeIndex(eligibility_frame.index)
        eligibility_frame=eligibility_frame.loc[(eligibility_frame.index<data_cutoff)&(eligibility_frame.index<=end_bound)].reindex(columns=names)
        if not eligibility_frame.index.is_unique:raise ValueError('Duplicate known eligibility dates')
        if not eligibility_frame.isin([False,True,0,1]).all().all():raise ValueError('Eligibility must be complete literal0/1, no future-label gate')
    e=_event_table(events,names,clock,end,data_cutoff);event_by_date={date:g for date,g in e.groupby('effective_date')}
    from distribution_claims import DistributionClaims
    claim_book=DistributionClaims(distribution_claims,names,dates[0],dates[-1])
    terminated=np.array([bool((e.ticker.eq(t)&e.effective_date.lt(dates[0])).any()) for t in names])
    shares=np.zeros(n);cash=previous_nav=1.;last=np.full(n,np.nan);segment=np.full(n,None,dtype=object);factor=np.full(n,np.nan)
    # Prior marks/factors are read only through the initial decision close.
    for j,t in enumerate(names):
        history=data[t].loc[data[t].index<dates[0]];valid=history.loc[history.__close_ok]
        if len(valid):
            row=valid.iloc[-1];last[j]=float(row.adj_close);segment[j]=row.tr_segment_id
            if 'raw_close' in valid and np.isfinite(row.raw_close) and row.raw_close>0:factor[j]=row.adj_close/row.raw_close
    previous_close=last.copy();schedule={};target_rows=[];decision_rows=[];records=[];position_rows=[];issues=[]
    callback_audit=[]
    def decide(date,initial=False):
        nonlocal schedule
        past=desired.loc[desired.index<=date]
        if past.empty:return
        base=past.iloc[-1].to_numpy(dtype=float);new=date in desired.index
        actual=np.zeros(n) if initial else _value(shares,last)/previous_nav
        b=None if risk is None or date not in risk.index else risk.loc[date].to_numpy(dtype=float)
        exposed=np.abs(actual)>1e-14;known=b is not None and np.isfinite(b[exposed]).all()
        abeta=float(actual[exposed]@b[exposed]) if known else None
        reference_beta=float(base.sum()) if controlled_long_bias else 0.
        mandatory=float(np.abs(actual).sum())>gross_guard+1e-12 or ((beta_neutral or controlled_long_bias) and (abs(abeta-reference_beta)>beta_guard+1e-12 if abeta is not None else exposed.any()))
        if controlled_long_bias:mandatory=mandatory or abs(float(actual.sum())-reference_beta)>beta_guard+1e-12
        declared=None
        if eligibility_frame is not None:
            if date not in eligibility_frame.index:raise ValueError('Missing decision-known eligibility date')
            declared=eligibility_frame.loc[date].to_numpy(dtype=bool)
        allowed=None if support_frame is None else support_frame.loc[date].to_numpy(dtype=bool)
        callback_detail=None;before_callback=None
        if target_callback is not None:
            before_callback=base.copy()
            quote_mask=quotes.row('__close_ok',date)&~terminated
            # A newly reset TR gauge is not a valid held-position quote merely
            # because its numeric close is positive; mirror the account's
            # already-known held-segment qualification at this decision close.
            held_break=np.array([shares[j]!=0 and segment[j] is not None and
                pd.notna(quotes.row('tr_segment_id',date)[j]) and quotes.row('tr_segment_id',date)[j]!=segment[j]
                for j,t in enumerate(names)])
            quote_mask &= ~held_break
            beta_values=np.full(n,np.nan) if b is None else b.copy()
            returned=target_callback(pd.Timestamp(date),pd.Series(actual.copy(),index=names),
                pd.Series(base.copy(),index=names),pd.Series(beta_values,index=names),pd.Series(quote_mask,index=names))
            if not isinstance(returned,tuple) or len(returned)!=2:raise ValueError('Callback must return(weights,diagnostics)')
            if isinstance(returned[1],dict) and returned[1].get('baseline_fallback_bypass_callback') is True:
                raise ValueError('Legacy class-specific warmup bypass is disabled in this explicit-boundary engine')
            base,callback_detail=_callback_weights(returned[0],returned[1],names,b,quote_mask,position_cap,sectors,sector_neutral,
                gross_cap,dollar_neutral,beta_neutral,declared,stock_slots,caps,allowed)
            callback_audit.append(dict(decision_date=str(pd.Timestamp(date).date()),diagnostics=json.loads(callback_detail)))
        missing_target=False;reproject=False;stale_risk_repaired=False;excluded_risk_names=0
        def project_current(value,beta_values,current,limit):
            if controlled_long_bias:
                return project_biased(value,beta_values,current,limit,position_cap,reference_beta)
            return _project_declared(value,beta_values,current,limit,position_cap,dollar_neutral,beta_neutral,names,sectors,sector_neutral,caps if asset_name_caps is not None else None)
        active=np.abs(base)>1e-14
        if beta_neutral or controlled_long_bias:
            if b is None or not np.isfinite(b[active]).all():
                if repair_stale_risk and not new and b is not None:
                    current=quotes.row('__close_ok',date)&~terminated&np.isfinite(b)
                    if declared is not None:current &= declared
                    budget=current.copy()
                    if allowed is not None:current &= allowed
                    excluded_risk_names=int(np.count_nonzero(active&~current))
                    limit=min(float(np.abs(base).sum()),gross_cap*min(1.,(budget&stock_slots).sum()/STOCK_SLOTS))
                    if current.sum()>=3:
                        base=project_current(base,b,current,limit);reproject=True;stale_risk_repaired=True
                    else:base=np.zeros(n);missing_target=True
                else:base=np.zeros(n);missing_target=True
            elif abs(float(base[active]@b[active])-reference_beta)>1e-8:
                if new:raise ValueError('Desired target not beta neutral at decision close')
                eligible=quotes.row('__close_ok',date)&~terminated&np.isfinite(b)
                if declared is not None:eligible &= declared
                budget_eligible=eligible.copy()
                if allowed is not None:eligible &= allowed
                if gross_cap==1 and position_cap<=.1 and dollar_neutral and declared is None:
                    # Preserve legacy default arithmetic exactly.
                    from stock_allocation import project_neutral
                    base,_=project_neutral(pd.Series(base,index=names),pd.Series(b,index=names),eligible=eligible,
                        sectors=sectors,sector_neutral=sector_neutral,max_abs=position_cap,gross_limit=max(1e-15,float(np.abs(base).sum())))
                    base=base.to_numpy()
                else:
                    limit=min(float(np.abs(base).sum()),gross_cap*min(1.,(budget_eligible&stock_slots).sum()/STOCK_SLOTS))
                    base=project_current(base,b,eligible,limit)
                reproject=True
        if declared is not None:
            current=declared&quotes.row('__close_ok',date)&~terminated
            if beta_neutral or controlled_long_bias:current &= np.isfinite(b) if b is not None else False
            limit=gross_cap*min(1.,(current&stock_slots).sum()/STOCK_SLOTS)
            if allowed is not None:current &= allowed
            invalid=bool((np.abs(base[~current])>1e-12).any() or np.abs(base).sum()>limit+1e-9)
            if invalid:
                if new and target_callback is None:raise ValueError('Desired target violates declared current eligibility/cohort gross cap')
                base=project_current(base,np.zeros(n) if b is None else b,current,limit);reproject=True
        directional_detail=None
        if directional_limits is not None:
            current=quotes.row('__close_ok',date)&~terminated&np.isfinite(b)
            if declared is not None:current &= declared
            limit=gross_cap*min(1.,current.sum()/STOCK_SLOTS)
            base=_directional_contract(base,b,current,limit,position_cap,directional_limits)
            actual_net=float(actual.sum());tolerance=directional_limits['guard_tolerance']
            bad_net=actual_net<directional_limits['min_net']-tolerance or actual_net>directional_limits['max_net']+tolerance
            bad_beta=abs(abeta)>directional_limits['max_abs_beta']+tolerance if abeta is not None else bool(exposed.any())
            bad_side=directional_limits['long_only'] and bool((actual < -1e-12).any())
            bad_member=declared is not None and bool((exposed&~declared).any())
            directional_guard=bool(bad_net or bad_beta or bad_side or bad_member)
            mandatory=bool(mandatory or directional_guard)
            reference_beta=float(base[current]@b[current])
            directional_detail=dict(directional_guard=directional_guard,net_guard=bool(bad_net),beta_bound_guard=bool(bad_beta),
                forbidden_short_guard=bool(bad_side),ineligible_holding_guard=bool(bad_member))
        exit_detail=None
        if position_exit_policy is not None:
            quote_mask=quotes.row('__close_ok',date)&~terminated
            if declared is not None:quote_mask &= declared
            quote_mask &= np.isfinite(b) if b is not None else False
            value,exit_detail=position_exit_policy(pd.Timestamp(date),actual.copy(),base.copy(),np.array(b,copy=True),quote_mask.copy(),initial)
            base=np.asarray(value,dtype=float)
            if base.shape!=(n,) or not np.isfinite(base).all() or abs(base[~quote_mask]).max(initial=0)>1e-10:raise ValueError('Invalid exit-policy target')
            if abs(base).max()>position_cap+1e-9 or abs(base).sum()>gross_cap*min(1.,quote_mask.sum()/STOCK_SLOTS)+1e-9:raise ValueError('Exit policy exceeds exposure limits')
            reference_beta=float(base.sum())
            if reference_beta < -1e-9 or reference_beta>.5+1e-9 or abs(base[quote_mask]@b[quote_mask]-reference_beta)>1e-9:raise ValueError('Exit policy violates controlled bias')
            mandatory=float(np.abs(actual).sum())>gross_guard+1e-12 or (abs(abeta-reference_beta)>beta_guard+1e-12 if abeta is not None else exposed.any()) or abs(float(actual.sum())-reference_beta)>beta_guard+1e-12 or bool(exit_detail.get('force_exit',False))
        daily_risk_detail=None
        if long_only_risk is not None:
            if directional_limits is None or not directional_limits['long_only']:raise ValueError('Daily risk requires long-only mandate')
            base,risk_mandatory,daily_risk_detail=long_only_risk(date,actual.copy(),base.copy(),b.copy(),current.copy(),initial)
            if not np.isfinite(base).all() or base.min() < -1e-12 or base[~current].max(initial=0)>1e-12:raise ValueError('Invalid daily risk target')
            if base.sum()>.99+1e-10 or base.max(initial=0)>.15+1e-10 or abs(base@np.nan_to_num(b))>1+1e-10:raise ValueError('Daily risk target violates mandate')
            mandatory=bool(mandatory or risk_mandatory);new=True
        distance=float(np.abs(base-actual).sum());reason='' 
        if not initial and not mandatory and not new:reason='no_new_target'
        elif not mandatory and distance<(buffer_l1 if buffer_schedule is None else float(buffer_schedule.loc[date])):reason='buffer'
        execution=clock[clock.searchsorted(date,side='right')]
        if execution not in dates:return
        if not reason:
            schedule[pd.Timestamp(execution)]=(base.copy(),len(decision_rows));target_rows.append(pd.Series(base,index=names,name=execution))
        decision_rows.append(dict(decision_date=date,execution_date=execution,desired_date=past.index[-1],
            initial_cash_decision=initial,actual_weights_json=_json(names,actual),base_weights_json=_json(names,base),
            target_weights_json=_json(names,base) if not reason else None,base_to_actual_l1=distance,
            actual_gross=float(np.abs(actual).sum()),actual_net=float(actual.sum()),actual_estimated_beta=abeta,
            target_gross=float(np.abs(base).sum()) if not reason else None,target_net=float(base.sum()) if not reason else None,
            target_estimated_beta=float(base[np.abs(base)>1e-14]@b[np.abs(base)>1e-14]) if b is not None and not reason else None,
            mandatory_guard=bool(mandatory),risk_missing_target=missing_target,guard_reprojection=reproject,
            stale_risk_repaired=stale_risk_repaired,excluded_risk_names=excluded_risk_names,reference_beta=reference_beta,
            submitted=not bool(reason),skip_reason=reason,execution_applied=False,execution_abort_reason='',
            cash_termination_canceled_count=0,cash_termination_canceled_gross=0.,
            execution_target_weights_json=None,execution_target_gross=None,execution_target_net=None))
        if daily_risk_detail is not None:decision_rows[-1].update(daily_risk_detail)
        if directional_detail is not None:decision_rows[-1].update(directional_detail)
        if position_exit_policy is not None:
            decision_rows[-1].update(position_exit_diagnostics_json=json.dumps(exit_detail,allow_nan=False))
        if target_callback is not None:
            decision_rows[-1].update(target_callback_used=True,base_desired_before_callback_json=_json(names,before_callback),
                target_callback_diagnostics_json=callback_detail)
        if buffer_schedule is not None:decision_rows[-1]['effective_buffer_l1']=float(buffer_schedule.loc[date])
    before=desired.index[desired.index<dates[0]]
    if len(before):decide(before[-1],True)
    for day,date in enumerate(dates):
        prior_shares=shares.copy();prior_cash=cash;event_cash=np.zeros(n);event_pnl=np.zeros(n);event_price=np.full(n,np.nan);event_fees=np.zeros(n);event_notional=np.zeros(n)
        claim=claim_book.on_day(date,prior_shares,previous_close,fee)
        issues.extend(claim['issues']);cash+=float(claim['cash'].sum()-claim['long_costs'].sum()-claim['short_costs'].sum())
        if date in event_by_date:
            for event in event_by_date[date].itertuples():
                j=names.index(event.ticker);supplied=getattr(event,'adjustment_factor',np.nan)
                conversion=float(supplied) if pd.notna(supplied) else factor[j]
                adjusted=getattr(event,'adjusted_cash_price',np.nan)
                payout=float(adjusted) if pd.notna(adjusted) else float(event.raw_cash_price)*conversion
                if not np.isfinite(payout) or payout<=0:
                    issues.append(dict(date=str(date.date()),ticker=event.ticker,kind='unresolved_cash_action_factor'));terminated[j]=True;continue
                if shares[j]!=0:
                    rights=getattr(event,'unpriced_rights',False)
                    cvr=getattr(event,'cvr_upper',0.)
                    conditional=getattr(event,'conditional_reason',None)
                    if (pd.notna(rights) and bool(rights)) or (pd.notna(cvr) and float(cvr)>0) or (isinstance(conditional,str) and conditional):
                        issues.append(dict(date=str(date.date()),ticker=event.ticker,kind='unresolved_conditional_corporate_value',
                            raw_share_equivalent=float(shares[j]*conversion),
                            reason=conditional if isinstance(conditional,str) else 'Unpriced contingent rights; explicit cash-only scenario',
                            cvr_upper=float(cvr) if pd.notna(cvr) else None))
                    event_cash[j]=shares[j]*payout;event_pnl[j]=shares[j]*(payout-previous_close[j]);cash+=event_cash[j]
                    if getattr(event,'event_type','cash_merger') in ['stock_merger_liquidation','otc_liquidation']:
                        stock_part=getattr(event,'stock_component',float('nan'))
                        fraction=float(stock_part)/float(event.raw_cash_price) if pd.notna(stock_part) else 1.
                        event_notional[j]=abs(event_cash[j])*fraction;event_fees[j]=fee*event_notional[j];cash-=event_fees[j]
                    shares[j]=0.
                terminated[j]=True;event_price[j]=payout;last[j]=payout
        quote_open=quotes.row('adj_open',date)
        quote_close=quotes.row('adj_close',date)
        open_ok=quotes.row('__open_ok',date)&~terminated
        close_ok=quotes.row('__close_ok',date)&~terminated
        seg=quotes.row('tr_segment_id',date)
        broken=np.array([shares[j]!=0 and segment[j] is not None and pd.notna(seg[j]) and seg[j]!=segment[j] for j in range(n)])
        open_ok &= ~broken;close_ok &= ~broken
        for j in np.flatnonzero(broken):issues.append(dict(date=str(date.date()),ticker=names[j],kind='held_segment_break'))
        opening=np.where(open_ok,quote_open,last);unpriced_open=(shares!=0)&~open_ok
        if np.any((shares!=0)&~np.isfinite(opening)):raise ValueError('Held asset has no prior mark')
        for j in np.flatnonzero(unpriced_open):issues.append(dict(date=str(date.date()),ticker=names[j],kind='unpriced_held_open'))
        distribution_factor=quotes.row('distribution_reinvestment_turnover',date)
        if ((distribution_factor>0)&(shares!=0)&~close_ok).any():raise ValueError('Cannot execute distribution reinvestment without observed close')
        distribution_notional=abs(_value(shares,np.where(close_ok,quote_close,0.)))*distribution_factor
        distribution_fees=fee*distribution_notional;cash-=float(distribution_fees.sum())
        old=_value(shares,opening);opening_shares=shares.copy();opening_nav=float(cash+old.sum());_old._positive_nav(opening_nav,date,'open')
        overnight=_value(shares,opening-previous_close)
        long_pnl=np.where(prior_shares>0,event_pnl,0.)+np.where(shares>0,overnight,0.)+claim['long_pnl']
        short_pnl=np.where(prior_shares<0,event_pnl,0.)+np.where(shares<0,overnight,0.)+claim['short_pnl']
        long_costs=np.where(prior_shares>0,event_fees+distribution_fees,0.)+claim['long_costs'];short_costs=np.where(prior_shares<0,event_fees+distribution_fees,0.)+claim['short_costs'];absolute=event_notional+distribution_notional+claim['notional'];open_trade=np.zeros(n);close_trade=np.zeros(n);rebalance=np.zeros(n)
        turnover=absolute.sum()/previous_nav;trade_events=int(np.count_nonzero(event_notional)+2*np.count_nonzero(distribution_notional)+np.count_nonzero(claim['notional']));terminal_cost=0.;margin_open=margin_close=False;forced_cover=forced_sale=0.;applied=False;aborted=False;abort_reason=''
        canceled_count=0;canceled_gross=0.;effective_scheduled=np.full(n,np.nan)
        short_open=float(-np.minimum(old,0).sum());margin_required=short_open>0 and opening_nav/short_open<maintenance_margin-1e-12
        terminal_rebalance_skipped=bool(execution_clock=='next_close' and day==len(dates)-1 and date in schedule)
        if terminal_rebalance_skipped:
            decision_rows[schedule[date][1]].update(execution_applied=False,execution_abort_reason='terminal_close_rebalance_skipped')
        if margin_required and not unpriced_open.any():
            trial=_old._margin_cover(cash,shares,np.where(np.isfinite(opening),opening,1.),fee,date,'open margin');delta=trial[2]
            if open_ok[np.abs(delta)>1e-14].all():
                margin_open=True;cash,shares,delta,cover,sale=trial;forced_cover+=cover;forced_sale+=sale;open_trade+=delta;absolute+=abs(delta)
                long_costs+=fee*np.where(old>0,abs(delta),0.);short_costs+=fee*np.where(old<0,abs(delta),0.);turnover+=abs(delta).sum()/opening_nav;trade_events+=np.count_nonzero(abs(delta)>1e-12)
                if date in schedule:decision_rows[schedule[date][1]].update(execution_applied=False,execution_abort_reason='margin_action_supersedes_rebalance')
        elif date in schedule and not terminal_rebalance_skipped:
            weights,decision_index=schedule[date]
            # A verified effective cash termination cancels only its queued
            # market-order leg. The other target fractions are unchanged: no
            # renormalization, anticipatory source filter or synthetic fill.
            weights=weights.copy();cancel=terminated&(np.abs(weights)>1e-14)
            canceled_count=int(cancel.sum());canceled_gross=float(np.abs(weights[cancel]).sum())
            weights[terminated]=0.;effective_scheduled=weights.copy()
            decision_rows[decision_index].update(cash_termination_canceled_count=canceled_count,
                cash_termination_canceled_gross=canceled_gross,
                execution_target_weights_json=_json(names,weights),
                execution_target_gross=float(np.abs(weights).sum()),execution_target_net=float(weights.sum()))
            values=_old._post_fee_values(opening_nav,old,weights,fee)
            if values is None:raise AccountInsolvencyError(date,'rebalance fees',opening_nav-fee*abs(old).sum())
            delta=values-old;required=np.abs(delta)>1e-14
            if unpriced_open.any():aborted=True;abort_reason='unpriced_held_NAV'
            elif not open_ok[required].all():aborted=True;abort_reason='required_open_unavailable_or_terminated'
            else:
                long_costs+=fee*abs(np.maximum(values,0)-np.maximum(old,0));short_costs+=fee*abs(np.minimum(values,0)-np.minimum(old,0))
                cash=float(opening_nav-values.sum()-fee*abs(delta).sum())
                shares=np.divide(values,opening,out=shares.copy(),where=open_ok);open_trade+=delta;rebalance=delta.copy();absolute+=abs(delta)
                turnover+=float(abs(delta).sum())/opening_nav;trade_events+=np.count_nonzero(abs(delta)>1e-12);applied=True
            decision_rows[decision_index].update(execution_applied=applied,execution_abort_reason=abort_reason)
        if margin_required and not margin_open:issues.append(dict(date=str(date.date()),kind='unresolved_margin_open'))
        if cash < -1e-10:raise ValueError('Cash borrowing after open/corporate cash settlement')
        cash=max(0.,float(cash));after_shares=shares.copy();after_values=_value(shares,opening);post_nav=float(cash+after_values.sum())
        _old._positive_nav(post_nav,date,'after open')
        # Valid opens can provisionally mark a missing close; no market fill uses this.
        current_mark=np.where(open_ok,quote_open,last);closing=np.where(close_ok,quote_close,current_mark)
        unpriced_close=(shares!=0)&~close_ok
        for j in np.flatnonzero(unpriced_close):issues.append(dict(date=str(date.date()),ticker=names[j],kind='unpriced_held_close'))
        intraday=_value(shares,closing-opening);long_pnl+=np.where(shares>0,intraday,0.);short_pnl+=np.where(shares<0,intraday,0.)
        values=_value(shares,closing);close_nav=float(cash+values.sum());_old._positive_nav(close_nav,date,'close before actions')
        held_net=float(values.sum()/close_nav);held_gross=float(abs(values).sum()/close_nav);held_long=float(np.maximum(values,0).sum()/close_nav);held_short=float(-np.minimum(values,0).sum()/close_nav)
        before_actions=values.copy();short_close=float(-np.minimum(values,0).sum())
        if short_close>0 and close_nav/short_close<maintenance_margin-1e-12:
            if not unpriced_close.any():
                trial=_old._margin_cover(cash,shares,np.where(np.isfinite(closing),closing,1.),fee,date,'close margin');delta=trial[2]
                if close_ok[np.abs(delta)>1e-14].all():
                    margin_close=True;cash,shares,delta,cover,sale=trial;forced_cover+=cover;forced_sale+=sale;close_trade+=delta;absolute+=abs(delta)
                    long_costs+=fee*np.where(values>0,abs(delta),0.);short_costs+=fee*np.where(values<0,abs(delta),0.);turnover+=abs(delta).sum()/close_nav;trade_events+=np.count_nonzero(abs(delta)>1e-12)
            if not margin_close:issues.append(dict(date=str(date.date()),kind='unresolved_margin_close'))
        if day==len(dates)-1:
            values=_value(shares,closing);before_terminal=float(cash+values.sum());delta=np.where(close_ok,-values,0.)
            if cash-delta.sum()-fee*abs(delta).sum() < -1e-10:
                issues.append(dict(date=str(date.date()),kind='unresolved_terminal_cash_funding'));delta=np.zeros(n)
            liquidation_costs=fee*np.abs(delta);terminal_cost=float(liquidation_costs.sum())
            cash-=float(delta.sum())+terminal_cost;long_costs+=fee*np.where(values>0,abs(delta),0.);short_costs+=fee*np.where(values<0,abs(delta),0.)
            close_trade+=delta;absolute+=abs(delta);turnover+=float(abs(delta).sum())/before_terminal;trade_events+=np.count_nonzero(abs(delta)>1e-12)
            shares=np.divide(values+delta,closing,out=shares.copy(),where=close_ok)
        closing_values=_value(shares,closing);nav=float(cash+closing_values.sum());_old._positive_nav(nav,date,'after close actions')
        costs=long_costs+short_costs;net_long=long_pnl-long_costs;net_short=short_pnl-short_costs;net_pnl=float(net_long.sum()+net_short.sum())
        if not np.isclose(nav-previous_nav,net_pnl,rtol=1e-10,atol=1e-11):raise ValueError('Stock P&L conservation failed')
        if not np.isclose(cash-prior_cash,event_cash.sum()+claim['cash'].sum()-open_trade.sum()-close_trade.sum()-costs.sum(),atol=1e-10):raise ValueError('Stock cash conservation failed')
        validated=not (unpriced_open.any() or unpriced_close.any() or broken.any() or any(i['date']==str(date.date()) and i['kind'].startswith('unresolved') for i in issues))
        records.append(dict(Date=date,nav=nav,cash=cash,opening_nav=opening_nav,post_trade_nav=post_nav,close_nav_before_actions=close_nav,daily_return=nav/previous_nav-1,
            delayed_distribution_cashflow=float(claim['cash'].sum()),delayed_distribution_cost=float(claim['long_costs'].sum()+claim['short_costs'].sum()),
            costs=float(costs.sum()),turnover=turnover,trade_events=int(trade_events),target_scheduled=date in schedule,target_applied=applied,
            execution_clock=execution_clock,terminal_rebalance_skipped=terminal_rebalance_skipped,
            cash_termination_canceled_count=canceled_count,cash_termination_canceled_gross=canceled_gross,
            order_batch_aborted=aborted,order_abort_reason=abort_reason,margin_call_open=margin_open,margin_call_close=margin_close,margin_calls=int(margin_open)+int(margin_close),
            forced_short_cover=forced_cover,forced_long_liquidation=forced_sale,terminal_cost=terminal_cost,held_exposure=held_net,held_gross_exposure=held_gross,held_long_exposure=held_long,held_short_exposure=held_short,
            net_exposure=float(closing_values.sum()/nav),gross_exposure=float(abs(closing_values).sum()/nav),long_notional=float(np.maximum(closing_values,0).sum()),short_notional=float(-np.minimum(closing_values,0).sum()),
            long_price_pnl=float(long_pnl.sum()),short_price_pnl=float(short_pnl.sum()),long_costs=float(long_costs.sum()),short_costs=float(short_costs.sum()),long_net_pnl=float(net_long.sum()),short_net_pnl=float(net_short.sum()),net_pnl=net_pnl,
            corporate_liquidation_cost=float(event_fees.sum()),distribution_reinvestment_cost=float(distribution_fees.sum()),distribution_reinvestment_notional=float(distribution_notional.sum()),corporate_cashflow=float(event_cash.sum()),corporate_price_pnl=float(event_pnl.sum()),unpriced_held_open=int(unpriced_open.sum()),unpriced_held_close=int(unpriced_close.sum()),held_segment_breaks=int(broken.sum()),pnl_validated=validated,borrow_cost=0.,cash_interest=0.,short_proceeds_interest=0.))
        if record_positions:
            for j,t in enumerate(names):position_rows.append(dict(Date=date,ticker=t,previous_closing_shares=prior_shares[j],opening_shares=opening_shares[j],opening_value=old[j],after_open_shares=after_shares[j],after_open_value=after_values[j],scheduled_weight=schedule[date][0][j] if date in schedule else np.nan,after_open_weight=after_values[j]/post_nav,
                delayed_distribution_cashflow=claim['cash'][j],delayed_distribution_cost=claim['long_costs'][j]+claim['short_costs'][j],
                open_trade_value=open_trade[j],rebalance_trade_value=rebalance[j],close_value_before_actions=before_actions[j],close_trade_value=close_trade[j],closing_shares=shares[j],closing_value=closing_values[j],costs=costs[j],absolute_trade_value=absolute[j],long_price_pnl=long_pnl[j],short_price_pnl=short_pnl[j],long_net_pnl=net_long[j],short_net_pnl=net_short[j],
                distribution_reinvestment_cost=distribution_fees[j],distribution_reinvestment_notional=distribution_notional[j],corporate_cashflow=event_cash[j],corporate_price_pnl=event_pnl[j],event_adjusted_cash_price=event_price[j],open_quote_valid=bool(open_ok[j]),close_quote_valid=bool(close_ok[j]),mark_provisional=bool(unpriced_close[j]),segment_break=bool(broken[j]),terminated=bool(terminated[j]),
                after_execution_shares=after_shares[j],after_execution_value=after_values[j],after_execution_weight=after_values[j]/post_nav,execution_quote_valid=bool(open_ok[j]),execution_clock=execution_clock,
                effective_scheduled_weight=effective_scheduled[j]))
            if position_sink is not None and len(position_rows)>=position_batch_rows:
                position_sink(pd.DataFrame(position_rows));streamed_position_rows+=len(position_rows);position_rows.clear()
        for j,t in enumerate(names):
            if close_ok[j] or (open_ok[j] and not broken[j]):last[j]=closing[j];segment[j]=seg[j]
            if close_ok[j] and 'raw_close' in data[t]:
                raw=quotes.row('raw_close',date)[j]
                if np.isfinite(raw) and raw>0:factor[j]=quote_close[j]/raw
        previous_nav=nav;previous_close=closing.copy()
        if day<len(dates)-1:decide(date)
    if position_sink is not None and position_rows:
        position_sink(pd.DataFrame(position_rows));streamed_position_rows+=len(position_rows);position_rows.clear()
    ledger=pd.DataFrame(records).set_index('Date');positions=pd.DataFrame(position_rows);targets=pd.DataFrame(target_rows,columns=names,dtype=float);targets.index=pd.DatetimeIndex(targets.index,name='Date');decisions=pd.DataFrame(decision_rows)
    unresolved=[names[j] for j in range(n) if shares[j]!=0]+[x['ticker']+':distribution_claim' for x in claim_book.open]
    audit=dict(arithmetic_passed=True,validated_performance=bool(ledger.pnl_validated.all() and not unresolved),execution_clock=execution_clock,
        exposure_clock='after next-session close execution; end-of-day exposure' if execution_clock=='next_close' else 'after next-session open execution; held close exposure',
        provisional_sessions=int((~ledger.pnl_validated).sum()),unresolved_terminal_assets=unresolved,issues=issues,
        aborted_order_batches=int(ledger.order_batch_aborted.sum()),corporate_cash_events=len(e),reserved_scored=False,
        delayed_distribution_events=claim_book.log,
        cash_termination_canceled_legs=int(ledger.cash_termination_canceled_count.sum()),
        quote_policy='No forward-filled execution quotes; execution-slot quote checked independently; stale valuation marks provisional only',
        event_clock='Effective session open before market orders; raw cash times last verified adjusted/raw factor',
        terminated_order_policy='Cancel terminated queued target legs at execution without rescaling others; target dollar/beta constraints may temporarily drift until next known-close guard',
        costs='One-way on actual market traded notional; cash-merger consideration has no fee; stock-merger received shares liquidated at recipient regular-session open with one-way fee/turnover',
        limitations=['Daily Sharpe/drawdown/performance must not be validated when any session uses a provisional held mark.',
                     'Estimated-beta constraints and next-open safeguards do not guarantee realized or instantaneous neutrality.'])
    if target_callback is not None:
        audit.update(target_callback_enabled=True,target_callback_calls=len(callback_audit),target_callback_audit=callback_audit,
            target_callback_clock='decision close only; actual marked holdings/NAV, base desired, current beta and known quote mask; no future prices passed')
    if gross_cap!=1 or position_cap!=.1 or dollar_neutral!=bool(neutral) or beta_neutral!=bool(neutral) or eligibility_frame is not None:
        audit.update(declared_gross_cap=float(gross_cap),declared_position_cap=float(position_cap),
            dollar_neutral=bool(dollar_neutral),beta_neutral=bool(beta_neutral),gross_guard=float(gross_guard),
            known_eligibility_frame_supplied=eligibility_frame is not None,
            funding='No cash loans; neutral gross3 may hold long1.5/short1.5 using existing short-proceeds/cash accounting; all interest0',
            target_basis='fractions of actual post-fee NAV, not scaled NAV returns; caps constrain targets, held exposure may drift')
    if stock_slot_mask is not None or asset_name_caps is not None or trade_support is not None:
        audit.update(stock_slot_mask=dict(zip(names,map(bool,stock_slots))),asset_name_caps=dict(zip(names,map(float,caps))),explicit_support_enforced=trade_support is not None,hedges_count_all_gross_costs=True)
    audit.update(data_cutoff_exclusive=str(data_cutoff.date()),requested_start=str(start_bound.date()),
        requested_end=str(end_bound.date()),first_observed_session=str(dates[0].date()),last_observed_session=str(dates[-1].date()),
        source_numeric_prefix_only=True,legacy_class_specific_bypass=False,spy_benchmark=bool(spy_benchmark))
    if position_sink is not None:audit.update(position_output_streamed=True,streamed_position_rows=streamed_position_rows)
    audit.update(repair_stale_risk=repair_stale_risk,controlled_long_bias=controlled_long_bias,
        stale_risk_repairs=int(decisions.stale_risk_repaired.sum()),repaired_excluded_names=int(decisions.excluded_risk_names.sum()))
    return ledger,positions,targets,decisions,audit
