"""Explicit-boundary63-session diag50 risk; unchanged sealed-v9 sizing math.

Imports read no data. The default boundary protects2024+; callers must pass a
frozen exclusive boundary to evaluate later data. Each covariance uses only
qualified completed-close returns through its own query. No price fills,
forecast fitting, performance selection or global cutoff patching occurs.
"""
from __future__ import annotations
STOCK_SLOTS=500
MAX_HISTORY_NAMES=2000 # Storage capacity; never the risk-budget denominator
import numpy as np
import pandas as pd

DEFAULT_DATA_CUTOFF=pd.Timestamp('2024-01-01')


def _cutoff(value):
    value=pd.Timestamp(value)
    if pd.isna(value) or value.tz is not None or value!=value.normalize():
        raise ValueError('Explicit exclusive normalized naive data_cutoff required')
    return value


def _clock(index,label):
    dates=pd.DatetimeIndex(index)
    if dates.tz is not None:raise ValueError('Naive observed-session dates required')
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Unique increasing '+label+' dates required')
    return dates


def _series(value,names,label,mask=False):
    if not isinstance(value,pd.Series) or not value.index.is_unique or set(value.index)!=set(names):
        raise ValueError('Full named '+label+' required')
    out=value.reindex(names)
    if mask:
        if not out.isin([False,True,0,1]).all():raise ValueError('Literal complete '+label+' mask required')
        return out.astype(bool)
    return out.astype(float)


def _slots(names,stock_slot_mask):
    if not names or len(names)>MAX_HISTORY_NAMES+1 or len(set(names))!=len(names):raise ValueError('Declared stock slots plus one hedge required')
    slots=pd.Series(True,index=names) if stock_slot_mask is None else _series(stock_slot_mask,names,'stock slots',True)
    if slots.sum()>MAX_HISTORY_NAMES:raise ValueError('At most110 declared stock slots')
    if any(name!='SPY' for name in slots.index[~slots]):raise ValueError('Only explicit SPY outside stock budget')
    return slots




def build_covariances(prices,names,dates,*,estimator='diag50',window=63,stock_slot_mask=None,data_cutoff='2024-01-01'):
    """Exact-date daily risk, with all63 consecutive qualified returns.

    Every source is bounded through the LAST requested query before numeric
    parsing. Earlier records then use their own past prefix. Invalid assets
    are zero matrix placeholders with a separate validity mask, never zero
    return observations included as valid stocks. The full-matrix ddof1 and
   50% diagonal shrinkage arithmetic is exactly the sealed v9 diag50 method.
    """
    boundary=_cutoff(data_cutoff)
    if estimator!='diag50' or window!=63:raise ValueError('Frozen63-session diag50 estimator only')
    names=list(names);slots=_slots(names,stock_slot_mask)
    queries=_clock(dates,'query')
    if (queries>=boundary).any():raise ValueError('Query must precede explicit data_cutoff')
    if 'SPY' not in prices:raise ValueError('Observed SPY clock required')
    if not len(queries):return {}
    last_query=queries[-1]
    calendar=pd.DatetimeIndex(prices['SPY'].index)
    calendar=_clock(calendar[(calendar<boundary)&(calendar<=last_query)],'SPY')
    if not queries.isin(calendar).all():raise ValueError('Query outside bounded observed SPY calendar')
    returns=pd.DataFrame(np.nan,index=calendar,columns=names)
    for name in names:
        source=prices[name];index=pd.DatetimeIndex(source.index)
        keep=(index<boundary)&(index<=last_query);frame=source.loc[keep].copy(deep=False)
        frame.index=_clock(index[keep],'source');frame=frame.reindex(calendar)
        if 'q' in frame:q=frame.q.eq(1)
        elif {'source_available','tr_chain_available'}.issubset(frame):q=frame.source_available.eq(1)&frame.tr_chain_available.eq(1)
        elif 'tr_qualified' in frame:q=frame.tr_qualified.eq(1)
        elif 'price_qualified' in frame:q=frame.price_qualified.eq(1)
        else:raise ValueError('Explicit quote/return-chain qualification required')
        close=pd.to_numeric(frame.adj_close,errors='raise')
        segment=frame.tr_segment_id if 'tr_segment_id' in frame else pd.Series(1.,index=calendar)
        good=q&np.isfinite(close)&close.gt(0)&segment.notna()
        returns[name]=(close/close.shift(1)-1).where(good&good.shift(1,fill_value=False)&segment.eq(segment.shift(1)))
    cache={}
    for date in queries:
        history=returns.loc[:date].tail(window);x=history.to_numpy(dtype=float)
        valid=np.isfinite(x).all(axis=0) if len(history)==window else np.zeros(len(names),dtype=bool)
        covariance=np.zeros((len(names),len(names)));shrinkage=None
        if valid.any():
            work=np.where(valid[None,:],x,0.);centered=work-work.mean(axis=0)
            sample=centered.T@centered/(window-1)
            covariance=.5*sample+.5*np.diag(np.diag(sample));shrinkage=.5
        covariance=(covariance+covariance.T)/2
        metadata=dict(decision_date=str(date.date()),window_start=str(history.index[0].date()),window_end=str(date.date()),
            max_source_date=str(date.date()),window_sessions=len(history),required_sessions=window,
            estimator='diag50',covariance_horizon=1,covariance_units='daily_return_squared',shrinkage=shrinkage,
            normalization='sample_ddof1',valid_assets=int(valid.sum()),valid_stock_assets=int((valid&slots.to_numpy()).sum()),
            hedge_names=list(slots.index[~slots]),stock_budget_denominator=STOCK_SLOTS,nonlinear_2017_replication=False,
            data_cutoff_exclusive=str(boundary.date()),numeric_prefix_end=str(last_query.date()))
        cache[pd.Timestamp(date)]=dict(names=tuple(names),pooled=covariance,valid=pd.Series(valid,index=names),
            stock_slot_mask=slots.copy(),metadata=metadata)
    return cache


def active_support_targets(base,beta,covariance,tradable,support,*,date=None,allow_spy=False,
        stock_slot_mask=None,vol_target=.12,gross_cap=3.,name_cap=.15,spy_cap=.75,variance_multiplier=1.,data_cutoff='2024-01-01'):
    """Joint dollar+beta projection restricted to selected stocks, optional SPY.

    Full-source tradable STOCK slots determine the capital budget; selected
    support is a distinct constraint. Unselected observed stocks do not become
    missing cash slots. Every hedge counts in gross, risk and actual account fees.
    One whole-vector scale enforces annual volatility/gross/heterogeneous caps.
    """
    if not isinstance(base,pd.Series) or not base.index.is_unique:raise ValueError('Named base vector required')
    data_cutoff=_cutoff(data_cutoff)
    names=list(base.index);slots=_slots(names,stock_slot_mask)
    date=pd.Timestamp(date if date is not None else base.name)
    if pd.isna(date) or date>=data_cutoff:raise ValueError('Current date must precede explicit data_cutoff')
    if not np.isfinite([gross_cap,name_cap,spy_cap,variance_multiplier]).all() or not 0<gross_cap<=3 or not 0<name_cap<=.15 or not 0<spy_cap<=.75 or variance_multiplier<=0:
        raise ValueError('Declared gross/name/SPY/risk-scale bounds required')
    if vol_target is not None and (not np.isfinite(vol_target) or vol_target<=0):raise ValueError('Positive annual target required')
    w0=_series(base,names,'base');b=_series(beta,names,'beta')
    if not np.isfinite(w0).all():raise ValueError('Finite base weights required')
    q=_series(tradable,names,'tradable',True);selected=_series(support,names,'support',True)
    if tuple(covariance['names'])!=tuple(names):raise ValueError('Risk cohort ordering differs')
    meta=covariance['metadata']
    if meta['covariance_horizon']!=1 or pd.Timestamp(meta['decision_date'])!=date or pd.Timestamp(meta['max_source_date'])>date or pd.Timestamp(meta['window_end'])>date:
        raise ValueError('Exact current daily risk required')
    valid=_series(covariance['valid'],names,'risk validity',True)
    known=q&np.isfinite(b)
    # Source availability determines risk-feasible stock budget independently of
    # desired support. Covariance validity is required for every held candidate.
    stock_known=known&slots
    limit=float(gross_cap*min(1.,stock_known.sum()/STOCK_SLOTS))
    candidate=stock_known&selected&valid
    hedges=~slots
    if allow_spy:
        if 'SPY' not in names or bool(slots.loc['SPY']):raise ValueError('Explicit non-stock SPY slot required')
        candidate |= hedges&known&valid
    account_eligible=stock_known.copy()
    if allow_spy:account_eligible |= hedges&known&valid
    caps=pd.Series(name_cap,index=names);caps.loc[hedges]=spy_cap
    out=pd.Series(0.,index=names)
    detail=dict(decision_date=str(date.date()),status='cash',stock_budget_denominator=STOCK_SLOTS,
        full_tradable_stock_count=int(stock_known.sum()),selected_stock_count=int((candidate&slots).sum()),
        selected_support_mask=(candidate&slots).to_dict(),eligible_mask=account_eligible.to_dict(),
        source_eligible_mask=stock_known.to_dict(),stock_slot_mask=slots.to_dict(),asset_name_caps=caps.to_dict(),
        gross_cap=float(gross_cap),gross_limit=limit,name_cap=float(name_cap),spy_cap=float(spy_cap),
        target_gross=0.,stock_gross=0.,hedge_gross=0.,target_net=0.,target_estimated_beta=0.,
        requested_annual_volatility=vol_target,ex_ante_volatility=0.,variance_multiplier=float(variance_multiplier),
        gross_cap_binding=False,name_cap_binding=False,vol_target_attained=False if vol_target else None,
        selected_stock_nonzero=0,hedge_nonzero=0,outside_support_stock_nonzero=0,
        dense_hedge_added_names=0,all_hedges_count_in_gross=True,data_cutoff_exclusive=str(data_cutoff.date()))
    def finish(status):detail['status']=status;return out,detail
    if candidate.sum()<3 or limit<=0:return finish('cash_insufficient_support')
    matrix=np.asarray(covariance['pooled'],dtype=float)
    if matrix.shape!=(len(names),len(names)):raise ValueError('Full covariance shape required')
    mask=candidate.to_numpy();sub=matrix[np.ix_(mask,mask)]*variance_multiplier
    if not np.isfinite(sub).all() or not np.allclose(sub,sub.T,atol=1e-10,rtol=1e-8):return finish('cash_invalid_covariance')
    if np.linalg.eigvalsh(sub).min()<-1e-10:return finish('cash_nonPSD_covariance')
    rows=np.vstack([np.ones(mask.sum()),b.loc[candidate].to_numpy()])
    _,singular,right=np.linalg.svd(rows,full_matrices=False)
    rank=int(np.count_nonzero(singular>max(rows.shape)*np.finfo(float).eps*singular[0]))
    values=w0.loc[candidate].to_numpy();direction=values-right[:rank].T@(right[:rank]@values)
    gross=float(abs(direction).sum())
    if gross<=1e-12:return finish('cash_empty_feasible_direction')
    annual_vol=float(np.sqrt(max(0.,252*direction@sub@direction)))
    if vol_target is not None and annual_vol<=1e-10:return finish('cash_degenerate_risk')
    requested=float(vol_target/annual_vol if vol_target is not None else 1.)
    gross_bound=limit/gross
    absolute=abs(direction);cap_values=caps.loc[candidate].to_numpy()
    name_bound=float(np.divide(cap_values,absolute,out=np.full_like(absolute,np.inf),where=absolute>1e-14).min())
    scale=min(requested,gross_bound,name_bound)
    if scale<requested:scale*=1-1e-12
    out.loc[candidate]=direction*scale
    vol=float(np.sqrt(max(0.,252*out.loc[candidate].to_numpy()@sub@out.loc[candidate].to_numpy())))
    detail.update(target_gross=float(out.abs().sum()),stock_gross=float(out.loc[slots].abs().sum()),hedge_gross=float(out.loc[hedges].abs().sum()),
        target_net=float(out.sum()),target_estimated_beta=float(out.loc[candidate]@b.loc[candidate]),
        ex_ante_volatility=vol,base_ex_ante_volatility=annual_vol,scale_requested=requested,scale_applied=float(scale),
        gross_cap_binding=bool(requested>gross_bound+1e-12 and gross_bound<=name_bound+1e-12),
        name_cap_binding=bool(requested>name_bound+1e-12 and name_bound<=gross_bound+1e-12),
        vol_target_attained=bool(abs(vol-vol_target)<1e-8) if vol_target else None,
        selected_stock_nonzero=int((out.loc[slots].abs()>1e-12).sum()),hedge_nonzero=int((out.loc[hedges].abs()>1e-12).sum()),
        outside_support_stock_nonzero=int((out.loc[slots&~selected].abs()>1e-12).sum()),constraint_rank=rank)
    if abs(out.sum())>1e-9 or abs(out.loc[candidate]@b.loc[candidate])>1e-9 or out.abs().sum()>limit+1e-9 or (out.abs()>caps+1e-9).any():raise AssertionError('Support hedge constraints failed')
    if detail['outside_support_stock_nonzero']:raise AssertionError('Unselected stock introduced')
    return finish('capped' if scale<requested else 'allocated')



# Primary full-support sizing preserves the EXACT sealed-v8 mask/budget convention.
def _named(value, names, label, *, finite=False):
    if not isinstance(value, pd.Series) or not value.index.is_unique or set(value.index) != set(names):
        raise ValueError('Full unique named '+label+' Series required')
    result = value.reindex(names)
    if finite:
        result = result.astype(float)
        if not np.isfinite(result).all():
            raise ValueError('Finite '+label+' required')
    return result


def _mask(value, names, label):
    result = _named(value, names, label)
    if not result.isin([False, True, 0, 1]).all():
        raise ValueError('Literal nonmissing '+label+' mask required')
    return result.astype(bool)


def _risk(record, names, date, conditional=False,data_cutoff='2024-01-01'):
    data_cutoff=_cutoff(data_cutoff)
    """Daily cache record only, with its independent qualified-history mask."""
    if tuple(record['names']) != tuple(names):
        raise ValueError('Covariance cohort ordering differs')
    metadata = record['metadata']
    if metadata.get('covariance_horizon') != 1:
        raise ValueError('Daily covariance required')
    date = pd.Timestamp(date)
    if date >= data_cutoff or (pd.Timestamp(metadata['window_end']) > date
                           or pd.Timestamp(metadata['max_source_date']) > date):
        raise ValueError('Future or beyond-boundary risk input')
    if pd.Timestamp(metadata['decision_date']) != date:
        raise ValueError('Exact current covariance date required')
    matrix = np.asarray(record['conditional' if conditional else 'pooled'], dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError('Full daily covariance shape required')
    return matrix, _mask(record['valid'], names, 'covariance validity'), dict(metadata)


def allocate_target(base_weights, beta, covariance, eligible, *, date=None,
                    vol_target=None, gross_cap=1., name_cap=.10,
                    gross_multiplier=1., conditional=False,
                    sector_map=None, sector_neutral=False,data_cutoff='2024-01-01'):
    """Project the declared base direction, then continuously scale its risk.

    vol_target is annualized decimal ex-ante volatility, NOT realized volatility.
    A value may scale UP. Upper bounds are gross_cap*eligible/110 and name_cap;
    an unattainable target remains below target with a binding diagnostic.
    gross_multiplier=2 with no vol_target is the static-scaling comparator.
    No account NAV, future realized volatility, or trading outcome is an input.
    """
    data_cutoff=_cutoff(data_cutoff)
    if conditional:raise ValueError('Only frozen pooled diag50 risk is declared')
    if not isinstance(base_weights, pd.Series) or not base_weights.index.is_unique:
        raise ValueError('Named base weights required')
    names = list(base_weights.index)
    if not 1 <= len(names) <= MAX_HISTORY_NAMES:
        raise ValueError('One-to110 declared cohort names required')
    if gross_cap not in (1., 2., 3.) or name_cap not in (.10, .15):
        raise ValueError('Declared sizing gross1/2/3 and name.10/.15 variants only')
    if not np.isfinite(gross_multiplier) or gross_multiplier <= 0:
        raise ValueError('Positive static multiplier required')
    if vol_target is not None and (not np.isfinite(vol_target) or vol_target <= 0):
        raise ValueError('Positive annual volatility target required')
    if vol_target is not None and gross_multiplier != 1:
        raise ValueError('Volatility and static scaling are separate declared arms')
    date = pd.Timestamp(date if date is not None else base_weights.name)
    if pd.isna(date) or date >= data_cutoff:
        raise ValueError('Decision must precede explicit data_cutoff')
    base = _named(base_weights, names, 'base weights', finite=True)
    b = _named(beta, names, 'fresh beta').astype(float)
    mask = _mask(eligible, names, 'eligibility') & np.isfinite(b)
    matrix, valid, metadata = _risk(covariance, names, date, conditional,data_cutoff)
    mask &= valid
    out = pd.Series(0., index=names)
    active = int(mask.sum())
    limit = float(gross_cap * min(1.,active / STOCK_SLOTS))
    diagnostics = dict(allocation='neutral_direction_risk_sizing', decision_date=str(date.date()),
        eligible_mask=mask.to_dict(), active_assets=active, cohort_size=STOCK_SLOTS,
        gross_cap=float(gross_cap), gross_limit=limit, cap=float(name_cap),
        requested_annual_volatility=vol_target, annualization_sessions=252,
        covariance_metadata=metadata, conditional_covariance=bool(conditional),
        static_gross_multiplier=float(gross_multiplier), scale_requested=0., scale_applied=0.,
        base_ex_ante_volatility=0., ex_ante_volatility=0., target_exante_vol=0.,
        gross_cap_binding=False, name_cap_binding=False,
        vol_target_attained=False if vol_target is not None else None,
        target_gross=0., target_net=0., target_estimated_beta=0., max_abs_weight=0.,
        base_nonzero_names=int(base.abs().gt(1e-12).sum()), target_nonzero_names=0,
        hedge_names_added=0, signal_signs_flipped=0,
        realized_volatility_used=False, source_names_without_complete_risk=int((~valid).sum()))
    diagnostics['data_cutoff_exclusive']=str(data_cutoff.date())
    def finish(status):
        diagnostics['status'] = status
        return out, diagnostics
    if active < 3:
        return finish('cash_insufficient_active_members')
    sub = matrix[np.ix_(mask, mask)]
    if not np.isfinite(sub).all() or not np.allclose(sub, sub.T, atol=1e-10, rtol=1e-8):
        return finish('cash_invalid_active_covariance')
    sub = (sub + sub.T) / 2
    minimum = float(np.linalg.eigvalsh(sub).min())
    diagnostics['covariance_min_eigenvalue'] = minimum
    if minimum < -1e-10:
        return finish('cash_nonPSD_covariance')
    rows = [np.ones(active), b.loc[mask].to_numpy()]
    if sector_neutral:
        if sector_map is None:
            raise ValueError('Explicit sector mapping required')
        labels = pd.Series(sector_map).reindex(names)
        if labels.loc[mask].isna().any():
            raise ValueError('Known sector required for active members')
        rows.extend(labels.loc[mask].eq(label).to_numpy(dtype=float) for label in sorted(labels.loc[mask].unique()))
    constraints = np.vstack(rows)
    _, singular, right = np.linalg.svd(constraints, full_matrices=False)
    rank = int(np.count_nonzero(singular > max(constraints.shape)*np.finfo(float).eps*singular[0]))
    diagnostics['constraint_rank'] = rank
    x = base.loc[mask].to_numpy()
    direction = x - right[:rank].T @ (right[:rank] @ x)
    gross = float(np.abs(direction).sum())
    largest = float(np.abs(direction).max(initial=0))
    if gross <= 1e-12 or largest <= 1e-12:
        return finish('cash_no_feasible_direction')
    annual_variance = float(252 * direction @ sub @ direction)
    annual_vol = float(np.sqrt(max(0., annual_variance)))
    diagnostics['base_ex_ante_volatility'] = annual_vol
    if vol_target is not None and annual_vol <= 1e-10:
        return finish('cash_degenerate_risk_estimate')
    requested = float(vol_target / annual_vol if vol_target is not None else gross_multiplier)
    gross_bound = limit / gross
    name_bound = name_cap / largest
    applied = min(requested, gross_bound, name_bound)
    # Tiny inward contraction only when a hard cap binds; equalities preserved.
    binding = applied < requested
    if binding:
        applied *= 1-1e-12
    out.loc[mask] = direction * applied
    realized_estimate = float(np.sqrt(max(0., 252*out.loc[mask].to_numpy() @ sub @ out.loc[mask].to_numpy())))
    diagnostics.update(scale_requested=requested, scale_applied=float(applied),
        ex_ante_volatility=realized_estimate, target_exante_vol=realized_estimate,
        gross_cap_binding=bool(requested > gross_bound+1e-12 and gross_bound <= name_bound+1e-12),
        name_cap_binding=bool(requested > name_bound+1e-12 and name_bound <= gross_bound+1e-12),
        vol_target_attained=bool(abs(realized_estimate-vol_target) <= 1e-8) if vol_target is not None else None,
        target_gross=float(out.abs().sum()), target_net=float(out.sum()),
        target_estimated_beta=float(out.loc[mask] @ b.loc[mask]), max_abs_weight=float(out.abs().max()),
        target_nonzero_names=int(out.abs().gt(1e-12).sum()),
        hedge_names_added=int((out.abs().gt(1e-12) & base.abs().le(1e-12)).sum()),
        signal_signs_flipped=int((out.mul(base).lt(-1e-12)).sum()))
    if abs(out.sum()) > 1e-9 or abs(out.loc[mask] @ b.loc[mask]) > 1e-9:
        raise AssertionError('Joint risk sizing neutrality failed')
    if out.abs().sum() > limit+1e-10 or out.abs().max() > name_cap+1e-10:
        raise AssertionError('Risk sizing caps failed')
    return finish('vol_target_capped' if vol_target is not None and binding else 'allocated')

