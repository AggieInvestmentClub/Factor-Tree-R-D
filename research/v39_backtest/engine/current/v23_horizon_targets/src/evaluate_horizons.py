"""Prespecified horizon/target/cadence comparisons on a complete common clock.

Primary family: fourteen alternatives versus matched-census H5 sector-risk.
Secondary family: six matched-cadence versus weekly-cadence paired contrasts.
Original Huber and factorial summaries are descriptive. No automatic promotion.
"""
from __future__ import annotations
import argparse
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from horizon_accounts import (ROOT,SOURCE,ORIGINAL,REFERENCE,HORIZONS,TARGETS,WEEKLY,MATCHED,
    ACCOUNTS,ALTERNATIVES,account_spec,current_contract,account_contract,sha,read,write)
from analyze_state_v20 import load_ledger, performance, holm

PERIODS = [('primary','2012-01-01','2023-12-31'),('early','2007-01-01','2011-12-31'),
    ('full','2007-01-01','2023-12-31'),('era_2012_2015','2012-01-01','2015-12-31'),
    ('era_2016_2019','2016-01-01','2019-12-31'),('era_2020_2023','2020-01-01','2023-12-31')]
SEED = 20261111
REPLICATIONS = 2000


def paired_comparisons(returns, contrasts, block, *, family_size, seed=SEED, replications=REPLICATIONS):
    """Paired circular bootstrap, sharing each draw across every contrast."""
    names = list(dict.fromkeys([name for pair in contrasts for name in pair]))
    data = returns[names].to_numpy(float)
    n = len(data)
    if n<=block or not np.isfinite(data).all():
        raise ValueError('Complete paired return clock required')
    mu = data.mean(axis=0)*252
    sigma = data.std(axis=0,ddof=1)*np.sqrt(252)
    if (sigma<=1e-15).any():
        raise ValueError('Zero-variance account Sharpe')
    sharpe = mu/sigma
    rng = np.random.default_rng(seed)
    bm, bs = np.empty((replications,len(names))),np.empty((replications,len(names)))
    for left in range(0,replications,50):
        size = min(50,replications-left)
        starts = rng.integers(0,n,size=(size,int(np.ceil(n/block))))
        index = ((starts[:,:,None]+np.arange(block))%n).reshape(size,-1)[:,:n]
        draw = data[index]
        means = draw.mean(axis=1)*252
        vol = draw.std(axis=1,ddof=1)*np.sqrt(252)
        if (vol<=1e-15).any():
            raise ValueError('Undefined bootstrap Sharpe')
        bm[left:left+size],bs[left:left+size] = means,means/vol
    records = []
    for candidate,reference in contrasts:
        j,k = names.index(candidate),names.index(reference)
        delta,mean_delta = sharpe[j]-sharpe[k],mu[j]-mu[k]
        dist,mean_dist = bs[:,j]-bs[:,k],bm[:,j]-bm[:,k]
        records.append(dict(arm=candidate,baseline=reference,period='primary',block_length=block,
            family_size=family_size,available_contrasts=len(contrasts),replications=replications,seed=seed,sessions=n,
            delta_sharpe=float(delta),ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),
            p_one_sided=float((1+np.sum(dist-delta>=delta))/(replications+1)),
            delta_annual_mean=float(mean_delta),mean_ci_low=float(np.quantile(mean_dist,.025)),mean_ci_high=float(np.quantile(mean_dist,.975)),
            p_mean_one_sided=float((1+np.sum(mean_dist-mean_delta>=mean_delta))/(replications+1)),
            block_role='primary' if block==60 else 'sensitivity',
            interval_scope='Pointwise unadjusted percentile95; not simultaneous',
            inference_scope='Exploratory reused development history; source-qualified; planned-family correction only'))
    result = pd.DataFrame(records)
    if len(result)>family_size:
        raise ValueError('Comparison family larger than declared')
    # An interim weekly report keeps the full planned family of fourteen.
    for source,target in [('p_one_sided','p_holm'),('p_mean_one_sided','p_mean_holm')]:
        padded = np.r_[result[source].to_numpy(),np.ones(family_size-len(result))]
        result[target] = holm(padded)[:len(result)]
    return result


def factorial_tables(perf):
    cells = perf.loc[perf.arm.isin(WEEKLY)].copy()
    cells['horizon'] = cells.arm.map(lambda a: account_spec(a)['horizon'])
    cells['target'] = cells.arm.map(lambda a: account_spec(a)['target'])
    rows = []
    for period,g in cells.groupby('period'):
        d = g.set_index('arm')
        for target in TARGETS:
            base = f'h5_{target}_weekly'
            for horizon in (10,21):
                candidate = f'h{horizon}_{target}_weekly'
                if candidate not in d.index or base not in d.index:
                    continue
                rows.append(dict(period=period,kind='horizon_with_target_fixed',candidate=candidate,baseline=base,
                    delta_sharpe=d.loc[candidate,'sharpe']-d.loc[base,'sharpe'],
                    delta_annual_mean=d.loc[candidate,'annual_mean_return']-d.loc[base,'annual_mean_return']))
        for horizon in HORIZONS:
            base = f'h{horizon}_sector_risk_weekly'
            for target in ('projected_scaled','projected_rank'):
                candidate = f'h{horizon}_{target}_weekly'
                if candidate not in d.index or base not in d.index:
                    continue
                rows.append(dict(period=period,kind='target_with_horizon_fixed',candidate=candidate,baseline=base,
                    delta_sharpe=d.loc[candidate,'sharpe']-d.loc[base,'sharpe'],
                    delta_annual_mean=d.loc[candidate,'annual_mean_return']-d.loc[base,'annual_mean_return']))
    contrasts = pd.DataFrame(rows)
    if len(contrasts):
        contrasts['interpretation'] = 'Prespecified descriptive factorial contrasts; no separate significance or model-selection claim'
    return cells,contrasts


def evaluate(stage='all'):
    protocol,contract = current_contract(),account_contract()
    expected = list((ORIGINAL,*WEEKLY) if stage=='weekly' else ACCOUNTS)
    baseline_original = read(SOURCE/f'accounts/{ORIGINAL}_complete.json')
    local_original = read(ROOT/f'accounts/{ORIGINAL}_complete.json')
    errors = local_original['baseline_identity_max_errors']
    assert errors and max(errors.values())<1e-10
    records,ledgers = {},{}
    for name in expected:
        info = account_spec(name)
        file = ROOT/f'accounts/{name}_ledger.csv.gz'
        meta = read(ROOT/f'accounts/{name}_complete.json')
        assert meta['protocol_sha256']==protocol and meta['account_contract_sha256']==contract
        assert meta['ledger_sha256']==sha(file) and meta['audit']['arithmetic_passed']
        for key in ('source_approval_sha256','prices_sha256','events_sha256','claims_sha256','noncash_census_sha256','account_code_hashes'):
            assert meta['source_status'][key]==baseline_original[key]
        gate = ROOT/f"account_inputs/original_huber_gate_cadence{info['cadence']}.csv"
        assert meta['gate_sha256']==sha(gate)
        ledgers[name] = load_ledger(file,name)
        records[name] = dict(**info,ledger_sha256=meta['ledger_sha256'],
            arithmetic_passed=meta['audit']['arithmetic_passed'],validated_performance=meta['audit']['validated_performance'],
            provisional_sessions=meta['audit']['provisional_sessions'],noncash_exposure_passed=meta['audit']['noncash_exposure_audit']['passed'],state=meta['state'])
    clock = ledgers[ORIGINAL].index
    assert all(clock.equals(ledger.index) for ledger in ledgers.values())
    returns = pd.DataFrame({name:ledger.daily_return for name,ledger in ledgers.items()})
    assert np.isfinite(returns.to_numpy()).all()
    rows,annual = [],[]
    for name,ledger in ledgers.items():
        prior = ledger.nav.shift(1,fill_value=1.)
        for period,left,right in PERIODS:
            subset = ledger.loc[left:right]
            rows.append(performance(name,subset,prior.reindex(subset.index),period,None))
        for year in range(2007,2024):
            subset = ledger.loc[str(year)]
            item = performance(name,subset,prior.reindex(subset.index),f'year_{year}',None)
            item['year'] = year
            annual.append(item)
    perf = pd.DataFrame(rows)
    primary = returns.loc['2012-01-01':'2023-12-31']
    contrasts = [(name,REFERENCE) for name in ALTERNATIVES if name in ledgers]
    family = pd.concat([paired_comparisons(primary,contrasts,block,family_size=14) for block in (20,60)],ignore_index=True)
    cadence_contrasts = [(name,name.replace('_matched','_weekly')) for name in MATCHED if name in ledgers]
    cadence = (pd.concat([paired_comparisons(primary,cadence_contrasts,block,family_size=6) for block in (20,60)],ignore_index=True)
               if cadence_contrasts else pd.DataFrame())
    cells,factorial = factorial_tables(perf)
    destination = ROOT/'reports/account_comparison'
    destination.mkdir(parents=True,exist_ok=True)
    for filename,frame in [('performance.csv',perf),('annual.csv',pd.DataFrame(annual)),
        ('paired_vs_matched_h5.csv',family),('paired_matched_vs_weekly.csv',cadence),
        ('factorial_weekly_cells.csv',cells),('factorial_weekly_contrasts.csv',factorial)]:
        frame.to_csv(destination/filename,index=False)
    returns.to_csv(destination/'common_daily_returns.csv.gz',index_label='date')
    # Common-census H5 versus old Huber is descriptive, not in the fourteen-arm family.
    original_compare = perf[perf.arm.isin([ORIGINAL,REFERENCE])].copy()
    original_compare.to_csv(destination/'original_vs_common_census_h5.csv',index=False)
    positive = set(ALTERNATIVES)
    for block in (20,60):
        q = family[family.block_length.eq(block)]
        positive &= set(q.loc[q.delta_sharpe.gt(0)&q.p_holm.lt(.05),'arm'])
    write(destination/'analysis_status.json',dict(completed=stage=='all',stage=stage,
        protocol_sha256=protocol,account_contract_sha256=contract,evaluator_sha256=sha(__file__),
        accounts=records,common_clock_sessions=len(clock),primary_sessions=len(primary),
        original_baseline_identity_max_errors=errors,primary_reference=REFERENCE,primary_family_size=14,
        completed_primary_contrasts=len(contrasts),cadence_family_size=6,completed_cadence_contrasts=len(cadence_contrasts),
        blocks=[20,60],bootstrap_replicates=REPLICATIONS,bootstrap_seed=SEED,
        positive_holm_both_blocks=sorted(positive),promotion=False,
        interpretation='Exploratory horizon/target/cadence factorial study with fixed original-Huber H5 efficacy gate; no new independent holdout',
        limitations=['Same-source provisional valuations retained; individual account audit flags remain binding',
                    'Net means after modeled2bps trading costs; excludes borrow/financing',
                    'Daily guards can rebalance between scheduled dates; cadence is not guaranteed fixed holding duration',
                    'Matched-cadence arms share original H5 gate, not horizon-matched efficacy estimates',
                    'Fourteen-alternative correction does not cover prior project searches; factorial tables are descriptive',
                    '2024 onward excluded here; earlier project inspected it, so not pristine']))
    print(perf.loc[perf.period.isin(['primary','early','full']),['arm','period','sharpe','cagr','volatility','max_drawdown']].to_string(index=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['weekly','all'],default='all')
    evaluate(parser.parse_args().stage)
