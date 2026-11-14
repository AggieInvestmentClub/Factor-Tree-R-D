"""Descriptive forecast/portfolio target bridge; never changes an order.

H5 overlapping forward basket payoffs are GROSS diagnostics, not investable
daily P&L. Incomplete future labels remain in the census, never drop a trade.
"""
import numpy as np,pandas as pd
from scipy.stats import rankdata

def measure(stages,panel,ungated,gates,names):
    outcomes=panel[panel.model_eligible].set_index(['decision_date','ticker'])
    columns=['target_risk_resid_h5','future_return_h5','future_spy_h5','beta_spy63','idio_vol21']
    records=[]
    for date in stages['raw_score'].index:
        if date>=pd.Timestamp('2024-01-01'):raise ValueError('Reserved outcome forbidden')
        raw=stages['raw_score'].loc[date];members=raw.index[raw.notna()]
        g=outcomes.reindex(pd.MultiIndex.from_product([[date],members],names=['decision_date','ticker']))
        complete=g.label_end_h5.notna().all() and g.label_end_h5.lt('2024-01-01').all() and np.isfinite(g[columns].to_numpy(float)).all()
        rec=dict(decision_date=date,members=len(members),complete_mature_query=bool(complete),scheduled=date in ungated.index)
        if complete:
            y=g.target_risk_resid_h5.to_numpy(float);r=g.future_return_h5.to_numpy(float);beta=g.beta_spy63.to_numpy(float)
            gate_y=r-beta*g.future_spy_h5.to_numpy(float);gate_y-=gate_y.mean();gate_y/=g.idio_vol21.to_numpy(float)*np.sqrt(5)
            def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-12 else np.nan
            for stage in ['raw_score','ewma_score']:
                s=stages[stage].loc[date,members].to_numpy(float)
                centered=rankdata(s)-(len(s)+1)/2;gross=abs(centered).sum()
                w=centered/gross if gross>0 else np.zeros_like(centered)
                rec[stage+'_training_target_rank_ic']=corr(rankdata(s),rankdata(y))
                rec[stage+'_gate_target_pearson_ic']=corr(s,gate_y)
                rec[stage+'_unit_rank_h5_bps']=float(w@r*10000)
            for stage in ['signal','projected']:
                w=stages[stage].loc[date,members].to_numpy(float)
                rec[stage+'_h5_bps']=float(w@r*10000);rec[stage+'_gross']=float(abs(w).sum())
            if date in ungated.index:
                w=ungated.loc[date,members].to_numpy(float);mult=float(gates.loc[date,'multiplier'])
                rec.update(ungated_sized_h5_bps=float(w@r*10000),gated_sized_h5_bps=float(mult*w@r*10000),gate_multiplier=mult,
                    sized_gross=float(abs(w).sum()))
        records.append(rec)
    return pd.DataFrame(records)
