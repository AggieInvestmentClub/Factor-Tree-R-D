from settings import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'h21_legacy':'Original H21 challenger','h21_repaired':'H21 + execution repair','old_rff':'Previous nonlinear model','rff_repaired':'Previous nonlinear + repair',
    'utility5':'Refit: risk aversion 5','utility1':'Refit: risk aversion 1','net_return':'Refit: maximize mean return','log_growth':'Refit: maximize log growth',
    'growth_v16':'Log growth, 16% risk target','growth_v20':'Log growth, 20% risk target','growth_long25_v12':'Log growth + 25% long, 12% risk',
    'growth_long25_v16':'Log growth + 25% long, 16% risk','growth_long50_v12':'Log growth + 50% long, 12% risk','growth_long50_v16':'Log growth + 50% long, 16% risk',
    'market_long25':'25% market basket only','market_long50':'50% market basket only',
    'rff_gated':'Nonlinear RFF, gated','rff_unchanged_no_gate':'Same RFF coefficients, gate removed','rff_continuous':'Nonlinear RFF, retrained continuous',
    'linear_gated':'Linear, retrained gated','linear_continuous':'Linear, retrained continuous','mlp_gated':'Neural MLP, retrained gated','mlp_continuous':'Neural MLP, retrained continuous'}

def fmt(x,kind='pct'):
    if pd.isna(x):return '—'
    return f'{x:.2%}' if kind=='pct' else f'{x:.3f}'

def table(frame,cols):
    rows=[]
    for _,r in frame.iterrows():
        cells=[]
        for c,title,kind in cols:
            value=LABELS.get(r[c],r[c]) if kind=='text' else fmt(r[c],kind)
            cells.append('<td>'+html.escape(str(value))+'</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for _,t,_ in cols)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'

def main():
    out=ROOT/'reports';p=pd.read_csv(out/'performance.csv');primary=p.loc[p.period.eq('primary')].set_index('arm');full=p.loc[p.period.eq('full')].set_index('arm')
    diagnostics=pd.read_csv(out/'account_diagnostics.csv').set_index('arm');returns=pd.read_csv(out/'daily_returns.csv.gz',index_col=0,parse_dates=True);q=read(out/'qa.json');status=read(out/'status.json')
    growth=pd.read_csv(out/'paired_growth.csv');borrow=pd.read_csv(out/'borrow_sensitivity.csv');annual=pd.read_csv(out/'annual.csv');assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(11,5));keys=['h21_legacy','h21_repaired','rff_repaired','log_growth','growth_long25_v12','growth_long50_v12']
    for n in keys:ax.plot((1+returns.loc['2012':'2023',n]).cumprod(),label=LABELS[n],lw=1.6)
    ax.set(ylabel='Growth of $1, after modeled trading costs',title='Primary period: 2012–2023');ax.legend(fontsize=8,loc='upper left',ncol=2);fig.tight_layout();fig.savefig(assets/'wealth.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5))
    for n,r in primary.iterrows():
        if n.startswith('market'):marker='s';color='#848b96'
        elif 'long' in n:marker='^';color='#c35b22'
        elif n.startswith('h21'):marker='D';color='#304e72'
        else:marker='o';color='#23826c'
        ax.scatter(r.volatility*100,r.cagr*100,color=color,marker=marker,s=60)
        short={'growth_long25_v12':'25% long / 12% target','growth_long25_v16':'25% long / 16% target','growth_long50_v12':'50% long / 12% target','growth_long50_v16':'50% long / 16% target','growth_v16':'16% risk target','growth_v20':'20% risk target',
            'h21_legacy':'H21 original','h21_repaired':'H21 repaired','old_rff':'Previous nonlinear','rff_repaired':'Nonlinear + repair','utility5':'Risk penalty 5','utility1':'Risk penalty 1','net_return':'Mean return','log_growth':'Log growth','market_long25':'25% market only','market_long50':'50% market only'}[n]
        offset,align={'growth_v16':((-5,-14),'right'),'growth_v20':((5,7),'left'),'rff_repaired':((-5,5),'right'),'utility5':((0,-13),'center'),'growth_long50_v12':((-5,6),'right'),'growth_long50_v16':((-5,-14),'right')}.get(n,((5,5),'left'))
        ax.annotate(short,(r.volatility*100,r.cagr*100),xytext=offset,textcoords='offset points',fontsize=7,ha=align)
    ax.set(xlabel='Realized annual volatility (%)',ylabel='CAGR (%)',title='2012–2023 growth and realized risk');fig.tight_layout();fig.savefig(assets/'risk_return.png',dpi=170);plt.close(fig)
    frame=primary.reset_index();frame['full_cagr']=frame.arm.map(full.cagr)
    cols=[('arm','Variant','text'),('cagr','CAGR','pct'),('sharpe','Sharpe','num'),('volatility','Volatility','pct'),('max_drawdown','Max drawdown','pct'),('spy_beta','Realized SPY beta','num'),('full_cagr','2007–23 CAGR','pct')]
    compare=frame.loc[frame.arm.isin(['h21_legacy','h21_repaired','old_rff','rff_repaired'])]
    positive=status['positive_growth_holm_both_blocks'];evidence='; '.join(LABELS[a]+' versus '+LABELS[b] for a,b in positive) if positive else 'None'
    funding=borrow.loc[borrow.period.eq('primary')].pivot(index='arm',columns='annual_borrow_bps',values='cagr').reindex(frame.arm).reset_index().rename(columns={0:'zero',100:'one',300:'three'})
    diag=diagnostics.reset_index();diagcols=[('arm','Variant','text'),('primary_flat_fraction','Cash time','pct'),('mean_target_vol','Mean forecast vol','pct'),('mean_target_bias','Mean net target','pct'),('estimated_beta_mean','Mean estimated beta','num'),('submissions','Submissions','num'),('margin_calls','Margin calls','num')]
    inf=growth.loc[growth.block_length.eq(60)];icols=[('arm','Candidate','text'),('baseline','Reference','text'),('delta_annual_log_growth','Δ annual log growth','pct'),('ci_low','95% lower','pct'),('ci_high','95% upper','pct'),('p_holm','Holm p','num')]
    er=p.loc[p.period.str.startswith('era')].pivot(index='arm',columns='period',values='cagr').reindex(frame.arm).reset_index();ercols=[('arm','Variant','text'),('era_2012_2015','2012–15 CAGR','pct'),('era_2016_2019','2016–19 CAGR','pct'),('era_2020_2023','2020–23 CAGR','pct')]
    rff=primary.loc['rff_repaired'];old=primary.loc['old_rff'];bias25=primary.loc['growth_long25_v12'];bias50=primary.loc['growth_long50_v12'];neutral=primary.loc['log_growth']
    decomposition=pd.read_csv(out/'exposure_decomposition.csv')
    gapfile=out/'surrogate_engine_gap.csv';gap=''
    if gapfile.exists():
        gf=pd.read_csv(gapfile);gap='<h2>Remaining training/account mismatch</h2><p>Identical fixed six-period diagnostic windows, cold-start accounts. These windows diagnose approximation error; they do not select models.</p>'+table(gf,[('year','Year','text'),('surrogate_net_return','21-session surrogate','pct'),('legacy_net_return','Original engine','pct'),('repaired_net_return','Repaired engine','pct'),('repaired_gap','Surrogate minus repaired','pct')])
    followup='';f32=ROOT.parent/'v32_continuous_exposure/reports/performance.csv'
    if f32.exists():
        c=pd.read_csv(f32);c=c.loc[c.period.eq('primary')&~c.arm.eq('h21_legacy')]
        followup='<h2>Follow-up: does the model need to relearn for less cash time?</h2><p>A separate fixed factorial compares linear, nonlinear-feature and small neural learners under the original gate and continuous exposure. A fixed-coefficient control separates gate removal from retraining. <a href="../v32_continuous_exposure/index.html">Read the full cash-regime experiment and its paired tests.</a></p>'+table(c,[('arm','Model/regime','text'),('cagr','CAGR','pct'),('sharpe','Sharpe','num'),('volatility','Volatility','pct'),('max_drawdown','Max drawdown','pct'),('flat_fraction','Cash time','pct')])
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v31 · Growth and controlled long bias</title>
    <style>body{{margin:0;background:#f5f7fa;color:#19283b;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 28px}}h1{{font-size:34px;line-height:1.15}}h2{{margin-top:38px;font-size:23px}}p{{max-width:1000px}}.note{{background:#fff3dd;border-left:4px solid #bc7b23;padding:14px 18px}}.card{{background:white;padding:18px 22px;border-radius:8px;margin:18px 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{padding:9px 10px;border-bottom:1px solid #e2e7ed;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{background:#e9eff5}}img{{width:100%;height:auto;background:white}}a{{color:#245d9b}}code{{font-size:12px;overflow-wrap:anywhere}}.small{{font-size:13px;color:#536477}}</style><main>
    <p class="small">v31 · Fixed pre-2024 data · 21-session research challenger · 15 new account runs, 68 annual fits</p>
    <h1>Growth objectives and controlled long bias</h1>
    <p>The original H21 benchmark remains unchanged. This study separates execution repair, objective choice, risk-target changes and deliberate market exposure. All numerical returns below are after the original 2 bps per side trading cost, with zero financing, borrow and cash interest unless explicitly labeled otherwise.</p>
    <div class="note"><strong>Research status:</strong> reused historical development sample, not a fresh holdout. Source valuation qualifications remain binding. {sum(v['provisional_sessions']>0 for v in q['accounts'].values())} of 15 new accounts contain provisional valuation sessions. No strategy was automatically promoted.</div>
    <h2>What changed</h2>
    <p>A stale target containing a stock with missing beta previously sent the entire portfolio to cash. The research engine excludes unavailable names and reprojects eligible holdings. It preserves next-close fills, unavailable-quote order aborts, corporate distributions, margin checks and fee accounting. With both new flags disabled, the H21 replay differs from the original by at most {q['baseline_replay_error']:.2g} across numeric ledger fields.</p>
    <p>On the previous nonlinear signal alone, this repair changes primary-period CAGR from {old.cagr:.2%} to {rff.cagr:.2%}, volatility from {old.volatility:.2%} to {rff.volatility:.2%}, and cash time from {old.flat_fraction:.1%} to {rff.flat_fraction:.1%}. Restored exposure is not automatically profitable exposure.</p>
    {table(compare,cols)}
    <p>The previous nonlinear model's annual arithmetic mean barely changes: {old.annual_mean_return:.2%} versus {rff.annual_mean_return:.2%} after repair. Higher variance therefore explains much of the lower compound growth. The paired return decomposition below groups days by the original account's closing cash state. It is additive arithmetic attribution, not a causal estimate of missed trading opportunities: changing holdings and guards affects later days as well.</p>
    {table(decomposition,[('original','Original model','text'),('category','Original closing state','text'),('sessions','Sessions','num'),('delta_annual_mean_contribution','Contribution to Δ annual mean','pct')])}
    <p>Keeping the old objective but selecting regularization with the repaired engine changes {len(q['utility5_regularization_changes'])} annual fits. All unchanged-penalty years reproduce their old targets exactly. This isolates validation-regime adaptation from new model architecture.</p>
    <h2>What controlled long bias means</h2>
    <p>At 3× gross, a 25% net-long book has 162.5% long and 137.5% short exposure; a 50% net-long book has 175% long and 125% short. Dollar net and market beta are distinct quantities. Here, a broad long-only basket is constructed to have estimated beta one, so its net capital allocation and intended beta contribution are both 0.25 or 0.50. Realized beta can differ.</p>
    <p>The market basket uses only eligible stocks and decision-known betas. Its exposure is reduced when necessary to meet the total forecast-risk or exposure budget. Neutral alpha is contracted to fit beside it; the overlay receives no extra risk allowance. It persists while the old alpha gate is off. Daily decisions check beta and dollar-net deviations of 0.05, with execution at the next close. Thus a guard is a trading rule, not a guarantee that realized exposures never cross a band.</p>
    <p>The 12% risk versions produce CAGR {bias25.cagr:.2%} with the 25% bias and {bias50.cagr:.2%} with the 50% bias, versus {neutral.cagr:.2%} for neutral log growth. Their descriptive SPY betas are {bias25.spy_beta:.3f}, {bias50.spy_beta:.3f}, and {neutral.spy_beta:.3f}, respectively. Market-only controls below help distinguish market exposure from the stock-selection component; they share caps but are not matched on realized volatility.</p>
    <h2>All fixed variants</h2>
    <p>Primary period 2012–2023. Sharpe uses the inherited zero cash-rate convention. Drawdowns are measured from a starting wealth of one within each period.</p>
    {table(frame,cols)}<img src="assets/wealth.png" alt="Primary-period wealth paths of six fixed variants">
    <img src="assets/risk_return.png" alt="Risk and return of every account">
    <h2>How the learner was changed</h2>
    <p>The same 58 causal price inputs, 32 fixed nonlinear features, seed, 100 training epochs, annual ten-year rolling windows and two regularization choices are retained. Four objectives are tested: mean net return minus 2.5 times variance; mean minus 0.5 times variance; mean return alone under the portfolio limits; and expected log wealth growth. Inner and outer label endpoints must mature strictly before their cutoffs. Regularization is selected with the repaired daily account engine and the corresponding objective. Early 2007/2008 fits use the fixed fallback penalty.</p>
    <p>Expected log growth aligns with compounding, but its empirical maximization can overfit. This motivates a bounded test, not an assumption that it will beat the old loss. See <a href="https://web.stanford.edu/~boyd/papers/pdf/robust_kelly.pdf">Sun and Boyd, Distributional Robust Kelly Gambling</a> for the growth objective and sensitivity to estimation error. The higher-risk variants rescale the learned log-growth direction; they are not separately trained at 16% or 20% risk.</p>
    <p>Gross exposure stays capped at 3×, reduced with the active cohort; each name stays capped at 15%; the old 20% L1 trading buffer, 3.3× gross guard, covariance estimator and 21-session schedule remain. A risk target is an upper allocation budget, not a promise to attain that volatility when the gross cap binds.</p>
    {table(diag,diagcols)}
    <h2>Growth inference and time stability</h2>
    <p>The fixed primary endpoint is annual mean log return. Paired circular block bootstrap uses 2,000 draws and 20/60-session blocks; Holm adjustment covers the 14 planned growth comparisons, but not previous project searches. Positive differences passing both block lengths: <strong>{html.escape(evidence)}</strong>. Intervals below are pointwise, not simultaneous. Secondary Sharpe comparisons are in the linked data.</p>
    {table(inf,icols)}{table(er,ercols)}
    <h2>Short-borrow sensitivity</h2>
    <p>These are approximate return-path charges of 100 or 300 basis points annually on prior-close short exposure. They hold trades fixed and do not rerun NAV-dependent guards or model name-specific borrow availability, financing, rebates or collateral yield. They are sensitivity scenarios, not historical net performance estimates.</p>
    {table(funding,[('arm','Variant','text'),('zero','0 bp borrow CAGR','pct'),('one','100 bp borrow CAGR','pct'),('three','300 bp borrow CAGR','pct')])}
    {gap}{followup}
    <h2>Validation and reproducibility</h2>
    <p>Six focused tests cover missing-risk repair, missing execution quotes, controlled-bias guards, joint allocation constraints, old-loss identity and log-growth gradients. Final checks verify {q['models']} model files, {q['inner_account_validations']} inner account validations, 3,060 scheduled target vectors, unchanged original-source hashes, target regeneration, strict label purges and {q['performance_rows']} independently recomputed metric rows. Maximum regenerated-target error: {q['recomputed_target_max_error']:.2g}; maximum metric error: {q['performance_max_error']:.2g}. Arithmetic and noncash exposure checks pass; these checks do not certify unresolved source valuations.</p>
    <p>The gradient still uses 21-session return segments rather than differentiating through every daily guard, quote failure or corporate cash event. The daily engine, not the training surrogate, determines the reported account outcomes. No post-2023 returns were used, and the old H21 benchmark and source engine remain unchanged.</p>
    <p><a href="protocol.json">Frozen protocol</a> · <a href="engine_changes.diff">Engine diff</a> · <a href="reports/qa.json">Validation audit</a> · <a href="reports/performance.csv">All period metrics</a> · <a href="reports/annual.csv">Annual metrics</a> · <a href="reports/paired_growth.csv">Growth inference</a> · <a href="reports/paired_sharpe_secondary.csv">Secondary Sharpe inference</a> · <a href="reports/account_diagnostics.csv">Execution diagnostics</a> · <a href="reports/borrow_sensitivity.csv">Borrow scenarios</a></p>
    <p class="small">Protocol SHA-256: <code>{status['protocol_sha256']}</code></p></main></html>'''
    (ROOT/'index.html').write_text(body,encoding='utf-8');print(ROOT/'index.html')
if __name__=='__main__':main()
