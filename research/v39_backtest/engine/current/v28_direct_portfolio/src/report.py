"""Standalone research report from completed, audited v28 results only."""
from pathlib import Path
import sys,json,hashlib,html,base64
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
NAMES={'h21_projected_scaled_matched':'Unchanged H21 challenger','h21_replay':'Unchanged H21 replay','huber_ridge_10y':'Original H5 baseline',
 'h5_sector_risk_weekly':'Common-census H5 baseline','sample_aligned_h21':'H21 regression / matched training dates',
 'direct_risk_only':'Direct: risk only','direct_risk_cost':'Direct: risk + costs',
 'direct_risk_cost_persist':'Direct: all-input persistence','direct_risk_cost_persist_target12':'All-input persistence / original 12% sizing',
 'direct_price_persistence':'Price-factor persistence / balanced regularization',
 'direct_price_persistence_target12':'Price-factor persistence / original 12% sizing'}
ORDER=['huber_ridge_10y','h21_projected_scaled_matched','sample_aligned_h21','direct_risk_only','direct_risk_cost','direct_price_persistence','direct_price_persistence_target12','direct_risk_cost_persist','direct_risk_cost_persist_target12','h5_sector_risk_weekly']
def table(frame):return '<div class="scroll">'+frame.to_html(index=False,border=0,escape=True)+'</div>'
def main():
    d=ROOT/'reports/account_comparison';status=read(d/'analysis_status.json');audit=read(ROOT/'reports/validation.json')
    independent=read(ROOT/'reports/independent_target_audit.json');combined=read(ROOT/'reports/combined_status.json')
    diagnostic=read(ROOT/'persistence_control/validation.json')
    assert status['completed'] and audit['passed'] and independent['passed'] and combined['completed'] and diagnostic['passed']
    perf=pd.read_csv(d/'all_performance.csv');pairs=pd.read_csv(d/'paired_primary.csv');annual=pd.read_csv(d/'all_annual.csv')
    returns=pd.read_csv(d/'all_daily_returns.csv.gz',index_col=0,parse_dates=True)
    spec=read(ROOT/'protocol.json');notes=read(ROOT/'reports/interpretation.json')
    figs=ROOT/'reports/figures';figs.mkdir(exist_ok=True);artifacts=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(figsize=(12,5));order=ORDER[:7];x=np.arange(len(order));width=.36
    for offset,period,color in [(-width/2,'primary','#176f78'),(width/2,'full','#a4b6bf')]:
        f=perf.loc[perf.period.eq(period)].set_index('arm').loc[order]
        ax.bar(x+offset,f.sharpe,width,label='2012–2023' if period=='primary' else '2007–2023',color=color)
    ax.axhline(0,color='#34485c',linewidth=.7);ax.set_xticks(x,[NAMES[n].replace(' / ','\n').replace(' + ',' +\n') for n in order],rotation=18,ha='right',fontsize=8)
    ax.set_ylabel('Sharpe after modeled trading costs');ax.legend(frameon=False);ax.set_title('Direct portfolio learning versus the unchanged challenger')
    fig.tight_layout();fig.savefig(figs/'sharpe.png',dpi=170);fig.savefig(figs/'sharpe.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8));chosen=[ORDER[0],ORDER[1],ORDER[5],ORDER[6]]
    colors=['#9a9a9a','#203e59','#138379','#b66d20']
    for ax,left,title in zip(axes,['2007','2012'],['Full research history','Primary window, reset to $1 at its start']):
        for n,c in zip(chosen,colors):ax.plot((1+returns.loc[left:,n]).cumprod(),label=NAMES[n],color=c,lw=1.3)
        ax.set_title(title);ax.set_ylabel('Compounded wealth');ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figs/'wealth.png',dpi=170);fig.savefig(figs/'wealth.pdf');plt.close(fig)
    rows=perf.loc[perf.period.eq('primary')].set_index('arm').loc[ORDER].reset_index()
    columns=['arm','sharpe','cagr','annual_mean_return','volatility','max_drawdown','turnover_per_year','cost_per_year']
    view=rows[columns].copy();view['arm']=view.arm.map(NAMES)
    for c in ('cagr','annual_mean_return','volatility','max_drawdown','cost_per_year'):view[c]=view[c].map(lambda v:f'{100*v:.2f}%')
    for c in ('sharpe','turnover_per_year'):view[c]=view[c].map(lambda v:f'{v:.3f}')
    view.columns=['Model','Sharpe','CAGR','Annual mean','Volatility','Max drawdown','Turnover/year','Cost/year']
    out=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct portfolio learning — controlled H21 study</title>',
        '<style>body{font:16px/1.6 system-ui;color:#203244;background:#f6f7f6;margin:0}main{max-width:1120px;margin:auto;padding:40px 24px;background:white}h1{font-size:34px;line-height:1.2}h2{margin-top:36px}p{max-width:1000px}.note{padding:16px;background:#fff5df;border-left:4px solid #b7892b}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:9px;border-bottom:1px solid #dce2e6;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#087683}.small{font-size:13px;color:#596877}</style></head><body><main>',
        '<p class="small">Completed research · 85 annual fits · 8 new account runs · 2007–2023 only</p>',
        '<h1>'+html.escape(notes['headline'])+'</h1>']
    for p in notes['summary']:out.append('<p>'+html.escape(p)+'</p>')
    out.append('<p class="note">These are conditional research results on reused development history. Every account retains its own valuation qualifications. Borrowing and financing are excluded. No model is automatically promoted.</p>')
    out.append('<h2>Primary comparison: 2012–2023</h2>'+table(view))
    for name in ('sharpe','wealth'):
        p=figs/f'{name}.png';out.append('<img alt="'+name+' comparison" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">')
        artifacts.extend([p,figs/f'{name}.pdf'])
    out.append('<h2>What was learned</h2>')
    for p in notes['findings']:out.append('<p>'+html.escape(p)+'</p>')
    out.append('<h2>Direct economic objective</h2><p>The policy learns the weights of characteristic portfolios directly. Its reward is portfolio return minus covariance risk, trading costs, and coefficient regularization. It does not regress a separate stock-return target.</p>')
    out.append('<p><b>Objective:</b> annualized mean portfolio return − 5/2 × annualized ex-ante variance − 2 bp × annualized turnover − λ/2 × squared coefficient norm.</p>')
    out.append('<p>The original 36 price, volume, availability, and sector inputs remain the information source. Each becomes a feasible dollar- and beta-neutral portfolio using the original 63-session covariance estimator. The persistence version adds a causal 21-session-half-life history of those same characteristics alongside the existing fast EWMA. Joint fitting rewards combinations whose subsequent returns justify their risk and trading.</p>')
    out.append('<p><b>Persistence identification control:</b> before account returns were inspected, fitted coefficients revealed that slow copies of static sector indicators duplicated existing inputs. One additional model therefore applies slow histories only to the 22 price/volume factors. Its regularizer gives exact duplicates the same effective penalty as a single feature. The first model and every original account remain reported; this follow-up is explicitly adaptive.</p>')
    out.append('<p>A signed coefficient L1 limit preserves the 3× gross, 15% name, and 12% risk limits. The economic learner can choose less risk. A separate account applies the original 12% sizing rule to the same learned direction; this is a post-training exposure control. All accounts retain next-close execution, the 21-session schedule, original efficacy gate, 0.2 trading buffer, eligibility rules, and daily risk guards.</p>')
    out.append('<p>Training uses the preceding ten years available from 2002, annual refits, and whole-date mature labels. The final two years of each training window choose regularization; inner labels must mature before validation starts. The sparse 2007 window uses the largest fixed penalty without tuning. A sample-aligned H21 regression control uses exactly the same scheduled training dates.</p>')
    out.append('<p class="note">The training loss is a fixed-notional holding-period approximation. Turnover uses the previous basket drifted by its realized holding returns. It does not differentiate through the account’s gate, buffer, intraperiod safeguards, cash distributions, or NAV normalization. Final results use the unchanged full accounting engine. The persistence comparison also adds model capacity; it does not identify a unique structural persistence parameter.</p>')
    out.append('<h2>Paired uncertainty</h2><p>Six comparisons were declared before account outcomes. Primary inference uses paired 60-session circular blocks; 20-session blocks are the sensitivity check. Holm correction applies only to this family, not earlier research or adaptive study selection. Confidence intervals are pointwise.</p>')
    for block in (60,20):
        q=pairs.loc[pairs.block_length.eq(block),['arm','baseline','delta_sharpe','ci_low','ci_high','p_holm','delta_annual_mean','p_mean_holm']].copy()
        q['arm']=q.arm.map(NAMES);q['baseline']=q.baseline.map(NAMES)
        for c in q.columns[2:]:q[c]=q[c].map(lambda v:f'{v:.4f}')
        out.append(('<h3>60-session blocks</h3>' if block==60 else '<details><summary>20-session sensitivity</summary>')+table(q)+('' if block==60 else '</details>'))
    out.append('<h3>Separate persistence-control family</h3><p>Three additional contrasts use their own Holm3 correction. This does not adjust for choosing the diagnostic after reviewing fitted coefficients.</p>')
    extra=pd.read_csv(d/'paired_persistence_control.csv')
    q=extra.loc[extra.block_length.eq(60),['arm','baseline','delta_sharpe','ci_low','ci_high','p_holm','delta_annual_mean','p_mean_holm']].copy()
    q['arm']=q.arm.map(NAMES);q['baseline']=q.baseline.map(NAMES)
    for c in q.columns[2:]:q[c]=q[c].map(lambda v:f'{v:.4f}')
    out.append(table(q))
    out.append('<h2>Validation and reproducibility</h2>')
    out.append('<p>All 85 annual model records share the declared date census where applicable and pass cutoff/provenance checks. All eight target paths pass risk and neutrality checks. The unchanged challenger replay matches its saved ledger with maximum numerical error '+f"{audit['unchanged_challenger_max_ledger_error']:.2e}"+'.</p>')
    counts=pd.DataFrame([dict(model=NAMES.get(a,a),**v) for a,v in combined['all_account_qualifications'].items()]);out.append(table(counts))
    out.append('<p>The economic objective follows the research direction in <a href="https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier">Jensen, Kelly, Malamud and Pedersen</a>. This small regularized linear policy is an implementation for this system, not a replication of their nonlinear empirical specification.</p>')
    out.append('<details><summary>All periods and annual results</summary>'+table(perf.assign(arm=perf.arm.map(NAMES)))+table(annual.assign(arm=annual.arm.map(NAMES)))+'</details>')
    links=['protocol.json','data_manifest.json','reports/validation.json','reports/independent_target_audit.json','reports/interpretation.json',
        'reports/account_comparison/all_performance.csv','reports/account_comparison/paired_primary.csv','reports/account_comparison/all_annual.csv',
        'reports/account_comparison/paired_persistence_control.csv','reports/descriptive_diagnostics.json','reports/coefficient_composition.csv',
        'reports/sector_exposure_diagnostics.csv','persistence_control/protocol.json','persistence_control/validation.json',
        'src/core.py','src/build_data.py','src/fit_policy.py','src/accounts.py','src/evaluate.py','src/test_policy.py','persistence_control/run.py','persistence_control/evaluate.py']
    out.append('<h2>Files and rerun commands</h2><ul>')
    for name in links:assert (ROOT/name).exists();out.append('<li><a href="'+name+'">'+name+'</a></li>')
    out.append('</ul><p class="small">From this study directory, with the existing research Python environment: <code>python -B src/fit_policy.py policy</code>; <code>python -B src/fit_policy.py aligned</code>; <code>python -B src/accounts.py prepare</code>; <code>python -B src/accounts.py run</code>; <code>python -B src/evaluate.py</code>. The persistence control uses <code>python -B persistence_control/run.py fit</code>, followed by <code>prepare</code>, <code>accounts</code>, then <code>python -B persistence_control/evaluate.py</code>. Existing source artifacts are required and their seals are checked.</p></main></body></html>')
    path=ROOT/'index.html';path.write_text(''.join(out),encoding='utf-8');artifacts.append(path)
    manifest=dict(complete=True,accounts=8,annual_fits=85,protocol_sha256=sha(ROOT/'protocol.json'),
        source_reports={str(p.relative_to(ROOT)):sha(p) for p in [d/'analysis_status.json',d/'all_performance.csv',d/'paired_primary.csv',d/'paired_persistence_control.csv',ROOT/'reports/validation.json',ROOT/'reports/combined_status.json',ROOT/'reports/interpretation.json']},
        output_hashes={str(p.relative_to(ROOT)):sha(p) for p in artifacts},generator_sha256=sha(Path(__file__)),new_tests_or_model_selection=False)
    (ROOT/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(path)

if __name__=='__main__':main()
