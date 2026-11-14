"""Self-contained research report with standalone scientific figures."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html,base64
V30=ROOT.parent/'v30_hybrid_stocks'
LABELS={CHALLENGER:'21-session challenger','replay_linear':'Execution-aware linear','replay_random':'Nonlinear random features','replay_mlp':'Small neural network',
    'absolute_pooled_price':'Pooled absolute · price + flags','absolute_pooled':'Pooled absolute · company data','absolute_local':'Separate stock models','absolute_hierarchical':'Shared + stock-specific hybrid','h21_hierarchical_blend':'50/50 challenger + hybrid'}
def table(frame):return frame.to_html(index=False,border=0,escape=True,classes='data')
def perf_table(p,names):
    a=p.loc[p.period.eq('primary')&p.arm.isin(names)].set_index('arm').reindex(names)
    full=p.loc[p.period.eq('full')].drop_duplicates('arm').set_index('arm')
    return table(pd.DataFrame({'Model':[LABELS[n] for n in names],'Sharpe':a.sharpe.map(lambda x:f'{x:.3f}').to_numpy(),'CAGR':a.cagr.map(lambda x:f'{100*x:.2f}%').to_numpy(),
        'Volatility':a.volatility.map(lambda x:f'{100*x:.2f}%').to_numpy(),'Max drawdown':a.max_drawdown.map(lambda x:f'{100*x:.2f}%').to_numpy(),
        'Annual trading fees':a.cost_per_year.map(lambda x:f'{100*x:.2f}%').to_numpy(),'2007–23 Sharpe':[f'{full.loc[n,"sharpe"]:.3f}' for n in names]}))
def img(path):return '<img alt="Net account growth over the primary evaluation period" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def plot(returns,names,file,title):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');colors=['#173a56','#147d80','#de8645','#8b64a9','#5b8c44','#be5668']
    r=returns.loc['2012':'2023'];nav=(1+r[names]).cumprod()
    for n,c in zip(names,colors):ax.plot(nav.index,nav[n],label=LABELS[n],color=c,lw=2.2 if n==CHALLENGER else 1.5)
    ax.set(title=title,ylabel='Growth of $1, after modeled trading fees',xlim=(pd.Timestamp('2012-01-01'),pd.Timestamp('2023-12-31')));ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=8,frameon=False,ncol=2)
    fig.savefig(file,dpi=170);fig.savefig(file.with_suffix('.svg'));plt.close(fig)
def main():
    out=ROOT/'reports';findings=read(out/'interpretation.json');q29=read(out/'validation.json');q30=read(V30/'reports/validation.json');blend=read(V30/'blend/status.json')
    p29=pd.read_csv(out/'performance.csv');p30=pd.read_csv(V30/'reports/performance.csv');pb=pd.read_csv(V30/'blend/performance.csv')
    p=pd.concat([p29,p30,pb]).drop_duplicates(['arm','period']);r29=pd.read_csv(out/'daily_returns.csv.gz',index_col=0,parse_dates=True);r30=pd.read_csv(V30/'reports/daily_returns.csv.gz',index_col=0,parse_dates=True)
    rb=pd.read_csv(V30/'accounts/h21_hierarchical_blend_ledger.csv.gz',index_col=0,parse_dates=True).daily_return
    returns=pd.concat([r29,r30,rb.rename('h21_hierarchical_blend')],axis=1);returns=returns.loc[:,~returns.columns.duplicated()]
    core=[CHALLENGER,*MODELS];hybrid=[CHALLENGER,'absolute_pooled_price','absolute_pooled','absolute_local','absolute_hierarchical','h21_hierarchical_blend']
    plot(returns,core,out/'nonlinear_equity.png','Nonlinear policies · same price data and account constraints')
    plot(returns,hybrid,out/'hybrid_equity.png','Absolute-return and stock-specific pilots · same account constraints')
    fm=pd.read_csv(V30/'reports/forecast_metrics.csv').query('period=="primary"');ft=[]
    for z in fm.itertuples():ft.append({'Model':LABELS[z.model],'Absolute RMSE':f'{100*z.absolute_rmse:.2f}%','Absolute R² vs zero':f'{100*z.absolute_r2_zero:.2f}%','Market-projected RMSE':f'{100*z.projected_rmse:.2f}%','Rank IC':f'{z.rank_ic:.4f}'})
    intervals=[]
    for basepath,family in [(out/'paired_primary.csv','Nonlinear · Holm5'),(V30/'reports/paired_primary.csv','Stock pilot · Holm4'),(V30/'blend/paired_primary.csv','Adaptive blend · Holm2')]:
        inf=pd.read_csv(basepath)
        for z in inf.loc[inf.block_length.eq(60)].itertuples():intervals.append({'Comparison':LABELS[z.arm]+' vs '+LABELS[z.baseline],'Δ Sharpe':f'{z.delta_sharpe:+.3f}','95% interval':f'[{z.ci_low:+.3f}, {z.ci_high:+.3f}]','Adjusted p':f'{z.p_holm:.3f}','Family':family})
    gaps=pd.read_csv(out/'surrogate_engine_gap.csv');gap_table=gaps.copy();gap_table['model']=gap_table.model.map(LABELS)
    for c in ['surrogate_net_return','engine_net_return','difference']:gap_table[c]=gap_table[c].map(lambda x:f'{100*x:.3f}%')
    history=read(V30/'models/absolute_hierarchical_2023.json')['stock_history_quantiles']
    paragraphs=''.join('<p>'+html.escape(s)+'</p>' for s in findings['findings'])
    recommendation=''.join('<li>'+html.escape(s)+'</li>' for s in findings['next_steps'])
    content=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nonlinear and stock-specific portfolio research</title>
    <style>body{{margin:0;background:#f4f6f7;color:#183346;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 28px}}h1{{font-size:36px;line-height:1.15}}h2{{font-size:25px;margin-top:38px}}.eyebrow{{letter-spacing:.12em;text-transform:uppercase;font-size:12px;color:#147d80}}.lead{{font-size:21px}}.note{{background:#fff3dc;border-left:4px solid #bd8c35;padding:16px 22px}}.card{{background:white;padding:22px;margin:22px 0;border:1px solid #dce5e8;border-radius:9px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce5e8;vertical-align:top}}th{{color:#147d80}}img{{width:100%;height:auto}}a{{color:#14697d}}code{{font-size:13px}}footer{{font-size:13px;border-top:1px solid #bccbd2;margin-top:32px;padding-top:18px}}li{{margin:9px 0}}</style></head><body><main>
    <div class="eyebrow">Research v29 + v30 · 2007–2023 · completed {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}</div>
    <h1>Nonlinear portfolio learning and stock-specific forecasts</h1><p class="lead">{html.escape(findings['headline'])}</p>
    <div class="note">All performance is conditional research output. The original data’s unresolved valuation sessions remain; accounting checks do not certify those marks. This history has already been used for development. No model was promoted.</div>
    <h2>What the results say</h2>{paragraphs}
    <h2>Nonlinear portfolio learners</h2><p>Annual rolling fits keep the 21-session challenger fixed. Linear, random-feature and neural policies use the same 58 causal price-based inputs: the original 36 features smoothed at α=.25, plus a slower bank for the 22 continuous inputs. The nonlinear model adds 32 fixed cosine features; the network uses eight tanh units with a linear skip.</p>
    <div class="card scroll">{perf_table(p,core)}</div>{img(out/'nonlinear_equity.png')}
    <h2>How the objective now follows trading</h2><p>The sequential loss carries holdings and NAV forward, solves post-fee position sizing, charges actual absolute turnover at two basis points per side, applies the hard trading buffer and shared efficacy gate, and measures ex-ante portfolio risk using the original covariance. Fast and slow feature histories let it value persistent signals. Risk sizing preserves dollar and SPY-beta neutrality, 12% target volatility, gross and name limits.</p>
    <p><strong>The gradient loss remains a surrogate.</strong> It works at 21-session boundaries, places its buffer check one close later than the actual decision, forces a scheduled rebalance when carried positions become risk-ineligible, and does not reproduce daily margin/guard actions or corporate cash timing. Ninety inner validation replays use the original daily engine to select regularization; final accounts use that engine too. This is closer execution alignment, not differentiation through the full accounting ledger.</p>
    <p>Every inner training endpoint precedes its validation window; every final training endpoint precedes its annual refit. The 2012 refit’s validation ends on 2011-12-13 to avoid an unsettled year-end distribution claim. The same shortened window applies to every candidate. Pre-2007 training uses gate multiplier one because the original out-of-sample gate is unavailable then.</p>
    <details class="card"><summary>Measured surrogate-to-engine gap on fixed six-period windows</summary><p>Same cold start and target path in each window. These are descriptive diagnostics, not additional tuning results. Daily safeguards and corporate mechanics can materially alter realized returns.</p><div class="scroll">{table(gap_table)}</div></details>
    <h2>The stock-specific hybrid</h2><p>The pilot estimates <code>absolute return = shared predictor + stock-specific adjustment</code>. Shared coefficients learn across stocks; identifier-specific deviations shrink toward that common predictor. Both components see the same 89 price, market, company, earnings, revision and annual financial inputs. A pooled model and independent stock models provide controls. The joint model is solved directly, so it does not stack on in-sample predictions.</p>
    <p>In the 2023 training window, the median historical identifier has {history['median']:.0f} usable monthly observations; the lower quartile has {history['p25']:.0f}. A separate 90-coefficient model, including its intercept, is demanding even with ridge shrinkage. Identifier changes reset local parameters. New identifiers use the shared prediction in the hierarchical model; standalone stock models default to zero.</p>
    <div class="card scroll">{perf_table(p,hybrid)}</div>{img(out/'hybrid_equity.png')}
    <p>The 50/50 combination mixes the actual challenger’s and hybrid’s portfolio directions, then uses the original allocator to restore the common risk budget before applying the gate. It is a fixed late combination, not a new trained forecaster. It was declared after viewing forecast diagnostics and before viewing portfolio results, and receives a separate exploratory correction.</p>
    <h2>Forecast accuracy versus useful portfolio information</h2><div class="card scroll">{table(pd.DataFrame(ft))}</div><p>Forecast metrics cover 130 complete primary-period monthly dates. RMSE is in 21-session return units; R² compares with a zero-return prediction. Market-projected errors remove contemporaneous dollar and beta components from predictions and realized outcomes for diagnosis only. No future realized market return enters a predictor or allocation. Absolute error, rank correlation and neutral portfolio profit answer different questions.</p>
    <h2>Uncertainty</h2><p>Primary period: 2012–2023. Early history: 2007–2011. Paired circular block bootstraps use 2,000 draws with 60-session blocks and a 20-session sensitivity check. Intervals below are pointwise, not simultaneous. Holm correction covers the stated comparison family only, not the broader adaptive research history.</p><div class="card scroll">{table(pd.DataFrame(intervals))}</div>
    <h2>Research connections</h2><ul><li><a href="https://academic.oup.com/rfs/article/39/10/3035/8524346">Jensen, Kelly, Malamud and Pedersen, RFS 2026</a> links nonlinear random features and economic portfolio objectives with trading costs and signal persistence. This experiment tests related mechanisms under our existing restrictions; it does not replicate their data, model scale or impact-cost framework.</li>
    <li><a href="https://cepr.org/publications/dp17123">Pesaran, Pick and Timmermann, panel forecasting</a> studies the tradeoff between heterogeneous coefficients and estimation uncertainty, motivating partial pooling rather than assuming each stock needs a fully independent model.</li>
    <li><a href="https://www.aeaweb.org/conference/2025/program/powerpoint/kidEK8Kh">Cong, Feng, He and Wang, Mosaics of Predictability, 2025 presentation</a> motivates grouping observations by predictability and market state. That suggests testing event or sector groups as an alternative to permanent stock-specific coefficients.</li>
    <li><a href="https://arxiv.org/abs/2609.25617">Yang et al., LiMT, September 2026 preprint</a> combines cross-stock context, each stock’s temporal history and multiple forecasting tasks. It motivates a shared representation with small specialized components; its Chinese-equity experiments are not a comparable benchmark for this strategy.</li></ul>
    <h2>Next tests justified by this run</h2><ol>{recommendation}</ol>
    <h2>Audit and reproducibility</h2><p>119 annual model fits; 90 additional inner fits and full-engine validation replays; nine final account replays including the unchanged benchmark and fixed blend. Target constraints, source hashes and label maturity pass. Twelve thousand hybrid forecasts were independently reconstructed. The benchmark ledger’s largest numeric replay error is {q29['benchmark_replay_max_error']:.2g}. Exact source allocator comparisons pass. These checks establish implementation consistency, not investment validity.</p>
    <p>Company data inherit the earlier conservative publication-date joins, missing-data flags and issuer quarantines. They remain modern reconstructions rather than certified original-vintage histories. Performance includes modeled two-basis-point trading fees, but excludes borrowing and financing costs, as required by the fixed benchmark constraints.</p>
    <p><a href="protocol.json">Nonlinear protocol</a> · <a href="reports/validation.json">Nonlinear audit</a> · <a href="reports/performance.csv">Nonlinear results</a> · <a href="reports/paired_primary.csv">Nonlinear paired tests</a> · <a href="../v30_hybrid_stocks/protocol.json">Stock-model protocol</a> · <a href="../v30_hybrid_stocks/reports/validation.json">Stock-model audit</a> · <a href="../v30_hybrid_stocks/reports/performance.csv">Stock-model results</a> · <a href="../v30_hybrid_stocks/reports/forecast_metrics.csv">Forecast diagnostics</a> · <a href="../v30_hybrid_stocks/blend/protocol.json">Blend add-on</a> · <a href="../v30_hybrid_stocks/blend/paired_primary.csv">Blend paired tests</a> · <a href="src/policy.py">Portfolio implementation</a> · <a href="../v30_hybrid_stocks/src/hybrid.py">Hybrid implementation</a></p>
    <footer>Primary metrics use a common daily account calendar. All capital growth charts include the original gate and trading costs. Existing data qualifications and exploratory multiple testing remain material.</footer></main></body></html>'''
    (ROOT/'index.html').write_text(content,encoding='utf-8')
    write(out/'report_status.json',dict(completed=True,html_sha256=sha(ROOT/'index.html'),v29_validation=q29,v30_validation=q30,blend_complete=blend['completed'],figures=['nonlinear_equity.png','hybrid_equity.png']))
    print('REPORT_CREATED',flush=True)
if __name__=='__main__':main()
