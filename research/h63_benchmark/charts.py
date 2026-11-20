"""Rebuild publication figures using only the public reference results."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
C={'H63':'#176b78','SPY':'#858b93'}
r=pd.read_csv(P/'reference/daily_returns.csv.gz',index_col=0,parse_dates=True)
def finish(fig,name,title):
    fig.suptitle(title,fontsize=16,fontweight='bold',x=.07,ha='left')
    fig.text(.07,.012,'Through Nov 11, 2026 • 2 bp modeled costs • conditional valuations retained • 2024 split already examined',fontsize=8,color='#555')
    fig.tight_layout(rect=(.02,.05,1,.93));fig.savefig(P/f'charts/{name}.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4.7))
for ax,sub,title in [(axes[0],r,'Full sample · linked accounts'),(axes[1],r.loc['2024':],'2024 onward · rebased')]:
    eq=(1+sub).cumprod()
    for c in eq:ax.plot(eq.index,eq[c],label=c,color=C[c],lw=1.8)
    ax.set_title(title);ax.set_ylabel('Growth of $1');ax.legend(frameon=False)
    locator=mdates.AutoDateLocator(minticks=4,maxticks=6);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
axes[0].axvline(pd.Timestamp('2024-01-02'),color='#b06c30',ls='--',lw=1.2,label='2024 boundary')
axes[0].text(pd.Timestamp('2024-01-02'),axes[0].get_ylim()[1]*.94,' 2024 boundary',color='#9c5b24',fontsize=8)
finish(fig,'equity','H63 and SPY | equity curves')
fig,axes=plt.subplots(1,2,figsize=(13,4.7))
for ax,sub,title in [(axes[0],r,'Full sample'),(axes[1],r.loc['2024':],'2024 onward')]:
    eq=(1+sub).cumprod();dd=eq/eq.cummax().clip(lower=1)-1
    for c in dd:ax.plot(dd.index,dd[c],label=c,color=C[c],lw=1.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(title);ax.legend(frameon=False)
    locator=mdates.AutoDateLocator(minticks=4,maxticks=6);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
finish(fig,'drawdown','H63 and SPY | drawdown')
fig,ax=plt.subplots(figsize=(13,4.7));annual=(1+r).groupby(r.index.year).prod()-1
annual.plot.bar(ax=ax,color=[C[c] for c in annual],width=.8);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('2026 is a partial year');ax.legend(frameon=False)
finish(fig,'annual_returns','H63 and SPY | calendar-year returns')
parts=[pd.read_csv(P/f'reference/{part}_ledger.csv.gz',index_col=0,parse_dates=True) for part in ('historical','forward')]
q=pd.concat(parts);b=pd.concat([pd.read_csv(P/f'reference/{part}_exposure.csv.gz',index_col=0,parse_dates=True) for part in ('historical','forward')])
fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True)
axes[0].plot(q.index,q.net_exposure,color=C['H63'],lw=1);axes[0].set_ylabel('Invested / NAV');axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].plot(b.index,b.raw_beta,color=C['H63'],lw=1);axes[1].set_ylabel('Estimated portfolio beta')
for ax in axes:ax.axvline(pd.Timestamp('2024-01-02'),color='#b06c30',ls='--',lw=1)
finish(fig,'exposure','H63 | cash demand and market exposure')
print('Four H63-only figures generated')
