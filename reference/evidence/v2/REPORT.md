This is the previous factor-tree version I mentioned. I tried a long/short system for a Treasury ETF, but it didn't work. 
# Swing Decision Tree Research V2

## Design

V2 retains SPY and NVDA as long-only weekly swing models and TLT as a weekly long/short model. Signals form after each week's final market close and execute at the next session open. The official run assumes zero slippage; we report 5- and 10-basis-point one-way sensitivities separately.

We select every candidate using 2022-2023 only. The 2024-present interval is a frozen historical comparison against V1. It is not described as a pristine holdout because V1's later-period results were already known when V2 was designed. The next untouched forward sample begins after September 25, 2026.

## Selected models

| Model | Selected specification | Sizing |
|---|---|---|
| SPY long-only | 5-session, depth-2 paper-feature tree | Binary 0/1 |
| NVDA long-only | 10-session, depth-3 technical plus changepoint tree | Binary 0/1 |
| TLT long/short | 40-session, depth-3 TLT curve tree | 12% volatility target, capped at 1x |

The broader multi-horizon ensembles and HMM feature sets were tested but were not selected. TLT's most important fitted features were the 10y-2y slope, 2y/10y/30y curvature, 10-year yield, 63-session drawdown, and 30-year yield.

## 2024-present historical comparison

| Model | V2 CAGR | V2 Sharpe | V2 max DD | V1 CAGR | V1 Sharpe | V1 max DD |
|---|---:|---:|---:|---:|---:|---:|
| SPY long-only | 21.09% | 1.53 | -10.59% | 21.09% | 1.53 | -10.59% |
| NVDA long-only | 33.61% | 1.40 | -13.50% | 58.76% | 1.53 | -24.03% |
| TLT long/short | 0.33% | 0.09 | -13.83% | -1.43% | -0.10 | -17.85% |

SPY rejected the added complexity and reproduced V1. NVDA's changepoint layer materially reduced drawdown but sacrificed upside participation and Sharpe versus V1. TLT improved all three headline metrics versus V1, but remained weak in absolute terms and lost 7.0% in 2026 through September 25.

The paired 20-session block bootstrap estimates a 68% probability that TLT V2's Sharpe exceeds V1, but its 90% interval is wide and crosses zero. This is a promising result rather than strong statistical evidence. NVDA's observed Sharpe decline is also statistically inconclusive.

At 10 basis points one way, later-period Sharpe is 1.41 for SPY, 1.33 for NVDA, and 0.04 for TLT. TLT's weak absolute edge is therefore especially sensitive to omitted financing, borrow, and implementation costs.

## Research translated into tests

- [Moreira and Muir, Volatility-Managed Portfolios](https://www.nber.org/papers/w22208): instrument-specific volatility scaling.
- [Gu, Kelly, and Xiu, Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276): shallow nonlinear interactions among momentum, liquidity, and volatility predictors.
- [Lim, Zohren, and Roberts, Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912): forecast and volatility-aware sizing evaluated jointly.
- [Wood, Roberts, and Zohren, Slow Momentum with Fast Reversion](https://arxiv.org/abs/2105.13727): causal changepoint severity features.
- [Allena, Confident Risk Premiums and Investments Using Machine Learning Uncertainties](https://academic.oup.com/rfs/article-abstract/39/5/1463/8287227): multi-horizon vote agreement and forecast-dispersion diagnostics.
- [Bianchi, Buchner, and Tamoni, Bond Risk Premiums with Machine Learning](https://academic.oup.com/rfs/article-abstract/34/2/1046/5843806): nonlinear TLT forecasts combining yield and macro information.
- [Cochrane and Piazzesi, Bond Risk Premia](https://www.nber.org/papers/w9178): Treasury slope and curvature information.
- [Bauer and Hamilton, Robust Bond Risk Premia](https://academic.oup.com/rfs/article-abstract/31/2/399/4210006): small-sample controls, compact feature blocks, and stability penalties.
- [Fons et al., Feature Saliency HMM for Smart Beta Investing](https://arxiv.org/abs/1902.10849): an economically restricted six-feature rate HMM rather than a full-feature HMM.
- [Bajo Traver, Harvesting Term and Credit Risk Premia](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6735418): annual expanding HMM fits and transparent rate-state sizing tests.

## Audit

Ten checks passed: delayed next-open execution, correct forward labels, weekly-only position changes, one-session FRED lag, changepoint future invariance, rate-HMM future invariance, equity long-only bounds, TLT long/short and leverage bounds, minimum multi-day holding, and credential scanning.

The result package includes the full candidate table, fitted tree rules, feature importance, annual and cost-sensitive performance, daily positions, HMM probabilities, regime diagnostics, and V1/V2 equity and drawdown curves.
