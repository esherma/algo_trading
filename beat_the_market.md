# Swiss Finance Institute Research Paper Series $N^{\circ} 24-97$ 

## Beat the Market: An Effective Intraday Momentum Strategy for S\&P500 ETF (SPY)

![img-0.jpeg](img-0.jpeg)

## Carlo Zarattini

Concretum Research Lugano

## Andrew Aziz

Peak Capital Trading and Bear Bull Traders, Canada

## Andrea Barbon

University of St. Gallen and Swiss Finance Institute

# Beat the Market An Effective Intraday Momentum Strategy for S\&P500 ETF (SPY) 

Carlo Zarattini ${ }^{1}$, Andrew Aziz ${ }^{2,3}$, Andrea Barbon ${ }^{4}$<br>${ }^{1}$ Concretum Research, Viale Carlo Cattaneo 1, 6900 Lugano, Switzerland<br>${ }^{2}$ Peak Capital Trading, 744 West Hastings Street, Vancouver, BC, Canada V6C 1A5<br>${ }^{3}$ Bear Bull Traders, 744 West Hastings Street, Vancouver, BC, Canada V6C 1A5<br>${ }^{4}$ University of St.Gallen, Dufourstrasse 50, 9000 St. Gallen, Switzerland<br>$\square{ }^{1}$ carlo@concretumgroup.com, ${ }^{3}$ andrew@peakcapitaltrading.com, ${ }^{4}$ andrea.barbon@unisg.ch<br>$\mathbb{X}:{ }^{1} @$ ConcretumR, ${ }^{2}$ @BearBullTraders , ${ }^{3}$ @Andrea_Barbon

First Version: May 10, 2024
This Version: April 29, 2025


#### Abstract

This paper investigates the profitability of a simple, yet effective intraday momentum strategy applied to SPY, one of the most liquid ETFs that tracks the S\&P 500. Unlike the academic literature that typically limits trading to the last 30 minutes of the trading session, our model initiates trend-following positions as soon as there is an indication of abnormal demand/supply imbalance in the intraday price action. Building on trading techniques commonly used by active day traders, which have been discussed in our previous papers, we introduce the use of dynamic trailing stops to mitigate downside risks while allowing for unlimited upside potential. From 2007 to early 2024, the resulting intraday momentum portfolio achieved a total return of $1,985 \%$ (net of costs), an annualized return of $19.6 \%$, and a Sharpe Ratio of 1.33. We conduct extensive statistical tests to examine whether the profitability of the strategy is affected by different market volatility regimes and whether the estimated gamma imbalance of dealers could predict changes in strategy profitability. We analyze the daily profitability of the intraday momentum strategy with respect to day-of-the-week effects. Additionally, we evaluate its performance against well-known technical daily patterns to understand its behavior under various market conditions. Given the short-term nature of the model, we also assess the impact of commissions and slippage on the overall profitability of the strategy.


Keywords: Day Trading, Day Trading Systems, Algo Trading, Momentum, Trend-Following, Intraday Momentum, Delta-Hedging

# 1 Introduction 

An object in motion tends to stay in motion, an object at rest tends to remain at rest.

- Isaac Newton

In a very influential paper, Jegadeesh and Titman [1] introduce a groundbreaking insight into the realm of stock market anomalies: the momentum effect. Contrary to the efficient market hypothesis, which suggests that past stock performance is not predictive of future returns, Jegadeesh and Titman demonstrate that stocks exhibiting strong performance over recent periods tend to continue outperforming in the short to medium term. The concept of momentum trading has revolutionized investment strategies by revealing the persistent trend-following behavior of stock prices. It also contradicts the efficient market hypothesis, which posits that stock prices reflect all available information and that past performance cannot predict future performance.

Jegadeesh and Titman's research shows that a portfolio that goes long the best performing stocks and shorts the worst performing stocks, generates significant positive returns over 3 - to 12 -month holding periods and the profitability of these portfolios is not explained by common risk factors. They provide empirical evidence supporting the profitability of momentum, which has since become a widely studied and implemented investment strategy in both academic and practical contexts.

Since then, momentum strategies have been implemented over longer time frames, with research focusing on daily, weekly, or monthly intervals. However, with the rise of highfrequency trading and increased accessibility to intraday data, there is growing interest in applying momentum strategies to shorter time frames, such as intraday intervals for day trading $[2,3,4,5]$. Gao et al. [6] demonstrates the potential of overnight returns to forecast the final half-hour returns of the S\&P 500, highlighting significant intraday time-series momentum. Extending these findings, Rosa [7] analyzes additional data post2013, noting a decline in predictability, which he attributes to market adaptations or the

ephemeral nature of previously exploitable inefficiencies. His application of a Markovswitching model underlines how the intraday momentum's effectiveness is contingent upon volatile market regimes.

The study by Baltussen et al. [8] further enriches this discussion by examining how options dealers' gamma hedging activities influences intraday price dynamics. This work illustrates the complex interplay between institutional hedging and equity market movements, shedding light on how such actions can amplify or mitigate intraday price trends, thus impacting the performance of momentum strategies.

Other global studies have expanded the scope of intraday momentum across various markets. For instance, Zhang et al. [9] and Jin et al. [10] confirm the presence of intraday momentum in Chinese equities and commodities, respectively, while Li et al. [11] provide evidence across multiple developed markets, underscoring the pervasive yet varied nature of intraday momentum influenced by localized market behaviors and regulations.

In our previous publications, we present findings from various day trading strategies, including the Opening Range Breakout (ORB) [2, 3, 4, 12] and Volume Weighted Average Price (VWAP) Trend Trading [5], which consistently outperform the simple buy-and-hold strategy on major US stock market indices such as SPY and QQQ. These results are robust, exhibiting significant Sharpe Ratios and statistical significance for our straightforward day trading approaches.

Building upon our prior research, this paper extends the model of Gao et al. [6] and Baltussen et al. [8]. Our aim is to study the economic and statistical significance of a day trading strategy that uses SPY, a very liquid ETF that tracks the S\&P500, to exploit intraday trends caused by sticky demand/supply imbalances. Unlike the aforementioned studies, our model does not limit the trading to the last 30 -minutes of the trading session; empirically, we have seen that intraday trends can start earlier and a strategy confined to trading only after 15:30 may prove to be suboptimal.

Our model is an instance of a time-series momentum strategy, focusing on a single in-

strument, namely the SPY, instead of relying on a portfolio of multiple assets sorted by their relative past performance. A research conducted by Moskowitz et al. [13] on timeseries momentum indicates that the effect is more pronounced at shorter horizons and tends to revert over longer time frames. It is, therefore, natural to test the profitability of time-series momentum at high frequencies, which motivates our intraday study.

While academic literature continues to debate the causes of momentum effects, one of the most convincing explanations is the under-reaction to news, which causes information to be slowly incorporated into prices. This under-reaction may arise from a variety of factors. For instance, Frazzini [14] proposes and tests an explanation based on a behavioral bias called the disposition effect, which is the tendency to capitalize gains from winning stocks and hold onto losing positions. The flows originating from this biased behavior may slow down the appreciation of stock prices following positive news and, symmetrically, decelerate the price drops following negative news. Another explanation for under-reaction, based on investors' limited attention to news, is proposed by Della Vigna and Pollet [15]. Such behavior may stem from limited cognitive abilities or be optional if information acquisition is costly [16]. Finally, Duffie [17] argues that underreaction may stem from slow-moving capital, caused by institutional impediments to fast trading, such as the search costs for counterparties or raising capital by intermediaries.

Our work contributes to the ongoing discourse surrounding market anomalies and the practical applications of momentum trading in intraday contexts. Through empirical analysis and rigorous backtesting, we evaluate the effectiveness of intraday momentum strategies, shedding light on their potential benefits and challenges in the modern trading landscape.

Furthermore, our evidence of sizeable momentum effects at an intraday frequency supports an explanation of momentum based on under-reaction to news, in line with previous literature on the topic.

# 2 Data 

The database for SPY and VIX (Chicago Board Options Exchange Volatility Index) has been constructed using 1-minute OHLCV (Open, High, Low, Close, and Volume) data from IQFeed, covering the period from May 2007 to April 2024. All backtests and statistical tests have been conducted using Matlab R2023a.

## 3 Strategy Description

The existence of intraday trends typically results from a persistent and significant imbalance between buyers and sellers throughout a trading session. Given the high level of noise in mature equity markets, such as the S\&P 500, an intraday momentum strategy should only establish positions when the price action clearly shows a deviation from demand-supply equilibrium. To effectively navigate this, we have to mathematically define a region where prices are expected to oscillate under conditions of balanced buying and selling pressures. We refer to this region as the Noise Area, an equilibrium zone where markets do not exhibit any exploitable intraday trend.

Practically, the market demonstrates a lack of abnormal imbalances when the intraday movement from the Open is less than the average movement observed on the same intraday time interval over the previous days. We define the Noise Area as the space between 2 boundaries. These boundaries are time-of-day dependent and are computed using the average movements recorded over the previous 14 days. Mathematically, the Noise Area can be compupted on day $t$ following these steps:

1. For each day $t-i$ and time-of-day $H H: M M$, calculate the absolute move from Open as

$$
\text { move }_{t-i, 9: 30-H H: M M}=\left|\frac{\text { Close }_{t-i, H H: M M}}{\text { Open }_{t-i, 9: 30}}-1\right|, \quad \text { where } i=[1,14]
$$

2. For each time-of-day $H H: M M$, calculate the average move over the last 14 days as

$$
\sigma_{t, 9: 30-H H: M M}=\frac{1}{14} \sum_{i=1}^{14} \text { move }_{t-i, 9: 30-H H: M M}
$$

3. Using the Open of day $t$, compute the Upper and Lower Boundaries as

$$
\begin{aligned}
& \text { UpperBound }_{t, H H: M M}=\text { Open }_{t, 9: 30} \times\left(1+\sigma_{t, 9: 30-H H: M M}\right) \\
& \text { LowerBound }_{t, H H: M M}=\text { Open }_{t, 9: 30} \times\left(1-\sigma_{t, 9: 30-H H: M M}\right)
\end{aligned}
$$

4. Compute the Noise Area as the area between the Upper and Lower Boundaries

$$
\text { NoiseArea }_{t, \text { HH:MM }}=\left[\text { LowerBound }_{t, \text { HH:MM }}, \text { UpperBound }_{t, \text { HH:MM }}\right]
$$

To make the evidence of potential imbalances more robust, we include the closing price in our calculation. This adjustment accounts for overnight gaps, which in themselves often signal imbalances. For instance, following a gap-down event, the Upper Boundary of the Noise Area is adjusted upward by a quantity equal to the gap size, and conversely for a gap-up event, the Lower Boundary is adjusted downward.

With these considerations, the revised mathematical equations for the Upper and Lower Boundaries, including adjustments for overnight gaps, are as follows:

$$
\begin{aligned}
& \text { UpperBound }_{t, H H: M M}=\max \left(\text { Open }_{t, 9: 30}, \text { Close }_{t-1,16: 00}\right) \times\left(1+\sigma_{t, 9: 30-H H: M M}\right) \\
& \text { LowerBound }_{t, H H: M M}=\min \left(\text { Open }_{t, 9: 30}, \text { Close }_{t-1,16: 00}\right) \times\left(1-\sigma_{t, 9: 30-H H: M M}\right)
\end{aligned}
$$

This enhancement ensures that our definition of the Noise Area dynamically reflects both the intraday fluctuations and the impact of the opening market conditions, providing a more comprehensive tool for identifying true market imbalances beyond mere noise.

Figure 1 provides a graphical representation of the Noise Area. As long as the market remains within this designated region, it is considered that demand and supply are in equilibrium, and thus, the strategy refrains from taking any positions. A crossing above the Upper Boundary, illustrated by the blue line, indicates abnormal buying pressure, potentially heralding an exploitable upward trend. Conversely, a crossing below the Lower

![img-1.jpeg](img-1.jpeg)

Figure 1: The time-varying boundary area for identifying the start of a new upward or downward trend in price. The shaded area represents noise or choppy price action.

Boundary, depicted by the red line, signals abnormal selling pressure, suggesting a possible downward trend.

A distinctive feature of this model is that the noise boundaries are dynamic and vary throughout the trading day. This variation means, for example, that the market movement required to indicate a demand/supply imbalance is different in the first 30 minutes of trading compared to the first 6 hours. Typically, the average intraday movement from the Open tends to increase over time, peaking at 16:00. As demonstrated in the simplified example in Figure 1, the movement required from the Open to 10:30 to indicate abnormal selling pressure was -0.30%, which was half of that required between the Open and 15:30.

The introduction of time-varying boundaries represents a novel feature that distinguishes this model from well-documented systems like the volatility breakout systems by Cabrel [18] and Kaufman [19], or the intraday momentum models described in academic papers by authors such as Rosa [7] and Baltussen et al. [8]. This adaptability enhances the strategy's responsiveness to varying market conditions throughout the trading day, providing a more precise and actionable framework for identifying significant market movements.

![img-2.jpeg](img-2.jpeg)

Figure 2: The time-varying boundary areas on SPY for 31 January 2022 (a) and 29 April 2022 (b) and the resulting trades based on our base-model. Please note that all positions are closed at the market Close (16:00).

As discussed earlier, if the market breaches the boundaries of the *Noise Area*, our strategy initiates positions in accordance with the prevailing intraday move—going long if the price is above the *Noise Area* and short if it is below. To mitigate the risk of overtrading caused by short-term market fluctuations, trading is restricted to semi-hourly intervals, specifically at *HH:00* and *HH:30*. This timing mechanism ensures that decisions are based on more stable price movements rather than transient spikes.

Positions are unwound either at market Close or if there is a crossover to the opposite boundary of the *Noise Area*. In the event of such a crossover, the existing position is closed, and a new one is initiated in the opposite direction to align with the latest evidence of demand/supply imbalance. Stop losses can also be triggered only on semi-hourly intervals.

Figures 2 illustrates the *Noise Areas* and the resulting trades on 2 days with clear trends that occurred in 2022. As shown by the dashed grid on the charts, trades are executed at 10:30, even though the price had moved outside the *Noise Area* a few minutes earlier. This delay in response reduces the risk that trading decisions are triggered by fleeting market movements. All positions are closed at market Close, adhering to the strategy's protocol to limit exposure to overnight market changes.

The strategy allocates exposure equal to $100 \%$ of the equity available at the beginning of each trading day. We calculate the number of shares that can be traded using the equation:

$$
\text { Shares }_{t}=\left\lfloor\frac{A U M_{t-1}}{\text { Open }_{t, 9: 30}}\right\rfloor
$$

where $A U M_{t-1}$ represents the equity (or Asset Under Management) available at the end of day $t-1$.

To assess the historical performance of our strategy, we simulate a portfolio starting with an initial capital of $\$ 100,000$. The backtest incorporates transaction costs, including a commission fee of $\$ 0.0035$ per share, which represents the entry-level commission charged by Interactive Brokers. Additionally, we account for a slippage of $\$ 0.001$ per share, which is consistent with a real-market case study recently conducted in our proprietary trading accounts ${ }^{1}$.

These operational parameters ensure that our backtesting reflects realistic trading conditions and provides a credible estimation of the strategy's potential performance.

Figure 3 displays the equity curve of the strategy in black, juxtaposed with a long passive exposure in SPY, shown in red. As set out in Table 1, from 2007 to 2024, the active intraday momentum strategy realized a total return of $178 \%$, with an average annual return of $6.2 \%$ and a volatility of $10.9 \%$. This resulted in a Sharpe Ratio of 0.61 . Despite the overall slight underperformance compared to the passive SPY exposure, the most concerning aspects are the worst daily return of $-10.33 \%$ and the negative skewness of the intraday momentum portfolio (more details are provided in Table A1, at the end of the paper). Typically, trend-following strategies are favored for their significant asymmetry between positive and negative daily returns.

[^0]
[^0]:    ${ }^{1}$ More details are provided in Section 4.6

![img-3.jpeg](img-3.jpeg)

Figure 3: Comparison of equity curves of intraday momentum strategy vs. SPY buy-hold strategy. Commission set at $0.0035 as per Interactive Brokers' entry-level rate.

By design, the active intraday strategy is vulnerable to sudden market reversals, which can be triggered by news releases or other fundamental catalysts that can provoke dramatic shifts in market character. A trailing stop set at the opposite side of the *Noise Area* exacerbates this vulnerability, as the portfolio can suffer dramatically on such days.

January 20, 2022 serves as a poignant example of the weaknesses inherent in our base model. As shown in Figure 4, the market strongly rallies in the first 30 minutes, triggering a long exposure. For some 4 hours, the price remains above the *Noise Area*, indicating continued abnormal buying pressure. However, around 14:00, the market begins to turn south, dropping almost 2% in the last trading hours; SPY closes the day below the Lower Boundary of the *Noise Area*, resulting in a 2.19% loss for the active strategy. This sce-

Table 1: Summary statistics of intraday momentum strategy with stop loss at opposite band and SPY Buy&Hold. Commission set at $0.0035 as per Interactive Brokers' entry-level rate. We highlight in bold coefficients that are statistically significant at 5% level or below.

|  Strategy | Stop | Size | Total Return | IRR | Vol | Sharpe Ratio | Hit Ratio | MDD | Alpha | Beta  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Momentum | Opp.Band | 100% | 178% | 6.2% | 10.9% | 0.61 | 54% | 21% | 7.1% | -0.05  |
|  SPY (Buy&Hold) |  | 100% | 227% | 7.2% | 20.2% | 0.45 | 54% | 56% |  |   |

![img-4.jpeg](img-4.jpeg)

Figure 4: SPY price action on 20 January 2022 illustrates our model strategy's entry and exits. The base strategy uses the opposite band as a trailing stop. Following a sharp reversal in SPY in the last 2 hours of trading, the base model realizes a loss of more than 2%.

nario highlights the need for adjustments in our base model to mitigate losses during abrupt market reversals.

A straightforward method to mitigate the risk of sudden market reversals is by utilizing the Upper Boundary of the *Noise Area* as a trailing stop for our long positions, and vice versa for short positions. As we demonstrate in Figure 5(a), this approach leads to the position being closed at 14:00, just as the market begins to weaken. This adjustment improves the return from -2.19% to -0.31%. While a tighter stop loss can significantly reduce downside risks, it is important to note that it may also substantially increase the number of trades due to false signals, and consequently, the associated transaction costs.

This observation underscores the need to balance the protective benefits of a tighter stop loss against the potential for increased operational costs, ensuring that the strategy remains economically viable while providing adequate risk management.

Moreover, as documented in a previous paper [5], Volume Weighted Average Price (VWAP) can serve as a powerful intraday indicator to identify shifts in the demand-supply imbalance.

![img-5.jpeg](img-5.jpeg)

Figure 5: Improvement in our strategy's exit by incorporating tighter trailing stop losses. In Figure (a), the trailing stop is based on the upper band of the *Noise Area*, while in Figure (b), it is based on the maximum between the VWAP and the upper band.

ance. Building upon insights from our previous publication, we have chosen to integrate VWAP as a trailing stop for intraday trend exposures<sup>2</sup>. Once a position is established, as soon as the price crosses either the current band or the VWAP, the position is closed. Mathematically, the trailing stops for long and short postions are defined as:

$$
\begin{aligned}
\text{Long TrailingStop}_{t,HH:MM} &= \max (UB_{t,HH:MM}, VWAP_{t,HH:MM}) \\
\text{Short TrailingStop}_{t,HH:MM} &= \min (LB_{t,HH:MM}, VWAP_{t,HH:MM})
\end{aligned}
$$

Figure 5(b) illustrates the revised approach in the case study of January 20, 2022. Including VWAP allows us to close the long exposure at 13:00, ultimately achieving a break-even trade. This adjustment represents a significant improvement, showcasing VWAP's efficacy as a dynamic stop loss mechanism that aligns closely with market momentum, thus enhancing the strategy's risk management capabilities.


To demonstrate the economic advantages of these trailing stop refinements, we have conducted a comprehensive historical backtest, the results of which are displayed in Figure 6 and Table 2. The improvements are substantial: the average annual return increases to 9.7%, while annual volatility decreases to 7.7%. The Sharpe Ratio doubles, improving from 0.61 to 1.24, while the Hit Ratio decreased to 43% due to tighter stops. As exhibited in Table A1, the skewness shifts to positive (from -1.24 to 1.29) and the worst daily re- turn sees a 5% improvement (from -10.3% to -4.8%). However, the tighter stop loss does result in a significant increase in the total number of trades, rising from 5,494 to 7,668 over the 17-year period (see Table 4). These results highlight the enhanced risk-return profile achieved by the strategic modifications, albeit with an increase in trading activity.

As a final refinement to our model, we have implemented a sizing methodology that dynamically adjusts the traded exposure based on daily market volatility. Instead of maintaining constant full notional exposure, this method targets a daily market volatility

Table 2: Summary statistics of intraday momentum strategy with a) stop loss at opposite band, b) current band with VWAP, c) SPY Buy&Hold. Commission set at $0.0035 as per Interactive Brokers' entry-level rate. We highlight in bold coefficients that are statistically significant at 5% level or below.

|  Strategy | Stop | Size | Total Return | IRR | Vol | Sharpe Ratio | Hit Ratio | MDD | Alpha | Beta  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Momentum | Opp.Band | 100% | 178% | 6.2% | 10.9% | 0.61 | 54% | 21% | 7.1% | -0.05  |
|  Momentum | Curr.Band + VWAP | 100% | 380% | 9.7% | 7.7% | 1.24 | 43% | 12% | 9.9% | -0.03  |
|  SPY (Buy&Hold) |  | 100% | 227% | 7.2% | 20.2% | 0.45 | 54% | 56% |  |   |

Table 3: Summary statistics of intraday momentum strategy with a) stop loss at opposite band, b) current band with VWAP, c) as in (b) with the additional dynamically adjusted share size based on daily market volatility, and d) SPY Buy&Hold Commission set at $0.0035 as per Interactive Brokers' entry-level rate. We highlight in bold coefficients that are statistically significant at 5% level or below.

| Strategy | Stop | Size | Total Return | IRR | Vol | Sharpe Ratio | Hit Ratio | MDD | Alpha | Beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Momentum | Opp.Band | 100% | 178% | 6.2% | 10.9% | 0.61 | 54% | 21% | 7.1% | -0.05 |
| Momentum | Curr.Band + VWAP | 100% | 380% | 9.7% | 7.7% | 1.24 | 43% | 12% | 9.9% | -0.03 |
| Momentum | Curr.Band + VWAP | Dyn. | 1,985% | 19.6% | 14.3% | 1.33 | 43% | 25% | 19.6% | -0.07 |
| SPY (Buy&Hold) |  | 100% | 227% | 7.2% | 20.2% | 0.45 | 54% | 56% | | |

of 2% (σ_{target} = 2%). Practically, if the recent daily volatility of SPY is 4%, we would trade with half of the capital; conversely, if it is 1%, we would utilize a leverage of 2. Mathematically, the number of shares traded on day t is computed as

$$Shares_t = \left\lfloor \frac{AUM_{t-1} \times \min(4, \sigma_{target} / \sigma_{SPY,t})}{Open_{t,9:30}} \right\rfloor,$$

where

$$\sigma_{SPY,t} = \sqrt{\frac{1}{13} \sum_{i=1}^{14} (ret_{t-i} - \mu_{SPY,t})^2}, \qquad\mu_{SPY,t} = \frac{1}{14} \sum_{i=1}^{14} (ret_{t-i})$$

This adjustment results in a more stable risk profile for the active portfolio, making it less dependent on overall market volatility regimes. In line with mainstream brokerage policies, the leverage for the active strategy is capped at 4x.

Figure 7 displays the equity line for the modified portfolio, while Table 3 details the main performance statistics. The active intraday momentum portfolio achieves a total return of 1,985%, an annualized return of 19.6%, with a volatility of 14.3%. The Sharpe Ratio improves to 1.33, indicative of a more stable risk profile. The maximum drawdown is contained at 25%, and the daily Hit Ratio stands at approximately 43%. The portfolio's best trade realizes a return of 9.1%, significantly outperforming the worst trade return of -2.9% (see Table 4).

To assess the outperformance of the active strategy compared to a passive exposure in SPY, we employ the following classic regression analysis:

![img-7.jpeg](img-7.jpeg)

Figure 7: Comparison of equity curves of intraday momentum strategy with a) stop loss at opposite band, b) current band with VWAP, c) as in (b) with the additional dynamically adjusted share size based on daily market volatility, and d) SPY buy-hold strategy. Commission set at $0.0035 as per Interactive Brokers' entry-level rate.

$$Ret_{Mom,t} = \alpha + \beta \times Ret_{SPY,t} + \epsilon_t.$$

The regression analysis reveals that the annualized alpha of the active strategy is 19.6% and it is highly statistically significant. This indicates that the strategy outperforms the baseline market passive exposure by a substantial margin on a consistent basis. Furthermore, the beta of the strategy is slightly below 0, which suggests there is a negative.

Table 4: Summary statistics at trade-level of intraday momentum strategy with a) stop loss at opposite band, b) current band with VWAP, c) as in (b) with the additional dynamically adjusted share size based on daily market volatility, and d) SPY Buy&Hold strategy Commission set at $0.0035 as per Interactive Brokers' entry-level rate.

| Strategy | Stop | Size | Trades | Avg. Trade x Day | Hit Ratio | Avg. PnL x Share | Max Loss Trade | Max Gain Trade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intra Mom. | Opp.Band | 100% | 5,494 | 1.3 | 53% | 0.10 | -8.6% (13-Nov-2008) | 6.0% (13-Oct-2008) |
| Intra Mom. | Curr.Band + VWAP | 100% | 7,668 | 1.8 | 37% | 0.09 | -4.0% (18-Mar-2020) | 4.6% (15-Oct-2008) |
| Intra Mom. | Curr.Band + VWAP | Dyn. | 7,668 | 1.8 | 37% | 0.09 | -2.9% (31-Jul-2013) | 9.1% (10-Oct-2018) |

correlation between the active strategy's returns and the overall market movements. This lack of correlation underscores the strategy's effectiveness in diversifying risk and achieving returns independent of market trends.

# 4 Further Investigations 

In this section, we outline the more in-depth statistical analysis we have conducted in order to assess how the profitability of the intraday momentum strategy changes in different market scenarios. Unless otherwise specified, all tests are conducted using the intraday momentum strategy that employs a trailing stop based on the current band and VWAP, and adjusts investment size according to recent market volatility.

### 4.1 Market Volatility Regimes

Even though our active intraday momentum strategy adjusts daily exposure based on recent market volatility, it is pertinent to investigate how the expected profitability of the strategy varies across different market volatility regimes. For this analysis, we utilize the VIX Index value at the opening of each trading session to categorize the days and subsequently assess the strategy's Sharpe Ratio specifically for days when the VIX is above certain thresholds.

Figure 8 displays the results using a simple bar chart. Consistent with findings by Rosa [7], the risk-adjusted returns tend to improve with increasing levels of market volatility. For instance, when the VIX is above 6, the Sharpe Ratio is approximately 1.50; remarkably, when the VIX exceeds 40, the Sharpe Ratio escalates to an impressive 3.50. However, for even higher levels of VIX, the Sharpe Ratio begins to decrease, likely due to the fewer observations available and higher dispersion among results on extremely volatile days.

This pattern suggests that a more sophisticated implementation of the strategy could benefit from dynamically adjusting exposure based on current market volatility. Such an adaptive approach would potentially enhance the strategy's performance by optimizing

![img-8.jpeg](img-8.jpeg)

Figure 8: Resulting Sharpe Ratio for different level of VIX at market Open.
risk exposure in response to fluctuating market conditions.

# 4.2 Daily Patterns 

Well-known active traders often rely on technical analysis to identify market setups that could signal an improvement in the expected profitability of a trading strategy. For instance, the presence of a daily hammer might indicate an upcoming enhancement in the effectiveness of an intraday momentum strategy. Considering the virtually infinite number of patterns that could be analyzed, we chose to focus on 8 common patterns widely recognized and utilized by experienced traders:

- NR4 (Narrow Range 4 Days): Identifies days where the range (high-low) is the smallest of the last 4 days.
- NR7 (Narrow Range 7 Days): Identifies days where the range is the smallest of the last 7 days.
- ID (Inside Days): Identifies days when the day's high is below the previous day's high, and the low is above the previous day's low.
- OD (Outside Days): Identifies days when the high is above the previous day's high and the low is below the previous day's low.
- Triangle: Identifies days when the high is below the highs of the previous 2 days, and the low is above the lows of the previous 2 days.
- Trend Days: Identifies days when the Open is below the 15th percentile of the daily range, and the Close is above the 85th percentile, suggesting a strong bullish

Table 5: Summary comparison of daily patterns and statistics achieved on our intraday momentum strategy applied on each pattern. We highlight in bold Avg.PnL that are statistically significant at 10% level or below.

|  Daily Pattern | Observations | Avg.PnL | t-stat | Hit | Sharpe  |
| --- | --- | --- | --- | --- | --- |
|   |  | (in bps) |  | Ratio | Ratio  |
|  Unconditional | 2620 | 12 | 5.34 | 43% | 1.7  |
|  NR4 | 660 | 22 | 5.14 | 45% | 3.2  |
|  NR7 | 367 | 16 | 3.07 | 43% | 2.5  |
|  ID | 260 | 5 | 0.85 | 41% | 0.8  |
|  OD | 260 | 7 | 1.05 | 43% | 1.0  |
|  Triangle | 664 | 14 | 3.19 | 43% | 2.0  |
|  Trend | 215 | -2 | -0.24 | 40% | -0.3  |
|  Big Tail | 85 | 19 | 1.17 | 47% | 2.0  |
|  Strong/Weak Closure | 422 | 14 | 2.04 | 43% | 1.6  |

trend (the reverse applies for bearish trends). The current range must also exceed the average range of the last 14 days.

- **Big Tail:** Identified on days with hammers or reverse hammers; where the Open and Close are within the top 75th percentile of the daily range (or the reverse for reverse hammers).
- **Strong/Weak Closure:** Identifies days where the closing price is above the 90th percentile (strong closure) or below the 10th percentile (weak closure) of the daily range.

Table 5 presents our findings, which include the number of days each trading pattern occurs, the average return per day expressed in basis points, the computed t-statistic, the Hit Ratio and the Sharpe Ratio. The first row provides aggregate results that are unconditional of specific daily patterns.

The interpretation of the t-statistics is crucial; a value between -1.65 and 1.65 suggests that the expected return is not statistically different from zero at a 10% confidence level. This indicates insufficient evidence to reject the null hypothesis of zero average return at this level of significance. In practice, if looking for reliable edges, the higher the t-statistics, the better (very negative t-statistics suggest that traders should flip the trading signal).

Among the patterns we analyze, the NR4 pattern emerges as the most promising, exhibiting an average return of 22 basis points with a t-statistic of 5.14 and a Hit Ratio of $45 \%$. This pattern demonstrates statistically significant profitability and warrants consideration for increased investment focus. The Triangle pattern also shows notable profitability with an average daily return of 14 basis points and a t-statistic of 3.19 .

Conversely, the strategy of riding intraday trends after a trend day does not show a statistically significant edge, suggesting it may not be effective as a stand-alone trading strategy. The Big Tail pattern, despite an attractive return profile, suffers from a lack of statistical reliability due to a limited number of observations ( 85 occurrences).

As reported in the first row, employing a trading strategy that allows for daily trading irrespective of specific setups, proves to be more reliable and statistically significant. However, in order to exploit higher expected returns, active day traders might consider adjusting bet sizes after observing NR4, NR7, and Triangle patterns.

# 4.3 Day-of-the-Week Effect 

The Day-of-the-Week Effect is a focal point for statistical arbitrageurs who leverage calendar seasonality to identify market inefficiencies. While this approach runs the risk of data-mining, the observed patterns could have substantial economic underpinnings tied to specific days. For instance, options expirations are traditionally more pronounced on Wednesdays and Fridays, which are historically the days most options expire in our dataset. With the emergence of zero-days-to-expiration (0-dte) options, the effects of options expirations are expected to be more evenly distributed across all weekdays.

In Table 6, we present the number of traded days, the average daily returns, t-statistics, Hit Ratios and Sharpe Ratio for each trading day. The data highlights that Wednesday, Thursday, and Friday stand out as the most profitable days with statistically significant results. Intriguingly, Monday shows no statistical significance, contrasting with our earlier findings of higher Monday profitability in the 5-minute Opening Range Breakout (ORB), as discussed in our previous publications [4, 20]. This discrepancy could be at-

Table 6: Summary statistics grouped by day-of-the-week for the momentum intraday strategy. We highlight in bold Avg.PnL that are statistically significant at 10% level or below.

|  Day | Observations | Avg.PnL (in bps) | t-stat | Hit Ratio | Sharpe Ratio  |
| --- | --- | --- | --- | --- | --- |
|  Unconditional | 2620 | 12 | 5.34 | 43% | 1.7  |
|  Monday | 472 | 9 | 1.84 | 43% | 1.4  |
|  Tuesday | 543 | 9 | 2.00 | 42% | 1.4  |
|  Wednesday | 530 | 18 | 3.42 | 45% | 2.4  |
|  Thursday | 588 | 12 | 2.39 | 43% | 1.6  |
|  Friday | 487 | 13 | 2.39 | 44% | 1.7  |

tributed to the weekend effect, where significant market movements on Monday often occur within the first 30 minutes of trading as traders rush to rebalance portfolios postweekend. Contrary to the 5-minute ORB, which aims to capture moves starting from 9:35, the intraday momentum strategy we outline here takes its first position at 10:00, potentially missing a significant portion of the early move. Additionally, the noted high profitability on Wednesday may be linked to significant intraday trends often triggered on Federal Open Market Committee (FOMC) days.

### 4.4 Volatility Multiplier

A central element of the strategy we discuss in this paper is the definition of the *Noise Area*, essential for identifying potential tradable trends. As we initially define it, a market movement is recognized as a possible trend when it exceeds the average movement recorded during the same intraday time interval over the previous 14 days. We now introduce a parameter we term as the *Volatility Multiplier* (VM), which adjusts the width of the *Noise Area*, making the model either more conservative (for values above 1) or more aggressive (for values below 1). A VM below 1 reduces the required movement in SPY to signal a potential trend, whereas a multiplier above 1 demands a larger movement before initiating any exposure.

$$ \text{UpperBound}_{t, \text{HH:MM}} = \max \left( \text{Open}_{t, 9:30}, \text{Close}_{t-1, 16:00} \right) \times (1 + \text{VM} \times \sigma_{t, 9:30-\text{HH:MM}}) $$

$$ \text{LowerBound}_{t, \text{HH:MM}} = \min \left( \text{Open}_{t, 9:30}, \text{Close}_{t-1, 16:00} \right) \times (1 - \text{VM} \times \sigma_{t, 9:30-\text{HH:MM}}) $$

![img-9.jpeg](img-9.jpeg)

Figure 9: The effect of Volatility Multiplier on the Sharpe Ratio and Total Return of the intraday momentum strategy

We have analyzed the profitability of the intraday momentum strategy across different settings of the VM and the results are presented in Figure 9. The bottom bar chart in Figure 9 illustrates that the total return negatively correlates with the VM. This suggests that a too conservative approach (i.e., a very high VM) may result in less exploitable trading opportunities. Conversely, the top bar chart in Figure 9 provides a more comprehensive view by considering not only average returns but also the volatility of the strategy's returns. This chart indicates that the best risk-adjusted returns are achieved with a VM of approximately 1.5, highlighting a subjective trade-off between the total return and risk-adjusted returns.

Given the complexity of the decision between higher returns and better risk-adjusted performance, we leave it to each trader to determine the most appropriate implementation. To maintain simplicity and statistical intuition, the strategies presented in this paper employ a VM of 1, even though this may not be the most efficient setting based on retrospective analysis.

# 4.5 Gamma Imbalance 

As Baltussen et al [8] and Barbon and Buraschi [21] note, the delta hedging activities carried out by financial institutions to manage the risk in their options portfolios can result in self-fulfilling trends in intraday price movements. This phenomenon becomes particularly pronounced when the overall gamma of dealers is negative, necessitating adjustments to the delta in the same direction as the market trend. Typically, it is assumed that dealers sell put options to investors seeking to hedge against market downturns, while simultaneously purchasing call options from large institutional investors engaged in call-overwriting strategies. Consequently, if the market hovers around the strike prices of predominantly purchased calls, dealers are compelled to counteract intraday movements to rebalance their portfolio's delta to zero-this involves selling on intraday rallies and buying on dips. Conversely, if the market approaches the strike prices of predominantly sold puts, dealers must enhance intraday movements to align their portfolio delta to zero, thus amplifying the intraday trend.

Although it is beyond the scope of this paper to provide a precise estimate of gamma imbalance, we hypothesize that utilizing a short-term Relative Strength Index (RSI) might allow us to gauge whether the market is near the bulk of call strikes or put strikes. To explore this, we propose running a regression analysis between the daily returns of our active momentum portfolio (dependent variable) and the 5-day RSI of SPY (explanatory variable) as of the previous day's close. We anticipate that a higher RSI indicates greater dealer gamma, suggesting a higher likelihood of intraday mean reversion and, consequently, lower returns for our intraday momentum strategy. Conversely, a lower RSI suggests reduced gamma, implying more pronounced intraday trends and potentially higher profitability for the momentum portfolio. By running the following regression ${ }^{3}$ :

$$
\operatorname{Ret}_{\operatorname{Mom}, t}=\alpha+\beta \times R S I_{\mathrm{SPY}, t-1}+\epsilon_{t}
$$

we obtain a value of beta of -3.25 with a p-value of 0.001 . The results indicate that the slope coefficient of the regression is negative and significantly different from zero. This

[^0]
[^0]:    ${ }^{3}$ To enhance the readability of the regression results, we express $\operatorname{Ret}_{\text {Mom, } t}$ in basis points. For instance, a return of 100 basis points is equivalent to a return of $1 \%$.

finding suggests a substantial negative linear relationship between the RSI level and the profitability of the intraday momentum strategy. More specifically, a higher RSI, indicating that the market has recently moved higher, correlates with lower expected returns from the strategy. Conversely, a lower RSI, suggesting a recent market decline, correlates with increased expected returns.

This pattern aligns with the hypothesis that when the market has been rising, there is a higher likelihood of dealers having a long gamma position, which could cause intraday trends to be short-lived due to their delta hedging practices. Conversely, a sharp decline in market levels in recent days may lead dealers to shift to a negative gamma position, thus initiating a trend-chasing mechanism in their delta hedging strategies.

# 4.6 The Impact of Commissions (and Slippage) 

When dealing with low-frequency trading strategies that involve trading liquid instruments less than once per month, the role of transaction costs and slippage is negligible. However, when evaluating the profitability of intraday trading strategies, it is very important to conduct an in-depth analysis of how commissions and slippage could impact a trading edge. Estimating commissions is straightforward, as most brokers provide transparent details about the cost per share charged for each transaction. What is usually more difficult to estimate is the slippage paid for each transaction.

As we anticipate in the previous section, in the base strategy, we use the entry-level commission charged by Interactive Brokers: $\$ 0.0035$ per share. Typically, the commission charged by the broker depends on the monthly trading volume by the client. For example, at Interactive Brokers, the commission decreases to $\$ 0.0020$ per share when the traded volume exceeds 300,000 shares per month. In our backtest, we estimate that on approximately $20 \%$ of the days, we benefit from lower commission fees due to a 1-month trailing volume exceeding 300,000 shares.

As for slippage, instead of using a random number, we have conducted a real-market experiment on a few trading days in April 2024. Using an Interactive Brokers account

![img-10.jpeg](img-10.jpeg)

Figure 10: Impact of commissions on total return of intraday momentum strategy.

funded with $100,000, we instruct an algorithm to send market orders on SPY for $100,000 just a few milliseconds before the start of minute HH:MM. The algorithm randomly takes long or short positions. The goal of the study is to evaluate the magnitude of the slippage relative to the Open price of minute HH:MM. Mathematically, we evaluate the distribution of the following variable:

$$Slippage = Side \times (ExecutedPrice - Open_{HH:MM})$$

After executing over 1,000 trades, involving more than $50 million in notional value, the average slippage recorded is approximately $0.001, with a median value even lower at $0.0005. We recognize that these figures might rise with an increase in portfolio size, yet we remain confident that employing advanced trading techniques can facilitate large orders without significantly impacting the market. For simplicity and consistency with our real-trading analysis, we have chosen to apply a standard slippage rate of $0.001 in our backtest.

To further assess the sensitivity of our results to varying commission levels and associated slippage, we rerun the backtest using a spectrum of commission values. Figure 10 illustrates how the total return of the intraday momentum strategy fluctuates across different commission levels, offering a more nuanced understanding of how costs impact profitability. Given the available diversity in commission structures and order methodologies, we recommend that traders evaluate the profitability of the strategy based on their specific circumstances.

# 5 Conclusion 

In conclusion, this paper delves into the realm of intraday momentum trading strategies, building upon the foundational work laid by Jegadeesh and Titman's momentum effect. Through empirical analysis and rigorous backtesting, we demonstrate the profitability and viability of implementing intraday momentum strategies within the context of the SPY ETF, tracking the S\&P 500 index.

Our findings reveal that a simple intraday momentum strategy, initiated upon indications of abnormal demand/supply imbalance in intraday price action and incorporating dynamic trailing stops, yields substantial returns over the period from 2007 to early 2024. The intraday momentum portfolio achieves a remarkable total return of $1,985 \%$, with an annualized return of $19.6 \%$, and a Sharpe Ratio of 1.33 , underscoring its robustness and potential as a trading approach.

Moreover, we have conducted extensive statistical tests to assess the impact of different market volatility regimes and dealer gamma imbalances on the strategy's profitability. Additionally, our analysis of day-of-the-week effects and performance against well-known technical patterns provide further insights into the behavior of intraday momentum strategies under various market conditions.

This study contributes to the ongoing discourse on market anomalies and the practical applications of momentum trading, particularly in intraday contexts. By extending Jegadeesh and Titman's momentum framework to intraday trading, we provide valuable insights for traders operating in fast-paced intraday markets. Our research underscores the persistent nature of momentum effects within shorter time frames and highlights opportunities for profitable trading strategies in today's dynamic trading landscape.

In summary, the results offer compelling evidence supporting the efficacy of intraday momentum strategies, further enriching our understanding of market dynamics and offering actionable insights for market participants seeking to capitalize on short-term price movements.

# Author Biography 

![img-11.jpeg](img-11.jpeg)

Andrew Aziz is a Canadian trader, investor, and official Forbes Council member. He has ranked as one of the top 100 best-selling authors in "Business and Finance" for 7 consecutive years from 2016 to 2023. Aziz's book on finance has been published in 13 different languages. Originally from Iran, Andrew moved to Canada in 2008 to pursue a PhD in chemical engineering, initiating a distinguished career in academia and industry. As a research scientist, Andrew made significant contributions to the field, authoring 13 papers and securing 3 US patents. Following a successful stint in research in chemical engineering and clean technology, he transitioned to the world of trading. Currently Andrew is a trader and proprietary fund manager at Peak Capital Trading in Vancouver, BC Canada.
![img-12.jpeg](img-12.jpeg)

Carlo Zarattini, originally from Italy, currently resides in Lugano, Switzerland. After completing his mathematics degree in Padova, he pursued a dual master's in quantitative finance at Imperial College London and USI Lugano. He formerly served as a quantitative analyst at BlackRock, where he developed volatility and trend-following trading strategies. Carlo later established Concretum Research, assisting institutional clients with both high and medium-frequency quantitative strategies in stocks, futures, and options. Additionally, he founded R-Candles.com, the first backtester for discretionary technical traders.
![img-13.jpeg](img-13.jpeg)

Andrea Barbon, born in Venice, currently resides in Zurich, Switzerland. He holds a Master degree in pure mathematics from the University of Amsterdam and a PhD in finance from the University of Lugano. He is Assistant Professor of Financial Technology at the FSI Center of the University of St. Gallen, Switzerland and at the Swiss Finance Institute. His research interests include asset pricing, monetary policy, fintech, blockchain, and machine learning. He is also head of AI at Concretum Research, and lead developer for the R-Candles web application.

# References 

[1] N. Jegadeesh and S. Titman. Returns to buying winners and selling losers: Implications for stock market efficiency. J Finance, 48(1):65-91, 1993.
[2] A. Aziz. Advanced Techniques in Day Trading: A Practical Guide to High Probability Strategies and Methods. AMS Publishing Group, 2018.
[3] A. Aziz. How to Day Trade for a Living: A Beginner's Guide to Trading Tools and Tactics, Money Management, Discipline and Trading Psychology. AMS Publishing Group, 2015.
[4] C. Zarattini and A. Aziz. Can day trading really be profitable? evidence of sustainable long-term profits from opening range breakout (orb) day trading strategy vs. benchmark in the us stock market. SSRN Electronic Journal, 2023.
[5] C. Zarattini and A. Aziz. Volume weighted average price (vwap) the holy grail for day trading systems. SSRN Electronic Journal, 2023.
[6] Li Gao, Yan Han, S. Zhengzi Li, and Guofu Zhou. Market intraday momentum. J financ econ, 129(2):394-414, 2018.
[7] C. Rosa. Understanding intraday momentum strategies. Journal of Futures Markets, 42(12):2218-2234, 2022.
[8] G. Baltussen, Z. Da, S. Lammers, and M. Martens. Hedging demand and market intraday momentum. Journal of Financial Economics, 142(1):377-403, 2021.
[9] Y. Zhang, F. Ma, and B. Zhu. Intraday momentum and stock return predictability: Evidence from china. Economic Modelling, 76:319-329, 2019.
[10] M. Jin, F. Kearney, Y. Li, and Y.C. Yang. Intraday time-series momentum: Evidence from china. Journal of Futures Markets, 40(4):632-650, 2020.
[11] Z. Li, A. Sakkas, and A. Urquhart. Intraday time series momentum: Global evidence and links to market characteristics. Journal of Financial Markets, 57:100619, 2022.
[12] A. Aaziznia and A. Aziz. A Beginner's Guide to Investing and Trading in the Modern Stock Market. 2020.
[13] Tobias J Moskowitz, Yao Hua Ooi, and Lasse Heje Pedersen. Time series momentum. Journal of Financial Economics, 104(2):228-250, 2012.
[14] Andrea Frazzini. The disposition effect and underreaction to news. The Journal of Finance, 61(4):2017-2046, 2006.
[15] Stefano DellaVigna and Joshua M Pollet. Investor inattention and friday earnings announcements. The Journal of Finance, 64(2):709-749, 2009.
[16] Andrew B Abel, Janice C Eberly, and Stavros Panageas. Optimal inattention to the stock market with information costs and transactions costs. Econometrica, 81(4):1455-1481, 2013.

[17] Darrell Duffie. Presidential address: Asset price dynamics with slow-moving capital. The Journal of Finance, 65(4):1237-1267, 2010.
[18] T. Crabel. Day trading with short term price patterns and opening range breakout. 1990 .
[19] P. J. Kaufman. Trading Systems and Methods. Wiley, 5 edition, 2020.
[20] C. Zarattini, A. Barbon, and A. Aziz. A profitable day trading strategy for the u.s. equity market. SSRN Electronic Journal, February 2024.
[21] A. Barbon and A. Buraschi. Gamma fragility. SSRN Electronic Journal, November 2020 .

# Frequently Asked Questions (FAQs) 

Following the publication of this paper, we received hundreds of insightful questions from readers. While we are unable to address every inquiry-particularly those requiring significant extensions of our research or the disclosure of specific trading rules that we utilize to enhance strategy efficiency-we have compiled answers to the most common questions. This section will be updated regularly to include new Q\&As that may arise in the future.

## Q1. Can you share the code used to backtest the strategy?

Yes! We shared the Matlab and Python versions of the code in www. concretumgroup. com/coding. Users can easily change the backtest parameters (look-back window, volatility multiplier, slippage assumption, and even ticker) to create tailored versions of the strategy. Moreover, to make the backtesting procedure more accessible to readers with low coding skills, we exploit GoogleColab so users can run the backtest even without installing any library or coding software. Below you find the links to the codes:

- Matlab: https://bit.ly/BeatMarketCode ${ }^{4}$
- Python: https://bit.ly/BTestPyt ${ }^{5}$

Please note that to run the backtests, you will need to subscribe (even if it's a free subscription) to a data vendor. See Q2 for more details.

## Q2. Where do you source your data? Can you share the database?

The backtest presented in this paper utilizes intraday data from IQFeed, a leading data provider known for its high-quality datasets, especially for quantitative research on intraday timeframes. While IQFeed is a premium option, there are other valuable and more affordable alternatives, such as Polygon (which offers 2 years of free intraday data) and Alpaca (which provides 7 years of free intraday data). However, due to contractual agreements, we are unable to share the specific database used in this research.

## Q3. How can I incorporate the Noise Area into my intraday charts?

If you're using TradingView, you can easily add the Noise Area to your charts by utilizing the free indicator called Concretum Bands. You can access the TradingView indicator from https://bit.ly/CBands ${ }^{6}$. In the spirit of transparency, the PineScript code is open-source and can be customized to suit individual needs. The indicator is applicable to any ticker but is designed specifically for intraday time frames. We have also included two adjustable parameters: one for the Volatility Multiplier and one for the lookback period.

[^0]
[^0]:    ${ }^{4}$ https://www.concretumgroup.com/backtesting-riding-intraday-trends-in-us-markets-using-matlab/
    5
    ${ }^{6}$ https://www.concretumgroup.com/python-backtesting-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/
    ${ }^{6}$ https://www.tradingview.com/script/COpWCZhe-Concretum-Bands/

Q4. Do you have a monthly performance table for the strategy?
Yes, see the following table. For a more updated table, please refer to Q24.

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Yearly |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 2007 |  |  |  |  | -3.4 | 2.9 | 1.9 | 1.0 | 0.7 | -0.6 | 1.9 | 0.1 | 4.5 |
| 2008 | 5.1 | -1.6 | 2.2 | -1.3 | 8.6 | 5.3 | 7.3 | 4.7 | 5.5 | 8.3 | 3.6 | 3.0 | 63.4 |
| 2009 | 0.9 | 4.4 | 2.6 | 1.4 | 0.4 | 2.3 | 3.8 | -5.4 | 0.2 | 9.9 | -1.8 | 0.1 | 19.6 |
| 2010 | 7.3 | -1.0 | -3.1 | 2.5 | -3.3 | 3.0 | 5.1 | 1.3 | 1.6 | -4.8 | -5.6 | -0.8 | 1.3 |
| 2011 | 0.8 | 2.5 | 4.9 | -5.5 | -0.4 | 8.2 | 0.7 | 6.7 | -0.8 | 3.0 | 1.8 | 1.7 | 25.7 |
| 2012 | 0.5 | -4.6 | -0.6 | 7.5 | 0.3 | 4.9 | 3.6 | -0.6 | 4.2 | 2.9 | -1.4 | 3.3 | 21.1 |
| 2013 | -1.4 | 8.4 | -5.8 | 6.4 | -6.9 | 2.9 | -5.8 | -2.5 | 1.8 | 4.9 | 6.2 | 2.5 | 9.6 |
| 2014 | 4.6 | 2.2 | 1.2 | 3.0 | -0.5 | -1.6 | 4.9 | -3.7 | 2.2 | 3.1 | -1.1 | -2.1 | 12.6 |
| 2015 | -2.2 | -1.3 | 8.5 | -5.6 | 1.2 | 4.6 | 5.5 | 3.4 | -0.4 | -0.7 | 3.2 | 1.7 | 18.4 |
| 2016 | -1.0 | -1.9 | 1.3 | -1.4 | 0.4 | -7.3 | -3.1 | -5.2 | 8.4 | -5.0 | 1.4 | 0.7 | -12.8 |
| 2017 | -3.3 | 0.9 | 1.7 | -3.3 | -2.4 | 0.4 | -1.8 | 3.6 | -1.1 | 2.9 | -2.2 | -2.3 | -6.9 |
| 2018 | 3.5 | 8.9 | 7.2 | 3.5 | -4.7 | 2.0 | 6.2 | -1.7 | -3.5 | 13.4 | 3.0 | 12.5 | 61.1 |
| 2019 | 0.9 | -0.7 | 2.9 | 0.9 | -5.1 | -0.5 | 1.4 | 6.8 | -1.4 | 2.6 | -1.7 | 1.0 | 6.9 |
| 2020 | 5.3 | 1.9 | -0.5 | -1.8 | -0.2 | 8.0 | 0.1 | -1.8 | 14.3 | 2.8 | -1.1 | -1.9 | 26.8 |
| 2021 | 7.8 | 3.1 | 0.7 | 6.1 | 2.6 | -1.0 | 2.8 | 1.7 | 2.8 | 3.3 | -3.2 | 4.1 | 34.8 |
| 2022 | -5.5 | 2.8 | 0.2 | 9.0 | 2.1 | 0.5 | 0.2 | 6.3 | -1.0 | 5.8 | 1.6 | 0.7 | 24.4 |
| 2023 | 2.9 | -1.3 | 7.8 | 1.8 | 2.9 | 4.1 | 2.2 | 6.0 | 2.9 | -1.1 | 0.5 | 3.8 | 37.2 |
| 2024 | 8.8 | -1.5 | -0.4 | 5.8 |  |  |  |  |  |  |  |  | 12.9 |

Q5. Do the long and short legs of the strategy contribute equally? Can you provide separate results for each leg?
Both the long and short legs of the strategy have contributed positively to the overall portfolio performance. Given the generally positive trend of the S\&P 500 during the backtested period, it's expected that the long leg would contribute slightly more to the total return of the strategy. This is indeed the case, as illustrated in the figure below, where we have plotted the equity lines for the long-only, short-only, and combined longshort strategies.

# Q6. How does the lookback period for sigma affect the results? 

The Sharpe Ratio of the strategy remains largely unaffected by variations in the lookback period used to compute the Noise Area. Empirical evidence shows that the highest Total Returns and Sharpe Ratios are achieved with very long windows (e.g., 90 days). The table below presents key performance statistics for different lookback periods, with the parameter used in the paper highlighted in bold.

| LookBack <br> (days) | Total <br> Return (\%) | IRR (\%) | Vol (\%) | Sharpe <br> Ratio | MDD (\%) |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 5 | 1,856 | 19.1 | 14.1 | 1.31 | 21 |
| 10 | 1,570 | 18.0 | 14.3 | 1.23 | 23 |
| $\mathbf{1 4}$ | $\mathbf{2 , 1 0 8}$ | $\mathbf{1 9 . 9}$ | $\mathbf{1 4 . 3}$ | $\mathbf{1 . 3 5}$ | $\mathbf{2 3}$ |
| 20 | 2,060 | 19.7 | 14.4 | 1.33 | 23 |
| 25 | 2,189 | 20.2 | 14.4 | 1.35 | 23 |
| 30 | 1,941 | 19.4 | 14.3 | 1.31 | 23 |
| 60 | 1,842 | 19.0 | 14.1 | 1.31 | 25 |
| 90 | 2,757 | 21.7 | 13.8 | 1.50 | 13 |

![img-14.jpeg](img-14.jpeg)

Q7. How did the strategy perform during the worst 10 quarters for the S\&P 500 ?

The intraday momentum strategy demonstrated exceptional performance during the worst 10 quarters for the S\&P 500. As shown in the next table, the strategy consistently delivered positive returns in each of these challenging periods. Empirical evidence suggests that the performance of the active intraday momentum strategy tends to improve during volatile periods, which are typically associated with negative market returns.

|  | The 10 Worst Quarters for S\&P500 |  |  |
| :-- | :-- | :-- | :-- |
| Quarter | Event | S\&P500 | Strategy |
| Q4 2008 | Financial Crisis | $-22 \%$ | $16 \%$ |
| Q1 2020 | COVID-19 | $-19 \%$ | $7 \%$ |
| Q2 2022 | Inflation and Rates | $-16 \%$ | $12 \%$ |
| Q3 2011 | Debt Ceiling Crisis | $-14 \%$ | $7 \%$ |
| Q4 2018 | Trade War Escalation | $-14 \%$ | $31 \%$ |
| Q2 2010 | European Debt Crisis | $-11 \%$ | $2 \%$ |
| Q1 2009 | Financial Crisis Bottom | $-11 \%$ | $8 \%$ |
| Q1 2008 | Financial Crisis Onset | $-9 \%$ | $6 \%$ |
| Q3 2008 | Lehman Collapse | $-9 \%$ | $19 \%$ |
| Q3 2015 | China Market Crash | $-6 \%$ | $9 \%$ |

Q8. Can we consider the strategy a hedge for a passive long market exposure?

By design, the intraday momentum strategy cannot be considered a straightforward hedge for a passive long market portfolio. For example, if the market experiences a significant overnight gap down but stabilizes during regular trading hours without further downward pressure, the intraday momentum model may not be triggered, making it ineffective as a

hedging program.

However, empirical evidence shows that during volatile, negative market periods, the intraday momentum strategy often generates abnormal positive returns. Statistically, the likelihood of trend days (either upward or downward) increases with market volatility, which typically rises when the market is trending downward. As a result, the intraday momentum strategy is well-positioned to capitalize on these trend days, often exhibiting a negative correlation with passive long market exposure.
![img-15.jpeg](img-15.jpeg)

# Q9. What software did you use for the historical backtest? 

We used Matlab but you can easily obtain the same results using other well-known coding languages such as Python or R.

Q10. I have tried to replicate the backtest, but my results differ. Can you help?

Unfortunately, we are unable to debug individual user backtesting code. We strongly encourage you to compare your code with the Matlab or Python code available at www. concretumgroup.com/code (see question Q1).

If you require further assistance, we can recommend a few skilled freelancers who can help. Please send us an email at info@concretumgroup.com.

# Q11. Is it possible to automate this strategy in a personal trading account? How? 

Yes, but please note that this is a simplified model that can be further enhanced with a few small adjustments. The goal of our papers is not to provide ready-to-use trading strategies, but rather to introduce traders and readers to potential trading edges.

We do trade a slightly modified version of this strategy in our accounts, leveraging the functionalities of the Interactive Brokers API. While there are many ways to automate a trading strategy, in our portfolios, we update signals at high frequency using IQFeed and then automatically send orders to Interactive Brokers via IBML, a specialized Matlab trading package designed for both financial institutions and individual quant traders. For more information, visit: https://undocumentedmatlab.com/ib-matlab.

If you're not a skilled coder and would like to automate your trading rules, we can recommend a few highly skilled freelancers who can assist you. Please send us an email at info@concretumgroup.com.

## Q12. Does the strategy work with other ETFs or stocks?

Absolutely YES! Below you find a chart that show the performance from 2018 until August 2024 for some well-known ETFs (including BITO, the Bitcoin ETF) and some of the most liquid US stocks. A diversified portfolio would have produced a Sharpe Ratio that exceeds 2 .
![img-16.jpeg](img-16.jpeg)

# Q13. Does the strategy work also in futures markets? 

We applied the same rules to 33 different futures markets globally, covering 21 commodities and 12 equity indices, between January 2007 and September 2024. Of these, only 2 instruments produced a negative total return. The average Sharpe Ratio across all instruments was 0.60 , while a well-diversified portfolio trading all assets simultaneously achieved an impressive Sharpe Ratio of 1.65. The main summary statistics for each instrument are presented in the table below.

For more detailed insights, feel free to contact us at carlo@concretumgroup.com.

| Asset | Asset Class | TotRet <br> (\%) | CAGR <br> (\%) | Vol <br> (\%) | SR | MDD <br> (\%) | HitRatio <br> (\%) | Skew |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| *** | Commodities | 1,102 | 4.2 | 14.3 | 0.29 | 56 | 39 | 0.8 |
| *** | Commodities | 1,188 | 6.3 | 11.7 | 0.54 | 28 | 40 | 1.7 |
| *** | Commodities | -47 | -2.9 | 9.7 | $-0.30$ | 59 | 36 | 2.2 |
| *** | Commodities | 1,155 | 5.6 | 14.4 | 0.39 | 37 | 40 | 1.8 |
| *** | Commodities | -26 | -1.8 | 14.0 | $-0.13$ | 57 | 39 | 1.5 |
| *** | Commodities | 1,039 | 14.5 | 13.0 | 1.11 | 18 | 43 | 2.0 |
| *** | Commodities | 67 | 3.1 | 13.2 | 0.23 | 38 | 41 | 1.3 |
| *** | Commodities | 283 | 8.1 | 13.0 | 0.62 | 24 | 39 | 2.8 |
| *** | Commodities | 86 | 3.9 | 12.1 | 0.32 | 18 | 43 | 0.6 |
| *** | Commodities | 461 | 10.6 | 10.9 | 0.97 | 17 | 44 | 1.6 |
| *** | Commodities | 507 | 11.1 | 13.1 | 0.85 | 22 | 43 | 1.6 |
| *** | Commodities | 307 | 8.2 | 11.2 | 0.73 | 19 | 42 | 1.0 |
| *** | Commodities | 64 | 2.9 | 12.3 | 0.24 | 54 | 40 | 2.2 |
| *** | Commodities | 147 | 5.4 | 10.4 | 0.52 | 17 | 42 | 1.5 |
| *** | Commodities | 694 | 12.5 | 12.7 | 0.98 | 15 | 42 | 2.0 |
| *** | Commodities | 78 | 2.8 | 12.3 | 0.23 | 46 | 38 | 2.6 |
| *** | Commodities | 190 | 6.4 | 12.7 | 0.50 | 35 | 40 | 1.2 |
| *** | Commodities | 82 | 3.6 | 13.2 | 0.27 | 37 | 40 | 1.0 |
| *** | Commodities | 75 | 3.3 | 12.2 | 0.27 | 46 | 39 | 1.7 |
| *** | Commodities | 135 | 5.2 | 13.6 | 0.38 | 46 | 39 | 1.7 |
| *** | Commodities | 238 | 7.3 | 12.4 | 0.59 | 23 | 40 | 1.4 |
| Nasdaq | Equity | 1,533 | 16.8 | 13.6 | 1.23 | 26 | 43 | 2.1 |
| Russell | Equity | 181 | 6.1 | 13.6 | 0.45 | 36 | 42 | 1.7 |
| S\&P 400 MidCap | Equity | 538 | 11.0 | 13.7 | 0.80 | 21 | 42 | 1.7 |
| S\&P500 | Equity | 2,501 | 19.8 | 14.9 | 1.32 | 23 | 43 | 2.9 |
| *** | Equity | 283 | 7.8 | 14.9 | 0.53 | 35 | 38 | 2.2 |
| *** | Equity | 140 | 7.0 | 11.1 | 0.63 | 21 | 40 | 2.2 |
| *** | Equity | 105 | 4.1 | 15.5 | 0.26 | 41 | 37 | 2.4 |
| *** | Equity | 771 | 12.9 | 15.9 | 0.81 | 19 | 38 | 2.2 |
| *** | Equity | 679 | 12.1 | 14.6 | 0.83 | 30 | 42 | 2.6 |
| *** | Equity | 57 | 2.6 | 15.6 | 0.16 | 48 | 36 | 2.4 |
| *** | Equity | 871 | 14.0 | 14.3 | 0.98 | 30 | 40 | 1.8 |
| *** | Equity | 675 | 12.5 | 11.5 | 1.09 | 17 | 40 | 2.8 |

## Q14. To me, the strategy seems over-optimized!

We respect your perspective, but in our experience, it's uncommon to find a profitable systematic trading strategy that relies on so few parameters.

The buy/sell signals for this strategy are generated by comparing the price of SPY to the boundaries of the Noise Area. The two main parameters that define the Noise Area are:

- The lookback period used to calculate the average move from open to HH:MM (set to 14 days in the paper).
- The volatility multiplier (set to 1 in the paper).

If the strategy were the result of ex-post optimization, we would expect these parameters to produce the highest possible returns. However, as demonstrated in Section 4.4 and Q6, this is not the case. In fact:

- The optimal lookback period is 90 days (versus 14 days), which increases the Sharpe Ratio from 1.35 to 1.50 .
- The optimal Volatility Multiplier is 1.5 (versus 1), which raises the Sharpe Ratio from 1.35 to 1.55 .

Over-optimization is more commonly associated with black-box machine learning algorithms, where the high number of parameters and lack of signal interpretability can contribute to this issue.

Q15. The slippage model you used seems inaccurate. How do the results change with a more advanced and recognized slippage model?

Even though our proprietary models often utilize more advanced quantitative techniques, we prioritize comprehension in our papers to make our writing accessible even to those without a strong background in quantitative finance.

We acknowledge that the slippage estimate used in the paper may seem inaccurate, especially for large AUM sizes, but we strongly disagree that the strategy's profitability vanishes once more sophisticated slippage models are included in the backtest.

To address this critique, we reran the backtest using the I-Star Market Impact Model originally developed by Kissell and Malamut.

I-Star is a cost allocation approach where participants incur costs based on the size of their order and the overall participation in market volumes. Mathematically, we can express the slippage as

$$
I_{b p}^{*}=a_{1} \cdot\left(\frac{Q}{A D V}\right)^{a_{2}} \cdot \sigma^{a_{3}}
$$

where $Q$ is the amount of shares to transact, $A D V$ is the 30-days average daily volume $\mathrm{amd} \sigma$ is the the 30 -days price volatility of the underlying.

In his book The Science of Algorithmic Trading and Portfolio Management, Robert Kissell provides estimated values for the parameters $a_{1}, a_{2}$, and $a_{3}$ obtained from real-market tests. Given that these parameters can vary from small-cap to large-cap stocks, Kissell grouped the estimates by market capitalization.

| Group | $a_{1}$ | $a_{2}$ | $a_{3}$ |
| :-- | :--: | :--: | :--: |
| All Data | 708 | 0.55 | 0.71 |
| Large Cap | 687 | 0.70 | 0.72 |
| Small Cap | 702 | 0.47 | 0.69 |

Since SPY is one of the most traded instruments, we decided to use the parameters estimated for large-cap stocks. This may even yield conservative results, given that the liquidity of SPY exceeds that of most large-cap stocks. After including these slippage assumptions, the strategy's profitability remains very strong. The Sharpe Ratio stands at 1.17 (versus 1.33), with the yearly alpha remaining well above $15 \%$ (versus $20 \%$ ).

We strongly encourage well-capitalized readers to rerun the backtest using the slippage model and parameter set they consider most reliable for their specific case.

# Q16. If the strategy is so profitable, why share it with the public? 

In our early years, we gained invaluable insights from books and papers authored by experienced quant researchers. We hope our work will similarly inspire the next generation of quant traders and researchers. Additionally, given the liquidity of the underlying instrument and the trend-following nature of the model, we believe the risk of alpha decay is minimal. Moreover, due to regulatory constraints, very few institutional investors can extract alpha from intraday time frames, where turnover and leverage often exceed their trading limits. Academic research has shown that momentum-based strategies tend to be less susceptible to alpha decay after publication.

## Q17. The results are great, can you manage my account using this strategy?

Absolutely NOT! We are not financial advisors or money managers. Our goal is to share advanced quantitative research to help readers/traders enhance their financial understanding. While we may trade similar strategies in our personal accounts, we do not offer wealth management services or manage assets for external investors.

## Q18. Does the Intraday Momentum strategy exhibit intraday seasonalities?

In other words, is the intraday trend strategy effective throughout the entire trading session, or are there times when its performance is weaker?
Our analysis (see chart below) shows a steady upward movement from 10:00 to noon, indicating positive expected returns for trend continuation during this period. While trends tend to pause during lunchtime, they resume strongly from 14:00 until market close. This suggests that although the intraday trend strategy is effective for most of the trading session, the effectivness of trend trading varies during specific time of day.

![img-17.jpeg](img-17.jpeg)

Q19. Does the profitability of short trades increase when market volatility is high?
We ran a simple linear regression between the PnL (in basis points) of each short trade and the average VIX level on the same day. Contrary to common belief, we found no significant relationship between VIX and short trade profitability. This is visually confirmed by the flat black line representing the fitted linear relation. Additionally, a p-value of 0.91 for the beta reinforces this conclusion.
![img-18.jpeg](img-18.jpeg)

Q20. How does total profitability change if we only trade when the VIX is above a certain threshold?

The next bar chart shows the total return of all short trades, conditioned on the VIX being above a certain threshold. In other words, how much money did short trades generate if we only traded when the VIX was sufficiently high? This chart further confirms the regression results: limiting short trades to high volatility environments does not benefit overall compounded performance.
![img-19.jpeg](img-19.jpeg)

Q21. Should we open short trades only when the market is trading below key SMAs?
Here we examined the common adage suggesting that traders should only trade in the direction of the higher-time frame trend. We identified trends using three different Simple Moving Averages: 100, 150, and 200 days. The charts suggest that traders who take short intraday trend exposures only during bear markets are sacrificing a substantial portion of overall profitability.
![img-20.jpeg](img-20.jpeg)

# Q22. How do the results change if we use only VWAP as a trailing stop? 

Switching to a VWAP-only trailing stop results in less restrictive exit conditions, leading to a slight increase in the average holding period per trade. Consequently, the number of trades decreases significantly from 7,964 (for the VWAP + Current Band version) to 6,728 (for the VWAP-only version). This reduction in trade frequency may enhance cost efficiency; however, the overall performance metrics, as shown in the table below, do not improve. Specifically, the strategy's IRR declines by $1.6 \%$ annually, while annualized volatility increases by $1.2 \%$. This results in a Sharpe ratio drop from 1.35 to 1.17 , indicating a clear deterioration in risk-adjusted returns. The table summarizes the strategy's performance from May 2007 to December 2024.

| Strategy | Stop | Leverage | Total <br> Return | IRR | Vol | Sharpe <br> Ratio | Hit <br> Ratio | MDD |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Intra Mom. | Opp.Band | 1 x | 169 | 5.8 | 10.7 | 0.58 | 54.0 | 21 |
| Intra Mom. | Curr.Band + VWAP | 1 x | 393 | 9.5 | 7.6 | 1.23 | 44.0 | 12 |
| Intra Mom. | Curr.Band + VWAP | Dynamic | 2382 | 19.9 | 14.3 | 1.35 | 44.0 | 25 |
| Intra Mom. | VWAP | Dynamic | 1846 | 18.3 | 15.5 | 1.17 | 46.0 | 27 |

## Q23. How does the average trade profitability change over the years?

As detailed in the main body of the paper, specifically in Table 4, the average profit per trade is $\$ 0.09$ per share. While this figure might initially seem low, it is important to consider that it is influenced by the price levels of SPY during the backtest period. From 2007 to 2024, the average SPY price was $\$ 247$, with a minimum of $\$ 68$ (March 2009) and a maximum of $\$ 608$ recorded in December 2024.
Thanks to the overall appreciation of SPY over the years, the same percentage move in SPY today (December 2024) translates into a higher dollar PnL per share for each trade. The figure below illustrates the average profit per share grouped by year, highlighting a clear trend of increasing profitability per share over time. Over the last seven years, the average profit per share has been a robust $\$ 0.18$. Current SPY price levels, combined with the high liquidity available intraday on SPY, are positive contributors to the strategy's sustainable profitability, even after accounting for transaction costs.
![img-21.jpeg](img-21.jpeg)

# Q24. How has the strategy performed since the paper was published? 

Here, on a regular basis, we provide updates to the performance of the strategy presented in the paper. Below, you'll find an updated table with the monthly and yearly performance, along with a chart showing the equity line. The green area highlights the post-publication period.

Intraday Momentum Strategy
![img-22.jpeg](img-22.jpeg)

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Yearly |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 2007 |  |  |  |  | $-3.4$ | 2.9 | 1.9 | 1.0 | 0.7 | $-0.6$ | 1.9 | 0.1 | 4.5 |
| 2008 | 5.1 | $-1.6$ | 2.2 | $-1.3$ | 8.6 | 5.3 | 7.3 | 4.7 | 5.5 | 8.3 | 3.6 | 3.0 | 63.4 |
| 2009 | 0.9 | 4.4 | 2.6 | 1.4 | 0.4 | 2.3 | 3.8 | $-5.4$ | 0.2 | 9.9 | $-1.8$ | 0.1 | 19.6 |
| 2010 | 7.3 | $-1.0$ | $-3.1$ | 2.5 | $-3.3$ | 3.0 | 5.1 | 1.3 | 1.6 | $-4.8$ | $-5.6$ | $-0.8$ | 1.3 |
| 2011 | 0.8 | 2.5 | 4.9 | $-5.5$ | $-0.4$ | 8.2 | 0.7 | 6.7 | $-0.8$ | 3.0 | 1.8 | 1.7 | 25.7 |
| 2012 | 0.5 | $-4.6$ | $-0.6$ | 7.5 | 0.3 | 4.9 | 3.6 | $-0.6$ | 4.2 | 2.9 | $-1.4$ | 3.3 | 21.1 |
| 2013 | $-1.4$ | 8.4 | $-5.8$ | 6.4 | $-6.9$ | 2.9 | $-5.8$ | $-2.5$ | 1.8 | 4.9 | 6.2 | 2.5 | 9.6 |
| 2014 | 4.6 | 2.2 | 1.2 | 3.0 | $-0.5$ | $-1.6$ | 4.9 | $-3.7$ | 2.2 | 3.1 | $-1.1$ | $-2.1$ | 12.6 |
| 2015 | $-2.2$ | $-1.3$ | 8.5 | $-5.6$ | 1.2 | 4.6 | 5.5 | 3.4 | $-0.4$ | $-0.7$ | 3.2 | 1.7 | 18.4 |
| 2016 | $-1.0$ | $-1.9$ | 1.3 | $-1.4$ | 0.4 | $-7.3$ | $-3.1$ | $-5.2$ | 8.4 | $-5.0$ | 1.4 | 0.7 | $-12.8$ |
| 2017 | $-3.3$ | 0.9 | 1.7 | $-3.3$ | $-2.4$ | 0.4 | $-1.8$ | 3.6 | $-1.1$ | 2.9 | $-2.2$ | $-2.3$ | $-6.9$ |
| 2018 | 3.5 | 8.9 | 7.2 | 3.5 | $-4.7$ | 2.0 | 6.2 | $-1.7$ | $-3.5$ | 13.4 | 3.0 | 12.5 | 61.1 |
| 2019 | 0.9 | $-0.7$ | 2.9 | 0.9 | $-5.1$ | $-0.5$ | 1.4 | 6.8 | $-1.4$ | 2.6 | $-1.7$ | 1.0 | 6.9 |
| 2020 | 5.3 | 1.9 | $-0.5$ | $-1.8$ | $-0.2$ | 8.0 | 0.1 | $-1.8$ | 14.3 | 2.8 | $-1.1$ | $-1.9$ | 26.8 |
| 2021 | 7.8 | 3.1 | 0.7 | 6.1 | 2.6 | $-1.0$ | 2.8 | 1.7 | 2.8 | 3.3 | $-3.2$ | 4.1 | 34.8 |
| 2022 | $-5.5$ | 2.8 | 0.2 | 9.0 | 2.1 | 0.5 | 0.2 | 6.3 | $-1.0$ | 5.8 | 1.6 | 0.7 | 24.4 |
| 2023 | 2.9 | $-1.3$ | 7.8 | 1.8 | 2.9 | 4.1 | 2.2 | 6.0 | 2.9 | $-1.1$ | 0.5 | 3.8 | 37.2 |
| 2024 | 8.8 | $-1.5$ | $-0.4$ | 5.8 | $-4.3$ | 1.6 | 8.2 | $-2.8$ | 4.1 | 6.7 | $-2.6$ | 5.7 | 32.2 |
| 2025 | $-1.2$ | 7.6 | $-0.2$ | 3.8 |  |  |  |  |  |  |  |  | 10.1 |

# Q25. I have a question that is not part of the above FAQs! 

If your question isn't addressed in the FAQs above, don't hesitate to contact us directly. We're here to help and will do our best to respond promptly. You can reach us through the main author's email at: carlo@concretumgroup.com.

# A Further Tables

Table A1: Summary statistics of intraday momentum strategy with a) stop loss at opposite band, b) current band with VWAP, c) as in (b) with the additional dynamically adjusted share size based on daily market volatility, and d) SPY buy-hold strategy Commission set at $\$ 0.0035$ as per Interactive Brokers' entry-level rate. We highlight in bold coefficients that are statistically significant at $5 \%$ level or below.

|  Strategy | Stop | Size | Total Return | IRR | Vol | Sharpe
Ratio | Hit Ratio | MDD | Alpha | Beta | Skewness | Worst Day | Best Day  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Momentum | Opp.Band | $100 \%$ | $178 \%$ | $6.2 \%$ | $10.9 \%$ | 0.61 | $54 \%$ | $21 \%$ | $\mathbf{7 . 1 \%}$ | $\mathbf{- 0 . 0 5}$ | -1.24 | $-10.3 \%$ | $6.1 \%$  |
|  Momentum | Curr.Band + VWAP | $100 \%$ | $380 \%$ | $9.7 \%$ | $7.7 \%$ | 1.24 | $43 \%$ | $12 \%$ | $\mathbf{9 . 9 \%}$ | $\mathbf{- 0 . 0 3}$ | 1.29 | $-4.8 \%$ | $5.1 \%$  |
|  Momentum | Curr.Band + VWAP | Dyn. | $1,985 \%$ | $19.6 \%$ | $14.3 \%$ | 1.33 | $43 \%$ | $25 \%$ | $\mathbf{1 9 . 6 \%}$ | $\mathbf{- 0 . 0 7}$ | 1.22 | $-4.5 \%$ | $9.1 \%$  |
|  SPY (Buy\&Hold) |  | $100 \%$ | $227 \%$ | $7.2 \%$ | $20.2 \%$ | 0.45 | $54 \%$ | $56 \%$ |  |  | 0.08 | $-10.9 \%$ | $14.5 \%$  |

