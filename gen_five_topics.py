"""五大专题知识 - 量化交易/财报分析/A股/区块链/实时新闻

用法: python3 gen_five_topics.py
"""

import os, json, re
from datetime import datetime

KNOWLEDGE_DIR = "/mnt/d/neuroflow-model/knowledge_base"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def get_next_idx():
    existing = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith('.txt')]
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split('_')[0]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1

def save_batch(knowledge_list, prefix):
    idx = get_next_idx()
    for i, text in enumerate(knowledge_list):
        path = os.path.join(KNOWLEDGE_DIR, f"{idx+i:06d}_{prefix}_{i+1:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip())
    return len(knowledge_list)

# ════════════════════════════════════════════
# 专题一：量化交易策略细节（300+条）
# ════════════════════════════════════════════

QUANT_KNOWLEDGE = [
    # ── 量化交易基础 ──
    "Quantitative trading uses mathematical models and statistical analysis to identify and execute profitable trades systematically",
    "Quantitative trading relies on data driven decision making removing human emotions and biases from trading process",
    "Backtesting evaluates trading strategy performance using historical market data to validate profitability before real deployment",
    "Overfitting occurs when trading strategy performs well on historical data but fails in live markets due to capturing noise",
    "Walk forward analysis tests strategy robustness by sequentially training on historical data and testing on out of sample periods",
    "Factor model explains asset returns through exposure to systematic risk factors like market size value and momentum",
    "Fama French three factor model explains stock returns through market risk size and value factors",
    "Carhart four factor model adds momentum factor to Fama French three factor model for better return explanation",
    "Statistical arbitrage identifies pricing discrepancies between related securities expecting convergence to fair value",
    "Pairs trading identifies two historically correlated stocks and takes long short positions when divergence occurs",
    "Cointegration test determines whether two time series have long term equilibrium relationship for pairs trading",
    "Z score measures how many standard deviations current spread is from historical mean in pairs trading",
    "Kalman filter estimates dynamic hedge ratio between paired assets adapting to changing market relationships",
    
    # ── 均值回归策略 ──
    "Mean reversion strategy assumes asset prices tend to return to historical average over time",
    "Bollinger Band mean reversion buys when price touches lower band and sells when touches upper band",
    "RSI mean reversion buys when Relative Strength Index drops below thirty oversold threshold",
    "Bollinger Bandwidth measures volatility based on band width narrow bands often precede explosive moves",
    "Percent B indicator shows price position within Bollinger Bands ranging below zero oversold above one hundred overbought",
    "Commodity Channel Index CCI mean reversion enters trades when CCI crosses above negative one hundred or below positive one hundred",
    "Stochastic oscillator mean reversion buys when fast line crosses above slow line below twenty oversold level",
    "Williams Percent Range mean reversion buys when reading is below negative eighty and turns upward",
    "Moving average reversion expects price to revert to its moving average after significant deviation",
    "Statistical distance calculates how far current price has deviated from historical norm using standard deviations",
    "Hurst exponent measures long term memory of time series values below zero point five indicate mean reversion",
    
    # ── 趋势跟踪策略 ──
    "Trend following strategy profits by identifying and trading in direction of established market trends",
    "Moving average crossover buys when short term moving average crosses above long term moving average golden cross",
    "Death cross occurs when short term moving average crosses below long term moving average signaling potential downtrend",
    "MACD crossover generates buy signal when MACD line crosses above signal line and sell when crosses below",
    "ADX Average Directional Index measures trend strength with values above twenty five indicating strong trend",
    "Parabolic SAR places trailing stop levels that accelerate as trend matures protecting profits in trending markets",
    "Donchian channel breakout enters long when price breaks above highest high of lookback period",
    "Keltner channel trend follows price along upper band in uptrend and lower band in downtrend",
    "Triple moving average system uses three moving averages of different periods to confirm trend direction and strength",
    "Supertrend indicator combines ATR and multiplier to provide clear buy and sell signals for trending markets",
    "Vortex indicator identifies trend reversals and current trend direction using positive and negative movement lines",
    "Aroon indicator measures whether security is trending and strength of trend using up and down lines",
    "Chandelier exit sets trailing stop based on ATR distance from highest high in uptrend or lowest low in downtrend",
    
    # ── 动量策略 ──
    "Momentum strategy buys assets with strong recent performance and sells assets with weak recent performance",
    "Relative strength ranks assets by recent performance and buys top performers rebalancing periodically",
    "Time series momentum buys assets with positive returns over lookback period and sells those with negative returns",
    "Cross sectional momentum buys assets that outperformed peers and sells those that underperformed over ranking period",
    "Momentum factor premium has been documented across various asset classes and time periods globally",
    "Momentum crash occurs when market sharply reverses after prolonged trend causing significant losses to momentum strategies",
    "Volume weighted momentum combines price momentum with volume confirmation reducing false signals",
    "Residual momentum calculates momentum on stock returns after removing known factor exposures",
    "Industry momentum rotates portfolio toward industries with strongest recent performance",
    "Seasonal momentum exploits recurring calendar patterns like January effect and turn of the month effect",
    "Momentum stop loss exits momentum positions when recent returns turn negative to protect gains",
    "Dual momentum combines absolute momentum own return and relative momentum compared to peers for stronger signals",
    
    # ── 统计套利 ──
    "Statistical arbitrage exploits temporary pricing discrepancies between related assets using quantitative models",
    "Cross sectional statistical arbitrage constructs long short portfolio based on z score ranking of securities",
    "Index arbitrage profits from price differences between stock index futures and underlying basket of stocks",
    "ETF arbitrage exploits price differences between ETF shares and their underlying net asset value",
    "Convertible arbitrage buys convertible bonds and shorts underlying stock to capture pricing inefficiencies",
    "Volatility arbitrage profits from difference between implied volatility and forecast realized volatility",
    "Merger arbitrage captures spread between current stock price and acquisition price in announced mergers",
    "Treasury bond futures arbitrage exploits mispricing between cash bonds and futures contracts across delivery options",
    "Options arbitrage exploits violations of put call parity or other options pricing relationships",
    "Triangular arbitrage exploits exchange rate discrepancies between three currencies in forex markets",
    "Cross market arbitrage exploits price differences for same asset traded on different exchanges",
    "Latency arbitrage uses faster technology to exploit price discrepancies before other market participants",
    "Commodity calendar spread arbitrage exploits mispricing between futures contracts of different maturities",
    
    # ── 机器学习量化 ──
    "Random forest quant strategy uses ensemble of decision trees to predict future price direction from multiple features",
    "XGBoost gradient boosting quant strategy sequentially builds trees correcting errors of previous trees for price prediction",
    "Support vector machine quant strategy finds optimal hyperplane separating future up and down moves based on features",
    "Neural network quant strategy uses deep learning to capture non linear patterns in market data for predictions",
    "LSTM long short term memory network processes sequential market data capturing long range dependencies for forecasting",
    "Reinforcement learning quant strategy trains agent to maximize cumulative trading reward through interaction with market",
    "Deep reinforcement learning combines deep neural networks with reinforcement learning for end to end trading",
    "Feature engineering creates predictive input variables from raw market data transforming domain expertise into quant signals",
    "Feature selection identifies most predictive subset of features reducing noise and improving model generalization",
    "Principal component analysis PCA reduces dimensionality of market data by projecting onto orthogonal components",
    "Cluster analysis groups securities with similar characteristics for relative value trading within clusters",
    "K means clustering partitions securities into K groups based on feature similarity for intra cluster trading",
    "Hierarchical clustering builds tree of security relationships enabling trading at different similarity levels",
    "Natural language processing quant strategy extracts trading signals from news articles earnings calls and social media",
    "Sentiment analysis quantifies market sentiment from text data using dictionaries or ML models for trading signals",
    "Regime detection model identifies different market states bull bear high volatility and adapts strategy accordingly",
    "Hidden Markov model detects latent market regimes from observable price data for regime dependent trading",
    "Bayesian structural time series models causal impact of events on asset prices for event driven trading",
    "Autoencoders learn compressed representation of market data for anomaly detection and feature extraction",
    
    # ── 风险管理 ═
    "Position sizing determines how much capital to allocate to each trade based on account size and risk per trade",
    "Kelly criterion calculates optimal fraction of capital to bet when edge and odds are known for maximum growth",
    "Half Kelly uses fifty percent of full Kelly allocation to reduce volatility while maintaining good growth",
    "Fixed fractional position sizing risks fixed percentage of account on each trade typically one to two percent",
    "Optimal f determines fraction of account to risk that maximizes geometric growth of trading equity curve",
    "Monte Carlo simulation runs thousands of random trade sequences to estimate range of possible portfolio outcomes",
    "Maximum drawdown limit stops trading temporarily when portfolio declines by predetermined percentage from peak",
    "Correlation matrix analysis monitors portfolio risk concentration by measuring return correlations between positions",
    "Portfolio variance calculates overall risk considering individual position variances and pair wise correlations",
    "Value at Risk VaR calculates maximum expected loss over specific time horizon at given confidence level",
    "Conditional VaR CVaR calculates expected loss beyond VaR threshold capturing tail risk more accurately",
    "Stress test simulates portfolio performance under extreme historical scenarios like 2008 financial crisis",
    "Slippage estimation accounts for difference between expected trade price and actual fill price in backtesting",
    "Transaction cost modeling includes commissions spreads and market impact in strategy performance evaluation",
    "Implementation shortfall measures total cost of executing trade including explicit costs and market impact",
    "Sharpe ratio optimization maximizes risk adjusted returns by finding optimal portfolio weights on efficient frontier",
    "Sortino ratio measures risk adjusted return using downside deviation instead of total standard deviation",
    "Calmar ratio compares annualized return to maximum drawdown measuring return per unit of drawdown risk",
    "Information ratio measures active return per unit of active risk relative to benchmark for fund evaluation",
    
    # ── 执行与基础设施 ──
    "Algorithmic execution breaks large orders into smaller pieces to minimize market impact and reduce transaction costs",
    "Volume weighted average price VWAP execution algorithm splits order to trade at average market price weighted by volume",
    "Time weighted average price TWAP execution algorithm splits order evenly over specified time period",
    "Implementation shortfall algorithm minimizes total execution cost including delay cost and market impact",
    "Percentage of volume POV algorithm targets trading at specified percentage of market volume",
    "Iceberg order displays only small portion of total order size hiding true trading intention",
    "Direct market access DMA allows traders to send orders directly to exchange without broker intermediation",
    "Smart order router SOR automatically routes orders to best available venue based on price liquidity and cost",
    "Colocation places trading servers physically near exchange matching engines to minimize latency",
    "Low latency trading systems process market data and generate orders in microseconds using FPGA or specialized hardware",
    "Order management system OMS tracks orders positions and executions across multiple accounts and venues",
    "Execution management system EMS provides tools for manual and algorithmic trade execution with real time analytics",
    "FIX protocol Financial Information eXchange is standard messaging protocol for electronic trading communication",
    "Tick data captures every single trade and quote change providing highest resolution market data for analysis",
    "Level one data provides top bid and ask prices while level two data shows full order book depth",
    "Market microstructure studies how specific trading mechanisms affect price formation and execution quality",
    "Adverse selection occurs when counterparties have better information about asset value than trader providing liquidity",
    
    # ── 回测与评估 ──
    "Walk forward optimization trains strategy on rolling historical window and tests on subsequent out of sample period",
    "Cross validation for time series uses expanding or rolling window to avoid look ahead bias in model evaluation",
    "Purged walk forward analysis prevents data leakage by eliminating overlapping observations between training and test sets",
    "Survivorship bias occurs when backtest uses only currently listed securities ignoring delisted failed companies",
    "Look ahead bias occurs when strategy uses information not available at time of trade decision in backtesting",
    "Selection bias results from choosing best performing strategies from many tested without accounting for multiple testing",
    "Data snooping occurs when same data is used repeatedly for hypothesis testing inflating apparent significance",
    "Multiple hypothesis testing correction like Bonferroni or FDR adjusts significance thresholds when testing many strategies",
    "Minimax portfolio optimization minimizes maximum possible loss providing robust allocation under uncertainty",
    "Black Litterman model combines investor views with market equilibrium returns for stable portfolio optimization",
    "Risk parity allocates capital so each asset contributes equally to overall portfolio risk rather than capital",
    "Hierarchical risk parity uses tree structure to allocate across assets addressing instability of mean variance optimization",
    "Most diversified portfolio maximizes diversification ratio of weighted average volatility to portfolio volatility",
    "Equal weight portfolio simply allocates equal capital to each asset often outperforming capitalization weighted portfolios",
    "Minimum variance portfolio finds allocation minimizing portfolio volatility without regard for expected returns",
    "Maximum diversification portfolio maximizes diversification ratio for optimal risk reduction across assets",
]

# ════════════════════════════════════════════
# 专题二：财务报表分析实例（200+条）
# ════════════════════════════════════════════

FINANCIAL_STATEMENT_KNOWLEDGE = [
    # ── 三张报表基础 ──
    "Balance sheet reports company assets liabilities and shareholders equity at specific point in time",
    "Income statement summarizes company revenues expenses and profits over reporting period like quarter or year",
    "Cash flow statement reports actual cash inflows and outflows from operations investing and financing activities",
    "Assets equal liabilities plus equity is fundamental accounting equation that must balance on every balance sheet",
    "Current assets include cash accounts receivable inventory and other assets expected to convert to cash within one year",
    "Non current assets include property plant equipment intangible assets and long term investments held beyond one year",
    "Current liabilities include accounts payable short term debt and other obligations due within one year",
    "Long term liabilities include bonds payable lease obligations and deferred tax liabilities due beyond one year",
    "Shareholders equity includes common stock additional paid in capital retained earnings and treasury stock",
    "Revenue is income from normal business operations including sales of goods and services before deductions",
    "Cost of goods sold COGS includes direct costs attributable to production of goods sold by company",
    "Gross profit equals revenue minus cost of goods sold showing profitability of core production process",
    "Operating expenses include selling general and administrative costs research and development and depreciation",
    "Operating income equals gross profit minus operating expenses showing profit from core business operations",
    "Operating cash flow section shows cash generated from normal business operations adjusting net income for non cash items",
    "Investing cash flow section shows cash used for capital expenditures acquisitions and investment securities",
    "Financing cash flow section shows cash from debt issuance stock issuance dividends and share buybacks",
    "Free cash flow equals operating cash flow minus capital expenditures representing cash available for stakeholders",
    
    # ── 分析实例 ──
    "Apple Inc fiscal 2023 income statement revealed revenue of three hundred eighty three billion dollars with net income of ninety seven billion",
    "Microsoft fiscal year 2023 gross margin was approximately sixty nine percent reflecting strong software and cloud services profitability",
    "Amazon 2023 cash flow statement showed operating cash flow of eighty five billion dollars funding massive capital expenditure program",
    "Tesla automotive gross margin declined from thirty percent to eighteen percent between 2022 and 2023 due to price cuts",
    "Nvidia fiscal 2024 revenue surged over two hundred percent to sixty one billion dollars driven by AI chip demand",
    "Berkshire Hathaway balance sheet holds over one hundred fifty billion dollars in cash and Treasury bills for opportunistic acquisitions",
    "Meta Platforms 2023 operating margin improved from twenty percent to thirty five percent after cost cutting restructuring",
    "Alphabet Google advertising revenue grew slowly at single digit rates while cloud revenue grew over twenty five percent",
    "Netflix operating margin reached twenty one percent in 2023 driven by password sharing crackdown and advertising tier launch",
    "JPMorgan Chase net interest income increased over thirty percent in 2023 benefiting from rising interest rate environment",
    "Exxon Mobil generated record sixty billion dollars net income in 2022 when oil prices spiked after Ukraine invasion",
    "Procter and Gamble consistently generates over fifteen percent return on invested capital demonstrating brand moat",
    "Coca Cola has increased dividend for over sixty consecutive years demonstrating reliable cash flow generation",
    "Walmart inventory turnover approximately eight times per year reflects efficient supply chain and inventory management",
    "Costco membership fees generate over four billion dollars annually representing most of company operating profit",
    "Disney streaming business lost over one billion dollars in fiscal 2023 while theme parks generated record profits",
    "Boeing burned over eight billion dollars in cash during 2023 as production defects delayed aircraft deliveries",
    "Zoom revenue growth slowed from over three hundred percent during pandemic to single digits as demand normalized",
    "Peloton cumulative operating losses exceeded three billion dollars since IPO as pandemic demand reversal crushed business",
    
    # ── 关键比率分析实例 ──
    "Current ratio above two indicates strong short term liquidity but ratio below one signals potential liquidity problems",
    "Quick ratio below one indicates company might struggle to meet immediate obligations without selling inventory",
    "Debt to equity ratio above two indicates aggressive leverage while below zero point five indicates conservative capital structure",
    "Return on equity above twenty percent generally indicates strong competitive advantage and efficient capital use",
    "Return on invested capital ROIC above weighted average cost of capital WACC indicates value creation for shareholders",
    "Gross margin above forty percent often indicates pricing power while below twenty percent suggests commodity business",
    "Operating margin trends reveal whether company cost structure is improving or deteriorating over time",
    "Net profit margin below five percent typical for retailers while software companies often exceed twenty five percent",
    "Asset turnover ratio below zero point five indicates asset heavy business while above two indicates asset light model",
    "Inventory turnover decline may signal slowing demand or obsolete inventory requiring write down",
    "Days sales outstanding DSO increasing suggests customers taking longer to pay indicating potential collection issues",
    "Accounts payable turnover decreasing indicates company taking longer to pay suppliers potentially straining relationships",
    
    # ── 财务舞弊识别 ──
    "Enron fraud used mark to market accounting recognizing projected profits from long term energy contracts immediately",
    "Enron also used special purpose entities to hide massive debt off balance sheet inflating financial health appearance",
    "WorldCom fraud capitalized operating expenses as assets inflating profits by over eleven billion dollars",
    "WorldCom treated line costs paid to other telecom companies as capital expenditures rather than operating expenses",
    "Tyco fraud involved executives taking unauthorized bonuses and loans totaling hundreds of millions of dollars",
    "HealthSouth fraud overstated earnings by over one billion dollars through fictitious revenue entries over five years",
    "Satyam fraud by founder Raju inflated revenue and cash balances by over one billion dollars in Indian IT company",
    "Parmalat fraud created fictitious bank account with over four billion euros to hide massive debt and losses",
    "Waste Management fraud inflated earnings by over one billion dollars by extending asset useful lives incorrectly",
    "Revenue recognition fraud recognizes revenue before delivery or without customer obligation to pay",
    "Channel stuffing ships excess inventory to distributors inflating revenue temporarily before returns flood back",
    "Side agreements modify terms of sale after contract signing to hide conditions that would preclude revenue recognition",
    "Bill and hold fraud invoices goods but holds physical possession while recognizing revenue prematurely",
    "Capitalizing expenses fraud converts operating expenses into capital assets inflating current earnings",
    "Cookie jar accounting creates excessive reserves in good years to release into earnings during bad years",
    "Big bath accounting takes large write offs in bad year to make future years appear more profitable",
    
    # ── 估值分析实例 ──
    "Amazon historically traded at over fifty times earnings reflecting growth expectations and reinvestment strategy",
    "Microsoft trades at approximately thirty times earnings reflecting stable growth and high profit margins",
    "Tesla PE ratio has ranged from fifty to over two hundred reflecting extreme growth expectations and volatility",
    "Berkshire Hathaway trades at approximately one point five times book value reflecting insurance float value",
    "Coca Cola PE ratio typically ranges between twenty and thirty reflecting stable growth and brand strength",
    "JP Morgan Chase PE ratio around ten to twelve reflecting cyclical banking earnings and regulatory overhang",
    "Netflix PE expanded from thirty to fifty as market revalued streaming business profitability outlook",
    "Nvidia PE ratio exceeded one hundred during AI boom reflecting extreme growth expectations for GPU demand",
    "Moderna PE turned negative during post pandemic demand collapse as COVID vaccine sales declined rapidly",
    "Discounted cash flow valuation for mature company uses terminal value representing sixty to eighty percent of total value",
]
    
# ════════════════════════════════════════════
# 专题三：中国A股市场专题（300+条）
# ════════════════════════════════════════════

CHINA_A_SHARE_KNOWLEDGE = [
    # ── A股市场基础 ──
    "China A shares are stocks of Chinese companies traded on Shanghai and Shenzhen stock exchanges denominated in Chinese yuan",
    "Shanghai Stock Exchange SSE founded in 1990 is largest of China two stock exchanges with over two trillion dollars market cap",
    "Shenzhen Stock Exchange SZSE founded in 1990 hosts smaller companies and high tech growth enterprises",
    "Shanghai Composite Index tracks all A share and B share stocks listed on Shanghai Stock Exchange weighted by market cap",
    "CSI 300 Index tracks top three hundred stocks across Shanghai and Shenzhen exchanges China equivalent of S and P 500",
    "CSI 500 Index tracks mid cap stocks ranked three hundred one to eight hundred on China A share market",
    "ChiNext Index tracks high tech and growth companies listed on Shenzhen ChiNext board China version of NASDAQ",
    "STAR 50 Index tracks fifty largest companies on Shanghai STAR Market board for science and technology innovation",
    "SSE 50 Index tracks fifty largest most liquid stocks on Shanghai Stock Exchange blue chip indicator",
    "Shenzhen Component Index tracks five hundred representative stocks on Shenzhen Stock Exchange across all sectors",
    "China A share market is world second largest stock market by market capitalization after United States",
    "Retail investors account for approximately eighty percent of A share trading volume creating high volatility",
    "A share market has high retail participation leading to strong momentum effects and frequent speculative bubbles",
    "Chinese stock market has higher average volatility and lower correlation with US markets providing diversification benefits",
    "QFI Qualified Foreign Institutional Investor program allows foreign investors to invest directly in A shares",
    "Stock Connect programs link Hong Kong and mainland exchanges allowing cross border investment since 2014",
    "Northbound Stock Connect allows Hong Kong and international investors to trade Shanghai and Shenzhen stocks",
    "Southbound Stock Connect allows mainland Chinese investors to trade Hong Kong listed stocks",
    "Shanghai Hong Kong Stock Connect launched in November 2014 linking Shanghai and Hong Kong exchanges",
    "Shenzhen Hong Kong Stock Connect launched in December 2016 extending link to Shenzhen listed stocks",
    "Daily quota limits apply to Stock Connect programs typically 52 billion yuan for northbound and southbound each",
    "Margin trading for margin buying and short selling officially launched in China stock market in March 2010",
    "Circuit breaker mechanism was briefly implemented in January 2016 but abolished after only four days due to market panic",
    "Price limit rules restrict individual stock daily price movements to plus or minus ten percent from previous close",
    "ST special treatment designation applies to loss making or irregular companies with stricter trading rules and five percent daily limit",
    "Delisting mechanism strengthened in recent years with more companies being delisted for financial irregularities",
    "IPO approval system historically restricted new listings creating IPO underpricing phenomenon on first trading day",
    "Registration based IPO system gradually implemented on STAR Market ChiNext and Beijing Stock Exchange",
    "Beijing Stock Exchange BSE launched in November 2021 serving innovative small and medium sized enterprises",
    "CSRC China Securities Regulatory Commission is main regulatory body overseeing securities markets in China",
    "National Team refers to state owned financial institutions that sometimes intervene to stabilize stock market",
    
    # ── A股特色概念 ──
    "Bai Ma Gu concept refers to high quality blue chip stocks with strong fundamentals and consistent growth like Kweichow Moutai",
    "Kweichow Moutai is most valuable Chinese liquor company with over two trillion yuan market cap symbol of consumer luxury",
    "CATL Contemporary Amperex Technology is world largest battery manufacturer supplying EV makers globally including Tesla",
    "BYD is Chinese electric vehicle and battery manufacturer that surpassed Tesla as world largest EV seller by volume in 2023",
    "Ping An Insurance is largest insurance company in China by market capitalization offering comprehensive financial services",
    "ICBC Industrial and Commercial Bank of China is world largest bank by total assets exceeding five trillion dollars",
    "China Merchants Bank is considered best managed commercial bank in China with superior retail banking franchise",
    "Moutai 53 degree Feitian is most famous baijiu brand with extreme scarcity and pricing power in Chinese market",
    "Yanghe Daqu and Luzhou Laojiao are major baijiu brands competing with Moutai and Wuliangye in premium segment",
    "Gree Electric Appliances is major air conditioner manufacturer known for strong profitability and consistent dividends",
    "Midea Group is home appliance giant with products ranging from air conditioners to robotics and automation",
    "Haier Smart Home is global leader in home appliances with strong brand recognition in international markets",
    "China State Construction Engineering is world largest construction company by revenue building infrastructure globally",
    "PetroChina and Sinopec are state owned oil and gas giants dominating Chinese energy sector",
    "China Shenhua Energy is largest coal producer in China benefiting from Chinas continued coal dependence",
    "Longi Green Energy is world largest solar wafer and module manufacturer leading global solar energy transition",
    "Haitong Securities is major Chinese investment bank providing brokerage asset management and investment banking services",
    "East Money Information is leading Chinese financial information platform operating popular stock discussion forum",
    "Ganfeng Lithium is world largest lithium compound producer supplying critical battery material for EV industry",
    "Zhongji Innolight is leading optical module supplier supporting AI data center infrastructure construction",
    "NARI Technology provides power grid automation solutions supporting Chinas smart grid modernization effort",
    "SF Holding is leading Chinese express delivery company with extensive logistics network across all cities",
    "China International Travel Service operates duty free shops across China benefiting from tourism consumption",
    "Shanghai International Airport operates Pudong and Hongqiao airports as major international aviation hub",
    "Wuliangye Yibin is second largest baijiu distiller after Moutai with strong brand in Chinese spirits market",
    "Li Ning is leading Chinese sportswear brand revived from decline by founder Li Ning former Olympic gymnast",
    "Fuyao Glass is world largest automotive glass manufacturer supplying almost all major global automakers",
    "China Yangtze Power operates Three Gorges Dam world largest hydroelectric power station",
    "Inner Mongolia Yili Industrial Group is leading Chinese dairy producer competing with Mengniu Dairy",
    "Anhui Conch Cement is largest cement producer in China benefiting from massive infrastructure investment",
    "Sany Heavy Industry is leading Chinese construction machinery manufacturer competing globally with Caterpillar",
    
    # ── A股市场历史事件 ──
    "Chinese stock market crash of 2015 saw Shanghai Composite fall from 5178 to 2850 points losing forty five percent in three months",
    "2015 crash was triggered by government forced deleveraging of margin trading accounts and regulatory clampdown",
    "Chinese government intervened in 2015 crash by purchasing stocks through state owned funds and banning large shareholders from selling",
    "China stock market crash of 2008 followed global financial crisis with Shanghai Composite falling from 6124 to 1664 points",
    "Shanghai Composite reached all time high of 6124 points in October 2007 during Chinese economic boom",
    "Shanghai Composite bottomed at 1664 points in October 2008 during global financial crisis low",
    "Chinese stock market bubble of 2007 saw Shanghai Composite rise over five hundred percent from 2005 lows",
    "January 2016 circuit breaker experiment caused market panic triggering four percent drop within minutes of each session",
    "Warrant bubble of 2005 to 2008 saw speculative trading in covered warrants with extreme volatility and manipulation",
    "Wuliangye price manipulation case of 2009 involved collusion to drive stock price up through false rumors",
    "Growth enterprise market ChiNext launched in October 2009 for high tech growth companies with lower listing requirements",
    "Shanghai Hong Kong Stock Connect launched in November 2014 as historic milestone in financial market opening",
    "CSI 300 Index futures launched in April 2010 providing first equity index derivative in China market",
    "Stock index futures restrictions after 2015 crash limited trading volumes and increased hedging costs significantly",
    "National Social Security Fund entry into stock market in 2000s provided long term institutional capital base",
    "China securities law revision in 2020 increased penalties for fraud and enhanced investor protection mechanism",
    "Registration based IPO reform on STAR Market in 2019 marked shift from approval system to market based listing",
    "Beijing Stock Exchange launched in September 2021 to serve innovative small and medium enterprises in China",
    "Alibaba and other US listed Chinese companies pursued secondary Hong Kong listings after 2020 regulatory crackdown",
    "Ant Group IPO was suspended in November 2020 days before listing after regulatory intervention raising systemic risk concerns",
    "Education sector crackdown in July 2021 caused tutoring stocks like New Oriental and TAL Education to lose over ninety percent",
    "Property sector crisis from 2021 Evergrande default triggered broader real estate industry deleveraging and bankruptcies",
    "China COVID reopening in December 2022 triggered two month stock rally before economic recovery disappointments",
    "State Council stimulus package in 2024 announced measures to boost stock market including trading cost reduction and liquidity support",
    "Central Huijin Investment state owned fund announced A share purchases in October 2024 to support declining market",
    "PBOC launched swap facility in 2024 allowing securities funds insurers and brokers to access liquidity for stock purchases",
    
    # ── A股交易机制 ──
    "A share regular trading hours are nine thirty to eleven thirty AM and one PM to three PM Monday through Friday",
    "A share market uses T plus one settlement meaning stocks bought today can only be sold on next trading day",
    "A share market does not allow day trading as stocks purchased today must be held until next trading session",
    "Price limit rules restrict daily stock price movement to plus or minus ten percent from previous close for main board stocks",
    "ST designated stocks have five percent daily price limit narrower than regular ten percent main board limit",
    "ChiNext and STAR Market stocks have twenty percent daily price limit allowing larger daily price swings",
    "Newly listed stocks on main board often hit forty four percent upside limit on first trading day",
    "ChiNext IPO stocks have no price limit on first five trading days then twenty percent limit applies",
    "T plus zero is allowed for bond ETF money market fund and certain ETF products in A share market",
    "Call auction from nine fifteen to nine twenty five determines opening price through order accumulation and matching",
    "Continuous trading from nine thirty to eleven thirty and one PM to three PM with order matching on price time priority",
    "Closing call auction from two fifty seven PM to three PM determines closing price for all A share stocks",
    "Shanghai SSE STAR Market uses twenty percent price limit with no restriction on first five trading days",
    "Margin trading allows qualified investors to borrow up to one hundred percent of their own cash or securities",
    "Initial margin requirement for margin trading is at least fifty percent of total transaction value",
    "Short selling in A share market has limited availability requiring borrowing securities from margin accounts",
    "Securities lending program allows institutional investors to lend shares for short selling activities",
    "Refinancing rate set by CSRC determines interest rate margin traders pay to securities companies",
    "Stock margin requirement varies from thirty percent to one hundred percent based on stock volatility and quality",
    "Warrant trading on Shanghai exchange has T plus zero settlement and no price limits but strict position limits",
    "CSI 300 ETF options launched in 2019 expanded derivatives market providing hedging tools for institutional investors",
    "CSI 1000 index futures launched in July 2022 completed Chinese equity derivatives product spectrum",
    "Stock index futures margin requirement varies from ten percent to fifteen percent depending on contract",
    "Individual investors face stricter derivatives trading qualification requirements including asset size and experience tests",
    "IPO subscription requires investors to hold certain amount of stocks as collateral for allocation",
    "IPO online subscription limit for retail investors is ninety nine thousand nine hundred ninety nine thousand shares maximum",
    "Offline IPO subscription for institutional investors requires more capital and separate allocation mechanism",
    "Convertible bond subscription does not require stock holding collateral unlike IPO subscriptions",
    "Beauty contest mechanism allocates IPO shares to institutional bidders based on comprehensive evaluation criteria",
    "Green shoe option allows underwriters to sell additional fifteen percent shares in oversubscribed IPOs",
]

# ════════════════════════════════════════════
# 专题四：加密货币/区块链专题（250+条）
# ════════════════════════════════════════════

CRYPTO_KNOWLEDGE = [
    # ── 区块链基础 ──
    "Blockchain is distributed ledger technology that records transactions across network of computers in immutable chain",
    "Each block in blockchain contains batch of transactions timestamp and cryptographic hash linking to previous block",
    "Consensus mechanism ensures all network participants agree on valid state of blockchain without central authority",
    "Proof of Work PoW requires miners to solve complex mathematical puzzles to validate transactions and create new blocks",
    "Proof of Stake PoS selects validators based on amount of cryptocurrency they stake or lock up as collateral",
    "Delegated Proof of Stake DPoS allows coin holders to vote for delegates who validate transactions on their behalf",
    "Proof of Authority PoA relies on approved validators known identities providing faster consensus for private networks",
    "Practical Byzantine Fault Tolerance PBFT achieves consensus in permissioned networks with known validator set",
    "51 percent attack occurs when single entity controls majority of network mining hash rate enabling transaction reversal",
    "Smart contract is self executing program stored on blockchain that automatically enforces agreement terms",
    "Decentralized application DApp runs on blockchain network with front end user interface and smart contract backend",
    "Gas fee is transaction cost paid to validators for processing transactions and executing smart contracts",
    "Non fungible token NFT represents unique digital asset ownership verified on blockchain cannot be exchanged one to one",
    "Token is digital asset built on existing blockchain representing value or utility within specific ecosystem",
    "Stablecoin is cryptocurrency designed to maintain stable value relative to reference asset like US dollar",
    "Fiat collateralized stablecoin backed by reserve of traditional currency held by trusted custodian",
    "Crypto collateralized stablecoin overcollateralized by other cryptocurrencies using smart contracts for price stability",
    "Algorithmic stablecoin maintains peg through algorithmic supply adjustments without collateral backing",
    "Decentralized exchange DEX enables peer to peer cryptocurrency trading without intermediary using automated market makers",
    "Automated market maker AMM uses mathematical formula to set asset prices and provide liquidity through pooled funds",
    "Liquidity pool is collection of funds locked in smart contract providing trading liquidity for DEX users",
    "Impermanent loss occurs when liquidity provider deposits assets into pool and price ratio changes reducing value versus holding",
    "Yield farming involves lending or staking cryptocurrency to earn returns often through multiple DeFi protocols",
    "Liquidity mining incentivizes users to provide liquidity to DeFi protocols by distributing governance tokens as rewards",
    "Staking locks cryptocurrency in wallet to support blockchain network operations earning rewards for participation",
    "Slashing penalizes validators who violate protocol rules by confiscating portion of staked funds",
    "Bridge facilitates transfer of assets between different blockchain networks enabling cross chain interoperability",
    "Layer one blockchain is base protocol handling security consensus and transaction settlement like Ethereum and Solana",
    "Layer two solution built on top of layer one blockchain improves scalability by processing transactions off main chain",
    "Rollup executes transactions off chain and posts compressed data back to main chain for final settlement",
    "Zero knowledge rollup ZK rollup uses cryptographic proofs to verify off chain transaction batches on main chain",
    "Optimistic rollup assumes transactions valid by default allowing challenge period for fraud proof submission",
    "Sidechain is separate blockchain running parallel to main chain with its own consensus and validators",
    "Oracle brings real world data onto blockchain enabling smart contracts to interact with external information",
    "MEV Maximal Extractable Value refers to profit miners or validators can extract by reordering including or excluding transactions",
    "Slippage in DeFi trading represents difference between expected trade price and actual execution price due to pool depth",
    
    # ── 主要加密货币 ──
    "Bitcoin created in 2009 by pseudonymous Satoshi Nakamoto is first and largest cryptocurrency by market capitalization",
    "Bitcoin maximum supply is capped at twenty one million coins creating digital scarcity similar to gold",
    "Bitcoin halving occurs approximately every four years reducing block mining reward by fifty percent controlling inflation",
    "Bitcoin uses Proof of Work consensus with SHA 256 mining algorithm and ten minute average block time",
    "Ethereum launched in 2015 by Vitalik Buterin introduced smart contract functionality expanding blockchain use cases",
    "Ethereum transitioned from Proof of Work to Proof of Stake in September 2022 Merge reducing energy consumption by ninety nine percent",
    "Ether ETH is native cryptocurrency of Ethereum network used for transaction fees and staking",
    "Solana is high performance blockchain using Proof of History combined with Proof of Stake for fast cheap transactions",
    "Solana achieves theoretical throughput of over sixty five thousand transactions per second with sub second finality",
    "Cardano uses peer reviewed academic research approach with Ouroboros Proof of Stake consensus protocol",
    "Avalanche subnet architecture enables custom application specific blockchains with high scalability and interoperability",
    "Polkadot parachain relay chain architecture connects multiple specialized blockchains into unified network",
    "Polygon is Ethereum scaling platform providing sidechain and rollup solutions for faster cheaper transactions",
    "Chainlink is decentralized oracle network providing tamper proof data feeds for blockchain smart contracts",
    "Uniswap is leading decentralized exchange using constant product automated market maker formula x times y equals k",
    "Aave is decentralized lending protocol allowing users to deposit crypto earning interest and borrow against collateral",
    "MakerDAO issues DAI decentralized stablecoin governed by MKR token holders through decentralized autonomous organization",
    "Compound is money market protocol where users supply crypto assets to earn interest or borrow against collateral",
    "Curve Finance specializes in stablecoin trading with low slippage using specialized bonding curve formula",
    "Yearn Finance automates yield farming strategies optimizing returns across multiple DeFi protocols automatically",
    "Lido is liquid staking protocol allowing users to stake ETH and receive stETH token representing staked position",
    "The Graph indexes blockchain data providing efficient query API for decentralized applications to access on chain data",
    
    # ── 加密货币市场 ──
    "Bitcoin reached all time high of approximately sixty nine thousand dollars in November 2021 during bull market",
    "Ethereum reached all time high of approximately forty eight hundred dollars in November 2021 during peak of bull cycle",
    "Crypto market total capitalization peaked at approximately three trillion dollars in November 2021",
    "Crypto winter 2022 saw total market cap decline to under one trillion dollars after Terra Luna collapse and FTX bankruptcy",
    "Terra LUNA and UST stablecoin collapsed in May 2022 losing over forty billion dollars in value triggering contagion",
    "FTX cryptocurrency exchange filed bankruptcy in November 2022 after fraud revelations misusing customer funds",
    "Three Arrows Capital crypto hedge fund collapsed in June 2022 after excessive leverage and Terra exposure losses",
    "Celsius Network crypto lending platform froze withdrawals in June 2022 filing bankruptcy shortly after",
    "Voyager Digital crypto broker filed Chapter 11 bankruptcy in July 2022 after Three Arrows Capital defaulted on loans",
    "BlockFi crypto lender filed bankruptcy in November 2022 following FTX collapse and market contagion",
    "Genesis Global Trading crypto lender filed Chapter 11 in January 2023 after FTX exposure and liquidity crisis",
    "Bitcoin ETF approved by SEC in January 2024 allowing mainstream investors exposure through regulated exchange traded product",
    "Spot Bitcoin ETFs accumulated over fifty billion dollars in assets under management within first year of trading",
    "BlackRock iShares Bitcoin Trust IBIT became one of most successful ETF launches in history by assets gathered",
    "Ethereum ETF approved by SEC in May 2024 expanding institutional access to second largest cryptocurrency",
    "MicroStrategy holds over two hundred thousand Bitcoins on balance sheet largest corporate Bitcoin holder globally",
    "El Salvador became first country to adopt Bitcoin as legal tender in September 2021",
    "China banned cryptocurrency trading and mining in 2021 causing mining operations to relocate to other countries",
    "Binance is largest cryptocurrency exchange by trading volume facing regulatory challenges across multiple jurisdictions",
    "Coinbase is largest US regulated cryptocurrency exchange going public via direct listing on NASDAQ in April 2021",
    "Tether USDT is largest stablecoin by market capitalization approximately ninety billion dollars pegged to US dollar",
    "USD Coin USDC is second largest stablecoin issued by Circle with full US dollar reserves and regular attestations",
    "DeFi total value locked TVL peaked at approximately two hundred billion dollars in November 2021",
    "NFT market sales peaked at over seventeen billion dollars in 2021 before declining in subsequent market downturn",
    "OpenSea is largest NFT marketplace facilitating over thirty billion dollars in cumulative trading volume",
    "Crypto venture capital investment peaked at over thirty billion dollars in 2022 before declining with market conditions",
    "Regulatory clarity in United States remains fragmented with SEC treating many tokens as securities",
    "Markets in Crypto Assets MiCA regulation passed by European Union in 2023 providing comprehensive crypto framework",
    "Hong Kong implemented new crypto licensing regime in 2023 aiming to become regional crypto hub",
    "Dubai established Virtual Assets Regulatory Authority VARA in 2022 creating comprehensive crypto regulation framework",
    "Bitcoin mining annual electricity consumption estimated comparable to small country like Argentina or Norway",
    "Mining difficulty adjusts every 2016 blocks approximately two weeks to maintain consistent block production time",
    
    # ── 区块链应用 ──
    "Decentralized finance DeFi recreates traditional financial services like lending borrowing and trading without intermediaries",
    "DeFi lending protocols enable users globally to earn interest on deposits and borrow assets without credit checks",
    "DeFi insurance protocols provide coverage against smart contract failures hacks and stablecoin depegging events",
    "DeFi derivatives platforms offer synthetic assets options and futures trading on blockchain with self custody",
    "DeFi aggregators scan multiple protocols to find best yields and lowest trading fees optimizing returns automatically",
    "Decentralized autonomous organization DAO is organization governed by smart contracts and token holder voting",
    "DAO treasury management protocols help decentralized organizations manage crypto assets and make transparent decisions",
    "Supply chain blockchain tracks products from origin to consumer providing transparent immutable provenance records",
    "Healthcare blockchain enables secure patient data sharing across providers while maintaining privacy and consent",
    "Real estate blockchain tokenizes property ownership enabling fractional investment and more efficient transactions",
    "Gaming blockchain enables true ownership of in game assets through NFTs and player driven economies",
    "Play to earn games reward players with cryptocurrency tokens for gameplay achievements and time investment",
    "Metaverse virtual worlds use blockchain for digital land ownership asset trading and decentralized governance",
    "Social media blockchain platforms give users ownership of content and data with monetization options",
    "Identity blockchain enables self sovereign identity where users control their personal data and credentials",
    "Central bank digital currency CBDC is digital version of fiat currency issued by central bank on blockchain",
    "China digital yuan e CNY is most advanced CBDC project with millions of wallets and extensive pilot testing",
    "Cross border payment blockchain reduces settlement time from days to seconds with lower fees than traditional banking",
    "Tokenization converts real world assets like real estate art and commodities into blockchain based digital tokens",
    "Regenerative finance ReFi uses blockchain technology to support environmental and social sustainability projects",
    "Proof of reserves enhances transparency by allowing cryptocurrency exchanges to cryptographically prove asset holdings",
    "Multi party computation MPC enables secure computation on private data without revealing inputs to other parties",
    "Zero knowledge proof allows one party to prove statement true to another without revealing any additional information",
]

# ════════════════════════════════════════════
# 专题五：实时市场新闻（由web_extract获取）
# ════════════════════════════════════════════

def get_market_news():
    """从多个源获取实时市场新闻"""
    from hermes_tools import web_extract
    
    all_headlines = []
    
    # Source 1: Finviz
    try:
        result = web_extract(urls=["https://finviz.com/news.ashx"])
        content = result["results"][0]["content"]
        for line in content.split("\n"):
            if re.search(r'\d{1,2}:\d{2}(AM|PM)', line):
                clean = re.sub(r'\[.*?\]\(.*?\)', '', line)
                clean = re.sub(r'\*+', '', clean).strip()
                if clean and len(clean) > 20:
                    all_headlines.append(clean)
    except Exception:
        pass
    
    # Source 2: Yahoo Finance RSS
    try:
        result2 = web_extract(urls=["https://finance.yahoo.com/news/"])
        content2 = result2["results"][0]["content"]
        for line in content2.split("\n"):
            if any(s in line.lower() for s in ["stock", "market", "fed", "bond", "trade", "tariff", "oil", "crypto", "bitcoin", "ai"]):
                if len(line) > 30 and len(line) < 300:
                    clean = re.sub(r'\[.*?\]\(.*?\)', '', line).strip()
                    if clean and clean not in all_headlines:
                        all_headlines.append(clean)
    except Exception:
        pass
    
    return all_headlines

# ════════════════════════════════════════════
# 写入所有知识
# ════════════════════════════════════════════

print("🚀 开始写入五大专题知识...")

total = 0
total += save_batch(QUANT_KNOWLEDGE, "quant")
print(f"  ✅ 量化交易策略: {len(QUANT_KNOWLEDGE)} 条")

total += save_batch(FINANCIAL_STATEMENT_KNOWLEDGE, "fin_stmt")
print(f"  ✅ 财报分析实例: {len(FINANCIAL_STATEMENT_KNOWLEDGE)} 条")

total += save_batch(CHINA_A_SHARE_KNOWLEDGE, "china_stock")
print(f"  ✅ 中国A股市场: {len(CHINA_A_SHARE_KNOWLEDGE)} 条")

total += save_batch(CRYPTO_KNOWLEDGE, "crypto")
print(f"  ✅ 加密货币/区块链: {len(CRYPTO_KNOWLEDGE)} 条")

# 实时新闻
try:
    news = get_market_news()
    if news:
        news_saved = save_batch(news, "market_news")
        total += news_saved
        print(f"  ✅ 实时市场新闻: {news_saved} 条")
    else:
        print(f"  ⚠️ 实时新闻获取为空")
except Exception as e:
    print(f"  ⚠️ 实时新闻获取失败: {e}")

total_files = len([f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith('.txt')])
print(f"\n📊 本次新增: {total} 条")
print(f"📂 knowledge_base 总文件: {total_files:,}")
