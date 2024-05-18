import backtrader as bt

class Customstrategy(bt.Strategy):
    """
    Custom Backtrader strategy with advanced technical indicators and sentiment analysis.

    Parameters:
    - fast_ma (int): Period for the fast moving average.
    - slow_ma (int): Period for the slow moving average.
    - rsi_period (int): Period for the Relative Strength Index (RSI).
    - rsi_oversold (float): RSI level considered as oversold for buying.
    - rsi_overbought (float): RSI level considered as overbought for selling.
    - bollinger_window (int): Period for Bollinger Bands.
    - bollinger_dev (float): Standard deviation factor for Bollinger Bands.
    - ema_window (int): Period for Exponential Moving Average (EMA).
    - envelopes_ema_window (int): Period for EMA used in Envelopes indicator.
    - envelopes_percentage (float): Percentage for Envelopes indicator.
    - macd_short_window (int): Short window period for MACD.
    - macd_long_window (int): Long window period for MACD.
    - macd_signal_window (int): Signal window period for MACD.
    - stochastic_k_window (int): Window period for Stochastic Oscillator %K.
    - stochastic_d_window (int): Window period for Stochastic Oscillator %D.
    """

    params = (
        ("fast_ma", 20),
        ("slow_ma", 50),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("bollinger_window", 20),
        ("bollinger_dev", 2),
        ("ema_window", 20),
        ("envelopes_ema_window", 20),
        ("envelopes_percentage", 5),
        ("macd_short_window", 12),
        ("macd_long_window", 26),
        ("macd_signal_window", 9),
        ("stochastic_k_window", 14),
        ("stochastic_d_window", 3),
    )

    def __init__(self, indicators=None):
        """
        Initializes the AdvancedStrategy.

        Creates and initializes the required technical indicators and sentiment data based on the selected indicators:
        - fast_ma: Fast Simple Moving Average (SMA)
        - slow_ma: Slow Simple Moving Average (SMA)
        - rsi: Relative Strength Index (RSI)
        - bollinger: Bollinger Bands
        - ema: Exponential Moving Average (EMA)
        - macd: Moving Average Convergence Divergence (MACD)
        - stochastic: Stochastic Oscillator
        - envelopes: Envelopes
        """
        self.indicators = indicators or {}
        
        if "ma" in self.indicators:
            self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_ma)
        if "ma" in self.indicators:
            self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_ma)
        if "rsi" in self.indicators:
            self.rsi = bt.indicators.RelativeStrengthIndex(self.data.close, period=self.params.rsi_period)
        if "bollinger" in self.indicators:
            self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_window, devfactor=self.params.bollinger_dev)
        if "ema" in self.indicators:
            self.ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_window)
        if "macd" in self.indicators:
            self.macd = bt.indicators.MACD(self.data.close, period_me1=self.params.macd_short_window, period_me2=self.params.macd_long_window, period_signal=self.params.macd_signal_window)
        if "stochastic" in self.indicators:
            self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.stochastic_k_window, period_dfast=self.params.stochastic_d_window)
        if "envelopes" in self.indicators:
            self.envelopes = bt.indicators.Envelope(self.data.close, period=self.params.envelopes_ema_window, devfactor=self.params.envelopes_percentage/100)

        self.sentiment = self.datas[0].signal if len(self.datas) > 0 else None
        self.transformer_sentiment = self.datas[0].transformer_sentiment if len(self.datas) > 0 else None
        self.ohlc_ta_model_sentiment = self.datas[0].ohlc_ta_model_sentiment if len(self.datas) > 0 else None
    def next(self):
        """
        Executes the trading logic on each iteration.
        """
        buy_signal = sell_signal = 0

        if "ma" in self.indicators and self.fast_ma > self.slow_ma:
            buy_signal += 1
        else:
            sell_signal += 1

        if "rsi" in self.indicators and self.rsi[0] < self.params.rsi_oversold:
            buy_signal += 1
        elif "rsi" in self.indicators and self.rsi[0] > self.params.rsi_overbought:
            sell_signal += 1

        if "bollinger" in self.indicators and self.data.close[0] < self.bollinger.lines.bot[0]:
            buy_signal += 1
        elif "bollinger" in self.indicators and self.data.close[0] > self.bollinger.lines.top[0]:
            sell_signal += 1

        if "ema" in self.indicators and self.data.close[0] > self.ema[0]:
            buy_signal += 1
        elif "ema" in self.indicators and self.data.close[0] < self.ema[0]:
            sell_signal += 1

        if "macd" in self.indicators and self.macd[0] > 0:
            buy_signal += 1
        elif "macd" in self.indicators and self.macd[0] < 0:
            sell_signal += 1

        if "stochastic" in self.indicators and self.stochastic[0] < self.stochastic.lines.d[0]:
            buy_signal += 1
        elif "stochastic" in self.indicators and self.stochastic[0] > self.stochastic.lines.d[0]:
            sell_signal += 1

        if "envelopes" in self.indicators and self.data.close[0] > self.envelopes.lines.erveh[0]:
            sell_signal += 1
        elif "envelopes" in self.indicators and self.data.close[0] < self.envelopes.lines.ervlo[0]:
            buy_signal += 1
        
        if "news_sentiment" in self.indicators and self.sentiment is not None and self.sentiment[0] > 0:
            buy_signal += 1
        elif "news_sentiment" in self.indicators and self.sentiment is not None and self.sentiment[0] < 0:
            sell_signal += 1
        
        if "transformer_sentiment" in self.indicators and self.transformer_sentiment is not None and self.transformer_sentiment[0] > 0:
            buy_signal += 1
        elif "transformer_sentiment" in self.indicators and self.transformer_sentiment is not None and self.transformer_sentiment[0] < 0:
            sell_signal += 1
        
        if "full_model_transformer_sentiment" in self.indicators and self.ohlc_ta_model_sentiment is not None and self.ohlc_ta_model_sentiment[0] > 0:
            buy_signal += 1
        elif "full_model_transformer_sentiment" in self.indicators and self.ohlc_ta_model_sentiment is not None and self.ohlc_ta_model_sentiment[0] < 0:
            sell_signal += 1

        if buy_signal > sell_signal:
            self.buy()
        elif sell_signal > buy_signal:
            self.sell()
