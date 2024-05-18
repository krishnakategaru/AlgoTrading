import yfinance as yf
import pandas as pd
from sentiment_analysis.sentiment_analysis_pipeline import do_sentiment_analysis
from sentiment_analysis.sentiment_analysis_trasformer_model import prepare_sentiment_from_transformer
from sentiment_analysis.sentiment_analysis_gru_model import train_model
class StockDataProcessor:
    def __init__(self, stock_ticker, start_date, end_date, news_sentiment,ohlc_sentiment,ohlc_ta_sentiment):
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        # self.sentiment_data_path = sentiment_data_path
        self.news_sentiment = news_sentiment
        self.ohlc_sentiment = ohlc_sentiment
        self.ohlc_ta_sentiment = ohlc_ta_sentiment
        self.data = self.download_stock_data()

    def download_stock_data(self):
        """
        Download stock data from Yahoo Finance.

        Returns:
            pd.DataFrame: Stock data.
        """
        return yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)

    def preprocess_sentiment_data(self):
        """
        Preprocess sentiment data and merge with stock data.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        if self.news_sentiment :
            news_sentiment_data,_ = do_sentiment_analysis(self.stock_ticker, self.start_date, self.end_date)
            # Create a column for buy/sell signals based on sentiment
            news_sentiment_data['signal'] = 0
            news_sentiment_data.loc[news_sentiment_data['sentiment'] == 'POSITIVE', 'signal'] = 1
            news_sentiment_data.loc[news_sentiment_data['sentiment'] == 'NEGATIVE', 'signal'] = -1

            # Assuming df is your existing DataFrame
            news_sentiment_data['timestamp'] = pd.to_datetime(news_sentiment_data['timestamp']).dt.date

            # Group by day and sum up 'Signal' values
            sentiment_daily_sum = news_sentiment_data.groupby('timestamp')['signal'].sum().reset_index()
            sentiment_daily_sum = sentiment_daily_sum.rename(columns={'timestamp': 'date', 'signal': 'signal'})

            sentiment_daily_sum['date'] = pd.to_datetime(sentiment_daily_sum['date'])

            sentiment_daily_sum.to_csv('data/sentiment_daily_sum.csv')
            # Merge DataFrames on 'Date'
            merged_df = pd.merge(self.data, sentiment_daily_sum, left_index=True, right_on='date', how='left')
            merged_df.set_index('date', inplace=True)
        
        #sentiment of transformer model
        if self.ohlc_sentiment:
            transformer_sentiment_data,_ = prepare_sentiment_from_transformer(self.stock_ticker, self.start_date, self.end_date,model = 'svm')
            print(transformer_sentiment_data.columns)
            transformer_sentiment_data = transformer_sentiment_data["transformer_sentiment"].copy()
            # transformer_sentiment_data['date'] = transformer_sentiment_data.iloc[:,0]
            if self.news_sentiment:
                merged_df = pd.merge(merged_df, transformer_sentiment_data,left_index=True, right_index = True, how ='left')
            else :
                merged_df = pd.merge(self.data, transformer_sentiment_data, left_index=True, right_index = True, how='left')
                # merged_df.set_index('date', inplace=True)
        
        if self.ohlc_ta_sentiment :
            full_model_transformer_sentiment_data,_ = train_model(self.stock_ticker, self.start_date, self.end_date,batch_size= 256, sequence_length= 30,stride = 1)
            # print(transformer_sentiment_data.columns)
            full_model_transformer_sentiment_data = full_model_transformer_sentiment_data["ohlc_ta_sentiment"].copy()
            # transformer_sentiment_data['date'] = transformer_sentiment_data.iloc[:,0]
            if self.news_sentiment or self.ohlc_sentiment:
                merged_df = pd.merge(merged_df, full_model_transformer_sentiment_data,left_index=True, right_index = True, how ='left')
            else :
                merged_df = pd.merge(self.data, full_model_transformer_sentiment_data, left_index=True, right_index = True, how='left')
                # merged_df.set_index('date', inplace=True)
         

        merged_df.to_csv('data/merged_df.csv')

        return merged_df
