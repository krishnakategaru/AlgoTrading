import os
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")
from processor.stock_data_processor import StockDataProcessor
from runner.backtest_runner import BacktestRunner
from sentiment_analysis.sentiment_analysis_pipeline import do_sentiment_analysis
from sentiment_analysis.sentiment_analysis_trasformer_model import prepare_sentiment_from_transformer

if __name__ == '__main__':
    # Configuration
    STOCK_TICKER = 'INFY'
    START_DATE = '2024-01-01'
    END_DATE = '2024-05-01'
    news_sentiment = True
    ohlc_sentiment = True
    SENTIMENT_DATA_PATH = 'data/stock_sentiment_data.csv'



    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Initialize the StockDataProcessor
    processor = StockDataProcessor(STOCK_TICKER, START_DATE, END_DATE, news_sentiment,ohlc_sentiment)

    # Download stock data
    stock_data = processor.download_stock_data()
    # Preprocess sentiment data and merge with stock data
    merged_df = processor.preprocess_sentiment_data()
    # Run backtest
    # BacktestRunner.run_backtest('data/merged_df.csv', STOCK_TICKER, START_DATE, END_DATE)

    # Define the list of indicators to use
    indicators = ['rsi','bollinger','macd','ema','fast_ma',"news_sentiment",'transformer_sentiment']

    # Generate all possible combinations of indicators
    indicator_combinations = list(itertools.chain.from_iterable(itertools.combinations(indicators, r) for r in range(1, len(indicators)+1)))

    results_df = []
    # Run backtest for each combination of indicators
    for indicators in indicator_combinations:
        print(f"Running backtest for indicators: {', '.join(indicators)}")
        results = BacktestRunner.run_backtest('data/merged_df.csv', STOCK_TICKER, START_DATE, END_DATE, indicators)
        
        # Convert the results dictionary to a DataFrame
        results_df.append(results)

        # Save the results DataFrame to an Excel file
        pd.DataFrame(results_df).to_excel('output/backtest_results_'+STOCK_TICKER+'_'+str(START_DATE)+'_'+str(END_DATE)+'.xlsx')
