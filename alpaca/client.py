from alpaca_trade_api import REST
import yaml
from datetime import datetime, timedelta
import requests

import sys
sys.path.append("D:\krishna\msdsm//trimister 6\Project\KrishnaProject\AlgoTrading")

with open('configuration.yaml', 'r') as file:
    config = yaml.safe_load(file)

class NewsAPI:
    def __init__(self):
        self.api_key = config['newsapi_api_key']
        
    
    def get_stock_news(self,stock_ticker, start_date, end_date):
        """
        Fetches financial news related to a specific stock ticker within a date range.

        Args:
            stock_ticker (str): The stock ticker symbol (e.g., AAPL, MSFT).
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            api_key (str): Your NewsAPI API key.

        Returns:
            list: A list of dictionaries containing news articles.
        """
       
        
        formatted_news = []
        current_date = start_date
        next_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


        # Loop through the date range in increments of one day
        while current_date <= end_date:

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": stock_ticker,
                "source" : "cnbc",
                "from": current_date,
                "to": next_date,
                "apiKey": self.api_key
            }

            response = requests.get(url, params=params)
            news_articles = response.json()
            #

            for article in news_articles:
                summary = article['content']
                title = article['title']
                timestamp = article['publishedAt']

                relevant_info = {
                    'timestamp': timestamp,
                    'title': title,
                    'summary': summary
                }

                formatted_news.append(relevant_info)

            # Move to the next day
            current_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            next_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")

        return formatted_news

class AlpacaNewsFetcher:
    """
    A class for fetching news articles related to a specific stock from Alpaca API.

    Attributes:
    - api_key (str): Alpaca API key for authentication.
    - api_secret (str): Alpaca API secret for authentication.
    - rest_client (alpaca_trade_api.REST): Alpaca REST API client.
    """

    def __init__(self):
        """
        Initializes the AlpacaNewsFetcher object.

        Args:
        - api_key (str): Alpaca API key for authentication.
        - api_secret (str): Alpaca API secret for authentication.
        """
        self.api_key = config['alpaca_api_key']
        self.api_secret = config['alpaca_api_secret']
        self.rest_client = REST(self.api_key, self.api_secret)

    def fetch_news(self, symbol, start_date, end_date):
        """
        Fetches news articles for a given stock symbol within a specified date range.

        Args:
        - symbol (str): Stock symbol for which news articles are to be fetched (e.g., "AAPL").
        - start_date (str): Start date of the range in the format "YYYY-MM-DD".
        - end_date (str): End date of the range in the format "YYYY-MM-DD".

        Returns:
        - list: A list of dictionaries containing relevant information for each news article.
        """
        formatted_news = []
        current_date = start_date
        next_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


        # Loop through the date range in increments of one day
        while current_date <= end_date:
            # Fetch news articles for the current date
            news_articles = self.rest_client.get_news(symbol, current_date, next_date)

            for article in news_articles:
                summary = article.summary
                title = article.headline
                timestamp = article.created_at

                relevant_info = {
                    'timestamp': timestamp,
                    'title': title,
                    'summary': summary
                }

                formatted_news.append(relevant_info)

            # Move to the next day
            current_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            next_date = (datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")

        return formatted_news
