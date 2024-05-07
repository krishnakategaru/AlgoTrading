# !pip install transformers
from transformers import pipeline
import sys
sys.path.append("D:\krishna\msdsm//trimister 6\Project\KrishnaProject\AlgoTrading")
from alpaca.client import AlpacaNewsFetcher
from openai import OpenAI


class NewsSentimentAnalysis:
    """
  A class for sentiment analysis of news articles using the Transformers library.

  Attributes:
  - classifier (pipeline): Sentiment analysis pipeline from Transformers.
  """

    def __init__(self):
        """
    Initializes the NewsSentimentAnalysis object.
    """
        self.classifier = pipeline('sentiment-analysis')

    def analyze_sentiment(self, news_article):
        """
    Analyzes the sentiment of a given news article.

    Args:
    - news_article (dict): Dictionary containing 'summary', 'headline', and 'created_at' keys.

    Returns:
    - dict: A dictionary containing sentiment analysis results.
    """
        summary = news_article['summary']
        title = news_article['title']
        timestamp = news_article['timestamp']

        relevant_text = summary + title
        sentiment_result = self.classifier(relevant_text)

        analysis_result = {
            'timestamp': timestamp,
            'title': title,
            'summary': summary,
            'sentiment': sentiment_result
        }

        return analysis_result
    
    def analyze_sentiment_using_mistral(self, symbol,news_article):
        summary = news_article['summary']
        title = news_article['title']
        timestamp = news_article['timestamp']
        relevant_text = summary + title
        prompt = f'''[INST]
        You are a sentiment analysis model that can classify news articles as having a positive or negative sentiment. You will be given a news article, and your task is to determine its sentiment based on the content of the article. Please provide a one-word answer, either "POSITIVE" or "NEGATIVE"
        [/INST]

        [userINST]
        {relevant_text}
        [userINST]

        "[Insert one-word sentiment classification here: "POSITIVE" or "NEGATIVE"]"
         '''


        completion = client.chat.completions.create(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": relevant_text}
            ],
            temperature=0.7,
        )

        assistant_response = completion.choices[0].message.content.strip()
        sentiment_result = assistant_response.split()[0]
        analysis_result = {
        'timestamp': timestamp,
        'title': title,
        'summary': summary,
        'sentiment': sentiment_result
    }

        return analysis_result



def do_sentiment_analysis(symbol,start_date,end_date):
    news_fetcher = AlpacaNewsFetcher()

    # Fetch news for AAPL from 2021-01-01 to 2021-12-31
    news_data = news_fetcher.fetch_news(symbol=symbol, start_date=start_date, end_date=end_date)

    
    import pandas as pd
    complete_news = []

    # Initialize the NewsSentimentAnalysis object
    news_sentiment_analyzer = NewsSentimentAnalysis()

    # Assume 'news_data' is a list of news articles (each as a dictionary)
    for article in news_data:

        sentiment_analysis_result = news_sentiment_analyzer.analyze_sentiment(article)
        complete_news.append({'timestamp': sentiment_analysis_result["timestamp"],'title': sentiment_analysis_result["title"],'sentiment': sentiment_analysis_result['sentiment'][0]['label']})
    pd.DataFrame(complete_news).set_index('timestamp').to_csv('data/sentiment_analysis.csv')

    return pd.DataFrame(complete_news),'data/sentiment_analysis.csv'

def do_sentiment_analysis_using_mistral(symbol,start_date,end_date):
    news_fetcher = AlpacaNewsFetcher()

    # Fetch news for AAPL from 2021-01-01 to 2021-12-31
    news_data = news_fetcher.fetch_news(symbol=symbol, start_date=start_date, end_date=end_date)

    import pandas as pd
    complete_news = []

    # Initialize the NewsSentimentAnalysis object
    news_sentiment_analyzer = NewsSentimentAnalysis()

    # Assume 'news_data' is a list of news articles (each as a dictionary)
    for article in news_data:

        sentiment_analysis_result = news_sentiment_analyzer.analyze_sentiment(article)
        complete_news.append({'timestamp': sentiment_analysis_result["timestamp"],'title': sentiment_analysis_result["title"],'sentiment': sentiment_analysis_result['sentiment'][0]['label']})
    pd.DataFrame(complete_news).set_index('timestamp').to_csv('data/sentiment_analysis.csv')

    return 'data/sentiment_analysis.csv'

