import backtrader as bt
class SentimentData(bt.feeds.GenericCSVData):
    """
    Custom Backtrader data feed class for sentiment data.

    Parameters:
    - dtformat (str): Date format for parsing the date column.
    - date (int): Column index for the date in the CSV file.
    - signal (int): Column index for the sentiment signal in the CSV file.
    - transformer_sentiment (int): Column index for the sentiment signal from the transformer model in the CSV file.
    - openinterest (int): Column index for the open interest in the CSV file.
    """

    lines = ('signal', 'transformer_sentiment')

    params = (
        ('dtformat', '%Y-%m-%d'),
        ('date', 0),
        ('signal', 7),
        ('transformer_sentiment', 8),
        ('openinterest', -1)
    )
