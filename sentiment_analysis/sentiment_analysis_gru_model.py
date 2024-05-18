#import libraries
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import warnings
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

warnings.filterwarnings("ignore")


# symbol_to_fetch = 'AAPL'
# start_date = '2020-01-01'
# end_date = '2024-05-01'
# # Parameters
# batch_size = 256
# sequence_length = 30

def fetch_ticker_data(symbol, start_date, end_date):
    """Fetches stock data for a given symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start='1980-01-01', end=end_date)
    return data

def label_data(data):
    # Calculate the percentage change in price from one day to the next
    data['pr_change_on_last_day'] = data['Close'].pct_change()
    data['pr_change_on_current_day'] = data['pr_change_on_last_day'].shift(-1)
    data.iloc[0,-2] = 0
    data['sentiment'] = pd.Series(np.where(data['pr_change_on_current_day'] > 0, 1, 0), index=data.index)
    # data['perc_change'] = data['Percentage Change']
    # # Drop any rows with missing values
    # data.dropna(inplace=True)
    data.drop('pr_change_on_current_day',axis=1 , inplace=True)
    return data
def train_model(symbol_to_fetch,start_date,end_date,batch_size,sequence_length,stride =1):
    stock = fetch_ticker_data(symbol_to_fetch, start_date, end_date)

    # Calculate deltas, moving averages, and Bollinger Bands
    for i in range(1, 90,5):
        stock[f"open_delta_{i}day"] = stock["Open"].diff(periods=i)
        stock[f"high_delta_{i}day"] = stock["High"].diff(periods=i)
        stock[f"low_delta_{i}day"] = stock["Low"].diff(periods=i)
        stock[f"close_delta_{i}day"] = stock["Close"].diff(periods=i)
        stock[f"rolling_mean_open_{i}day"] = stock["Open"].rolling(window=i).mean()
        stock[f"rolling_mean_high_{i}day"] = stock["High"].rolling(window=i).mean()
        stock[f"rolling_mean_low_{i}day"] = stock["Low"].rolling(window=i).mean()
        stock[f"rolling_mean_close_{i}day"] = stock["Close"].rolling(window=i).mean()
        stock[f"rolling_std_open_{i}day"] = stock["Open"].rolling(window=i).std()
        stock[f"rolling_std_high_{i}day"] = stock["High"].rolling(window=i).std()
        stock[f"rolling_std_low_{i}day"] = stock["Low"].rolling(window=i).std()
        stock[f"rolling_std_close_{i}day"] = stock["Close"].rolling(window=i).std()

    stock['fast_ma'] = stock['Close'].rolling(window=20).mean()
    stock['slow_ma'] = stock['Close'].rolling(window=50).mean()
    stock['bollinger_high'] = stock['Close'].rolling(window=20).mean() + (2 * stock['Close'].rolling(window=20).std())
    stock['bollinger_low'] = stock['Close'].rolling(window=20).mean() - (2 * stock['Close'].rolling(window=20).std())
    stock['ema'] = stock['Close'].ewm(span=20, adjust=False).mean()
    stock['envelope_high'] = stock['Close'].rolling(window=20).mean() * (1 + 0.05)
    stock['envelope_low'] = stock['Close'].rolling(window=20).mean() * (1 - 0.05)
    stock['macd_line'] = stock['Close'].ewm(span=12, adjust=False).mean() - stock['Close'].ewm(span=26, adjust=False).mean()
    stock['macd_signal'] = stock['macd_line'].ewm(span=9, adjust=False).mean()

    # RSI calculation
    def calculate_rsi(data, rsi_period):
        delta = data['Close'].diff().dropna()
        gain = delta.where(delta > 0, 0).dropna()
        loss = -delta.where(delta < 0, 0).dropna()
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    stock['rsi'] = calculate_rsi(stock, 14)

    # # Stochastic Oscillator calculation
    # def calculate_stochastic(data, k_window, d_window):
    #     high_low = data[['High', 'Low']]
    #     c = data['Close']
    #     highest = high_low.rolling(window=k_window).max()
    #     lowest = high_low.rolling(window=k_window).min()
    #     print(((c - lowest) / (highest - lowest)) * 100)
    #     stochastic_k = ((c - lowest) / (highest - lowest)) * 100
    #     stochastic_d = stochastic_k.rolling(window=d_window).mean()
    #     return stochastic_k, stochastic_d
    # stock['stochastic_k'], stock['stochastic_d'] = calculate_stochastic(stock, 14, 3)

    # stock['stochastic_k']= calculate_stochastic(stock, 14, 3)[0]
    # stock['stochastic_d']= calculate_stochastic(stock, 14, 3)[1]
    stock['day'] = pd.to_datetime(stock.index).day
    stock['month'] = pd.to_datetime(stock.index).month
    stock['year'] = pd.to_datetime(stock.index).year
    stock['weekday'] = pd.to_datetime(stock.index).weekday
    stock['dayofyear'] = pd.to_datetime(stock.index).dayofyear
    stock = stock.fillna(method="ffill", axis=0)
    stock = stock.fillna(method="bfill", axis=0)
    stock.index = stock.index.date
    # Split the data into training and test sets

    # df = stock.copy()

    # # Calculate pairwise correlation
    # corr_matrix = df.corr()

    # # Identify highly correlated columns
    # redundant_cols = set()
    # for i in range(5,len(corr_matrix.columns)-1):
    #     for j in range(i+1, len(corr_matrix.columns)):
    #         if corr_matrix.iloc[i,j] > 0.8 and corr_matrix.columns[i] not in redundant_cols:
    #             redundant_cols.add(corr_matrix.columns[j])

    # # Remove one of the redundant columns
    # for col in redundant_cols:
    #     df = df.drop(col, axis=1)

    # # Print the updated DataFrame
    # print(df)

    # stock = df.copy()
    train_data_index = np.searchsorted(stock.index.values, np.datetime64(start_date))
    train_data = stock.iloc[:int(0.9*train_data_index)].copy()
    val_data  = stock.iloc[int(0.9*train_data_index)-sequence_length:train_data_index].copy()
    test_data = stock.iloc[train_data_index:].copy()
    train_data = label_data(train_data)
    val_data = label_data(val_data)
    test_data = label_data(test_data)
    train_data.fillna(0,axis = 0, inplace=True)
    val_data.fillna(0,axis = 0, inplace=True)
    test_data.fillna(0,axis = 0, inplace=True)

    #trian & test data
    X_train_data = train_data.iloc[:,:-1]
    y_train_data = train_data.iloc[:,-1]
    #trian & test data
    X_val_data = val_data.iloc[:,:-1]
    y_val_data = val_data.iloc[:,-1]
    X_test_data = test_data.iloc[:,:-1]
    y_test_data = test_data.iloc[:,-1]
    print(len(X_test_data), len(X_test_data.columns))
    from keras.utils import to_categorical

    # Convert targets to one-hot encoding
    y_train_onehot = to_categorical(y_train_data, num_classes=3)
    y_val_onehot = to_categorical(y_val_data, num_classes=3)

    y_train_data = to_categorical(y_train_data)
    y_test_data = to_categorical(y_test_data)
    y_val_data = to_categorical(y_val_data)

    # Normalize the data
    normalizer = MinMaxScaler()
    X_train_data_normalizer = normalizer.fit_transform(X_train_data)
    X_val_data_normalizer = normalizer.fit_transform(X_val_data)
    X_test_data_normalizer = normalizer.transform(X_test_data)

    # # # Reshape X_train_data_normalizer
    X_train_reshaped = X_train_data_normalizer.reshape(X_train_data_normalizer.shape[0], X_train_data_normalizer.shape[1], 1)
    X_val_reshaped = X_val_data_normalizer.reshape(X_val_data_normalizer.shape[0], X_val_data_normalizer.shape[1], 1)
    X_test_reshaped = X_test_data_normalizer.reshape(X_test_data_normalizer.shape[0], X_test_data_normalizer.shape[1], 1)



    train_dataset = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_train_data_normalizer,
        y_train_data,
        length = 3,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )
    val_dataset = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_val_data_normalizer,
        y_val_data,
        length = sequence_length,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )
    test_dataset = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_test_data_normalizer,
        y_test_data,
        length = sequence_length,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size=batch_size
    )
    try :
        # Load the best saved model
        best_model = tf.keras.models.load_model('best_model '+symbol_to_fetch+'gru_model'+'.keras')
    except :
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        # Define the GRU model
        gru_model = tf.keras.Sequential([
        tf.keras.layers.GRU(256, input_shape=(sequence_length, X_train_data_normalizer.shape[1]), return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(32, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(16, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(8, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(4, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation='sigmoid') # 3 neurons for the 3 classes
        ])
        # Compile the model
        gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'recall','f1_score','precision'],)

        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='max', restore_best_weights=True)

        # Define model checkpoint to save the best model during training
        model_checkpoint = ModelCheckpoint('models/best_model '+symbol_to_fetch+'gru_model'+'.keras', save_best_only=True, monitor='val_loss', mode='max')

        # Train the model
        history = gru_model.fit(train_dataset, epochs=100, batch_size=64, validation_data=val_dataset, callbacks=[early_stopping, model_checkpoint])

        # Load the best saved model
        best_model = tf.keras.models.load_model('models/best_model '+symbol_to_fetch+'gru_model'+'.keras')
        

    # Make predictions on the test set
    test_predictions = best_model.predict(test_dataset)
    test_predictions_binary = np.argmax(test_predictions, axis=1)
    y_test_data_binary = np.argmax(y_test_data, axis=1)

    from sklearn.metrics import accuracy_score,classification_report
    # Calculate accuracy
    accuracy = accuracy_score(y_test_data_binary[sequence_length:], test_predictions_binary)

    print('Accuracy:', accuracy)

    print(test_predictions )
    print(classification_report(y_test_data_binary[sequence_length:], test_predictions_binary))
    # test_data = test_data.iloc[sequence_length-1:,:].copy()
    test_data['ohlc_ta_sentiment'] = y_test_data_binary
    test_data.to_csv('data/dragaon_model_transformer_sentiment.csv')
    return test_data,'data/dragon_model_transformer_sentiment.csv'
