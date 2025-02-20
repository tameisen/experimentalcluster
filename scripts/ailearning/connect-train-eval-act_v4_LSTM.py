import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import subprocess
import time
import os
from influxdb_client import InfluxDBClient

# InfluxDB connection details
bucket = "prometheus"
org = "university"
token = "T7MsfbYpOut-0wwrf6PIWV67wSh2vMlTrpdnWDe7o8vIiSoG_TIdQAenbKfOl-B8iQhOr5u3cA76x4rAi7c12g=="
url = "http://192.168.113.16:8086"

if token is None:
    raise ValueError("No token found in the environment variable INFLUX_TOKEN_FULL")

# Establish connection to InfluxDB
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

query_cpu_freq = '''
from(bucket: "prometheus")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "prometheus")
  |> filter(fn: (r) => r["_field"] == "node_cpu_scaling_frequency_hertz")
  |> filter(fn: (r) => r["url"] == "http://192.168.113.14:9100/metrics" or r["url"] == "http://192.168.113.15:9100/metrics" or r["url"] == "http://192.168.113.16:9100/metrics")
  |> filter(fn: (r) => r["cpu"] == "1" or r["cpu"] == "0" or r["cpu"] == "2" or r["cpu"] == "3")
|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

query_cpu_temp = '''
from(bucket: "prometheus")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_field"] == "node_thermal_zone_temp")
  |> filter(fn: (r) => r["url"] == "http://192.168.113.14:9100/metrics" or r["url"] == "http://192.168.113.15:9100/metrics" or r["url"] == "http://192.168.113.16:9100/metrics")
|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

query_cpu_freq_latest = '''
from(bucket: "prometheus")
  |> range(start: -20s)
  |> filter(fn: (r) => r["_measurement"] == "prometheus")
  |> filter(fn: (r) => r["_field"] == "node_cpu_scaling_frequency_hertz")
  |> filter(fn: (r) => r["url"] == "http://192.168.113.14:9100/metrics" or r["url"] == "http://192.168.113.15:9100/metrics" or r["url"] == "http://192.168.113.16:9100/metrics")
  |> filter(fn: (r) => r["cpu"] == "1" or r["cpu"] == "0" or r["cpu"] == "2" or r["cpu"] == "3")
|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

query_cpu_temp_latest = '''
from(bucket: "prometheus")
  |> range(start: -20s)
  |> filter(fn: (r) => r["_field"] == "node_thermal_zone_temp")
  |> filter(fn: (r) => r["url"] == "http://192.168.113.14:9100/metrics" or r["url"] == "http://192.168.113.15:9100/metrics" or r["url"] == "http://192.168.113.16:9100/metrics")
|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

def get_data_from_influxdb(query):
    try:
        result = query_api.query(org=org, query=query)
        data = []
        for table in result:
            for record in table.records:
                data.append(record.values)
        df = pd.DataFrame(data)  # Convert data to DataFrame
        #print("Data columns:", df.columns)
        return df
    except Exception as e:
        print(f"Error fetching data from InfluxDB: {e}")
        return None

def preprocess_data(data, data_temp,scaler=None):
    # Handle missing values
    #print("Current state of data:")
    #print(data.head())
    #print("preprocess_data: Before merge")
    #print(data.shape)
    
    # Drop unnecessary columns
    data = data.drop(columns=['_start', '_stop','result','zone','type','host','_measurement'], errors='ignore')
    data_temp = data_temp.drop(columns=['_start', '_stop','result','zone','type','host','_measurement'], errors='ignore')

    data['_time'] = data['_time'].apply(lambda x: x.timestamp())
    data_temp['_time'] = data_temp['_time'].apply(lambda x: x.timestamp())


    # Apply One-Hot-Encoding to categorical columns if they exist
    if 'url' in data.columns:
        data = pd.get_dummies(data, columns=['url'], drop_first=True)
    if 'cpu' in data.columns:
        data = pd.get_dummies(data, columns=['cpu'], drop_first=True)
        
            # Apply One-Hot-Encoding to categorical columns if they exist
    if 'url' in data_temp.columns:
        data_temp = pd.get_dummies(data_temp, columns=['url'], drop_first=True)
    if 'cpu' in data_temp.columns:
        data_temp = pd.get_dummies(data_temp, columns=['cpu'], drop_first=True)
    
    # Merge data and data_temp based on the timestamp
    data = pd.merge(data, data_temp, on='_time', how='inner').dropna()

    # Check if the necessary columns are in the DataFrame
    required_columns = ['node_cpu_scaling_frequency_hertz', 'node_thermal_zone_temp']
    if not all(column in data.columns for column in required_columns):
        raise KeyError(f"Required columns are missing: {required_columns}")

    # Select only numeric columns
   # numeric_columns = data.select_dtypes(include=[np.number]).columns
    #data = data[numeric_columns]
    
    # Select columns to use as input to the model
    input_columns = ['_time','node_cpu_scaling_frequency_hertz', 'node_thermal_zone_temp']
    data = data[input_columns]

    # Scale data using StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data[data.columns])  # Fit the scaler to the data
        data[data.columns] = scaler.transform(data[data.columns])

    data.to_csv("preprocessed_data.csv", index=False)
    return data


def preprocess_data(data, sequence_length=60):
    # Daten normalisieren
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    # Daten in Sequenzen aufteilen
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape für LSTM (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_model(data):
    sequence_length = 60

    # Daten vorverarbeiten
    X, y, scaler = preprocess_data(data, sequence_length)
    
    # Daten aufteilen (80% Training, 10% Validierung, 10% Test)
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Modell erstellen
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Modell kompilieren
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early Stopping und Model Checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Modell trainieren
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    return model, history, scaler, X_test, y_test

# Beispiel: Daten abrufen und in die Funktion übergeben
# Angenommen, `data` ist ein numpy-Array mit deinen Zeitreihendaten
data = np.random.rand(1000)  # Beispiel-Daten
model, history, scaler, X_test, y_test = train_model(data)

# Modell evaluieren
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')


def main():
    # Train the model with historical data
    #scaler = StandardScaler()
    #data_freq = get_data_from_influxdb(query_cpu_freq)
    #data_temp = get_data_from_influxdb(query_cpu_temp)



    model = train_model(data)

        while True:
            # Get latest data from InfluxDB
            latest_data_freq = get_data_from_influxdb(query_cpu_freq_latest)
            latest_data_temp = get_data_from_influxdb(query_cpu_temp_latest)

            if latest_data_freq is not None and latest_data_temp is not None:
                # Preprocess latest data
                latest_data = preprocess_data(latest_data_freq, latest_data_temp, scaler)
                latest_data_data_without_frequency = data.drop(columns=['node_cpu_scaling_frequency_hertz'])
                latest_data.to_csv("latest_preprocessed_data.csv", index=False)
                

                # Predict optimal CPU frequency
                #optimal_frequency = model.predict(latest_data_data_without_frequency)[0][0]
                optimal_frequency = model.predict(latest_data)[0][0]
                print ("Optimal Frequency:")
                readablefrequency = "{:,.0f}".format(optimal_frequency)
                print (readablefrequency)
                
                # Set CPU frequency
                # set_cpu_frequency(0, optimal_frequency)

                # Wait for 1 minute before the next iteration
                time.sleep(20)

if __name__ == "__main__":
    main()
