import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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

def train_model(data):
    # Split data into features (X) and target (y)
    X = data[['node_thermal_zone_temp', '_time','node_cpu_scaling_frequency_hertz']]
    y = data['node_cpu_scaling_frequency_hertz']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and compile the model
  #  model = Sequential([
  #      Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
   #     Dense(64, activation='relu'),
  #      Dense(1, activation='linear')
  #  ])
   # model.compile(optimizer='adam', loss='mse')
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='rmsprop', loss='mse')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test))

    return model

def main():
    # Train the model with historical data
    scaler = StandardScaler()
    data_freq = get_data_from_influxdb(query_cpu_freq)
    data_temp = get_data_from_influxdb(query_cpu_temp)


    if data_freq is not None and data_temp is not None:
        data = preprocess_data(data_freq, data_temp,scaler)
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
                set_cpu_frequency(0, 2400000000)
                
                # Wait for 1 minute before the next iteration
                time.sleep(20)

if __name__ == "__main__":
    main()
