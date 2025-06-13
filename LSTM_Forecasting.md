```python

# Import necessary libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

```


```python
# Upload the dataset (Google Colab specific)

from google.colab import files
uploaded = files.upload()
```



     <input type="file" id="files-e9937d91-18ce-414f-9e6e-0579f09d4502" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-e9937d91-18ce-414f-9e6e-0579f09d4502">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving AirPassengers.csv to AirPassengers.csv



```python
# Load the dataset

df = pd.read_csv("AirPassengers.csv", delimiter=',')
df.head()
```





  <div id="df-3c2329d3-452a-4968-a96f-f07b604f00ad" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>#Passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05</td>
      <td>121</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3c2329d3-452a-4968-a96f-f07b604f00ad')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3c2329d3-452a-4968-a96f-f07b604f00ad button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3c2329d3-452a-4968-a96f-f07b604f00ad');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-a073e09c-cd1b-4107-aa1b-1eca8b0313ba">
      <button class="colab-df-quickchart" onclick="quickchart('df-a073e09c-cd1b-4107-aa1b-1eca8b0313ba')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-a073e09c-cd1b-4107-aa1b-1eca8b0313ba button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
print(df.columns)
```

    Index(['Month', '#Passengers'], dtype='object')


 ## Exploratory Data Analysis (EDA)**


```python
# Basic info
print(df.info())
print(df.describe())

# Convert 'Month' to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# 3. Check for missing values
print(df.isnull().sum())

# Plot the time series
plt.figure(figsize=(12,6))
plt.plot(df.index, df['#Passengers'], label='Passengers')
plt.title('Monthly Number of Airline Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# Check seasonality & trend using rolling mean
df['Rolling_Mean_12'] = df['#Passengers'].rolling(window=12).mean()
df['Rolling_Std_12'] = df['#Passengers'].rolling(window=12).std()

plt.figure(figsize=(12,6))
plt.plot(df['#Passengers'], label='Passengers')
plt.plot(df['Rolling_Mean_12'], label='12-month Rolling Mean', color='orange')
plt.plot(df['Rolling_Std_12'], label='12-month Rolling Std', color='green')
plt.title('Passengers with Rolling Mean & Std Deviation')
plt.legend()
plt.show()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['#Passengers'], model='multiplicative')
result.plot()
plt.show()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 144 entries, 0 to 143
    Data columns (total 2 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Month        144 non-null    object
     1   #Passengers  144 non-null    int64 
    dtypes: int64(1), object(1)
    memory usage: 2.4+ KB
    None
           #Passengers
    count   144.000000
    mean    280.298611
    std     119.966317
    min     104.000000
    25%     180.000000
    50%     265.500000
    75%     360.500000
    max     622.000000
    #Passengers    0
    dtype: int64



    
![png](Airline_Passenger_LSTM_Forecasting%20%283%29_files/Airline_Passenger_LSTM_Forecasting%20%283%29_5_1.png)
    



    
![png](Airline_Passenger_LSTM_Forecasting%20%283%29_files/Airline_Passenger_LSTM_Forecasting%20%283%29_5_2.png)
    



    
![png](Airline_Passenger_LSTM_Forecasting%20%283%29_files/Airline_Passenger_LSTM_Forecasting%20%283%29_5_3.png)
    



```python
print(df.columns)

```

    Index(['#Passengers', 'Rolling_Mean_12', 'Rolling_Std_12'], dtype='object')


## Preprocess data for LSTM



```python

# Extract passengers as a numpy array and scale
data = df['#Passengers'].values.reshape(-1, 1)  # Use '#Passengers' column (Month is index)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences and labels for supervised learning
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 12  # Use last 12 months to predict the next month
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split train and test sets (80% train)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], SEQ_LENGTH, 1))
X_test = X_test.reshape((X_test.shape[0], SEQ_LENGTH, 1))
```

## Build and train LSTM model


```python
model = Sequential([
    Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

from tensorflow.keras.callbacks import EarlyStopping

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define EarlyStopping callback to stop training if val_loss doesn't improve for 10 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,             # increased epochs, EarlyStopping will stop training early if needed
    batch_size=16,
    callbacks=[early_stop]
)

```

    Epoch 1/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 138ms/step - loss: 0.1055 - val_loss: 0.1883
    Epoch 2/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 28ms/step - loss: 0.0333 - val_loss: 0.0229
    Epoch 3/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0188 - val_loss: 0.0368
    Epoch 4/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0139 - val_loss: 0.0430
    Epoch 5/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0122 - val_loss: 0.0206
    Epoch 6/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.0103 - val_loss: 0.0208
    Epoch 7/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0074 - val_loss: 0.0213
    Epoch 8/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0083 - val_loss: 0.0210
    Epoch 9/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0064 - val_loss: 0.0207
    Epoch 10/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0081 - val_loss: 0.0219
    Epoch 11/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0085 - val_loss: 0.0208
    Epoch 12/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.0093 - val_loss: 0.0202
    Epoch 13/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0071 - val_loss: 0.0205
    Epoch 14/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0094 - val_loss: 0.0202
    Epoch 15/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0079 - val_loss: 0.0225
    Epoch 16/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0068 - val_loss: 0.0199
    Epoch 17/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.0098 - val_loss: 0.0196
    Epoch 18/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0091 - val_loss: 0.0196
    Epoch 19/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0067 - val_loss: 0.0195
    Epoch 20/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0086 - val_loss: 0.0189
    Epoch 21/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.0080 - val_loss: 0.0215
    Epoch 22/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.0061 - val_loss: 0.0210
    Epoch 23/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.0055 - val_loss: 0.0239
    Epoch 24/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 0.0088 - val_loss: 0.0246
    Epoch 25/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 52ms/step - loss: 0.0060 - val_loss: 0.0184
    Epoch 26/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 47ms/step - loss: 0.0079 - val_loss: 0.0180
    Epoch 27/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 30ms/step - loss: 0.0108 - val_loss: 0.0183
    Epoch 28/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0058 - val_loss: 0.0175
    Epoch 29/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0064 - val_loss: 0.0197
    Epoch 30/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0074 - val_loss: 0.0196
    Epoch 31/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0079 - val_loss: 0.0167
    Epoch 32/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.0074 - val_loss: 0.0164
    Epoch 33/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0045 - val_loss: 0.0186
    Epoch 34/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0062 - val_loss: 0.0156
    Epoch 35/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0046 - val_loss: 0.0175
    Epoch 36/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 0.0050 - val_loss: 0.0151
    Epoch 37/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0037 - val_loss: 0.0148
    Epoch 38/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0044 - val_loss: 0.0162
    Epoch 39/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.0053 - val_loss: 0.0154
    Epoch 40/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0055 - val_loss: 0.0141
    Epoch 41/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0051 - val_loss: 0.0137
    Epoch 42/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0064 - val_loss: 0.0134
    Epoch 43/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0053 - val_loss: 0.0141
    Epoch 44/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0049 - val_loss: 0.0142
    Epoch 45/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0050 - val_loss: 0.0121
    Epoch 46/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0071 - val_loss: 0.0121
    Epoch 47/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0051 - val_loss: 0.0121
    Epoch 48/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0054 - val_loss: 0.0121
    Epoch 49/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.0062 - val_loss: 0.0107
    Epoch 50/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0044 - val_loss: 0.0169
    Epoch 51/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0060 - val_loss: 0.0090
    Epoch 52/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0040 - val_loss: 0.0099
    Epoch 53/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0041 - val_loss: 0.0086
    Epoch 54/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0046 - val_loss: 0.0121
    Epoch 55/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0045 - val_loss: 0.0080
    Epoch 56/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0050 - val_loss: 0.0096
    Epoch 57/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0031 - val_loss: 0.0068
    Epoch 58/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.0052 - val_loss: 0.0078
    Epoch 59/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.0034 - val_loss: 0.0064
    Epoch 60/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0055 - val_loss: 0.0066
    Epoch 61/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0038 - val_loss: 0.0051
    Epoch 62/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.0044 - val_loss: 0.0060
    Epoch 63/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0033 - val_loss: 0.0053
    Epoch 64/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.0025 - val_loss: 0.0104
    Epoch 65/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 48ms/step - loss: 0.0047 - val_loss: 0.0069
    Epoch 66/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 51ms/step - loss: 0.0043 - val_loss: 0.0053
    Epoch 67/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.0034 - val_loss: 0.0081
    Epoch 68/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.0035 - val_loss: 0.0044
    Epoch 69/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0046 - val_loss: 0.0067
    Epoch 70/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0028 - val_loss: 0.0065
    Epoch 71/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.0053 - val_loss: 0.0055
    Epoch 72/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0031 - val_loss: 0.0074
    Epoch 73/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0023 - val_loss: 0.0049
    Epoch 74/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0026 - val_loss: 0.0045
    Epoch 75/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.0032 - val_loss: 0.0043
    Epoch 76/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0030 - val_loss: 0.0042
    Epoch 77/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0035 - val_loss: 0.0035
    Epoch 78/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0028 - val_loss: 0.0041
    Epoch 79/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.0029 - val_loss: 0.0025
    Epoch 80/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0024 - val_loss: 0.0034
    Epoch 81/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0026 - val_loss: 0.0028
    Epoch 82/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0024 - val_loss: 0.0110
    Epoch 83/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.0037 - val_loss: 0.0064
    Epoch 84/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.0040 - val_loss: 0.0068
    Epoch 85/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.0022 - val_loss: 0.0024
    Epoch 86/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0026 - val_loss: 0.0020
    Epoch 87/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0016 - val_loss: 0.0045
    Epoch 88/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0028 - val_loss: 0.0027
    Epoch 89/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0030 - val_loss: 0.0061
    Epoch 90/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0032 - val_loss: 0.0020
    Epoch 91/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0020 - val_loss: 0.0019
    Epoch 92/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0032 - val_loss: 0.0021
    Epoch 93/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0038 - val_loss: 0.0162
    Epoch 94/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.0036 - val_loss: 0.0026
    Epoch 95/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0020 - val_loss: 0.0035
    Epoch 96/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.0026 - val_loss: 0.0034
    Epoch 97/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0027 - val_loss: 0.0022
    Epoch 98/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.0029 - val_loss: 0.0024
    Epoch 99/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.0018 - val_loss: 0.0022
    Epoch 100/100
    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.0016 - val_loss: 0.0031


## Evaluate on the Test Set


```python
# Predict on the test set
y_pred = model.predict(X_test)

# Inverse transform predictions and true values back to original scale
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R^2: {r2:.3f}")

# Plot true vs predicted values
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.title('True vs Predicted Passengers')
plt.xlabel('Time step')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 481ms/step
    MSE: 520.522
    RMSE: 22.815
    MAE: 19.409
    R^2: 0.918



    
![png](Airline_Passenger_LSTM_Forecasting%20%283%29_files/Airline_Passenger_LSTM_Forecasting%20%283%29_12_1.png)
    

