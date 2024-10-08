# Predictive Maintenance Project

This repository contains a Jupyter Notebook that demonstrates predictive maintenance using machine learning. The notebook includes data preprocessing, feature engineering, model training, and evaluation. The goal is to predict potential failures in machinery before they happen, thus enabling proactive maintenance and reducing downtime.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
---

## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/Predictive-Maintenance.git
cd Predictive-Maintenance
pip install -r requirements.txt
```

Make sure you have Python and pip installed. The required packages are listed in the `requirements.txt` file.

---

## Usage

Open the Jupyter Notebook `Predictive_maintainance.ipynb` to explore the data analysis, model building, and prediction steps.

You can run the notebook using the following command:

```bash
jupyter notebook Predictive_maintainance.ipynb
```

Make sure to follow the steps in the notebook sequentially for accurate results.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure your code follows the project's coding guidelines.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
## About Dataset

A company has a fleet of devices transmitting daily sensor readings. They would like to create a predictive maintenance solution to proactively identify when maintenance should be performed. This approach promises cost savings over routine or time based preventive maintenance, because tasks are performed only when warranted.

The task is to build a predictive model using machine learning to predict the probability of a device failure. When building this model, be sure to minimize false positives and false negatives. The column you are trying to Predict is called failure with binary value 0 for non-failure and 1 for failure

## Loading The Data


```python
import pandas as pd
# Load the dataset
data = pd.read_csv(r"https://raw.githubusercontent.com/Virendra2499/MISC/main/predictive_maintenance_dataset.csv")
# Replace with your actual column names


# Display the first few rows of the dataset
data.head()
```





 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>device</th>
      <th>failure</th>
      <th>metric1</th>
      <th>metric2</th>
      <th>metric3</th>
      <th>metric4</th>
      <th>metric5</th>
      <th>metric6</th>
      <th>metric7</th>
      <th>metric8</th>
      <th>metric9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2015</td>
      <td>S1F01085</td>
      <td>0</td>
      <td>215630672</td>
      <td>55</td>
      <td>0</td>
      <td>52</td>
      <td>6</td>
      <td>407438</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/1/2015</td>
      <td>S1F0166B</td>
      <td>0</td>
      <td>61370680</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>403174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/1/2015</td>
      <td>S1F01E6Y</td>
      <td>0</td>
      <td>173295968</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>237394</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/1/2015</td>
      <td>S1F01JE0</td>
      <td>0</td>
      <td>79694024</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>410186</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/1/2015</td>
      <td>S1F01R2B</td>
      <td>0</td>
      <td>135970480</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>313173</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
    

 ```python
data['date'].value_counts()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1/1/2015</th>
      <td>1163</td>
    </tr>
    <tr>
      <th>1/2/2015</th>
      <td>1163</td>
    </tr>
    <tr>
      <th>1/3/2015</th>
      <td>1163</td>
    </tr>
    <tr>
      <th>1/4/2015</th>
      <td>1162</td>
    </tr>
    <tr>
      <th>1/5/2015</th>
      <td>1161</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>10/27/2015</th>
      <td>31</td>
    </tr>
    <tr>
      <th>10/29/2015</th>
      <td>31</td>
    </tr>
    <tr>
      <th>10/30/2015</th>
      <td>31</td>
    </tr>
    <tr>
      <th>10/31/2015</th>
      <td>31</td>
    </tr>
    <tr>
      <th>11/2/2015</th>
      <td>31</td>
    </tr>
  </tbody>
</table>
<p>304 rows × 1 columns</p>
</div><br><label><b>dtype:</b> int64</label>



## Preprocessing and splitting the Dataset in testing and training sets


```python
# Drop the columns from the DataFrame
data = data.drop(columns=['date','device'])
```


```python
from sklearn.model_selection import train_test_split

# Assuming 'data' is the DataFrame
# Shuffle the DataFrame
df_shuffled = data.sample(frac=1, random_state=42)  # Shuffle the data

# Step 1: Perform the initial split (80% train, 20% test)
split_index = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split_index]
test_df = df_shuffled[split_index:]

# Step 2: Use train_test_split on the 80% train data to create train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

# Display the shapes to confirm
train_df.shape, val_df.shape, test_df.shape
```




    ((74696, 10), (24899, 10), (24899, 10))




```python
from sklearn.model_selection import train_test_split

# df_shuffled is the shuffled DataFrame from the previous step
# Step 1: Perform the initial split (80% train, 20% test)
split_index = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split_index]
test_df = df_shuffled[split_index:]

# Step 2: Use train_test_split on the 80% train data to create train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

# Display the shapes to confirm
train_df.shape, val_df.shape, test_df.shape

```




    ((74696, 10), (24899, 10), (24899, 10))




```python
# 'train_df', 'val_df', 'test_df' have the last column as the target variable
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np

# Splitting the features and labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_val = val_df.iloc[:, :-1].values
y_val = val_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values
```


```python
from sklearn.preprocessing import StandardScaler

# Import the StandardScaler class
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

## Building the Neural Network


```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_val = encoder.transform(y_val.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Step 3: Define the model architecture
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer + Hidden layer 1
    Dense(32, activation='relu'),                              # Hidden layer 2
    Dense(y_train.shape[1], activation='softmax')              # Output layer
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

    /usr/local/lib/python3.10/dist-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


## Training the Model


```python
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)
```

    Epoch 1/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 3ms/step - accuracy: 0.7522 - loss: 1.4087 - val_accuracy: 0.7928 - val_loss: 0.9327
    Epoch 2/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.7932 - loss: 0.9001 - val_accuracy: 0.7988 - val_loss: 0.8638
    Epoch 3/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8037 - loss: 0.8338 - val_accuracy: 0.8025 - val_loss: 0.8279
    Epoch 4/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 2ms/step - accuracy: 0.8063 - loss: 0.8036 - val_accuracy: 0.8022 - val_loss: 0.8129
    Epoch 5/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 3ms/step - accuracy: 0.8084 - loss: 0.7818 - val_accuracy: 0.8078 - val_loss: 0.7820
    Epoch 6/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 2ms/step - accuracy: 0.8090 - loss: 0.7678 - val_accuracy: 0.8081 - val_loss: 0.7639
    Epoch 7/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8096 - loss: 0.7543 - val_accuracy: 0.8075 - val_loss: 0.7586
    Epoch 8/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 2ms/step - accuracy: 0.8114 - loss: 0.7363 - val_accuracy: 0.8092 - val_loss: 0.7386
    Epoch 9/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 2ms/step - accuracy: 0.8124 - loss: 0.7273 - val_accuracy: 0.8101 - val_loss: 0.7320
    Epoch 10/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 2ms/step - accuracy: 0.8159 - loss: 0.7060 - val_accuracy: 0.8099 - val_loss: 0.7180
    Epoch 11/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8149 - loss: 0.7044 - val_accuracy: 0.8139 - val_loss: 0.7162
    Epoch 12/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8182 - loss: 0.6952 - val_accuracy: 0.8134 - val_loss: 0.7176
    Epoch 13/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8183 - loss: 0.6957 - val_accuracy: 0.8145 - val_loss: 0.7047
    Epoch 14/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 2ms/step - accuracy: 0.8224 - loss: 0.6773 - val_accuracy: 0.8163 - val_loss: 0.6876
    Epoch 15/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 2ms/step - accuracy: 0.8211 - loss: 0.6811 - val_accuracy: 0.8187 - val_loss: 0.6886
    Epoch 16/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8231 - loss: 0.6700 - val_accuracy: 0.8220 - val_loss: 0.6804
    Epoch 17/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8224 - loss: 0.6688 - val_accuracy: 0.8208 - val_loss: 0.6824
    Epoch 18/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8231 - loss: 0.6663 - val_accuracy: 0.8228 - val_loss: 0.6680
    Epoch 19/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 3ms/step - accuracy: 0.8232 - loss: 0.6604 - val_accuracy: 0.8249 - val_loss: 0.6665
    Epoch 20/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8256 - loss: 0.6554 - val_accuracy: 0.8268 - val_loss: 0.6550
    Epoch 21/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8243 - loss: 0.6576 - val_accuracy: 0.8262 - val_loss: 0.6798
    Epoch 22/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 3ms/step - accuracy: 0.8286 - loss: 0.6457 - val_accuracy: 0.8263 - val_loss: 0.6536
    Epoch 23/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8267 - loss: 0.6450 - val_accuracy: 0.8283 - val_loss: 0.6494
    Epoch 24/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8265 - loss: 0.6436 - val_accuracy: 0.8283 - val_loss: 0.6474
    Epoch 25/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 2ms/step - accuracy: 0.8303 - loss: 0.6362 - val_accuracy: 0.8284 - val_loss: 0.6436
    Epoch 26/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 3ms/step - accuracy: 0.8306 - loss: 0.6323 - val_accuracy: 0.8287 - val_loss: 0.6517
    Epoch 27/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 2ms/step - accuracy: 0.8306 - loss: 0.6314 - val_accuracy: 0.8285 - val_loss: 0.6476
    Epoch 28/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 3ms/step - accuracy: 0.8294 - loss: 0.6304 - val_accuracy: 0.8306 - val_loss: 0.6520
    Epoch 29/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.8310 - loss: 0.6282 - val_accuracy: 0.8322 - val_loss: 0.6393
    Epoch 30/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8331 - loss: 0.6219 - val_accuracy: 0.8263 - val_loss: 0.6445
    Epoch 31/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 2ms/step - accuracy: 0.8288 - loss: 0.6310 - val_accuracy: 0.8316 - val_loss: 0.6319
    Epoch 32/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8305 - loss: 0.6263 - val_accuracy: 0.8305 - val_loss: 0.6313
    Epoch 33/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 3ms/step - accuracy: 0.8347 - loss: 0.6142 - val_accuracy: 0.8317 - val_loss: 0.6318
    Epoch 34/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8323 - loss: 0.6190 - val_accuracy: 0.8291 - val_loss: 0.6473
    Epoch 35/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 3ms/step - accuracy: 0.8319 - loss: 0.6211 - val_accuracy: 0.8338 - val_loss: 0.6289
    Epoch 36/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8322 - loss: 0.6189 - val_accuracy: 0.8309 - val_loss: 0.6281
    Epoch 37/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8315 - loss: 0.6148 - val_accuracy: 0.8338 - val_loss: 0.6254
    Epoch 38/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8333 - loss: 0.6115 - val_accuracy: 0.8344 - val_loss: 0.6165
    Epoch 39/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.8359 - loss: 0.6051 - val_accuracy: 0.8334 - val_loss: 0.6273
    Epoch 40/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8317 - loss: 0.6188 - val_accuracy: 0.8334 - val_loss: 0.6326
    Epoch 41/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 3ms/step - accuracy: 0.8346 - loss: 0.6057 - val_accuracy: 0.8328 - val_loss: 0.6191
    Epoch 42/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 3ms/step - accuracy: 0.8329 - loss: 0.6105 - val_accuracy: 0.8320 - val_loss: 0.6291
    Epoch 43/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 2ms/step - accuracy: 0.8353 - loss: 0.6110 - val_accuracy: 0.8354 - val_loss: 0.6221
    Epoch 44/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8351 - loss: 0.5988 - val_accuracy: 0.8346 - val_loss: 0.6200
    Epoch 45/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8338 - loss: 0.6044 - val_accuracy: 0.8321 - val_loss: 0.7176
    Epoch 46/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.8351 - loss: 0.6052 - val_accuracy: 0.8341 - val_loss: 0.6359
    Epoch 47/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 3ms/step - accuracy: 0.8323 - loss: 0.6039 - val_accuracy: 0.8356 - val_loss: 0.6116
    Epoch 48/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.8321 - loss: 0.6061 - val_accuracy: 0.8349 - val_loss: 0.6168
    Epoch 49/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 3ms/step - accuracy: 0.8383 - loss: 0.5917 - val_accuracy: 0.8362 - val_loss: 0.6078
    Epoch 50/50
    [1m2335/2335[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 3ms/step - accuracy: 0.8358 - loss: 0.5977 - val_accuracy: 0.8364 - val_loss: 0.6113


## Evaluating The Model


```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')
```

    [1m779/779[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8311 - loss: 0.6159
    Test Accuracy: 0.8332


## Making Predictions


```python
# Make predictions
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=-1)

# Example: Print the first 10 predictions
print(predicted_classes[:10])
```

    [1m779/779[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step
    [0 0 0 0 0 0 0 0 0 0]



```python
y_prob = model.predict(X_test)
```

    [1m779/779[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step



```python
y_pred = y_prob.argmax(axis=1)
```


```python
# Import the necessary function
from sklearn.metrics import accuracy_score

# Convert y_test to multiclass
y_test_multiclass = y_test.argmax(axis=1)  # Assuming y_test is one-hot encoded
accuracy_score(y_test_multiclass, y_pred) # Now this line should work as well
```




    0.8331659906020322



## Plotting Some graphs for analysis


```python
import matplotlib.pyplot as plt # Import the matplotlib library and give it the alias 'plt'

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```




    




    
![png](Loss.png)
    



```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```




    




    
![png](Accuracy.png)
    
