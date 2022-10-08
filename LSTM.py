#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


#import pandas_datareader as pdr
#key="3702aa55d1a230198256d4009540e5be47270562"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout, Embedding
from numpy import array
from sklearn.preprocessing import MinMaxScaler
plt.style.use('fivethirtyeight')


# # Create the different dataframes

# In[2]:


#df = pdr.get_data_tiingo('TSLA', api_key=key)
#df.to_csv('TSLA.csv')
df=pd.read_csv('TSLA.csv')
df


# In[3]:


df1 = df.drop(range(1158,1258))
data=df.reset_index()['close']
data1=df1.reset_index()['close']


# # Spliting the test and training data

# In[4]:


data2 = np.array(data1).reshape(-1,1)
training_size=int(len(data2)*0.7)
test_size=len(data2)-training_size
train_data, test_data = data2[0:training_size,:], data2[training_size:len(data1),:1]


# # Scale the data

# In[5]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.fit_transform(test_data)
scaled_data = scaler.fit_transform(data2)


# # Get the X and y train and test data

# In[6]:


pred_days = 175

X_train = []
y_train = []

for i in range(pred_days, len(train_data)):  
    X_train.append(scaled_train_data[i-pred_days:i, 0] )
    y_train.append(scaled_train_data[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []

for i in range(pred_days, len(test_data)):  
    X_test.append(scaled_test_data[i-pred_days:i, 0] )
    y_test.append(scaled_test_data[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)


# # Reshape data to fit the model

# In[7]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print(X_train.shape), print(X_test.shape)


# In[8]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(175,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['mean_absolute_percentage_error'])


# In[9]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100, batch_size=64,verbose=1)


# # Get predictions

# In[10]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# # Plot the predictions

# In[11]:


plt.figure(figsize=(20,10))

#plot predictions for train
days_predicted=175
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[days_predicted:len(train_predict)+days_predicted, :] = train_predict


#plot predictions for test
testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(days_predicted*2):len(scaled_data), :] = test_predict


#show the plot for the original plus the predictions
plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[12]:


len(scaled_test_data)


# In[13]:


fut_inp = scaled_test_data[173:]
fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)
fut_inp.shape


# # Loop to get the next 100 days

# In[14]:


tmp_inp = tmp_inp[0].tolist()

lst_output=[]
n_steps=175
i=0
while(i<100):
    
    if(len(tmp_inp)>175):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

#print(lst_output)


# In[15]:


plot_new=np.arange(1,101)
plot_pred=np.arange(101,201)


# In[16]:


len(data2)


# # Plot the next 100 days

# In[17]:


plt.plot(plot_new, scaler.inverse_transform(scaled_data[1058:]))
plt.plot(plot_pred, scaler.inverse_transform(lst_output))
plt.plot(plot_pred, (data[1158:]))


# In[18]:


plot_new_1=np.arange(1,1159)
plot_pred_1=np.arange(1159,1259)
plt.figure(figsize=(25,15))
plt.plot(plot_new_1, scaler.inverse_transform(scaled_data[0:]))
plt.plot(plot_pred_1, scaler.inverse_transform(lst_output))
plt.plot(data[1160:])


# In[19]:


#model.save('LSTM_V2.h5')


# In[ ]:




