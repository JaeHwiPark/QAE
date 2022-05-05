#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import pandas as pd
import numpy as np
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")


# In[31]:


filepath = 'D:\QAE\FDS\data'
fn_csv = os.path.join(filepath, 'creditcard.csv')
df = pd.read_csv(fn_csv)
df.head()


# In[39]:


df.columns


# In[5]:


df.info()


# In[42]:


df_stat = df.describe()
# df_stat.to_csv(f"{filepath}/statistics.csv", encoding='utf-8-sig')


# PCA 변환 없이 Raw 데이터 그대로 있는 변수:
# - Time
# - Amount
# - Class (1: fraud, 0: not_fraud)

# In[7]:


LABELS = ['Normal','Fraud']

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[43]:


# df.Class.value_counts()
df['Class'].value_counts()


# 사기와 정상 데이터 나누기

# In[44]:


df_fraud = df[df['Class']==1]
df_normal = df[df['Class']==0]

print(f"Shape of Fraudulant transactions: {df_fraud.shape}")
print(f"Shape of Non-Fraudulant transactions: {df_normal.shape}")


# 결제 금액 비교

# In[10]:


df_sumry_amount = pd.concat([df_fraud.Amount.describe(), df_normal.Amount.describe()], axis=1)
df_sumry_amount.columns=['Fraud', 'Normal']  ## 컬럼 이름 지정
df_sumry_amount


# 결제 시간 비교

# In[11]:


df_sumry_time = pd.concat([df_fraud.Time.describe(), df_normal.Time.describe()], axis=1)
df_sumry_time.columns=['Fraud', 'Normal']
df_sumry_time


# 시각화

# In[12]:


plt.figure(figsize=(10,8))

# plot the time feature
ax1 = plt.subplot(1, 2, 1)
plt.title('Time Distribution (Seconds)')
sns.distplot(df.Time, color='blue', bins=20, ax=ax1);

#plot the amount feature
ax2 = plt.subplot(1, 2, 2)
plt.title('Distribution of Amount')
sns.distplot(df.Amount, color='blue', bins=20, ax=ax2);


# 시간 분포 비교

# In[13]:


plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
df[df.Class == 1].Time.hist(bins=20, color='blue', alpha=0.6, label="Fraudulant Transaction")
plt.xlabel("Time")
plt.legend()

plt.subplot(1, 2, 2)
df[df.Class == 0].Time.hist(bins=20, color='blue', alpha=0.6, label="Non Fraudulant Transaction")
plt.xlabel("Time")
plt.legend()


# In[45]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
df[df.Class == 1].Amount.hist(bins=20, color='blue', alpha=0.6, label="Fraudulant Transaction")
plt.xlabel("Amount")
plt.legend()

plt.subplot(1, 2, 2)
df[df.Class == 0].Amount.hist(bins=20, color='blue', alpha=0.6, label="Non Fraudulant Transaction")
plt.xlabel("Amount")
plt.legend()


# 전체 데이터의 분포

# In[46]:


df.hist(figsize=(20, 20));


# 변수간 상관성 확인

# In[47]:


df.corr()


# In[48]:


# heatmap to find any high correlations

plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), cmap="seismic")
plt.show();


# In[17]:


from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df.Class

# (Train, Validation), Test 데이터 나누기 (Train : Test = 7 : 3)
X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, test_size=0.2, random_state=1)


# 각 데이터셋의 크기 확인
print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n{'-'*60}")
print(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}\n{'-'*60}")
print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")


# In[18]:


from sklearn.preprocessing import StandardScaler 

# 표준 졍규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)

# Class 구성비율
w_p = np.round(y_train.value_counts()[0] / len(y_train), 4)
w_n = np.round(y_train.value_counts()[1] / len(y_train), 4)

print(f"Fraudulant transaction weight: {w_n}")
print(f"Non-Fradulant transaction weights: {w_p}")


# In[19]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

def model_report(y_real, y_pred):
    clf_report = pd.DataFrame(classification_report(y_real, y_pred, output_dict=True)).iloc[:,1]
    clf_report.drop('support', inplace=True)
    clf_report['accuracy'] = accuracy_score(y_real, y_pred)
    
    return clf_report


# RandomForest 모델

# In[58]:


model_rf.feature_importances_


# In[50]:


print("-"*50)


# XGBoost 모델

# In[51]:


from xgboost import XGBClassifier

# Modeling and Training
model_xgb = XGBClassifier(random_state=123)
model_xgb.fit(X_train,  y_train, 
              eval_set=[(X_validate, y_validate)], 
              early_stopping_rounds = 50,
              verbose=True)

# Predict
y_pred = model_xgb.predict(X_test)
y_real = y_test

# Model report
report_xbg = model_report(y_real, y_pred)
report_xbg.name = "XGBoost (7:3)"
print("-"*20)
print(report_xbg)
print("-"*20)


# Neural Network 모델

# In[22]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# import tensorflow.keras.backend as K
# K.clear_session()

# Input and Output data size
n_inputs = X_train.shape[1]
n_output = 2

# Neural Network 모형
keras.backend.clear_session()
model_nn = keras.Sequential([
    keras.layers.Dense(64, activation='tanh', input_shape=(X_train.shape[-1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(n_output, activation='softmax'),                     
])

model_nn.compile(optimizer = keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_nn.summary()


# In[52]:


# 출력값을 2-columns 형태로 변형
y_train2=y_train.to_frame()
y_train2.rename(columns={"Class":"fraud"}, inplace=True)
y_train2['normal'] = 1-y_train2['fraud']
y_train2 = np.array(y_train2[['normal','fraud']].values.tolist())

y_validate2=y_validate.to_frame()
y_validate2.rename(columns={"Class":"fraud"}, inplace=True)
y_validate2['normal'] = 1-y_validate2['fraud']
y_validate2 = np.array(y_validate2[['normal','fraud']].values.tolist())


# Train
model_nn.fit(
   X_train, y_train2,
   batch_size=100,
   validation_data = (X_validate, y_validate2),
   epochs=10
)

# Predict
y_pred = model_nn.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_real = y_test 

# Model report
report_nn = model_report(y_real, y_pred)
report_nn.name='Neural Network (7:3)'

print("-"*20)
print(report_nn)
print("-"*20)


# Train, validation, test dataset

# In[53]:


from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df.Class

# Train, Validation, Test 데이터 나누기 (Train : Test = 1 : 9)
X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.9, random_state=1)
X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, test_size=0.2, random_state=1)

# 표준 졍규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)


# 각 데이터셋의 크기 확인
print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n{'-'*60}")
print(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}\n{'-'*60}")
print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")


# Encoder-Decoder

# In[54]:


# K.clear_session()

n_inputs = X_train.shape[1]
n_outputs = 2
n_latent = 64

# Input
inputs = keras.layers.Input(shape=(n_inputs, ))
x = keras.layers.Dense(128, activation='tanh')(inputs)
# x = keras.layers.Dense(64, activation='tanh')(x)
latent = keras.layers.Dense(n_latent, activation = 'tanh')(x)

# Encoder
encoder = keras.models.Model(inputs, latent, name='encoder')
encoder.summary()

latent_inputs = keras.layers.Input(shape=(n_latent, ))
x = keras.layers.Dense(128, activation='tanh')(latent_inputs)
outputs = keras.layers.Dense(n_inputs, activation='sigmoid')(x)

# Decoder
decoder = keras.models.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# Model Training
# 정상 결제건만 이용 Auto-Encoder를 훈련

# In[55]:


X_train_norm = X_train[y_train==0]
X_validate_norm = X_validate[y_validate==0]

# Auto-Encoder Model
model_ae = keras.models.Model(inputs, decoder(encoder(inputs)))
model_ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae')

# Train
model_ae.fit(
   X_train_norm, X_train_norm,
   validation_data = (X_validate_norm, X_validate_norm),
   batch_size=100,
   epochs=50
)


# 학습됩 Latent Vector로 정상/사기 분류 모델 생성

# In[56]:


# 출력값을 2-columns 형태로 변형
y_train2 = y_train.to_frame()
y_train2.rename(columns={"Class":"fraud"}, inplace=True)
y_train2['normal'] = 1-y_train2['fraud']
y_train2 = np.array(y_train2[['normal','fraud']].values.tolist())

y_validate2 = y_validate.to_frame()
y_validate2.rename(columns={"Class":"fraud"}, inplace=True)
y_validate2['normal'] = 1-y_validate2['fraud']
y_validate2 = np.array(y_validate2[['normal','fraud']].values.tolist())

# Build Model
model_hybrid = keras.Sequential([
    keras.layers.Dense(64, activation='tanh', input_dim=n_latent),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(n_outputs, activation = 'softmax')
])

model_hybrid.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model_hybrid.summary()

# Train
encoded_train = encoder.predict(X_train)
encoded_validate = encoder.predict(X_validate)

model_hybrid.fit(
   encoded_train, y_train2,
   validation_data = (encoded_validate, y_validate2),
   batch_size=100,
   epochs=10
)


# Predict
encoded_test = encoder.predict(X_test)
y_pred = model_hybrid.predict(encoded_test)
y_real = y_test 

# Model report
y_pred = y_pred.argmax(axis=1)
report_hybrid = model_report(y_real, y_pred)
report_hybrid.name='AutoEncoder + Neural Network (1:9)'

print("-"*20)
print(report_hybrid)
print("-"*20)


# 클래스 불균형이 심각한 데이터에서, Auto-Encoder를 이용하면 비교적 적은 학습데이터로도 F1-score가 80% 모델

# 모델 비교

# In[57]:


report_df = pd.concat([report_rf, report_xbg, report_nn, report_hybrid], axis=1)
report_df.plot(kind='barh', figsize=(15,8))


# In[28]:




