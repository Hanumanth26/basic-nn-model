 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

In this model we will discuss with a neural network with 3 layers of neurons excluding input . First hidden layer with 5 neurons , Second hidden layer with 4 neurons and final Output layer with 1 neuron to predict the regression case scenario.

we use the relationship between input and output which is output = input * 2 + 1 and using epoch of about 50 to train and test the model and finnaly predicting the output for unseen test case.


# Neural Network Model



![Screenshot 2024-03-01 204933](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/86cb0d0b-a957-4be3-961c-0079b42e6349)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Name: hanumanth 
Reg.no: 212222240016
```

### Dependenices:
``` python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
### Data from sheets:
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```

### Data visulization:
```python
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
df.head()
X = df[['Input']].values
Y = df[['Output']].values
```

### Data split and preprocessing:
```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
### Regressive model:
```python

ai=Sequential([Dense(5,input_shape=[1]),
               Dense(4,activation='relu'),
Dense(1)])
ai.compile(optimizer="rmsprop",loss="mse")
ai.fit(X_train1,Y_train,epochs=50)
```
### Loss Calculation:
```python
loss_df=pd.DataFrame(ai.history.history)
loss_df.plot()
```
### Evaluate the model:
```python
ai.evaluate(X_test,Y_test)
```
### Prediction
```python
X_n1 = [[5]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```
## Dataset Information:

![Screenshot 2024-03-01 204047](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/749052ec-313b-4d34-8fec-57f63b43b5a8)


## OUTPUT

### Training Loss Vs Iteration Plot


![Screenshot 2024-03-01 204133](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/e56f2ee5-ab25-413b-9d47-bd70ee9ca137)

### Architecture and Training:

![Screenshot 2024-03-01 204215](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/4f0dda86-f5cd-490a-94c1-2cf8c1f61475)


### Test Data Root Mean Squared Error



![Screenshot 2024-03-01 204239](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/af69c60b-74a3-4098-b886-46f3d7fd2d7f)


### New Sample Data Prediction

![Screenshot 2024-03-01 204247](https://github.com/Hanumanth26/basic-nn-model/assets/121033192/eb750ca8-accf-40fb-831e-7e3e11b28df1)


## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
