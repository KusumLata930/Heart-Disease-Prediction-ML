import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data=pd.read_csv(r'C:\Users\91798\Downloads\heart_cleveland_upload.csv')

heart_data.head()

heart_data.tail()
heart_data.shape
heart_data.info()
heart_data.isnull().sum()
heart_data.describe()
heart_data['condition'].value_counts()
X = heart_data.drop(columns=['condition'])
Y=heart_data['condition']

print(X)

print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
model=LogisticRegression()
model.fit(X_train,Y_train)
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data:",training_data_accuracy)
#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data:",test_data_accuracy)

input_data=(61,1,0,134,234,0,0,145,0,2.6,1,2,0)
#change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the data as we are predicting for only one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)  

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The person does not have a heart disease")
else:
    print("The person has heart disease")

import pickle

pickle.dump(model, open("heart_model.pkl", "wb"))

print("Model Saved!")