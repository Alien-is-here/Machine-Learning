import pandas as pd
import numpy as np
from KNNeighbours import KNNeighbours

df = pd.read_csv('Social_Network_Ads.csv')
df.head()

X = df.iloc[:,2:4].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

Knn = KNNeighbours(k=3)
Knn.fit(X_train,y_train)

def predict_new():
    age = int(input('Age: '))
    salary = int(input('Salary: '))
    X_new = np.array([[age,salary]]).reshape(1,2)
    X_new = ss.transform(X_new)
    y_pred = Knn.predict(X_new)
    if y_pred == 0:
       print('Not Buy')
    else:
        print('Buy')

predict_new()