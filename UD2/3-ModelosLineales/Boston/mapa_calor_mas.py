import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

from sklearn.linear_model import LinearRegresion
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2,:2]])
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
target = raw_df.values[1::2, 2]

df = np.hstack([data, target.reshape(-1, 1)])

df = pd.DataFrame(df, columns=column_names)

lstat=np.array(df['STAT'])
lstat=np.transpose(lstat)

medv=np.array(df['MEDV'])
medv=np.transpose(medv)

X_train, y_train, X_test, y_test = train_test_split()

lr=LinearRegresion()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
y_pred2=lr.predict([[0]])


print('W1',lr.coef_)
print('W',lr.intecept_)

plt.scater(X_test,y_test,color='black')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')

plt.plot(X_test,y_pred,color='blue',linewidth=2)

coef_training=round(tr.score(X_train,y_train),2)

coef_testing=round(tr.score(X_test,y_test),2)




