# ML-projects
ML projects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('Advertising.csv')
df.head()
df.columns
df.dtypes
df.info()
df.isnull().sum()
df['total_sales']=df['TV']+df['radio']+df['newspaper']
df.head()
df.drop(columns=['TV','radio','newspaper'],inplace=True)
df
df.describe()
sns.pairplot(df)
df.corr()
x=df.drop(columns='sales')
y=df['sales']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model.intercept_,model.coef_
train_predictions=model.predict(x_train)
test_predictions=model.predict(x_test)
plt.scatter(x_train,y_train,c='red')
train_predictions=model.predict(x_train)
plt.plot(x_train,train_predictions,c='black')
plt.show()
train_res=y_train-train_predictions
test_res=y_test-test_predictions
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_test,test_predictions)
MAE
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test,test_predictions)
MSE
RMSE=np.sqrt(MSE)
RMSE
model.score(x_train,y_train)
model.score(x_test,y_test)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x,y,cv=5)
scores
scores.mean()
import statsmodels.formula.api as smf
m=smf.ols('y~x',data=df).fit()
m.summary()
plt.scatter(y_test,test_res)
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.show()
sns.distplot(test_res,kde=True)
plt.show()
plt.scatter(test_predictions,test_res,c='black')
plt.axhline(y=0,c='blue')
plt.xlabel('predicted values')
plt.ylabel('ERRORS')
plt.show()
final_model=LinearRegression()
final_model.fit(x,y)
final_model.coef_
final_model.predict([[200]])
from joblib import dump
dump(final_model,'sales_model.joblib')
from joblib import load
loaded_model=load('sales_model.joblib')
loaded_model.intercept_
loaded_model.coef_
loaded_model.predict([[200]])
