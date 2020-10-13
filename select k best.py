import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

lreg=LogisticRegression()
le=LabelEncoder() 

names=['preg','plas','pres','skin','test','mass','pedi','age']
dataframe=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/breast_cancer.csv')
out=dataframe['diagnosis']
out=le.fit_transform(out)
dataframe.drop(['id','diagnosis'],axis=1,inplace=True)
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]

#feature extraction
test=SelectKBest(score_func=chi2, k=4)
fit=test.fit(dataframe,out)

#summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features=fit.transform(dataframe)

#summarize selected features
print(features[0:5,:])

xtrain,xval,ytrain,yval=train_test_split(features,out,test_size=0.2,random_state=60)
lreg.fit(xtrain,ytrain)
out_pred=lreg.predict(xval)
ac=accuracy_score(yval,out_pred)