#Recursive feature elimination


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#load the data
le=LabelEncoder()
data=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/breast_cancer.csv')
X=data.drop(['id','diagnosis'],axis=1)
Y=le.fit_transform(data['diagnosis'])

#feature extraction
model=LogisticRegression()
acc1=[]
for i in range(1,10):
    rfe=RFE(model,i) #so we take i value as 3
    fit=rfe.fit(X,Y)
    print(rfe.n_features_)
    print(rfe.support_)
    print(fit.ranking_)
    features=fit.transform(X)

#with X the acc is 98% and with features its 97.3
    xtrain,xval,ytrain,yval=train_test_split(features,Y,test_size=0.2,random_state=60)
    model.fit(xtrain,ytrain)
    out_pred=model.predict(xval)

    acc=accuracy_score(yval,out_pred)
    acc1.append(acc)







import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


