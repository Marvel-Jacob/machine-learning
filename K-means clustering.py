#using K means

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

data=pd.DataFrame({
        'Age':[22,21,27,25,25,22,24,28,28,24,25,26,22,25,21,24,22,28,23,21,26,22,22,24,26],
        'Balance':[10,20,10,10,35,30,12,5,25,20,55,40,14,20,12,39,23,10,50,5,15,5,10,10,5]})

kmeans=KMeans(n_clusters=2)
kmeans.fit(data)

labels=kmeans.predict(data)
centroids=kmeans.cluster_centers_

labels1=pd.DataFrame(labels)
data1=data.join(labels1)
data1=data1.rename(columns={0:'Label'})

#colmap={1:'r',2:'b'}
fig=plt.figure(figsize=(5,5))

plt.scatter(data['Age'],data['Balance'],c=labels,s=50,cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],marker="*",s=100)

##########using voting data
voting=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/voting.csv',
                   index_col=0)

voting=voting.fillna(12)

kmeans=KMeans(n_clusters=2)
kmeans.fit(voting)
group=kmeans.predict(voting)
centroids=kmeans.cluster_centers_

country_arr=list(voting.index)
res=np.array([country_arr,group]).T #transform
res1=list(res)
res2=pd.DataFrame(res)
f=res2.columns.values.tolist()
final=res2.sort_values(by=f[1]) #in place of f we can use res2


#on a picture

from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import cluster
import numpy as np

img=ndimage.imread('C:/Users/Administrator/Desktop/IMG_20171230_141537.jpg')
plt.imshow(img)
x,y,z=img.shape
img1=img.reshape(x*y,z) # have to this code cuz k means wont take more than 2 dimensions

clust=cluster.KMeans(n_clusters=3)
clust.fit(img1)
cent=clust.cluster_centers_
lab=clust.labels_

final=cent[lab].reshape(x,y,z)
plt.figure()
plt.imshow(cent[lab].astype(int).reshape(x,y,z))# cent[lab] creates a single array so we need to change its dimension

np.place(final,final==cent[1],0)
np.place(final,final==cent[2],0)
plt.figure()
plt.imshow(final)

final1=cent[lab].reshape(x,y,z)
np.place(final1,final1==cent[0],0)
np.place(final1,final1==cent[2],0)
plt.figure()
plt.imshow(final1)

final2=cent[lab].reshape(x,y,z)
np.place(final2,final2==cent[0],0)
np.place(final2,final2==cent[1],0)
plt.figure()
plt.imshow(final2)

count=0
im=final[:,:,0]
for i in (0,x):
    for j in (0,y):
        if im[i,j]!=0:
            count=count+1
cluster_perc=(count/(x+y))*100
 
#######  ELBOW CURVE  #############    
#when the number of clusters are more the error start reducing it will reduce till
#the time the k value reaches the value of number of datapoints. at some point it 
#will have a sharp point which we consider an elbow curve.


