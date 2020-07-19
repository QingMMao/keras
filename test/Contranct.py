import numpy as np  
from sklearn.datasets import load_iris  
import matplotlib.pyplot as plt  
  
#load datasets  
iris = load_iris()  
data = iris.data[:,:2]  
target = iris.target  
print (data.shape)#(150,2)  
print (data[:10])  
print (target[:10])  
  
label = np.array(target)  
index_0 = np.where(label==0)  
plt.scatter(data[index_0,0],data[index_0,1],marker='x',color = 'b',label = '0',s = 15)  
index_1 =np.where(label==1)  
plt.scatter(data[index_1,0],data[index_1,1],marker='o',color = 'r',label = '1',s = 15)  
index_2 =np.where(label==2)  
plt.scatter(data[index_2,0],data[index_2,1],marker='s',color = 'g',label = '2',s = 15)  
plt.xlabel('X1')  
plt.ylabel('X2')  
plt.legend(loc = 'upper left')  
plt.show()  

import os
os.getcwd()
import KNN

from sklearn.model_selection import train_test_split  
X,X_test,y,y_test = train_test_split(data,target,test_size=0.2,random_state=1)  
print (X.shape,X_test.shape)  
  
#cross validation  
folds = 4  
k_choices = [1,3,5,7,9,13,15,20,25]  
  
X_folds = []  
y_folds = []  
  
X_folds = np.vsplit(X,folds)  
y_folds = np.hsplit(y,folds)  
  
accuracy_of_k = {}  
for k in k_choices:  
    accuracy_of_k[k] = []  
#split the train sets and validation sets  
for i in range(folds):  
    classify = KNN.KNearestNeighbor()  
    X_train =np.vstack(X_folds[:i] + X_folds[i+1:])   
    X_val = X_folds[i]  
    y_train = np.hstack(y_folds[:i] + y_folds[i+1:])  
    y_val = y_folds[i]  
    print (X_train.shape,X_val.shape,y_train.shape,y_val.shape)  
    classify.train(X_train,y_train)  
    for k in k_choices:  
        y_val_pred = classify.predict(X_val,k = k)  
        accuracy = np.mean(y_val_pred == y_val)  
        accuracy_of_k[k].append(accuracy)  
  
for k in sorted(k_choices):  
    for accuracy in accuracy_of_k[k]:  
        print ('k = %d,accuracy = %f' %(k,accuracy) )
        
        
import matplotlib.pyplot as plt  
#show the accuracy   
for k in k_choices:  
    plt.scatter([k]*len(accuracy_of_k[k]), accuracy_of_k[k])  
accuracies_mean = np.array([np.mean(v) for k,v in sorted(accuracy_of_k.items())])  
accuracies_std = np.array([np.std(v) for k,v in sorted(accuracy_of_k.items())])  
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)  
plt.title('Cross-validation on k')  
plt.xlabel('k')  
plt.ylabel('Cross-validation accuracy')  
plt.show()  

best_k = 13  
classify = KNN.KNearestNeighbor()  
classify.train(X_train,y_train)  
y_test_pred = classify.predict(X_test,k=best_k)  
num_correct = np.sum(y_test==y_test_pred)  
accuracy_test = np.mean(y_test==y_test_pred)  
print ('test accuracy is %d/%d = %f' %(num_correct,X_test.shape[0],accuracy_test))  