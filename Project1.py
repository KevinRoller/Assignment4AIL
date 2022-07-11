from ast import excepthandler
import numpy as np
import pandas as pd
import re
import math
from  scipy.optimize import minimize as opt
import matplotlib.pyplot as plt
import seaborn as sns 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearnex import patch_sklearn 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

patch_sklearn()
patternTime='^(?:[0-9]+)[:;,.](?:[0-9]+[:;,.])?(?:[0-9]+)$'
def loadData(file:str):
    df=pd.read_csv(file,delimiter=',',encoding_errors='ignore')
    return df
def rightBeginEnd(row):
    if (re.search(patternTime,str(row['start time']))!=None) and (re.search(patternTime,str(row['end time']))!=None):
        return True
    else:
        return False
def getSecond(s:str):
    s=s.replace(' ','')
    tempList=re.split("[:;,.]",s)
    power=1
    result=0
    try:
        for i in reversed(tempList):
            result+=int(i)*power
            power*=60
    except:
        print(s)
    return result
def getDuration(row):
    return getSecond(row['end time'])-getSecond(row['start time'])
def makeDuration(df:pd.DataFrame):
    count=0
    i=0
    df=df[[(rightBeginEnd(df.iloc[i,:]))for i in range(df.shape[0])]]
    df['duration']=df.apply(getDuration,axis=1)
    return df
#def cleanSpace(value):

def oneHot(df:pd.DataFrame):
    for i in ["venue","container"]:
        temp=pd.get_dummies(df[i])
        df.drop(i,axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
    return df
def func(row):
    return str(row).replace(' ','').lower()
from sklearn.ensemble import RandomForestClassifier
def cleanData(df:pd.DataFrame):
    df=df[['start time','end time','number of in','venue','container','describe how to make it',"viewer feeling of youtuber's style "]]
    df['venue'][(df['venue']=='x')|(df['venue']=='home')]='other'
    df['venue'][df['venue']=='boat restaurant']='fine restaurant'
    df['container'][df['container']=='Bag']='bag'
    df['container'][(df['container']=='cup')|(df['container']=='plastic glass')]='glass'
    df['container'][(df['container']=='x')|(df['container']=='no')]='other'
    df['container'][df['container']=='hand']='hands-on'
    df['container'][df['container']=='clay bot']='pot'
    df['container'][df['container']=='tray ']='tray'
    df.drop(df[(df["viewer feeling of youtuber's style "]=='x')|(df["viewer feeling of youtuber's style "]=='0')].index, inplace=True)
    #print(df["viewer feeling of youtuber's style "].unique())
    df=df.dropna()
    # print("------------------------------------------")
    # print("Values list of venue: "+str(df['venue'].unique()))
    # print("------------------------------------------")
    # print("Values list of container: "+str(df['container'].unique()))
    # print("------------------------------------------")
    # print("Values list of viewer's feeling: "+str(df["viewer feeling of youtuber's style "].unique()))
    # print("------------------------------------------")
    df=makeDuration(df)
    # df['duration']=(df['duration']-df['duration'].mean())/(df['duration'].max()-df['duration'].min())
    # df['number of in']=(df['number of in']-df['number of in'].mean())/(df['number of in'].max()-df['number of in'].min())
    df['duration']=df['duration']
    df['venue']=df['venue'].apply(func)
    df['container']=df['container'].apply(func)
    temp=df["viewer feeling of youtuber's style "]
    df.drop("viewer feeling of youtuber's style ",axis=1,inplace=True)
    df=oneHot(df)
    #print(df.shape)
    df.insert(loc=19,column="viewer feeling",value=temp)
    #print(df.columns)
    df['describe how to make it']=pd.to_numeric(df['describe how to make it'],errors='coerce')
    df['viewer feeling']=pd.to_numeric(df['viewer feeling'],errors='coerce')
    
    return df

df=loadData(r"D:\học python\AIL\Annotation_AllVideos_FPT_Ver1.csv")
df=cleanData(df)
npData=(df.iloc[:,2:]).to_numpy()
print(npData.shape)
x_train, x_test, y_train, y_test = train_test_split(npData[:,:17],npData[:,17],test_size=0.2,stratify=npData[:,17],shuffle=True)


print(np.unique(y_train))
# sm = SMOTE()
# x_train, y_train=sm.fit_resample(x_train,y_train.ravel())
# x_test, y_test=sm.fit_resample(x_test,y_test.ravel())
# plt.figure(1)
# plt.hist(y_train)
# plt.figure(2)
# plt.hist(y_test)
# plt.show()
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)

rbf = svm.SVC(decision_function_shape='ovo',kernel='rbf',gamma=10,C=100).fit(x_train, y_train)
rbf_pred = rbf.predict(x_test)

linear = svm.SVC(decision_function_shape='ovo',kernel='linear',C=100).fit(x_train, y_train)
linear_pred = linear.predict(x_test)

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
sns.heatmap(confusion_matrix(y_test,rbf_pred), annot=True,xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('Confusion matrix of RBF kernel')
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.show()

linear_accuracy = accuracy_score(y_test, linear_pred)
linear_f1 = f1_score(y_test, linear_pred, average='weighted')
print('Accuracy (linear Kernel): ', "%.2f" % (linear_accuracy*100))
print('F1 (linear Kernel): ', "%.2f" % (linear_f1*100))
sns.heatmap(confusion_matrix(y_test,linear_pred), annot=True,xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('Confusion matrix of Linear kernel')
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.show()


param_grid = {'C': [0.1,1, 10, 100],
              'gamma': [100,10, 1],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(x_train, y_train)
grid_predictions = grid.predict(x_test)
print(classification_report(y_test, grid_predictions))
sns.heatmap(confusion_matrix(y_test,grid_predictions), annot=True,xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('Confusion matrix of RBF kernel with grid search')
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.show()
