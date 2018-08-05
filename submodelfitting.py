import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets, neighbors, linear_model
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics,linear_model
import pickle

def read_data(filepath):
    f=open(filepath,'r')
    reader=csv.reader(f,delimiter=',')
    datagrid=[]
    next(reader, None)
    next(reader, None)
    for row in reader:
        datagrid.append(row)
    return datagrid

def split_stem_nonstem(datagrid):
    stem=[]
    nonstem=[]
    for row in datagrid:
        if(row[5]=='1'):
            stem.append(row)
        else:
            nonstem.append(row)
    return stem,nonstem

def get_complete_data(datagrid,indices):
    datagrid=np.array(datagrid)
    count=0
    flag = False
    all_data=[]
    without_data=[]
    for row in datagrid:
        for index in indices:
            if(len(row[index])==0):
                flag=True
                break

        if(flag!=True):

            all_data.append(row[indices])
            count+=1
        else:
            without_data.append(row[indices])
        flag = False

    return all_data,without_data

def normalize(arr,value):
    arr=np.array(arr,dtype=int)
    arr=arr/value
    return arr

def visualize(pairs):
    #pairs is an array of data containing x,y,color,label

    plt.scatter(pairs[0],pairs[1],marker='.',color=str(pairs[2][0]),label=str(pairs[3][0]))

    plt.legend(loc='upper left')
    plt.show()
def get_error_accuracy(y_true,y_pred):
    print('Error: ',mean_squared_error(y_true, y_pred))
    print('Accuracy: ',accuracy_score(y_true,y_pred))



def check_allnot_zero(row,indices):

    for i in indices:
        if(len(row[i])==0):
            return False
    return True

def check_any_nonzero(row,indices):
    for i in indices:
        if(len(row[i])!=0):
            return True
    return False

def get_mock_average(row,indices):

    count=0
    sum=0
    avg=-1
    if(check_any_nonzero(row,indices)):
        for i in indices:
            if(len(row[i])!=0):
                sum+=float(row[i])
                count+=1
        avg=sum/count

    return avg

def get_from_row(row,index):
    if(len(row[index])==0):
        return -1
    else:
        return row[index]

def check_validity(dataarr):
    for val in dataarr:
        if(val==-1):
            return False
    return True

def get_regressor():
    #regressor = SVR(kernel='rbf', C=1e3, gamma=0.000001)
    #regressor = DecisionTreeRegressor(max_depth=5)
    #regressor = SVR(kernel = 'linear', C = 1e3)
    #regressor = linear_model.Ridge(alpha=.1)
    regressor=linear_model.Lasso(alpha=0.1)
    #clf=SVR(kernel='linear', C=1e3)

    return regressor

def get_ind_var(datagrid):
    count=0
    x=[]
    y=[]
    for row in datagrid:
        gaokao_mockavg=get_mock_average(row,[28,39,50])

        zhongkao=get_from_row(row,54)
        gaokao=get_from_row(row,20)
        sum_3_subjects,x_subjects=get_subject_gaokao_scores(row)
        terms=get_mock_average(row,[72,76,125,137])
        female=get_from_row(row,105)
        county = get_from_row(row, 3)
        school = get_from_row(row, 4)
        state = get_from_row(row, 140)
        urban = get_from_row(row, 8)
        age=get_from_row(row,104)
        onechild=get_from_row(row,97)
        freshgraduate=get_from_row(row,11)
        ethnicity=get_from_row(row,7)
        zhongkaomock=get_from_row(row,94)
        year=get_from_row(row,0)
        #female, freshgraduate, state, urban, school
        dataarr = [gaokao_mockavg, gaokao]

        #dataarr = [zhongkao, gaokao_mockavg, sum_3_subjects,female, freshgraduate, state, age, urban, school, x_subjects]
        dataarr2=dataarr
        #dataarr2 = [zhongkao, gaokao_mockavg, female, freshgraduate,state, age, urban, school, gaokao]
        #dataarr2 = [zhongkao, gaokao_mockavg, female, state, school, gaokao]
        #dataarr2 = [zhongkao, terms,gaokao_mockavg, female,gaokao]
        if(check_validity(dataarr)):
            x.append(dataarr2[:-1])
            y.append(dataarr2[-1])

    return x,y

def get_max_at_indices(datagrid,indices):
    datagrid=np.array(datagrid)
    datagrid=datagrid[:,indices]
    maxvals=np.max(datagrid,axis=1)
    return maxvals

def get_subject_gaokao_scores(row):
    indices=[15,16,17,18]
    if(check_allnot_zero(row,indices)):
        sum_3_subject=float(row[15])+float(row[16])+float(row[17])
        x_subjects=float(row[18])
        return sum_3_subject,x_subjects
    return -1,-1

def read_binary_model(filename):
    f=open(filename,'rb')
    regressor=pickle.load(f)
    f.close()

def load_models():
    modelfiles=['stemregressor0','nonstemregressor0','stemregressor1','nonstemregressor1',
                'stemregressor2', 'nonstemregressor2','stemregressor3','nonstemregressor3']
    models=[]
    for file in modelfiles:
        models.append(read_binary_model(file))

    return models

def predict(datagrid):
    variables_models=[[4,8,11,54,104,105,140,20],[8,11,54,140,20],[8,54,140,20],[140,20]]

    models_index=[] #for all data rows, will contain the regressor to be used

    for row in datagrid:
        gaokao_mockavg = get_mock_average(row, [28, 39, 50])
        row = np.array(row)
        x=[str(gaokao_mockavg)]
        if(gaokao_mockavg!=-1):
            for i,indices in enumerate(variables_models):
                if(check_allnot_zero(row,indices)):
                    x=x+[val for val in row[indices]]
                    models_index.append(i)
                    break
        else:
            print('No gaokao mock scores available')
if __name__=='__main__':
    filepath='data_track_choice.csv'

    normalizedvalues=[2016,3,5,25,1,1,750,750,800,800,1,43,1,1]

    datagrid=read_data(filepath)



    datagrid=datagrid
    stem,nonstem=split_stem_nonstem(datagrid)



    x,y=get_ind_var(stem)
    x=np.array(x)
    y=np.array(y)

    #visualize([x[:100,1],y[:100],['red'],['Gaokao Mock']])
    #visualize([x[:100, 2], y[:100], ['red'], ['Terms']])

    print('Total len:',len(x))
    trainlen = int(len(x) * 0.8)

    xtrain=x[:trainlen]
    ytrain=y[:trainlen]

    stemxtest=np.array(x[trainlen:],dtype=float)
    stemytest=y[trainlen:]

    # stemregressor=get_regressor()
    # stemregressor.fit(xtrain,ytrain)
    # f = open('stemregressor1', 'wb')
    # pickle.dump(stemregressor, f)
    # f.close()
    f=open('stemregressor1','rb')
    stemregressor=pickle.load(f)
    f.close()
    y_pred=stemregressor.predict(stemxtest)

    stemytest=[float(x) for x in np.array(stemytest)]
    y_pred=[float(x) for x in np.array(y_pred)]


    print('Mean Squared Error:', metrics.mean_squared_error(stemytest,y_pred))
    print('R2 score:',metrics.r2_score(stemytest,y_pred))

    #non stem regressor
    x, y = get_ind_var(nonstem)
    x = np.array(x)
    y = np.array(y)

    # visualize([x[:100,1],y[:100],['red'],['Gaokao Mock']])
    # visualize([x[:100, 2], y[:100], ['red'], ['Terms']])

    print('Total len:', len(x))
    trainlen = int(len(x) * 0.8)

    xtrain = x[:trainlen]
    ytrain = y[:trainlen]

    nonstemxtest = np.array(x[trainlen:],dtype=float)

    nonstemytest = y[trainlen:]

    # nonstemregressor = get_regressor()
    # nonstemregressor.fit(xtrain, ytrain)
    # f=open('nonstemregressor1','wb')
    # pickle.dump(nonstemregressor,f)
    # f.close()
    f = open('nonstemregressor1', 'rb')
    nonstemregressor = pickle.load(f)
    f.close()
    y_pred = nonstemregressor.predict(nonstemxtest)


    nonstemytest = [float(x) for x in np.array(nonstemytest)]
    y_pred = [float(x) for x in np.array(y_pred)]

    print('Mean Squared Error:', metrics.mean_squared_error(nonstemytest, y_pred))
    print('R2 score:', metrics.r2_score(nonstemytest, y_pred))


    print('Stem students taking non stem:\n')
    y_pred = np.array(nonstemregressor.predict(stemxtest))
    print(sum(y_pred>stemytest))



    y_pred = np.array(stemregressor.predict(nonstemxtest))

    print('NonStem students taking stem:\n')
    print(sum(y_pred>nonstemytest))

    predict(datagrid[1000])
    # xindices=[105,20,3,4]
    # #xindices = [105,20,5, 3, 4, 8, 104,140]
    # yindices=[20]
    #
    #
    # rows,_=get_complete_data(datagrid,xindices+yindices)
    #
    # print(len(rows))
    # rows=np.array(rows)
    # trainlen = int(len(rows)*0.8)
    #
    # x=np.array(rows[:trainlen,:len(xindices)])
    # xtest=np.array(rows[trainlen:,:len(xindices)])
    #
    # y=np.array(rows[:trainlen,len(xindices):])
    # y = np.array([int(element) for element in y])
    #
    # ytest = np.array(rows[trainlen:, len(xindices):])
    # ytest = np.array([int(element) for element in ytest])
    #
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # #y_rbf = svr_rbf.fit(x, y).predict(xtest)
    # #y_rbf=[round(x) for x in y_rbf]
    #
    #
    # #
    # #
    # # clf=get_classifier()
    # # clf.fit(x,y)
    # # ret=clf.score(x,y)
    # # print(ret)
    #
    #
    # #y_rbf[y_rbf>=0.5]=1
    # #y_rbf[y_rbf < 0.5] = 0
    # print(svr_rbf.score(x,y))
    # #get_error_accuracy(ytest,y_rbf)
    # # for i in range(len(ret)):
    # #     print(ret[i],y[i] )
    #
    #
    #
    #
    # # plt.plot(x, y_rbf, color='navy', lw=lw, label='RBF model')
    # # plt.xlabel('data')
    # # plt.ylabel('target')
    # # plt.title('Support Vector Regression')
    # # plt.legend()
    # # plt.show()
