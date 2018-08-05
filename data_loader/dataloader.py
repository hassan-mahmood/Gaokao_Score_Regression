import csv
import matplotlib.pyplot as plt
import numpy as np

def read_data(filepath):
    f=open(filepath,'r')
    reader=csv.reader(f,delimiter=',')
    datagrid=[]
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
    #pairs is an array of data containing x,y,label
    print(pairs[1],pairs[2])

    plt.scatter(pairs[0][0],pairs[0][1],marker='.',color=str(pairs[1]),label=str(pairs[2]))

    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    filepath='../data_track_choice.csv'
    datagrid=read_data(filepath)
    stem,nonstem=split_stem_nonstem(datagrid)
    print(len(stem))
    print(len(nonstem))
    #mockindices are mock1,mock2,mock3,gaokao,zhongkao,term1m,term1f,zhongkaomock,female,age,term2m,term2f
    #mockindices=[28,39,50,20,54,72,76,94,105,104,125,137,3,4,11,7,8,140,97]
    mockindices = [28, 39, 50, 20, 54, 72, 76, 94, 105, 104, 125, 137, 3, 4, 11, 8, 140]
    #county,school,freshgraduate,ethnicity,urban,anhui state,onechild
    demographics=[3,4,11,7,8,140,97]

    #all_data,without_data=get_complete_data(stem,[28,39,50,54])

    all_data, without_data = get_complete_data(stem, [5,97])
    all_data=np.array(all_data)

    cont=0
    for d in all_data:
        if(d[0]=='0'):
            cont+=1
    print(cont)
    #print(len(all_data[0]=='1'))
    #print(len(all_data[0] == '0'))

    #arr=np.array([all_data,'red','One Child'])
    #visualize(arr)


    # print('for stem:')
    # print(len(all_data))
    # print(len(without_data))
    # all_data,without_data=get_complete_data(without_data,[9])
    # print(len(all_data))
    # print(len(without_data))
    # all_data, without_data = get_complete_data(nonstem,indices)
    # # print('for non stem:')
    # # print(len(all_data))
    # # print(len(without_data))