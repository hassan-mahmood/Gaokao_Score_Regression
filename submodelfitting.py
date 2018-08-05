import csv
import numpy as np
from sklearn import metrics,linear_model
import pickle


class Model:
    def __init__(self,csvfilepath):
        self.csvfilepath=csvfilepath
        self.data_loaded=False


    def check_validity(dataarr):
        for val in dataarr:
            if (val == -1):
                return False
        return True

    def get_mock_average(self,row, indices):

        count = 0
        sum = 0
        avg = -1
        if (self.check_any_nonzero(row, indices)):
            for i in indices:
                if (len(row[i]) != 0):
                    sum += float(row[i])
                    count += 1
            avg = sum / count

        return avg

    def check_allnot_zero(self,row, indices):

        for i in indices:
            if (len(row[i]) == 0):
                return False
        return True

    def check_any_nonzero(self,row, indices):
        for i in indices:
            if (len(row[i]) != 0):
                return True
        return False

    def read_csv(self,filepath):
        f = open(filepath, 'r')
        reader = csv.reader(f, delimiter=',')
        datagrid = []
        next(reader, None)
        for row in reader:
            datagrid.append(row)
        return datagrid

    def split_stem_nonstem(self,datagrid):
        stem = []
        nonstem = []
        for row in datagrid:
            if (row[5] == '1'):
                stem.append(row)
            else:
                nonstem.append(row)
        return stem, nonstem

    def get_organized_data(self,datagrid):
        count = 0
        x = []
        y = []
        variables_models = [[4, 8, 11, 54, 104, 105, 140, 20], [8, 11, 54, 140, 20], [8, 54, 140, 20], [140, 20]]
        organized_data=[[[],[]],[[],[]],[[],[]],[[],[]]]#it will contain x and y data values separated per regressor type

        for row in datagrid:
            row=np.array(row)
            gaokao_mockavg = self.get_mock_average(row, [28, 39, 50])
            if(gaokao_mockavg!=-1):
                x = [str(gaokao_mockavg)]
                for i, indices in enumerate(variables_models):
                    if (self.check_allnot_zero(row, indices)):
                        values = x + [val for val in row[indices]]
                        values=np.array(values,dtype=float)
                        organized_data[i][0].append(values[:-1])
                        organized_data[i][1].append(values[-1])

                        for idx in range(i+1,4):
                            values = x + [val for val in row[variables_models[idx]]]
                            values = np.array(values, dtype=float)
                            organized_data[idx][0].append(values[:-1])
                            organized_data[idx][1].append(values[-1])

                        break

        return organized_data

    def load_data(self):
        datagrid = self.read_csv(self.csvfilepath)
        self.xlsxdata=datagrid
        # datatypes are stem and nonstem
        tracksdata = [[], []]

        for idx, data in enumerate(self.split_stem_nonstem(datagrid)):
            tracksdata[idx] = np.array(self.get_organized_data(data))


        self.datagrid=tracksdata
        self.data_loaded=True
        print('Data Loaded')

    def get_regressor(self):
        #regressor = SVR(kernel='rbf', C=1e3, gamma=0.000001)
        #regressor = DecisionTreeRegressor(max_depth=5)
        #regressor = linear_model.Ridge(alpha=.1)
        regressor=linear_model.Lasso(alpha=0.1)
        return regressor

    def train_test_split(self,data):

        x,y=data
        trainlen=int(len(x)*0.8)
        trainx=x[:trainlen]
        trainy=y[:trainlen]
        testx=x[trainlen:]
        testy=y[trainlen:]
        return trainx,trainy,testx,testy

    def fit_and_store(self,data,name):                              #name='' when you do not want to serialize the regressor
        trainx,trainy,testx,testy=self.train_test_split(data)
        trainx=np.array(trainx)
        trainy = np.array(trainy)
        testx = np.array(testx)
        testy = np.array(testy)

        regressor=self.get_regressor()
        regressor.fit(trainx,trainy)
        self.evaluate_model(regressor,testx,testy)
        if(len(name)!=0):
            f=open(name,'wb')
            pickle.dump(regressor,f)
            f.close()


        #store the model
    def convert_arr_to_float(self,arr):
        arr=[float(x) for x in np.array(arr)]
        return arr

    def evaluate_model(self,regressor,test_x,test_y):

        y_pred=regressor.predict(test_x)

        test_y = self.convert_arr_to_float(test_y)
        y_pred = self.convert_arr_to_float(y_pred)

        print('Mean Squared Error:', metrics.mean_squared_error(test_y,y_pred))
        print('R2 score:',metrics.r2_score(test_y,y_pred))

    def train(self):
        if(not self.data_loaded):
            self.load_data()
        tracksdata=np.array(self.datagrid)

        flag=False
        tracktypes = ['stem', 'nonstem']

        for idx,track in enumerate(tracksdata):
            #track will have stem and non stem [4 elements]
            for i,datagroup in enumerate(track):
                print(str(tracktypes[idx])+' '+str(i))
                self.fit_and_store(datagroup,str(tracktypes[idx])+str(i))


    def predict(self,datagrid,models_dict):

        variables_models=[[4,8,11,54,104,105,140,20],[8,11,54,140,20],[8,54,140,20],[140,20]]


        tempdatagrid=[]
        for row in datagrid:
            name='stem'
            if(row[5]=='1'):   #for stem students, we want to predict non stem scores
                name='nonstem'

            gaokao_mockavg = self.get_mock_average(row, [28, 39, 50])
            row = np.array(row)
            x=[str(gaokao_mockavg)]
            if(gaokao_mockavg!=-1):
                for i,indices in enumerate(variables_models):
                    if(self.check_allnot_zero(row,indices)):
                        x=x+[val for val in row[indices]]
                        x=[float(elem) for elem in x]
                        x=x[:-1]
                        predicted_val=float(models_dict[name+str(i)].predict([x]))
                        row=np.append(row, str(predicted_val))

                        tempdatagrid.append(row)
                        #models_dict[name+str(i)]
                        break
            else:
                row = np.append(row, '')
                tempdatagrid.append(row)
                #print('No gaokao mock scores available')
        tempdatagrid=np.array(tempdatagrid)
        #print('tempdatagrid shape after', tempdatagrid.shape)
        return tempdatagrid

    def load_models(self):
        tracks=['stem','nonstem']
        subindex=['0','1','2','3']
        models=dict()
        for track in tracks:
            for idx in subindex:
                name=track+idx
                f=open(name,'rb')
                models[name]=pickle.load(f)
                f.close()
        return models

    def write_predictions(self,):

        models_dict=self.load_models()
        predictions=self.predict(self.xlsxdata,models_dict)
        f=open('predictions.csv','w')
        writer=csv.writer(f,delimiter=',')
        for row in predictions:
            writer.writerow(row)
        print('Predictions written to predictions.csv')

        #write to csv





if __name__=='__main__':
    filepath='data_track_choice.csv'

    normalizedvalues=[2016,3,5,25,1,1,750,750,800,800,1,43,1,1]

    model=Model(filepath)
    model.load_data()
    #organized_data=model.train()
    model.write_predictions()

