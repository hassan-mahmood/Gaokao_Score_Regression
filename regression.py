import csv
import matplotlib.pyplot as plt

def calculate_average(arr):
    length=0
    sum=0.0
    for val in arr:
        if len(val)!=0:
            sum+=float(val)
            length+=1
    if(length!=0):
        return (sum/length)
    else:
        return -1

def CheckConditions(arr):
    length=len(arr)
    for val in arr:
        if(len(val)==0):
            return False
    return True

if __name__=='__main__':
    f=open('data_track_choice.csv','r')
    reader=csv.reader(f,delimiter=',')
    next(reader,None)
    plt.figure()
    totallength=0
    x=[]
    y=[]
    zhongkao=[]
    sponsorship=[]
    urban_rural=[]
    counties=[]
    one_child_arr=[]
    stem_non_stem=[]
    for row in reader:
        totallength+=1
        # if(totallength>500):
        #     break
        county=row[3]
        total_gaokao=row[20]
        mock1_gaokao=row[28]
        mock2_gaokao = row[39]
        mock3_gaokao=row[50]
        total_zhongkao=row[54]
        need_sponsorship=row[99]
        one_child=row[97]
        urban=row[8]
        stem=row[5]

        #val=calculate_average([mock3_gaokao,mock2_gaokao,mock1_gaokao])
        val=2
        if(val!=-1):
            if(CheckConditions([stem])):
                #counties.append(int(county))
                #sponsorship.append(int(need_sponsorship))
                #urban_rural.append(int(urban))
                x.append(val / 750)
                one_child_arr.append(one_child)
                stem_non_stem.append(stem)
                #y.append(float(total_gaokao) / 750)
                #zhongkao.append(float(total_zhongkao) / 750)

    yticks = [i for i in range(0, 750, 50)]
    xticks = [i for i in range(0, 750, 50)]
    plt.xlabel('Average Mock Test')
    plt.ylabel('Actual Gaokao Test')

    #plt.axis([0, 700, 0, 700])
    #plt.scatter(x,y,marker='.',color='red',label='Gaokao Scores')
    plt.scatter(x, one_child_arr, marker='.', color='black', label='Urban')
    #plt.scatter(x, counties, marker='.', color='black', label='County')
    #plt.scatter(x, sponsorship, marker='.', color='blue', label='Sponsorship needed')
    #plt.scatter(x, zhongkao, marker='.', color='brown', label='Zhongkao Scores')
    plt.legend(loc='upper left')
    #plt.savefig('sponsorship_visualization.png',dpi=300)
    plt.show()





    def get_ind_var(self,datagrid):
        count = 0
        x = []
        y = []
        for row in datagrid:
            gaokao_mockavg = get_mock_average(row, [28, 39, 50])
            zhongkao = get_from_row(row, 54)
            gaokao = get_from_row(row, 20)
            sum_3_subjects, x_subjects = get_subject_gaokao_scores(row)
            terms = get_mock_average(row, [72, 76, 125, 137])
            female = get_from_row(row, 105)
            county = get_from_row(row, 3)
            school = get_from_row(row, 4)
            state = get_from_row(row, 140)
            urban = get_from_row(row, 8)
            age = get_from_row(row, 104)
            onechild = get_from_row(row, 97)
            freshgraduate = get_from_row(row, 11)
            ethnicity = get_from_row(row, 7)
            zhongkaomock = get_from_row(row, 94)
            year = get_from_row(row, 0)
            # female, freshgraduate, state, urban, school
            dataarr = [gaokao_mockavg, gaokao]

            # dataarr = [zhongkao, gaokao_mockavg, sum_3_subjects,female, freshgraduate, state, age, urban, school, x_subjects]
            dataarr2 = dataarr
            # dataarr2 = [zhongkao, gaokao_mockavg, female, freshgraduate,state, age, urban, school, gaokao]
            # dataarr2 = [zhongkao, gaokao_mockavg, female, state, school, gaokao]
            # dataarr2 = [zhongkao, terms,gaokao_mockavg, female,gaokao]
            if (check_validity(dataarr)):
                x.append(dataarr2[:-1])
                y.append(dataarr2[-1])

        return x, y