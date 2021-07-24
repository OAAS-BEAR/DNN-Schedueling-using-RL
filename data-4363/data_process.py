import csv
import copy
from queue import PriorityQueue

for i in range(80):
    with open("test_"+str(i)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag','during_time'])


        with open("test"+str(i)+".csv",'r') as f:
           f_csv=csv.reader(f)
           j=0
           for row in f_csv:

                if j!=0 and row[6]!='':

                    if float(row[5])<float(row[6]):
                        row[5] = float(row[5])
                        row[6] = float(row[6])
                        list=copy.deepcopy(row)
                        list2 = copy.deepcopy(row)
                        list.pop(6)
                        list.append(1)
                        list.append(float(row[6]) - float(row[5]))
                        f1_csv.writerow(list)
                        list2.pop(5)
                        list2.append(0)
                        list2.append(float(row[6]) - float(row[5]))

                        f1_csv.writerow(list2)
                j=j+1


for i in range(80):
    customers = PriorityQueue()
    with open("test_" + str(i) + ".csv", 'r') as f:

        f_csv = csv.reader(f)
        j = 0
        for row in f_csv:
           # print(row[5])
            if j != 0:

                prio1=float(row[5])
                prio2=float(row[6])
                customers.put(((prio1,prio2),row))
            j=j+1

    with open("test_"+str(i)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag','during_time'])
        while (not customers.empty()):
            row = customers.get()[1]
            if (round(float(row[5])) <= 86400):
                f1_csv.writerow(row)
