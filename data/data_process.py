import csv
import copy
from queue import PriorityQueue

for i in range(4):
    with open("test_"+str(i+1)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag'])


        with open("test"+str(i+1)+".csv",'r') as f:
           f_csv=csv.reader(f)
           j=0
           for row in f_csv:

                if j!=0:
                    if int(row[5])<int(row[6]):
                        list=copy.deepcopy(row)
                        list2 = copy.deepcopy(row)
                        list.pop(6)
                        list.append(1)
                        f1_csv.writerow(list)
                        list2.pop(5)
                        list2.append(0)

                        f1_csv.writerow(list2)
                j=j+1


for i in range(4):
    customers = PriorityQueue()
    with open("test_" + str(i + 1) + ".csv", 'r') as f:

        f_csv = csv.reader(f)
        j = 0
        for row in f_csv:
           # print(row[5])
            if j != 0:
                prio1=int(row[5])
                prio2=int(row[6])
                customers.put(((prio1,prio2),row))
            j=j+1

    with open("test_"+str(i+1)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag'])
        while not customers.empty():
            f1_csv.writerow(customers.get()[1])





