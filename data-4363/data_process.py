import csv
import copy
from queue import PriorityQueue
import numpy as np
import pandas as pd
CPURealTimeW = [0.12593318, 0.168585066, 0.215241714, 0.449792373, 0.378377307, 0.072351546, 0.008672056]  # the computing time on the CPU in the warm water area
CPURealTimeC = [0.121515624, 0.169374289, 0.216643645, 0.30870271, 0.233688547, 0.065995284, 0.008512394]  # the computing time on the CPU in the cold water area
GPURealTimeW = [0.028054602, 0.035798463, 0.040783417, 0.0267817, 0.040369528, 0.008844684, 0.025547585]  # the computing time on the GPU in the warm water area
GPURealTimeC = [0.027557281, 0.035004458, 0.038275455, 0.020054302, 0.036708363, 0.008514694, 0.025139011]  # the computing time on the GPU in the cold water area


for i in range(20):
    with open("val_"+str(i)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag','during_time'])


        with open("val"+str(i)+".csv",'r') as f:
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


for i in range(20):
    customers = PriorityQueue()
    with open("val_" + str(i) + ".csv", 'r') as f:

        f_csv = csv.reader(f)
        j = 0
        for row in f_csv:
           # print(row[5])
            if j != 0:
                DNNType = int(row[3]) % 7
                QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
                QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
                QoS = 0
                while True:
                    QoS = round(np.random.normal((QoSMin * 1.2 + 1.5 * QoSMax) / 2.0,
                                                 (1.5 * QoSMax - QoSMin * 1.2) / 6.0), 6)  # QoS requirement
                    if QoSMin * 1.1 <= QoS <= 1.5 * QoSMax:
                        break
                row.append(QoS)

                prio1=float(row[5])
                prio2=float(row[6])
                customers.put(((prio1,prio2),row))
            j=j+1

    with open("val_"+str(i)+".csv",'w') as f1:
        f1_csv = csv.writer(f1)
        f1_csv.writerow(['','vmId','tenantId','vmTypeId','priority','time','flag','during_time','qos'])
        while (not customers.empty()):
            row = customers.get()[1]
            if (round(float(row[5])) <= 86400):
                f1_csv.writerow(row)
