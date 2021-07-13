import pandas as pd
import numpy as np
np.random.seed(10)


'''
read trace data from sqlite, the trace is downloaded from https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
'''
# import sqlite3
# with sqlite3.connect('packing_trace_zone_a_v1.sqlite') as con:
    # df = pd.read_sql_query("SELECT * FROM (SELECT * FROM vm where starttime >= 0 and starttime < 1 order by random() limit 2182) order by starttime", con=con)
    
    # i=0
    # num=0
    # timeup=0
    # tmp=0
    # maxnum = np.zeros(2182)
    # for indexs in df.index:
    #     if round(df.loc[indexs].values[-2]*86400)>i:
    #         tmp =round(df.loc[indexs].values[-2]*86400)
    #         timeup = 1
    #     num = num +1
    #     if timeup == 1:
    #         for j in range(indexs):
    #             if np.isnan(df.loc[j].values[-1]*86400):
    #                 continue
    #             if round(df.loc[j].values[-1]*86400) > i and round(df.loc[j].values[-1]*86400) <=tmp:
    #                 num = num -1
    #     maxnum[indexs] = num
    #     timeup = 0
    #     i =tmp
    # print(np.max(maxnum))
    
    # df.to_csv("test5.csv")

'''
training DRL
'''

# there are 20 DNN models in total
CPUrealtimeW = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the computing time on the CPU in the warm water area
CPUrealtimeC = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the computing time on the CPU in the cold water area
GPUrealtimeW = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the computing time on the GPU in the warm water area
GPUrealtimeC = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the computing time on the GPU in the cold water area
CPUenergyW = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the energy consumption on the CPU in the warm water area
CPUenergyC = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the energy consumption on the CPU in the cold water area
GPUenergyW = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the energy consumption on the GPU in the warm water area
GPUenergyC = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] # the energy consumption on the GPU in the cold water area

LayerComp = [[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],\
    [50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10],[50,20,10]] # percentage of CONV, FC, and RC layers in each DNN model

hardwareNumber = [0,0,0,0] # the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C


for i in range(4):
    df = pd.read_csv("./data/test"+str(i+1)+".csv")
    print(df.shape)
    print(df.dtypes)
    for indexs in df.index:
        userType = df.loc[indexs].values[1] % 20 # user number, there are 20 users in total
        DNNType = df.loc[indexs].values[2] % 20 # DNN model number, there are 20 DNN models in total
        inTime = df.loc[indexs].values[-2] # request starts
        outTime = df.loc[indexs].values[-1] # request ends
        pCONV = LayerComp[DNNType][0]
        pFC = LayerComp[DNNType][1]
        pRC = LayerComp[DNNType][2]
        QoSmin = min(CPUrealtimeW[DNNType],CPUrealtimeC[DNNType],GPUrealtimeW[DNNType],GPUrealtimeC[DNNType])
        QoSmax = max(CPUrealtimeW[DNNType],CPUrealtimeC[DNNType],GPUrealtimeW[DNNType],GPUrealtimeC[DNNType])
        while True:
            QoS = round(np.random.gauss(QoSmax,(QoSmax-QoSmin)/3.0)) # QoS requirement
            if (QoS >= QoSmin*1.1 and QoS <= 2*QoSmax):
                break
        
        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        # 训练过程中记录loss曲线
        

'''
testing DRL
'''