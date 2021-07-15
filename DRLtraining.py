import pandas as pd
import numpy as np
from DQN import DQN
import torch
import copy
np.random.seed(10)
MEMORY_CAPACITY=100

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
s=[0,0,0,0,0,0,0,0,0,0,0,0]    #state
s[8]=hardwareNumber[0]
s[9]=hardwareNumber[1]
s[10]=hardwareNumber[2]
s[11]=hardwareNumber[3]
processing_request={}   #正在处理的requests      request_id->action
dqn=DQN()
episode=0
for i in range(4):
    df = pd.read_csv("./data/test_"+str(i+1)+".csv")
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    for indexs in df.index:
        episode+=1        #eposide 自增
        request_id=df.loc[indexs].values[0]  #请求的id
        userType = df.loc[indexs].values[1] % 20 # user number, there are 20 users in total
        DNNType = df.loc[indexs].values[2] % 20 # DNN model number, there are 20 DNN models in total
        inTime = df.loc[indexs].values[-2] # request starts
        flag = df.loc[indexs].values[-1] # request ends
        pCONV = LayerComp[DNNType][0]
        pFC = LayerComp[DNNType][1]
        pRC = LayerComp[DNNType][2]
        QoSmin = min(CPUrealtimeW[DNNType],CPUrealtimeC[DNNType],GPUrealtimeW[DNNType],GPUrealtimeC[DNNType])
        QoSmax = max(CPUrealtimeW[DNNType],CPUrealtimeC[DNNType],GPUrealtimeW[DNNType],GPUrealtimeC[DNNType])
        while True:
            QoS = round(np.random.normal(QoSmax,(QoSmax-QoSmin)/3.0)) # QoS requirement
            print(QoS,QoSmax,QoSmin)
            if (QoS >= QoSmin*1.1 and QoS <= 2*QoSmax):
                break

        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if flag==0:   # 某一个请求结束,则更改STATE，即hardwareNumber
            action=processing_request[request_id]

            s[action+8]+=1
            processing_request.remove()

        else:    #某一请求开始，更改state，然后run DQN
            s[0]=CPUrealtimeW[DNNType]
            s[1]=CPUrealtimeC[DNNType]
            s[2]=GPUrealtimeW[DNNType]
            s[4]=GPUrealtimeC[DNNType]
            s[5]=pCONV
            s[6]=pFC
            s[7]=pRC
            action = dqn.choose_action(s)
            s_=copy.deepcopy(s)
            s_[action+8]-=1   #获取新的state,即更改hardwareNumber
            reward=0
            '''
            获取reward的代码
            '''
            processing_request[request_id]=action   #添加到正在处理的请求中
            dqn.store_transition(s, action, reward, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            s=s_   #更新state









        # 训练过程中记录loss曲线


'''
testing DRL
'''