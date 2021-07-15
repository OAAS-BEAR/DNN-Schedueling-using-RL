import copy

import numpy as np
import pandas as pd

from DQN import DQN

np.random.seed(10)
MEMORY_CAPACITY = 100

'''
read trace data from sqlite, the trace is downloaded from 
https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
'''

'''
training DRL
'''

# there are 20 DNN models in total
CPURealTimeW = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                200]  # the computing time on the CPU in the warm water area
CPURealTimeC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                200]  # the computing time on the CPU in the cold water area
GPURealTimeW = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                200]  # the computing time on the GPU in the warm water area
GPURealTimeC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                200]  # the computing time on the GPU in the cold water area
CPUEnergyW = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200]  # the energy consumption on the CPU in the warm water area
CPUEnergyC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200]  # the energy consumption on the CPU in the cold water area
GPUEnergyW = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200]  # the energy consumption on the GPU in the warm water area
GPUEnergyC = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200]  # the energy consumption on the GPU in the cold water area

LayerComp = [[50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10],
             [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10],
             [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10], [50, 20, 10],
             [50, 20, 10]]  # percentage of CONV, FC, and RC layers in each DNN model

hardwareNumber = [0, 0, 0, 0]  # the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # state
s[8] = hardwareNumber[0]
s[9] = hardwareNumber[1]
s[10] = hardwareNumber[2]
s[11] = hardwareNumber[3]
processing_request = {}  # 正在处理的requests      request_id->action
dqn = DQN()
episode = 0
for i in range(4):
    df = pd.read_csv("./data/test_" + str(i + 1) + ".csv")
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    for indexes in df.index:
        episode += 1  # episode 自增
        request_id = df.loc[indexes].values[0]  # 请求的id
        userType = df.loc[indexes].values[1] % 20  # user number, there are 20 users in total
        DNNType = df.loc[indexes].values[2] % 20  # DNN model number, there are 20 DNN models in total
        inTime = df.loc[indexes].values[-2]  # request starts
        flag = df.loc[indexes].values[-1]  # request ends
        pCONV = LayerComp[DNNType][0]
        pFC = LayerComp[DNNType][1]
        pRC = LayerComp[DNNType][2]
        QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        while True:
            QoS = round(np.random.normal(QoSMax, (QoSMax - QoSMin) / 3.0))  # QoS requirement
            if QoSMin * 1.2 <= QoS <= 2 * QoSMax:
                break

        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if flag == 0:  # 某一个请求结束,则更改STATE，即hardwareNumber
            action = processing_request[request_id]

            s[action + 8] += 1
            processing_request.pop(request_id)

        else:  # 某一请求开始，更改state，然后run DQN
            s[0] = CPURealTimeW[DNNType]
            s[1] = CPURealTimeC[DNNType]
            s[2] = GPURealTimeW[DNNType]
            s[3] = GPURealTimeC[DNNType]
            s[4] = pCONV
            s[5] = pFC
            s[6] = pRC
            s[7] = QoS
            action = dqn.choose_action(s)
            s_ = copy.deepcopy(s)
            s_[action + 8] -= 1  # 获取新的state,即更改hardwareNumber
            reward = 0
            '''
            获取reward的代码
            '''
            processing_request[request_id] = action  # 添加到正在处理的请求中
            dqn.store_transition(s, action, reward, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            s = s_  # 更新state

        # 训练过程中记录loss曲线

'''
testing DRL
'''
