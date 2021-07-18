import copy
import math

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

M=[1,1,1,1]    # number of hardware of each server rack
hardwareNumber = [0, 0, 0, 0]  # the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
activated_server_racks=[]     #记录已经激活的server rack里硬件的信息
s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # state
s[8] = hardwareNumber[0]
s[9] = hardwareNumber[1]
s[10] = hardwareNumber[2]
s[11] = hardwareNumber[3]
processing_request = {}  # 正在处理的requests      request_id->（action,rack_id,hardware_id)
dqn = DQN()
episode = 0

idle_power = 0
gama = 1  # gama = CoolingEnergy / ITEnergy
alpha = 1
beta = 1

#新激活一个server rack
def activate_server_rack(activated_server_racks,M,state):
    new_server_rack=[] #新开辟的server_rack
    for i in range(4):
        hardwares=[]  #server_rack 每一类的硬件对应的都对应一个list
        for i in range(M[i]):
            hardwares.append((1,0.0))  #list里每一项对应一个硬件状态(idle是否为真，finish_time)
        new_server_rack.append(hardwares)   #设置单个rack里的硬件信息
        state[i+8]+=M[i]    #更新总的空闲硬件数量

    activated_server_racks.append(new_server_rack)


#执行action
def act(request_id,action,state,activated_server_racks,processing_request,M,finish_time):
    s_ = copy.deepcopy(state)
    flag=0
    for i in range(len(activated_server_racks)):
        for j in  range(activated_server_racks[i][action]):
            if activated_server_racks[i][action][j][0]==1:
                activated_server_racks[i][action][j][0]=0 #寻找一个机架来执行任务
                activated_server_racks[i][action][j][1]=finish_time
                s_[action+8]-=1    #更新总的空闲硬件数目硬件
                processing_request[request_id]=(action,i,j)   #添加任务信息到processing_request
                flag=1
                return s_

    if flag==0:   #没有空余硬件可用，需要新激活一个server rack
        activate_server_rack(activated_server_racks,M,s_)
        rack_id=len(activated_server_racks)-1
        activated_server_racks[rack_id][action][0][0] =0
        activated_server_racks[rack_id][action][0][1]=finish_time
        s_[action+8] -= 1
        processing_request[request_id] = (action, rack_id,0)
    return s_

#硬件完成task后，设置为idle状态
def release_hardware(request_id,activated_server_racks,state,processing_request):
    action=processing_request[request_id][0]
    rack_id=processing_request[request_id][1]
    hardware_id=processing_request[request_id][2]
    activated_server_racks[rack_id][action][hardware_id][0] == 1  #完成task后，硬件重新设置为idle
    activated_server_racks[rack_id][action][hardware_id][1]==0.0  #finish_time设置为0
    state[action+8] += 1  #更新总的空闲硬件数目硬件
    processing_request.pop(request_id)



def greedy_reward(computing_time, energy_consumption, finish_time, start_time, qos):
    request_num = math.ceil((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power * (qos - computing_time)) * request_num
    e_self_cooling = gama * e_self_it
    q = math.log(1 + math.exp(100 * (computing_time - qos) / qos))
    g_reward = -alpha * (e_self_it + e_self_cooling) - beta * q
    return g_reward


#reward function using DRL
def reward(activated_server_racks,processing_request,request_id,compute_time,energy_consumption, finish_time, start_time, qos):
    request_num = math.ceil((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power * (qos - compute_time)) * request_num
    e_self_cooling = gama * e_self_it
    rack_id=processing_request[request_id][1]
    this_hardware_id=processing_request[request_id][2]
    this_action=processing_request[request_id][0]
    max_finish=0

    #计算max finish time
    for action in range(4):
        for hardware_id in range(len(activated_server_racks[rack_id][action])):
            if activated_server_racks[rack_id][action][hardware_id][1]>max_finish and  not (this_action==action and this_hardware_id==hardware_id):
                max_finish=activated_server_racks[rack_id][action][hardware_id][1]
    if finish_time>max_finish:
        time=finish_time-max_finish
    else:
        time=0

    #compute e_other_it
    e_other_it=0
    for h in range(4):
        if abs(h-this_action)%2==0:
            if h==this_action:
                x=1
            else:
                x=0
            e_other_it+=(M[h]-x)*time

    e_other_cooling=gama * e_other_it
    q = math.log(1 + math.exp(100 * (compute_time - qos) / qos))
    reward = -alpha * (e_self_it + e_self_cooling+e_other_cooling+e_other_it) - beta * q
    return reward


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
        inTime = df.loc[indexes].values[-3]  # request starts
        flag = df.loc[indexes].values[-2]  # request begin or end
        if flag==1:
            finish_time =df.loc[indexes].values[-3]+df.loc[indexes].values[-1]
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
            release_hardware(request_id, activated_server_racks,s, processing_request)

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
            print(action)
            s_=act(request_id,action,s,activated_server_racks,processing_request,M,finish_time) # 获取新的state,即更改hardwareNumber
            # reward = 0
            '''
            获取reward的代码
            '''
            energy = [CPUEnergyW[DNNType], CPUEnergyC[DNNType], GPUEnergyW[DNNType], GPUEnergyC[DNNType]]
            #reward = greedy_reward(s[action], energy[action], flag, inTime, QoS, ph_idle, gama, alpha, beta)
            compute_time=s[action]
            energy_consumption=energy[action]
            reward=reward(activated_server_racks,processing_request,request_id,compute_time,energy_consumption, finish_time, inTime, QoS)
            dqn.store_transition(s, action, reward, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            s = s_  # 更新state

        # 训练过程中记录loss曲线

'''
testing DRL
'''
