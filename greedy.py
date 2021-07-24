import copy
import math
import sys

import numpy as np
import pandas as pd

from DQN import DQN
from DQN import doubleDQN

# np.random.seed(10)
MEMORY_CAPACITY = 100

'''
read trace data from sqlite, the trace is downloaded from 
https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
'''

'''
training DRL
'''

# there are 12 DNN models in total
# ResNetV1-50, ResNetV1-101, ResNetV1-152, EfficientNet-B1, EfficientNet-B3, EfficientNet-B5,
# EfficientNet-B7, Unet, YoloV3-416, YoloV3-spp, YoloV3-tiny, and NER

CPURealTimeW = [0.099118, 0.194185, 0.279248, 0.192070, 0.229745, 0.457346, 1.002424, 0.233919, 0.250791, 0.256342,
                0.045661, 0.005784]  # the computing time on the CPU in the warm water area
CPURealTimeC = [0.097433, 0.192931, 0.276896, 0.188478, 0.227395, 0.451082, 0.898043, 0.176990, 0.214053, 0.226121,
                0.043751, 0.005650]  # the computing time on the CPU in the cold water area
GPURealTimeW = [0.037001, 0.057690, 0.086847, 0.073057, 0.088970, 0.124025, 0.201542, 0.027709, 0.049020, 0.048426,
                0.011849, 0.014269]  # the computing time on the GPU in the warm water area
GPURealTimeC = [0.033925, 0.051543, 0.071645, 0.064489, 0.082196, 0.108451, 0.182225, 0.022258, 0.041965, 0.046916,
                0.009727, 0.014012]  # the computing time on the GPU in the cold water area
CPUEnergyW = [6.161884, 12.071303, 18.150614, 7.967937, 11.601355, 33.376023, 79.719643, 19.230621, 19.678410,
              20.852059, 4.029731, 0.311649]  # the energy consumption on the CPU in the warm water area
CPUEnergyC = [7.111787, 12.240676, 18.289973, 8.049952, 11.861565, 33.488322, 81.943463, 25.966349, 24.844868,
              24.600319, 4.516407, 0.309255]  # the energy consumption on the CPU in the cold water area
GPUEnergyW = [2.665347, 4.700274, 6.795247, 5.298865, 5.725534, 12.917292, 19.418534, 3.252527, 4.808302, 5.288533,
              1.154375, 0.847006]  # the energy consumption on the GPU in the warm water area
GPUEnergyC = [2.462627, 4.252362, 6.279354, 5.540303, 8.499656, 14.834170, 30.017568, 4.468464, 6.518914, 6.571887,
              1.286960, 0.909147]  # the energy consumption on the GPU in the cold water area

# percentage of CONV, POOL, FC, Batch, and RC layers in each DNN model
LayerComp = [[0.505, 0.019, 0.010, 0.467, 0.000], [0.502, 0.010, 0.005, 0.483, 0.000],
             [0.502, 0.006, 0.003, 0.489, 0.000], [0.550, 0.115, 0.005, 0.330, 0.000],
             [0.551, 0.114, 0.004, 0.331, 0.000], [0.553, 0.114, 0.003, 0.330, 0.000],
             [0.554, 0.114, 0.002, 0.331, 0.000], [0.857, 0.143, 0.000, 0.000, 0.000],
             [0.510, 0.000, 0.000, 0.490, 0.000], [0.500, 0.020, 0.000, 0.480, 0.000],
             [0.433, 0.200, 0.000, 0.367, 0.000], [0.000, 0.000, 0.500, 0.000, 0.500]]

M = [10, 10, 20, 20]  # number of hardware in each server rack, CPU-W,CPU-C,GPU-W,GPU-C

if sys.argv[1] == 'd':
    dqn = DQN()
else:
    dqn = doubleDQN()
episode = 10

idle_power = [24, 24, 10, 10]
gama = [0.01, 0.26, 0.01, 0.26]  # gama = CoolingEnergy / ITEnergy
alpha = 0.000001


# 新激活一个server rack
def activate_server_rack(activated_server_racks, M, state, action, activated_server_racks_flags):
    for i in range(len(activated_server_racks_flags[str(action)])):
        if activated_server_racks_flags[str(action)][i] == 0:
            activated_server_racks_flags[str(action)][i] = 1
            activated_server_racks_flags[str((action+2) % 4)][i] = 1
            return i
    new_server_rack = [[],[],[],[]]  # 新开辟的server_rack
    for x in range(4):
        for ii in range(M[x]):
            new_server_rack[x].append([1, 0.0])  # list里每一项对应一个硬件状态(idle是否为真，finish_time)
    if action == 0 or action == 2:
        activated_server_racks['0'].append(new_server_rack[0])  # 设置单个rack里的硬件信息,list里每一项对应一个硬件状态(0表示idle,finish_time)
        activated_server_racks['2'].append(new_server_rack[2])  # 设置单个rack里的硬件信息
        activated_server_racks_flags['0'].append(1)
        activated_server_racks_flags['2'].append(1)
        state[10] += M[0]  # 更新总的空闲硬件数量
        state[12] += M[2]  # 更新总的空闲硬件数量
    else:
        activated_server_racks['1'].append(new_server_rack[1])
        activated_server_racks['3'].append(new_server_rack[3])
        activated_server_racks_flags['1'].append(1)
        activated_server_racks_flags['3'].append(1)
        state[11] += M[1]
        state[13] += M[3]
    return len(activated_server_racks[str(action)]) - 1


# 执行action
def act(request_id, action, state, activated_server_racks, processing_request, M, finish_time, activated_server_racks_flags):
    s_new = copy.deepcopy(state)
    for x in range(len(activated_server_racks[str(action)])):  # 机架
        if activated_server_racks_flags[str(action)][x] == 1:
            for j in range(len(activated_server_racks[str(action)][x])):  # 硬件
                if activated_server_racks[str(action)][x][j][0] == 1:
                    activated_server_racks[str(action)][x][j][0] = 0  # 寻找一个机架来执行任务
                    activated_server_racks[str(action)][x][j][1] = finish_time
                    s_new[action + 10] -= 1  # 更新总的空闲硬件数目硬件
                    processing_request[request_id] = [action, x, j, finish_time] # 添加任务信息到processing_request
                    return s_new
    # 没有空余硬件可用，需要新激活一个server rack
    rack_id = activate_server_rack(activated_server_racks, M, s_new, action, activated_server_racks_flags)
    activated_server_racks[str(action)][rack_id][0][0] = 0
    activated_server_racks[str(action)][rack_id][0][1] = finish_time
    s_new[action + 10] -= 1
    processing_request[request_id] = [action, rack_id, 0, finish_time]
    return s_new


# 硬件完成task后，设置为idle状态
def release_hardware(request_id, activated_server_racks, state, processing_request, activated_server_racks_flags):
    action = processing_request[request_id][0]
    rack_id = processing_request[request_id][1]
    hardware_id = processing_request[request_id][2]
    activated_server_racks[str(action)][rack_id][hardware_id][0] = 1  # 完成task后，硬件重新设置为idle
    activated_server_racks[str(action)][rack_id][hardware_id][1] = 0.0  # finish_time设置为0
    state[action + 10] += 1  # 更新总的空闲硬件数目硬件

    # deactivate racks
    restNum = 0
    action1 = action
    action2 = (action+2) % 4
    for ii in range(len(activated_server_racks[str(action1)][rack_id])):
        restNum += activated_server_racks[str(action1)][rack_id][ii][0]
    for ii in range(len(activated_server_racks[str(action2)][rack_id])):
        restNum += activated_server_racks[str(action2)][rack_id][ii][0]
    if restNum == (M[action1]+M[action2]):
        activated_server_racks_flags[str(action1)][rack_id] = 0
        activated_server_racks_flags[str(action2)][rack_id] = 0
        state[action1 + 10] -= M[action1]
        state[action2 + 10] -= M[action2]
    
    E1 = processing_request[request_id][4]
    E2 = processing_request[request_id][5]
    Q = processing_request[request_id]
    processing_request.pop(request_id)
    return E1, E2, Q

def greedy_reward(computing_time, action,energy_consumption, finish_time, start_time, qos):
    '''
    模拟采取每种action下产生的后果
    '''
    activated_server_rackstmp = copy.deepcopy(activated_server_racks)
    processing_requesttmp = copy.deepcopy(processing_request)
    activated_server_racks_flagstmp = copy.deepcopy(activated_server_racks_flags)
    
    exit_flag = False
    for x in range(len(activated_server_rackstmp[str(action)])): # 机架
        if activated_server_racks_flags[str(action)][x] == 1:
            for j in range(len(activated_server_rackstmp[str(action)][x])): # 硬件
                if activated_server_rackstmp[str(action)][x][j][0] == 1:
                    activated_server_rackstmp[str(action)][x][j][0] = 0  # 寻找一个机架来执行任务
                    activated_server_rackstmp[str(action)][x][j][1] = finish_time
                    processing_requesttmp[request_id] = [action, x, j, finish_time]  # 添加任务信息到processing_request
                    exit_flag = True
                    break
        if exit_flag:
            break
    # 没有空余硬件可用，需要新激活一个server rack
    if exit_flag is False:
        new_server_rack = [[],[],[],[]]  # 新开辟的server_rack
        for x in range(4):
            for ii in range(M[x]):
                new_server_rack[x].append([1, 0.0])  # list里每一项对应一个硬件状态(idle是否为真，finish_time)
        if action == 0 or action == 2:
            activated_server_rackstmp['0'].append(new_server_rack[0])  # 设置单个rack里的硬件信息,list里每一项对应一个硬件状态(0表示idle,finish_time)
            activated_server_rackstmp['2'].append(new_server_rack[2])  # 设置单个rack里的硬件信息
            activated_server_racks_flagstmp['0'].append(1)
            activated_server_racks_flagstmp['2'].append(1)
        else:
            activated_server_rackstmp['1'].append(new_server_rack[1])
            activated_server_rackstmp['3'].append(new_server_rack[3])
            activated_server_racks_flagstmp['1'].append(1)
            activated_server_racks_flagstmp['3'].append(1)
        rack_id = len(activated_server_rackstmp[str(action)]) - 1
        activated_server_rackstmp[str(action)][rack_id][0][0] = 0
        activated_server_rackstmp[str(action)][rack_id][0][1] = finish_time
        processing_requesttmp[request_id] = [action, rack_id, 0, finish_time]

    '''
    计算4种能耗值
    '''
    c_action = action
    if computing_time>qos:
        return -float('inf')
    else:
        request_num = math.floor((finish_time - start_time) / qos)
        e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (finish_time - start_time - qos * request_num)*idle_power[c_action]
        e_self_cooling = gama[c_action] * e_self_it
    
    rack_id = processing_requesttmp[request_id][1]
    this_hardware_id = processing_requesttmp[request_id][2]
    this_action = processing_requesttmp[request_id][0]
    max_finish = 0

    # 计算max finish time
    for action in [this_action, (this_action+2) % 4]:
        for hardware_id in range(len(activated_server_rackstmp[str(action)][rack_id])):
            if activated_server_rackstmp[str(action)][rack_id][hardware_id][1] > max_finish and not (
                    this_action == action and this_hardware_id == hardware_id):
                max_finish = activated_server_rackstmp[str(action)][rack_id][hardware_id][1]
    if finish_time > max_finish:
        time = finish_time - max_finish
    else:
        time = 0

    # compute e_other_it
    e_other_it = 0
    for h in [this_action, (this_action+2) % 4]:
        if h == this_action:
            x = 1
        else:
            x = 0
        e_other_it += (M[h] - x) * time * idle_power[h]

    e_other_cooling = gama[c_action] * e_other_it
    
    g_reward = -alpha * (e_self_it + e_self_cooling + e_other_cooling + e_other_it)
    return g_reward

def request_add_item(request_id, computing_time, energy_consumption, finish_time, start_time, qos):
    '''
    确定采取某动作后，计算自身的IT和cooling能耗，更新processing_request
    '''
    c_action = processing_request[request_id][0]
    assert(computing_time<=qos)
    request_num = math.floor((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (finish_time - start_time - qos * request_num)*idle_power[c_action]
    e_self_cooling = gama[c_action] * e_self_it

    processing_request[request_id].extend([e_self_it, e_self_cooling, start_time, computing_time, qos,
                                          energy_consumption])

for i in range(episode):
    activated_server_racks = {'0':[],'1':[],'2':[],'3':[]}  # 记录已经激活的server rack里硬件的信息
    activated_server_racks_flags = {'0': [], '1': [], '2': [], '3': []}  # 1表示对应的server rack已被激活，0表示又重新deactivate
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # state, the last four numbers are the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
    processing_request = {}  # 正在处理的requests      request_id->(action,rack_id,hardware_id)
    QoS_satisfy = []
    E_IT = 0
    E_Cooling = 0
    timeline = 0

    df = pd.read_csv("./data-4363/test.csv")
    # print(df.shape)
    # print(df.dtypes)
    # print(df.index)
    for indexes in df.index:
        request_id = int(df.loc[indexes].values[0])  # 请求的id
        userType = int(df.loc[indexes].values[2] % 12)  # user number, there are 12 users in total
        DNNType = int(df.loc[indexes].values[3] % 12 ) # DNN model number, there are 12 DNN models in total
        inTime = df.loc[indexes].values[-4] # request starts
        flag = int(df.loc[indexes].values[-3])  # request begin or end
        pCONV = LayerComp[DNNType][0]
        pPOOL = LayerComp[DNNType][1]
        pFC = LayerComp[DNNType][2]
        pBatch = LayerComp[DNNType][3]
        pRC = LayerComp[DNNType][4]
        QoS = float(df.loc[indexes].values[-1])
        # QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        # QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        
        # while True:
        #     QoS = round(np.random.normal((QoSMin * 1.2 + 1.5 * QoSMax) / 2.0,
        #                                  (1.5 * QoSMax - QoSMin * 1.2) / 6.0),6)  # QoS requirement
        #     if QoSMin * 1.2 <= QoS <= 1.5 * QoSMax:
        #         break
        
        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if inTime > timeline:
            for iii in range(4):
                Num_tmp = 0
                for iiii in range(len(activated_server_racks[str(iii)])):
                    if activated_server_racks_flags[str(iii)][iiii] == 1:
                        for iiiii in range(len(activated_server_racks[str(iii)][0])):
                            Num_tmp += activated_server_racks[str(iii)][iiii][iiiii][0]
                E_IT += Num_tmp * idle_power[iii] * (inTime - timeline)
                E_Cooling += Num_tmp * idle_power[iii] * gama[iii] * (inTime - timeline)
        timeline = inTime  # timeline 增加

        if flag == 0:  # 某一个请求结束,则更改STATE，即hardwareNumber
            E1, E2, Q = release_hardware(request_id, activated_server_racks, s, processing_request, activated_server_racks_flags)
            E_IT += E1
            E_Cooling += E2
            QoS_satisfy.append(Q)
        else:  # 某一请求开始，更改state，然后run DQN
            s[0] = CPURealTimeW[DNNType]
            s[1] = CPURealTimeC[DNNType]
            s[2] = GPURealTimeW[DNNType]
            s[3] = GPURealTimeC[DNNType]
            s[4] = pCONV
            s[5] = pPOOL
            s[6] = pFC
            s[7] = pBatch
            s[8] = pRC
            s[9] = QoS
            outTime = df.loc[indexes].values[-4] + df.loc[indexes].values[-2]
            energy = [CPUEnergyW[DNNType], CPUEnergyC[DNNType], GPUEnergyW[DNNType], GPUEnergyC[DNNType]]
            e_consumption = np.average(energy)
            rewards = []
            for action in range(4):
                c_time = s[action]
                reward = greedy_reward(c_time,action,
                                e_consumption, outTime, inTime, QoS)
                rewards.append(reward)
            
            '''
            greedy选择
            '''
            action=rewards.index(max(rewards))
            # 当多个选择具有相同的最大reward时，随机选择一个
            actionlist = np.argwhere(np.array(rewards)==max(rewards))
            index = np.random.randint(0,len(actionlist))
            action = actionlist[index][0]
            
            '''
            随机从满足QoS的action中选择一个action
            '''
            # index = np.random.randint(0,len(rewards))
            # while(rewards[index] == -float('inf')):
            #     index = np.random.randint(0,len(rewards))
            # action = index

            c_time = s[action]
            e_consumption = energy[action]
            # 获取新的state,即更改hardwareNumber
            s_ = act(request_id, action, s, activated_server_racks, processing_request, M, outTime, activated_server_racks_flags)
            request_add_item(request_id, c_time,e_consumption, outTime, inTime, QoS)
            
            s = s_  # 更新state
    
    for request_id in processing_request:
        c_action = processing_request[request_id][0]
        qos = processing_request[request_id][8]
        Q = (processing_request[request_id][7]-processing_request[request_id][8])/processing_request[request_id][8]
        computing_time = processing_request[request_id][7]
        energy_consumption = processing_request[request_id][9]
        start_time = processing_request[request_id][6]
        finish_time = timeline
        if qos >= computing_time:
            request_num = math.floor((finish_time - start_time) / qos)
            e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (finish_time - start_time - qos * request_num)*idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it
        else:
            request_num = math.floor((finish_time - start_time) / computing_time)
            e_self_it = energy_consumption * request_num + (finish_time - start_time - computing_time * request_num)*idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it

        E_IT += e_self_it
        E_Cooling += e_self_cooling
        QoS_satisfy.append(processing_request[request_id])

        # 训练过程中记录loss曲线
    
    print('==============', i, E_IT/3600000, E_Cooling/3600000, '==============')
    
    #QoS_satisfy to file
    QoSsatisfy = pd.DataFrame(columns=['action', 'rack_id', 'hardware_id', 'finish_time', 'e_self_it', 'e_self_cooling', 'start_time', \
                 'computing_time','qos', 'energy_consumption'], data=QoS_satisfy)
    QoSsatisfy.to_csv("./result_greedy/QoS_" + str(i) + ".csv", index=False)

    # processing_request: action, rack_id, hardware_id, finish_time, e_self_it, e_self_cooling, start_time, computing_time, qos, energy_consumption

'''
testing DRL
'''