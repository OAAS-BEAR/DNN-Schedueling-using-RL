import copy
import math

import numpy as np
import pandas as pd

from DQN import DQN

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

CPURealTimeW = [99.118, 194.185, 279.248, 192.070, 229.745, 457.346, 1002.424, 233.919, 250.791, 256.342, 45.661, 5.784]  # the computing time on the CPU in the warm water area
CPURealTimeC = [97.433, 192.931, 276.896, 188.478, 227.395, 451.082, 898.043, 176.990, 214.053, 226.121, 43.751, 5.650]  # the computing time on the CPU in the cold water area
GPURealTimeW = [37.001, 57.690, 86.847, 73.057, 88.970, 124.025, 201.542, 27.709, 49.020, 48.426, 11.849, 14.269]  # the computing time on the GPU in the warm water area
GPURealTimeC = [33.925, 51.543, 71.645, 64.489, 82.196, 108.451, 182.225, 22.258, 41.965, 46.916, 9.727, 14.012]  # the computing time on the GPU in the cold water area
CPUEnergyW = [6.161884, 12.071303, 18.150614, 7.967937, 11.601355, 33.376023, 79.719643, 19.230621, 19.678410, 20.852059, 4.029731, 0.311649]  # the energy consumption on the CPU in the warm water area
CPUEnergyC = [7.111787, 12.240676, 18.289973, 8.049952, 11.861565, 33.488322, 81.943463, 25.966349, 24.844868, 24.600319, 4.516407, 0.309255]  # the energy consumption on the CPU in the cold water area
GPUEnergyW = [2.665347, 4.700274, 6.795247, 5.298865, 5.725534, 12.917292, 19.418534, 3.252527, 4.808302, 5.288533, 1.154375, 0.847006]  # the energy consumption on the GPU in the warm water area
GPUEnergyC = [2.462627, 4.252362, 6.279354, 5.540303, 8.499656, 14.834170, 30.017568, 4.468464, 6.518914, 6.571887, 1.286960, 0.909147]  # the energy consumption on the GPU in the cold water area

LayerComp = [[0.505, 0.019, 0.010, 0.467, 0.000], [0.502, 0.010, 0.005, 0.483, 0.000], [0.502, 0.006, 0.003, 0.489, 0.000], [0.550, 0.115, 0.005, 0.330, 0.000], 
             [0.551, 0.114, 0.004, 0.331, 0.000], [0.553, 0.114, 0.003, 0.330, 0.000], [0.554, 0.114, 0.002, 0.331, 0.000], [0.857, 0.143, 0.000, 0.000, 0.000], 
             [0.510, 0.000, 0.000, 0.490, 0.000], [0.500, 0.020, 0.000, 0.480, 0.000], [0.433, 0.200, 0.000, 0.367, 0.000], [0.000, 0.000, 0.500, 0.000, 0.500]]  # percentage of CONV, POOL, FC, Batch, and RC layers in each DNN model

M = [10, 10, 20, 20]  # number of hardware of each server rack
hardwareNumber = [0, 0, 0, 0]  # the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
activated_server_racks = []  # 记录已经激活的server rack里硬件的信息
s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # state
s[10] = hardwareNumber[0]
s[11] = hardwareNumber[1]
s[12] = hardwareNumber[2]
s[13] = hardwareNumber[3]
processing_request = {}  # 正在处理的requests      request_id->（action,rack_id,hardware_id)
dqn = DQN()
episode = 4
timeline = 0

idle_power = [26,26,10,10]
gama = [0.01,0.26,0.01,0.26]  # gama = CoolingEnergy / ITEnergy
alpha = 0.001
beta = 0.05


# 新激活一个server rack
def activate_server_rack(activated_server_racks, M, state):
    new_server_rack = []  # 新开辟的server_rack
    for x in range(4):
        hardware = []  # server_rack 每一类的硬件对应的都对应一个list
        for ii in range(M[x]):
            hardware.append([1, 0.0])  # list里每一项对应一个硬件状态(idle是否为真，finish_time)
        new_server_rack.append(hardware)  # 设置单个rack里的硬件信息
        state[x + 10] += M[x]  # 更新总的空闲硬件数量

    activated_server_racks.append(new_server_rack)


# 执行action
def act(request_id, action, state, activated_server_racks, processing_request, M, finish_time):
    s_new = copy.deepcopy(state)
    for x in range(len(activated_server_racks)):
        for j in range(len(activated_server_racks[x][action])):
            if activated_server_racks[x][action][j][0] == 1:
                activated_server_racks[x][action][j][0] = 0  # 寻找一个机架来执行任务
                activated_server_racks[x][action][j][1] = finish_time
                s_new[action + 10] -= 1  # 更新总的空闲硬件数目硬件
                processing_request[request_id] = (action, x, j)  # 添加任务信息到processing_request
                return s_new
    # 没有空余硬件可用，需要新激活一个server rack
    activate_server_rack(activated_server_racks, M, s_new)
    rack_id = len(activated_server_racks) - 1
    activated_server_racks[rack_id][action][0][0] = 0
    activated_server_racks[rack_id][action][0][1] = finish_time
    s_new[action + 10] -= 1
    processing_request[request_id] = (action, rack_id, 0)
    return s_new


# 硬件完成task后，设置为idle状态
def release_hardware(request_id, activated_server_racks, state, processing_request):
    action = processing_request[request_id][0]
    rack_id = processing_request[request_id][1]
    hardware_id = processing_request[request_id][2]
    activated_server_racks[rack_id][action][hardware_id][0] = 1  # 完成task后，硬件重新设置为idle
    activated_server_racks[rack_id][action][hardware_id][1] = 0.0  # finish_time设置为0
    state[action + 10] += 1  # 更新总的空闲硬件数目硬件
    processing_request.pop(request_id)


def greedy_reward(computing_time, energy_consumption, finish_time, start_time, qos):
    request_num = math.ceil((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power * (qos - computing_time)) * request_num
    e_self_cooling = gama * e_self_it
    q = math.log(1 + math.exp(100 * (computing_time - qos) / qos))
    g_reward = -alpha * (e_self_it + e_self_cooling) - beta * q
    return g_reward


# reward function using DRL
def get_reward(activated_server_racks, processing_request, request_id, computing_time, energy_consumption, finish_time,
               start_time, qos):
    request_num = math.ceil((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power * (qos - computing_time)) * request_num
    e_self_cooling = gama * e_self_it
    rack_id = processing_request[request_id][1]
    this_hardware_id = processing_request[request_id][2]
    this_action = processing_request[request_id][0]
    max_finish = 0

    # 计算max finish time
    for action in range(4):
        for hardware_id in range(len(activated_server_racks[rack_id][action])):
            if activated_server_racks[rack_id][action][hardware_id][1] > max_finish and not (
                    this_action == action and this_hardware_id == hardware_id):
                max_finish = activated_server_racks[rack_id][action][hardware_id][1]
    if finish_time > max_finish:
        time = finish_time - max_finish
    else:
        time = 0

    # compute e_other_it
    e_other_it = 0
    for h in range(4):
        if abs(h - this_action) % 2 == 0:
            if h == this_action:
                x = 1
            else:
                x = 0
            e_other_it += (M[h] - x) * time

    e_other_cooling = gama * e_other_it
    q = math.log(1 + math.exp(100 * (computing_time - qos) / qos))
    g_reward = -alpha * (e_self_it + e_self_cooling + e_other_cooling + e_other_it) - beta * q
    return g_reward


for i in range(episode):
    df = pd.read_csv("./data/test_" + str(i + 1) + ".csv")
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    for indexes in df.index:
        request_id = df.loc[indexes].values[0]  # 请求的id
        userType = df.loc[indexes].values[1] % 12  # user number, there are 12 users in total
        DNNType = df.loc[indexes].values[2] % 12  # DNN model number, there are 12 DNN models in total
        inTime = df.loc[indexes].values[-3]  # request starts
        timeline = inTime  # timeline 增加
        flag = df.loc[indexes].values[-2]  # request begin or end
        pCONV = LayerComp[DNNType][0]
        pPOOL = LayerComp[DNNType][1]
        pFC = LayerComp[DNNType][2]
        pBatch = LayerComp[DNNType][3]
        pRC = LayerComp[DNNType][4]

        QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        while True:
            QoS = round(np.random.normal((QoSMin * 1.2 + 1.5 * QoSMax)/2.0, (1.5 * QoSMax - QoSMin * 1.2) / 6.0))  # QoS requirement
            if QoSMin * 1.2 <= QoS <= 1.5 * QoSMax:
                break
            else:
                QoS = np.random.randint(QoSMin - 1, QoSMax + 1)
                break

        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if flag == 0:  # 某一个请求结束,则更改STATE，即hardwareNumber
            release_hardware(request_id, activated_server_racks, s, processing_request)
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
            action = dqn.choose_action(s)
            outTime = df.loc[indexes].values[-3] + df.loc[indexes].values[-1]

            # 获取新的state,即更改hardwareNumber
            s_ = act(request_id, action, s, activated_server_racks, processing_request, M, outTime)
            # reward = 0
            '''
            获取reward的代码
            '''
            energy = [CPUEnergyW[DNNType], CPUEnergyC[DNNType], GPUEnergyW[DNNType], GPUEnergyC[DNNType]]
            # reward = greedy_reward(s[action], energy[action], flag, inTime, QoS, ph_idle, gama, alpha, beta)
            c_time = s[action]
            e_consumption = energy[action]
            reward = get_reward(activated_server_racks, processing_request, request_id, c_time,
                                e_consumption, outTime, inTime, QoS)
            dqn.store_transition(s, action, reward, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            s = s_  # 更新state

        # 训练过程中记录loss曲线

'''
testing DRL
'''
