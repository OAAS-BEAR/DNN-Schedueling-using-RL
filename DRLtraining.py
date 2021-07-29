import copy
import math
import sys

import pandas as pd

from DQN import DQN
from DQN import doubleDQN

# np.random.seed(10)
MEMORY_CAPACITY = 200

'''
read trace data from sqlite, the trace is downloaded from 
https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
'''

'''
training DRL
'''

# there are 7 DNN models in total
# ResNetV1-50, EfficientNet-B1, EfficientNet-B3, Unet, YoloV3-spp, YoloV3-tiny, and NER

CPURealTimeW = [0.12593318, 0.168585066, 0.215241714, 0.449792373, 0.378377307, 0.072351546, 0.008672056]  # the computing time on the CPU in the warm water area
CPURealTimeC = [0.121515624, 0.169374289, 0.216643645, 0.30870271, 0.233688547, 0.065995284, 0.008512394]  # the computing time on the CPU in the cold water area
GPURealTimeW = [0.028054602, 0.035798463, 0.040783417, 0.0267817, 0.040369528, 0.008844684, 0.025547585]  # the computing time on the GPU in the warm water area
GPURealTimeC = [0.027557281, 0.035004458, 0.038275455, 0.020054302, 0.036708363, 0.008514694, 0.025139011]  # the computing time on the GPU in the cold water area
CPUEnergyW = [6.773681904, 4.808776556, 8.623915022, 25.82014818, 22.02489374, 4.150088693, 0.307646299]  # the energy consumption on the CPU in the warm water area
CPUEnergyC = [7.032627357, 4.723595155, 8.497495705, 40.93891925, 35.33893188, 4.351672798, 0.304127587]  # the energy consumption on the CPU in the cold water area
GPUEnergyW = [2.109958108, 2.980562943, 3.493269943, 2.535316654, 4.149884927, 1.064684258, 1.650573132]  # the energy consumption on the GPU in the warm water area
GPUEnergyC = [2.576643546, 3.584285763, 4.008268109, 5.0207346, 5.963033842, 1.252211195, 1.66986749]  # the energy consumption on the GPU in the cold water area

# percentage of CONV, POOL, FC, Batch, Activation, and RC layers in each DNN model
LayerComp = [[0.3081395348837209, 0.011627906976744186, 0.005813953488372093, 0.28488372093023256, 0.38953488372093026, 0.0],
             [0.26558891454965355, 0.05542725173210162, 0.0023094688221709007, 0.15935334872979215, 0.5173210161662818, 0.0],
             [0.2653061224489796, 0.05510204081632653, 0.0020408163265306124, 0.15918367346938775, 0.5183673469387755, 0.0],
             [0.4, 0.06666666666666667, 0.0, 0.0, 0.5333333333333333, 0.0],
             [0.29457364341085274, 0.011627906976744186, 0.0, 0.28294573643410853, 0.4108527131782946, 0.0],
             [0.3023255813953488, 0.13953488372093023, 0.0, 0.2558139534883721, 0.3023255813953488, 0.0],
             [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]]

M = [10, 10, 20, 20]  # number of hardware in each server rack, CPU-W,CPU-C,GPU-W,GPU-C

if sys.argv[1] == 'd':
    dqn = DQN()
else:
    dqn = doubleDQN()
episode = 320

idle_power = [26, 26, 10, 10]
gama = [0.01, 0.26, 0.01, 0.26]  # gama = CoolingEnergy / ITEnergy
alpha = 0.00001
beta = 0.00001


# 新激活一个server rack
def activate_server_rack(activated_server_racks, M, state, action, activated_server_racks_flags):
    for i in range(len(activated_server_racks_flags[str(action)])):
        if activated_server_racks_flags[str(action)][i] == 0:
            activated_server_racks_flags[str(action)][i] = 1
            activated_server_racks_flags[str((action + 2) % 4)][i] = 1
            return i
    new_server_rack = [[], [], [], []]  # 新开辟的server_rack
    for x in range(4):
        for ii in range(M[x]):
            new_server_rack[x].append([1, 0.0])  # list里每一项对应一个硬件状态(idle是否为真，finish_time)
    if action == 0 or action == 2:
        activated_server_racks['0'].append(new_server_rack[0])  # 设置单个rack里的硬件信息,list里每一项对应一个硬件状态(0表示idle,finish_time)
        activated_server_racks['2'].append(new_server_rack[2])  # 设置单个rack里的硬件信息
        activated_server_racks_flags['0'].append(1)
        activated_server_racks_flags['2'].append(1)
        state[11] += M[0]  # 更新总的空闲硬件数量
        state[13] += M[2]  # 更新总的空闲硬件数量
    else:
        activated_server_racks['1'].append(new_server_rack[1])
        activated_server_racks['3'].append(new_server_rack[3])
        activated_server_racks_flags['1'].append(1)
        activated_server_racks_flags['3'].append(1)
        state[12] += M[1]
        state[14] += M[3]
    return len(activated_server_racks[str(action)]) - 1


# 执行action
def act(request_id, action, state, activated_server_racks, processing_request, M, finish_time,
        activated_server_racks_flags):
    s_new = copy.deepcopy(state)
    for x in range(len(activated_server_racks[str(action)])):  # 机架
        if activated_server_racks_flags[str(action)][x] == 1:
            for j in range(len(activated_server_racks[str(action)][x])):  # 硬件
                if activated_server_racks[str(action)][x][j][0] == 1:
                    activated_server_racks[str(action)][x][j][0] = 0  # 寻找一个机架来执行任务
                    activated_server_racks[str(action)][x][j][1] = finish_time
                    s_new[action + 11] -= 1  # 更新总的空闲硬件数目硬件
                    processing_request[request_id] = [action, x, j, finish_time]  # 添加任务信息到processing_request
                    return s_new
    # 没有空余硬件可用，需要新激活一个server rack
    rack_id = activate_server_rack(activated_server_racks, M, s_new, action, activated_server_racks_flags)
    activated_server_racks[str(action)][rack_id][0][0] = 0
    activated_server_racks[str(action)][rack_id][0][1] = finish_time
    s_new[action + 11] -= 1
    processing_request[request_id] = [action, rack_id, 0, finish_time]
    return s_new


# 硬件完成task后，设置为idle状态
def release_hardware(request_id, activated_server_racks, state, processing_request, activated_server_racks_flags):
    action = processing_request[request_id][0]
    rack_id = processing_request[request_id][1]
    hardware_id = processing_request[request_id][2]
    activated_server_racks[str(action)][rack_id][hardware_id][0] = 1  # 完成task后，硬件重新设置为idle
    activated_server_racks[str(action)][rack_id][hardware_id][1] = 0.0  # finish_time设置为0
    state[action + 11] += 1  # 更新总的空闲硬件数目硬件

    # deactivate racks
    restNum = 0
    action1 = action
    action2 = (action + 2) % 4
    for ii in range(len(activated_server_racks[str(action1)][rack_id])):
        restNum += activated_server_racks[str(action1)][rack_id][ii][0]
    for ii in range(len(activated_server_racks[str(action2)][rack_id])):
        restNum += activated_server_racks[str(action2)][rack_id][ii][0]
    if restNum == (M[action1] + M[action2]):
        activated_server_racks_flags[str(action1)][rack_id] = 0
        activated_server_racks_flags[str(action2)][rack_id] = 0
        state[action1 + 11] -= M[action1]
        state[action2 + 11] -= M[action2]

    E1 = processing_request[request_id][4]
    E2 = processing_request[request_id][5]
    Q = processing_request[request_id]
    processing_request.pop(request_id)
    return E1, E2, Q


# reward function using DRL
def get_reward(activated_server_racks, processing_request, request_id, computing_time, energy_consumption, finish_time,
               start_time, qos):
    c_action = processing_request[request_id][0]
    if qos >= computing_time:
        request_num = math.floor((finish_time - start_time) / qos)
        e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (
                finish_time - start_time - qos * request_num) * idle_power[c_action]
        e_self_cooling = gama[c_action] * e_self_it
    else:
        request_num = math.floor((finish_time - start_time) / computing_time)
        e_self_it = energy_consumption * request_num + (finish_time - start_time - computing_time * request_num) * \
                    idle_power[c_action]
        e_self_cooling = gama[c_action] * e_self_it
    rack_id = processing_request[request_id][1]
    this_hardware_id = processing_request[request_id][2]
    this_action = processing_request[request_id][0]

    processing_request[request_id].extend([e_self_it, e_self_cooling, start_time, computing_time, qos,
                                           energy_consumption])  # 便于计算系统总能耗开销
    max_finish = 0

    # 计算max finish time
    for action in [this_action, (this_action + 2) % 4]:
        for hardware_id in range(len(activated_server_racks[str(action)][rack_id])):
            if activated_server_racks[str(action)][rack_id][hardware_id][1] > max_finish and not (
                    this_action == action and this_hardware_id == hardware_id):
                max_finish = activated_server_racks[str(action)][rack_id][hardware_id][1]
    if finish_time > max_finish:
        time = finish_time - max_finish
    else:
        time = 0

    # compute e_other_it
    e_other_it = 0
    for h in [this_action, (this_action + 2) % 4]:
        if h == this_action:
            x = 1
        else:
            x = 0
        e_other_it += (M[h] - x) * time * idle_power[h]

    e_other_cooling = gama[c_action] * e_other_it
    try:
        q = math.log(1 + math.exp(100 * (computing_time - qos) / qos)) * computing_time * request_num
    except:
        q = (100 * (computing_time - qos) / qos) * computing_time * request_num
    g_reward = -alpha * (e_self_it + e_self_cooling + e_other_cooling + e_other_it) - beta * q
    g_reward = math.log(-1.0 / g_reward)  # 减小reward发散的情况，加快收敛速度
    # print('#############',e_self_it,e_self_cooling,e_other_cooling,e_other_it,q)
    return g_reward



for i in range(80):
    activated_server_racks = {'0': [], '1': [], '2': [], '3': []}  # 记录已经激活的server rack里硬件的信息
    activated_server_racks_flags = {'0': [], '1': [], '2': [], '3': []}  # 1表示对应的server rack已被激活，0表示又重新deactivate
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,0]  # state, the last four numbers are the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
    processing_request = {}  # 正在处理的requests      request_id->(action,rack_id,hardware_id)
    QoS_satisfy = []
    E_IT = 0
    E_Cooling = 0
    timeline = 0

    df = pd.read_csv("./data-4363/test_" + str(i) + ".csv")
    # print(df.shape)
    # print(df.dtypes)
    # print(df.index)
    for indexes in df.index:
        request_id = int(df.loc[indexes].values[0])  # 请求的id
        userType = int(df.loc[indexes].values[2] % 7)  # user number, there are 12 users in total
        DNNType = int(df.loc[indexes].values[3] % 7)  # DNN model number, there are 12 DNN models in total
        inTime = df.loc[indexes].values[-4]  # request starts
        flag = int(df.loc[indexes].values[-3])  # request begin or end
        pCONV = LayerComp[DNNType][0]
        pPOOL = LayerComp[DNNType][1]
        pFC = LayerComp[DNNType][2]
        pBatch = LayerComp[DNNType][3]
        pAC = LayerComp[DNNType][4]
        pRC = LayerComp[DNNType][5]
        QoS = float(df.loc[indexes].values[-1])
        # QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        # QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])

        # while True:
        #     QoS = round(np.random.normal((QoSMin * 1.2 + 1.5 * QoSMax) / 2.0,
        #                                  (1.5 * QoSMax - QoSMin * 1.2) / 6.0),6)  # QoS requirement
        #     if QoSMin * 1.1 <= QoS <= 1.5 * QoSMax:
        #         break

        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if inTime > timeline:
            for x in range(4):
                Num_tmp = 0
                for y in range(len(activated_server_racks[str(x)])):
                    if activated_server_racks_flags[str(x)][y] == 1:
                        for z in range(len(activated_server_racks[str(x)][0])):
                            Num_tmp += activated_server_racks[str(x)][y][z][0]
                E_IT += Num_tmp * idle_power[x] * (inTime - timeline)
                E_Cooling += Num_tmp * idle_power[x] * gama[x] * (inTime - timeline)
        timeline = inTime  # timeline 增加

        if flag == 0:  # 某一个请求结束,则更改STATE，即hardwareNumber
            E1, E2, Q = release_hardware(request_id, activated_server_racks, s, processing_request,
                                         activated_server_racks_flags)
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
            s[8] = pAC
            s[9] = pRC
            s[10] = QoS
            action = dqn.choose_action(s,False)
            outTime = df.loc[indexes].values[-4] + df.loc[indexes].values[-2]

            # 获取新的state,即更改hardwareNumber
            s_ = act(request_id, action, s, activated_server_racks, processing_request, M, outTime,
                     activated_server_racks_flags)
            # reward = 0
            '''
            获取reward的代码
            '''
            energy = [CPUEnergyW[DNNType], CPUEnergyC[DNNType], GPUEnergyW[DNNType], GPUEnergyC[DNNType]]
            # reward = greedy_reward(s[action], energy[action], flag, inTime, QoS, ph_idle, gama, alpha, beta)
            c_time = s[action]
            e_consumption = energy[action]
            reward = get_reward( activated_server_racks, processing_request, request_id, c_time,
                                e_consumption, outTime, inTime,QoS)
            dqn.store_transition(s, action, reward, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                loss = dqn.learn()
                # if indexes % 500 == 0:
                # print('reward ' + str(reward))
                # print('epcoh ' + str(i) + ' step ' + str(indexes) + ' : ' + ' , LOSS =' + str(loss.item())) # 训练过程中记录loss曲线
            s = s_  # 更新state

    for request_id in processing_request:
        c_action = processing_request[request_id][0]
        qos = processing_request[request_id][8]
        Q = (processing_request[request_id][7] - processing_request[request_id][8]) / processing_request[request_id][8]
        computing_time = processing_request[request_id][7]
        energy_consumption = processing_request[request_id][9]
        start_time = processing_request[request_id][6]
        finish_time = timeline
        if qos >= computing_time:
            request_num = math.floor((finish_time - start_time) / qos)
            e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (
                    finish_time - start_time - qos * request_num) * idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it
        else:
            request_num = math.floor((finish_time - start_time) / computing_time)
            e_self_it = energy_consumption * request_num + (finish_time - start_time - computing_time * request_num) * \
                        idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it

        E_IT += e_self_it
        E_Cooling += e_self_cooling
        QoS_satisfy.append(processing_request[request_id])

    print('==============', i, E_IT / 3600000, E_Cooling / 3600000, '==============')

    # QoS_satisfy to file
    QoSsatisfy = pd.DataFrame(
        columns=['action', 'rack_id', 'hardware_id', 'finish_time', 'e_self_it', 'e_self_cooling', 'start_time',
                 'computing_time', 'qos', 'energy_consumption'], data=QoS_satisfy)
    QoSsatisfy.to_csv("./result/QoS_" + str(i) + ".csv", index=False)

    # processing_request: action, rack_id, hardware_id, finish_time, e_self_it, e_self_cooling, start_time,
    # computing_time, qos, energy_consumption

dqn.save_model()

'''
testing DRL
'''
for i in range(10):
    activated_server_racks = {'0': [], '1': [], '2': [], '3': []}  # 记录已经激活的server rack里硬件的信息
    activated_server_racks_flags = {'0': [], '1': [], '2': [], '3': []}  # 1表示对应的server rack已被激活，0表示又重新deactivate
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,0]  # state, the last four numbers are the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
    processing_request = {}  # 正在处理的requests      request_id->(action,rack_id,hardware_id)
    QoS_satisfy = []
    E_IT = 0
    E_Cooling = 0
    timeline = 0

    df = pd.read_csv("./data-4363/val_" + str(i % 10) + ".csv")
    # print(df.shape)
    # print(df.dtypes)
    # print(df.index)
    for indexes in df.index:
        request_id = int(df.loc[indexes].values[0])  # 请求的id
        userType = int(df.loc[indexes].values[2] % 7)  # user number, there are 12 users in total
        DNNType = int(df.loc[indexes].values[3] % 7)  # DNN model number, there are 12 DNN models in total
        inTime = df.loc[indexes].values[-4]  # request starts
        flag = int(df.loc[indexes].values[-3])  # request begin or end
        pCONV = LayerComp[DNNType][0]
        pPOOL = LayerComp[DNNType][1]
        pFC = LayerComp[DNNType][2]
        pBatch = LayerComp[DNNType][3]
        pAC = LayerComp[DNNType][4]
        pRC = LayerComp[DNNType][5]
        QoS = float(df.loc[indexes].values[-1])
        # QoSMin = min(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])
        # QoSMax = max(CPURealTimeW[DNNType], CPURealTimeC[DNNType], GPURealTimeW[DNNType], GPURealTimeC[DNNType])

        # while True:
        #     QoS = round(np.random.normal((QoSMin * 1.2 + 1.5 * QoSMax) / 2.0,
        #                                  (1.5 * QoSMax - QoSMin * 1.2) / 6.0),6)  # QoS requirement
        #     if QoSMin * 1.1 <= QoS <= 1.5 * QoSMax:
        #         break

        # 需判断是否有旧请求在这一时刻结束，如有，则更改STATE，即hardwareNumber
        '''
        DRL code here
        '''
        if inTime > timeline:
            for x in range(4):
                Num_tmp = 0
                for y in range(len(activated_server_racks[str(x)])):
                    if activated_server_racks_flags[str(x)][y] == 1:
                        for z in range(len(activated_server_racks[str(x)][0])):
                            Num_tmp += activated_server_racks[str(x)][y][z][0]
                E_IT += Num_tmp * idle_power[x] * (inTime - timeline)
                E_Cooling += Num_tmp * idle_power[x] * gama[x] * (inTime - timeline)
        timeline = inTime  # timeline 增加

        if flag == 0:  # 某一个请求结束,则更改STATE，即hardwareNumber
            E1, E2, Q = release_hardware(request_id, activated_server_racks, s, processing_request,
                                         activated_server_racks_flags)
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
            s[8] = pAC
            s[9] = pRC
            s[10] = QoS
            action = dqn.choose_action(s,True)
            outTime = df.loc[indexes].values[-4] + df.loc[indexes].values[-2]

            # 获取新的state,即更改hardwareNumber
            s_ = act(request_id, action, s, activated_server_racks, processing_request, M, outTime,
                     activated_server_racks_flags)
            # reward = 0
            '''
            获取reward的代码
            '''
            energy = [CPUEnergyW[DNNType], CPUEnergyC[DNNType], GPUEnergyW[DNNType], GPUEnergyC[DNNType]]
            # reward = greedy_reward(s[action], energy[action], flag, inTime, QoS, ph_idle, gama, alpha, beta)
            c_time = s[action]
            e_consumption = energy[action]
            reward = get_reward(activated_server_racks, processing_request, request_id, c_time,
                                e_consumption, outTime, inTime,QoS)
            s = s_  # 更新state

    for request_id in processing_request:
        c_action = processing_request[request_id][0]
        qos = processing_request[request_id][8]
        Q = (processing_request[request_id][7] - processing_request[request_id][8]) / processing_request[request_id][8]
        computing_time = processing_request[request_id][7]
        energy_consumption = processing_request[request_id][9]
        start_time = processing_request[request_id][6]
        finish_time = timeline
        if qos >= computing_time:
            request_num = math.floor((finish_time - start_time) / qos)
            e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (
                    finish_time - start_time - qos * request_num) * idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it
        else:
            request_num = math.floor((finish_time - start_time) / computing_time)
            e_self_it = energy_consumption * request_num + (finish_time - start_time - computing_time * request_num) * \
                        idle_power[c_action]
            e_self_cooling = gama[c_action] * e_self_it

        E_IT += e_self_it
        E_Cooling += e_self_cooling
        QoS_satisfy.append(processing_request[request_id])

    print('==============', i, E_IT / 3600000, E_Cooling / 3600000, '==============')

    # QoS_satisfy to file
    QoSsatisfy = pd.DataFrame(
        columns=['action', 'rack_id', 'hardware_id', 'finish_time', 'e_self_it', 'e_self_cooling', 'start_time',
                 'computing_time', 'qos', 'energy_consumption'], data=QoS_satisfy)
    QoSsatisfy.to_csv("./result/QoS_val_" + str(i) + ".csv", index=False)

    # processing_request: action, rack_id, hardware_id, finish_time, e_self_it, e_self_cooling, start_time,
    # computing_time, qos, energy_consumption
