import copy
import math
import os

import numpy as np
import pandas as pd

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

CPURealTimeC = [0.097433, 0.192931, 0.276896, 0.188478, 0.227395, 0.451082, 0.898043, 0.176990, 0.214053, 0.226121,
                0.043751, 0.005650]  # the computing time on the CPU in the cold water area
GPURealTimeC = [0.033925, 0.051543, 0.071645, 0.064489, 0.082196, 0.108451, 0.182225, 0.022258, 0.041965, 0.046916,
                0.009727, 0.014012]  # the computing time on the GPU in the cold water area
CPUEnergyC = [7.111787, 12.240676, 18.289973, 8.049952, 11.861565, 33.488322, 81.943463, 25.966349, 24.844868,
              24.600319, 4.516407, 0.309255]  # the energy consumption on the CPU in the cold water area
GPUEnergyC = [2.462627, 4.252362, 6.279354, 5.540303, 8.499656, 14.834170, 30.017568, 4.468464, 6.518914, 6.571887,
              1.286960, 0.909147]  # the energy consumption on the GPU in the cold water area

# percentage of CONV, POOL, FC, Batch, and RC layers in each DNN model
LayerComp = [[0.505, 0.019, 0.010, 0.467, 0.000], [0.502, 0.010, 0.005, 0.483, 0.000],
             [0.502, 0.006, 0.003, 0.489, 0.000], [0.550, 0.115, 0.005, 0.330, 0.000],
             [0.551, 0.114, 0.004, 0.331, 0.000], [0.553, 0.114, 0.003, 0.330, 0.000],
             [0.554, 0.114, 0.002, 0.331, 0.000], [0.857, 0.143, 0.000, 0.000, 0.000],
             [0.510, 0.000, 0.000, 0.490, 0.000], [0.500, 0.020, 0.000, 0.480, 0.000],
             [0.433, 0.200, 0.000, 0.367, 0.000], [0.000, 0.000, 0.500, 0.000, 0.500]]

M = [10, 20]  # number of hardware in each server rack, CPU-W,CPU-C,GPU-W,GPU-C

episode = 1

idle_power = [24, 10]
gama = [0.26, 0.26]  # gama = CoolingEnergy / ITEnergy
alpha = 0.000001


# ???????????????server rack
def activate_server_rack(activated_server_racks, M, state, action, activated_server_racks_flags):
    for t in range(len(activated_server_racks_flags[str(action)])):
        if activated_server_racks_flags[str(action)][t] == 0:
            activated_server_racks_flags[str(action)][t] = 1
            activated_server_racks_flags[str((action + 1) % 2)][t] = 1
            return t
    new_server_rack = [[], []]  # ????????????server_rack
    for t in range(2):
        for ii in range(M[t]):
            new_server_rack[t].append([1, 0.0])  # list????????????????????????????????????(idle???????????????finish_time)
    activated_server_racks['0'].append(new_server_rack[0])  # ????????????rack??????????????????,list????????????????????????????????????(0??????idle,finish_time)
    activated_server_racks['1'].append(new_server_rack[1])  # ????????????rack??????????????????
    activated_server_racks_flags['0'].append(1)
    activated_server_racks_flags['1'].append(1)
    state[10] += M[0]  # ??????????????????????????????
    state[11] += M[1]  # ??????????????????????????????
    return len(activated_server_racks[str(action)]) - 1


# ??????action
def act(request_id, action, state, activated_server_racks, processing_request, M, finish_time,
        activated_server_racks_flags):
    s_new = copy.deepcopy(state)
    for t in range(len(activated_server_racks[str(action)])):  # ??????
        if activated_server_racks_flags[str(action)][t] == 1:
            for j in range(len(activated_server_racks[str(action)][t])):  # ??????
                if activated_server_racks[str(action)][t][j][0] == 1:
                    activated_server_racks[str(action)][t][j][0] = 0  # ?????????????????????????????????
                    activated_server_racks[str(action)][t][j][1] = finish_time
                    s_new[action + 8] -= 1  # ????????????????????????????????????
                    processing_request[request_id] = [action, t, j, finish_time]  # ?????????????????????processing_request
                    return s_new
    # ????????????????????????????????????????????????server rack
    rack_id = activate_server_rack(activated_server_racks, M, s_new, action, activated_server_racks_flags)
    activated_server_racks[str(action)][rack_id][0][0] = 0
    activated_server_racks[str(action)][rack_id][0][1] = finish_time
    s_new[action + 8] -= 1
    processing_request[request_id] = [action, rack_id, 0, finish_time]
    return s_new


# ????????????task???????????????idle??????
def release_hardware(request_id, activated_server_racks, state, processing_request, activated_server_racks_flags):
    action = processing_request[request_id][0]
    rack_id = processing_request[request_id][1]
    hardware_id = processing_request[request_id][2]
    activated_server_racks[str(action)][rack_id][hardware_id][0] = 1  # ??????task???????????????????????????idle
    activated_server_racks[str(action)][rack_id][hardware_id][1] = 0.0  # finish_time?????????0
    state[action + 8] += 1  # ????????????????????????????????????

    # deactivate racks
    restNum = 0
    action1 = action
    action2 = (action + 1) % 2
    for ii in range(len(activated_server_racks[str(action1)][rack_id])):
        restNum += activated_server_racks[str(action1)][rack_id][ii][0]
    for ii in range(len(activated_server_racks[str(action2)][rack_id])):
        restNum += activated_server_racks[str(action2)][rack_id][ii][0]
    if restNum == (M[action1] + M[action2]):
        activated_server_racks_flags[str(action1)][rack_id] = 0
        activated_server_racks_flags[str(action2)][rack_id] = 0
        state[action1 + 8] -= M[action1]
        state[action2 + 8] -= M[action2]

    E1 = processing_request[request_id][4]
    E2 = processing_request[request_id][5]
    Q = processing_request[request_id]
    processing_request.pop(request_id)
    return E1, E2, Q


def greedy_reward(computing_time, action, energy_consumption, finish_time, start_time, qos):
    """
    ??????????????????action??????????????????
    """
    activated_server_racks_tmp = copy.deepcopy(activated_server_racks)
    processing_request_tmp = copy.deepcopy(processing_request)
    activated_server_racks_flags_tmp = copy.deepcopy(activated_server_racks_flags)

    exit_flag = False
    for x in range(len(activated_server_racks_tmp[str(action)])):  # ??????
        if activated_server_racks_flags[str(action)][x] == 1:
            for j in range(len(activated_server_racks_tmp[str(action)][x])):  # ??????
                if activated_server_racks_tmp[str(action)][x][j][0] == 1:
                    activated_server_racks_tmp[str(action)][x][j][0] = 0  # ?????????????????????????????????
                    activated_server_racks_tmp[str(action)][x][j][1] = finish_time
                    processing_request_tmp[request_id] = [action, x, j, finish_time]  # ?????????????????????processing_request
                    exit_flag = True
                    break
        if exit_flag:
            break
    # ????????????????????????????????????????????????server rack
    if exit_flag is False:
        new_server_rack = [[], []]  # ????????????server_rack
        for x in range(2):
            for ii in range(M[x]):
                new_server_rack[x].append([1, 0.0])  # list????????????????????????????????????(idle???????????????finish_time)
        activated_server_racks_tmp['0'].append(new_server_rack[0])  # ????????????rack??????????????????,list????????????????????????????????????(0??????idle,finish_time)
        activated_server_racks_tmp['1'].append(new_server_rack[1])  # ????????????rack??????????????????
        activated_server_racks_flags_tmp['0'].append(1)
        activated_server_racks_flags_tmp['1'].append(1)
        rack_id = len(activated_server_racks_tmp[str(action)]) - 1
        activated_server_racks_tmp[str(action)][rack_id][0][0] = 0
        activated_server_racks_tmp[str(action)][rack_id][0][1] = finish_time
        processing_request_tmp[request_id] = [action, rack_id, 0, finish_time]

    '''
    ??????4????????????
    '''
    c_action = action
    if computing_time > qos:
        return -float('inf')
    else:
        request_num = math.floor((finish_time - start_time) / qos)
        e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (
                finish_time - start_time - qos * request_num) * idle_power[c_action]
        e_self_cooling = gama[c_action] * e_self_it

    rack_id = processing_request_tmp[request_id][1]
    this_hardware_id = processing_request_tmp[request_id][2]
    this_action = processing_request_tmp[request_id][0]
    max_finish = 0

    # ??????max finish time
    for ac in [this_action, (this_action + 1) % 2]:
        for hardware_id in range(len(activated_server_racks_tmp[str(ac)][rack_id])):
            if activated_server_racks_tmp[str(ac)][rack_id][hardware_id][1] > max_finish and not (
                    this_action == ac and this_hardware_id == hardware_id):
                max_finish = activated_server_racks_tmp[str(ac)][rack_id][hardware_id][1]
    if finish_time > max_finish:
        time = finish_time - max_finish
    else:
        time = 0

    # compute e_other_it
    e_other_it = 0
    for h in [this_action, (this_action + 1) % 2]:
        if h == this_action:
            x = 1
        else:
            x = 0
        e_other_it += (M[h] - x) * time * idle_power[h]

    e_other_cooling = gama[c_action] * e_other_it

    g_reward = -alpha * (e_self_it + e_self_cooling + e_other_cooling + e_other_it)
    return g_reward


def request_add_item(request_id, computing_time, energy_consumption, finish_time, start_time, qos):
    """
    ??????????????????????????????????????????IT???cooling???????????????processing_request
    """
    c_action = processing_request[request_id][0]
    assert (computing_time <= qos)
    request_num = math.floor((finish_time - start_time) / qos)
    e_self_it = (energy_consumption + idle_power[c_action] * (qos - computing_time)) * request_num + (
            finish_time - start_time - qos * request_num) * idle_power[c_action]
    e_self_cooling = gama[c_action] * e_self_it

    processing_request[request_id].extend([e_self_it, e_self_cooling, start_time, computing_time, qos,
                                           energy_consumption])


for i in range(episode):
    activated_server_racks = {'0': [], '1': []}  # ?????????????????????server rack??????????????????
    activated_server_racks_flags = {'0': [], '1': []}  # 1???????????????server rack???????????????0???????????????deactivate
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0]  # state, the last four numbers are the number of idle hardware in each area, CPU-W,CPU-C,GPU-W,GPU-C
    processing_request = {}  # ???????????????requests      request_id->(action,rack_id,hardware_id)
    QoS_satisfy = []
    E_IT = 0
    E_Cooling = 0
    timeline = 0

    df = pd.read_csv("./data-4363/test.csv")
    # print(df.shape)
    # print(df.dtypes)
    # print(df.index)
    for indexes in df.index:
        request_id = int(df.loc[indexes].values[0])  # ?????????id
        userType = int(df.loc[indexes].values[2] % 12)  # user number, there are 12 users in total
        DNNType = int(df.loc[indexes].values[3] % 12)  # DNN model number, there are 12 DNN models in total
        inTime = df.loc[indexes].values[-4]  # request starts
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

        # ?????????????????????????????????????????????????????????????????????STATE??????hardwareNumber
        '''
        DRL code here
        '''
        if inTime > timeline:
            for x in range(2):
                Num_tmp = 0
                for y in range(len(activated_server_racks[str(x)])):
                    if activated_server_racks_flags[str(x)][y] == 1:
                        for z in range(len(activated_server_racks[str(x)][0])):
                            Num_tmp += activated_server_racks[str(x)][y][z][0]
                E_IT += Num_tmp * idle_power[x] * (inTime - timeline)
                E_Cooling += Num_tmp * idle_power[x] * gama[x] * (inTime - timeline)
        timeline = inTime  # timeline ??????

        if flag == 0:  # ?????????????????????,?????????STATE??????hardwareNumber
            E1, E2, Q = release_hardware(request_id, activated_server_racks, s, processing_request,
                                         activated_server_racks_flags)
            E_IT += E1
            E_Cooling += E2
            QoS_satisfy.append(Q)
        else:  # ???????????????????????????state?????????run DQN
            s[0] = CPURealTimeC[DNNType]
            s[1] = GPURealTimeC[DNNType]
            s[2] = pCONV
            s[3] = pPOOL
            s[4] = pFC
            s[5] = pBatch
            s[6] = pRC
            s[7] = QoS
            outTime = df.loc[indexes].values[-4] + df.loc[indexes].values[-2]
            energy = [CPUEnergyC[DNNType], GPUEnergyC[DNNType]]
            e_consumption = np.average(energy)
            rewards = []
            for action in [0, 1]:
                c_time = s[action]
                reward = greedy_reward(c_time, action,
                                       e_consumption, outTime, inTime, QoS)
                rewards.append(reward)

            '''
            greedy??????
            '''
            # ????????????????????????????????????reward????????????????????????
            actionList = np.argwhere(np.array(rewards) == max(rewards))
            index = np.random.randint(0, len(actionList))
            action = actionList[index][0]

            c_time = s[action]
            e_consumption = energy[action]
            # ????????????state,?????????hardwareNumber
            s_ = act(request_id, action, s, activated_server_racks, processing_request, M, outTime,
                     activated_server_racks_flags)
            request_add_item(request_id, c_time, e_consumption, outTime, inTime, QoS)

            s = s_  # ??????state

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

        # ?????????????????????loss??????

    print('==============', i, E_IT / 3600000, E_Cooling / 3600000, '==============')

    # QoS_satisfy to file
    p = "./result_baseline/"
    if not os.path.exists(p):     # ????????????????????????????????????????????????new?????????
        os.makedirs(p)
    fl = open(p + "QoS_" + str(i) + ".csv", "w")
    fl.close()
    QoSSatisfy = pd.DataFrame(
        columns=['action', 'rack_id', 'hardware_id', 'finish_time', 'e_self_it', 'e_self_cooling', 'start_time',
                 'computing_time', 'qos', 'energy_consumption'], data=QoS_satisfy)
    QoSSatisfy.to_csv(p + "QoS_" + str(i) + ".csv", index=False)

    # processing_request: action, rack_id, hardware_id, finish_time, e_self_it, e_self_cooling, start_time,
    # computing_time, qos, energy_consumption

'''
testing DRL
'''
