import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as tnf
import numpy as np
import os
cpu_num = 6 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

BATCH_SIZE = 32
LR = 0.9  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.5  # reward discount
TARGET_REPLACE_ITER = 20  # target update frequency
MEMORY_CAPACITY = 100
N_ACTIONS = 4  # number of actions
N_STATES = 14  # dimensions of states
device = torch.device('cuda:0')

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64).to(device)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s):
        s = self.fc1(s)
        s = tnf.relu(s)
        s = self.fc2(s)
        s = tnf.relu(s)
        actions_value = self.out(s)
        return actions_value  # return Q(S,A)


class DQN(object):
    def __init__(self):
        self.target_net = Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化memory
        self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # return action
    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        if np.random.uniform() < EPSILON:
            actions_value = self.target_net.forward(x)
            # print(torch.max(actions_value, 1)[1].data.numpy()[0])
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # set s,a,r,s_ the array of 1*N
        transition = np.hstack((s, [a, r], s_))
        # another type of the h_stack , the result is the same
        np.hstack((s, a, r, s_))
        # use % can loop from 0 to 2000
        index = self.memory_counter % MEMORY_CAPACITY
        # save the info_this_step into memory
        self.memory[index, :] = transition
        self.memory_counter += 1

    def save_model(self):
        torch.save(self.target_net, './model.pkl')

    def learn(self):
        # by random , choose the row's number from memory_capacity , total row's number is batch_size(32)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # in the memory, the 1st---4th column is state_now , the 5th is action , the 6th is reward
        # the final 4 column is state_next
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])).to(device)
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))).to(device)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])).to(device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.target_net(b_s).gather(1, b_a)  # shape (batch, 1)

        q_eval_test = self.target_net(b_s_)
        # argmax axis = 0 means column , 1 means row
        # we choose the max action value , the action is column , so axis = 1
        q1_argmax = np.argmax(q_eval_test.cpu().data.numpy(), axis=1)

        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagation
        q_next = self.target_net(b_s_)

        q_next_numpy = q_next.cpu().data.numpy()

        q_update = np.zeros((BATCH_SIZE, 1))
        for iii in range(BATCH_SIZE):
            q_update[iii] = q_next_numpy[iii, q1_argmax[iii]]

        q_update = GAMMA * q_update
        q_update = torch.FloatTensor(q_update).to(device)

        variable11 = Variable(q_update)
        q_target = b_r + variable11
        # q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


class doubleDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # return action
    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).to(device)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1)[1].data.numpy()[0])
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        # set s,a,r,s_ the array of 1*N
        transition = np.hstack((s, [a, r], s_))
        # another type of the h_stack , the result is the same
        np.hstack((s, a, r, s_))
        # use % can loop from 0 to 2000
        index = self.memory_counter % MEMORY_CAPACITY
        # save the info_this_step into memory
        self.memory[index, :] = transition
        self.memory_counter += 1

    def save_model(self):
        torch.save(self.target_net, './model.pkl')
        
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # by random , choose the row's number from memory_capacity , total row's number is batch_size(32)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # in the memory, the 1st---4th column is state_now , the 5th is action , the 6th is reward
        # the final 4 column is state_next
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])).to(device)
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))).to(device)
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])).to(device)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        q_eval_test = self.eval_net(b_s_)
        # argmax axis = 0 means column , 1 means row
        # we choose the max action value , the action is column , so axis = 1
        q1_argmax = np.argmax(q_eval_test.cpu().data.numpy(), axis=1)

        # q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagation
        q_next = self.target_net(b_s_)

        q_next_numpy = q_next.cpu().data.numpy()

        q_update = np.zeros((BATCH_SIZE, 1))
        for iii in range(BATCH_SIZE):
            q_update[iii] = q_next_numpy[iii, q1_argmax[iii]]

        q_update = GAMMA * q_update
        q_update = torch.FloatTensor(q_update).to(device)

        variable11 = Variable(q_update)
        q_target = b_r + variable11
        # q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
