## Table of Contents

- [Background](#background)
- [Environment install](#environment)
- [Usage](#usage)
- [Contributing](#contributer)
- [Reference](#reference)

## background
This is a research project on Energy-efficient Deep Neural Network Inference Scheduling At the Edge , conducted by a research group of the [Laboratory of service computing technology and systems, Ministry of Education](http://grid.hust.edu.cn/kydw/gdyjry.htm).

The research group consists of [Prof. Fangming Liu](http://faculty.hust.edu.cn/liufangming/zh_CN/index.htm), Qingyu Pei, Qunli Li(research intern), Dawei zhou(research intern)

The main contribution of our work:

1. According to our knowledge,we are the first to leverage Deep Reinforcement Learning to develop energy-efficient DNN inference scheduling algorithm.

2. We designed approriate State, action, and reward function, environment for  reinforcement learning scheduling algorithm

3. Our scheduling algorithm significantly outperforms traditional greedy scheduling algorithm by 10 percent on average, in terms of energy efficiency.

## environment


## usage
 train

   ```
   
   # prepare data
   python data_u.py
   # train our model 
   # If the GPU related running environment is installed and configured, the command line --cuda can be added to use GPU training
   python run.py
   ```


## contributer
Qunli Li @ Huazhong University of Science and Technology

Qiangyu Pei @ Huazhong University of Science and Technology

Dawei Zhou @ Huazhong University Of Science and Technology

Prof. Fangming Liu @ Huazhong University Of Science and Technology



## reference


   
