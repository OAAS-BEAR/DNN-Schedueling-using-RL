## Table of Contents

- [Background](#background)
- [Usage](#usage)
- [Contributing](#contributer)
- [Reference](#reference)

## background
This is a group research project I worked on when I was a research intern in [Laboratory of service computing technology and systems, Ministry of Education](http://grid.hust.edu.cn/kydw/gdyjry.htm). The research group consists of [Prof. Fangming Liu](http://faculty.hust.edu.cn/liufangming/zh_CN/index.htm), Qingyu Pei,   Qunli Li(research intern), Dawei zhou(research intern)

The research is about Energy-efficient Deep Neural Network Inference Scheduling At the Edge. We leverage Deep Reinforment Learning(DRL) to  reduce total energy consumption of the DNN inference task. The traditional greedy strategy can not capture the relevance between inference tasks, because it only focuses on the current optimal choice,  We believe that reinforcement learning can capture this relevance.

We also compared the performance of different DRL algorithms: DQN and double DQN, the experienment shows that double DQN slightly outperforms DQN, this is probably because double DQN can alleviate the over-estimation problem. We also conducted a lot of experienment on the neural network achitectures, the form of reward function and the learning rate.

The main contribution of our work:

1. According to our best knowledge,we are the first to leverage Deep Reinforcement Learning to develop energy-efficient DNN inference scheduling algorithm.

2. We designed approriate State, action, reward and the environment for  the reinforcement learning scheduling algorithm

3. Our scheduling algorithm significantly outperforms traditional greedy-based  DNN inference scheduling  by 10 percent on average, in terms of energy efficiency.


## usage
trian deep DRL model and evaluate model performance on task of DNN inference scheduling
  ```
  python DRLtraining.py -d  #using Deep Q learning as the underlying DRL algorithm
  
  python DRLtraining.py  #using Double Deep Q learning as the underlying DRL algorithm
  
  ```
evaluate greedy-based DNN inference scheduling 

```
   python greedy.py
```


## contributer
Qunli Li @ Huazhong University of Science and Technology

Qiangyu Pei @ Huazhong University of Science and Technology

Dawei Zhou @ Huazhong University Of Science and Technology

Prof. Fangming Liu @ Huazhong University Of Science and Technology



## reference


   
