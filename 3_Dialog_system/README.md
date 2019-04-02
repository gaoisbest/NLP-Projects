# Dialog system categories
- **Task-oriented** (in closed domain) vs **Non-task-oriented** (i.e., chat bots [8], in open domain) [1]
- **Generative** vs **retrieval-based**

# 1. Task-oriented
## 1.1 Pipeline
  - ASR -> NLU -> DM -> NLG
  - NLU: maps the utterance into semantic slots
    - **Domain identification**: text classification
    - **User domain specific intent detection**: text classification (We can incorporate preceding text information during perform classification, see [7])
    - **Slot filling**: sequence labeling
        - **Parameters** of the INTENT
        - Since **slots are intent-specific**, we can [**Joint learning of intent identification and slot filling**](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)
    - Industrial
        - To avoid OOV
            - Character-based classification and sequence labeling
            - Fasttext sub n-gram
            - Combine CNN or LSTM on character embedding and word embedding
        - New intent or new domain
            - Zero-shot learning
            ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/Rokid_new_intent.png)
  - DM: Dialogue Management
    - **Dialogue State Tracking**: estimates the goal at every turn. A probability distribution over each slot for each turn.
        
    - **Policy Learning**: generates next action
        - Supervised learning
        - Reinforcement learning
            - Entity: **agent** and **environment** 
            - Rule: agent takes **action** and environment gives **reward** and **state**
            - **Policy**: a rule that the agent should follow to select actions given the current state
    - [Models](https://www.bilibili.com/video/av30291543?from=search&seid=6125249948345166413)
        - **Rule-based** is popular in production system [8]
            - [Frame based](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec10.pdf)
![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/DM_Frame_based.png)
            - Finite state machine
                - Node is action
                - Edge is slot status
        ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/DM_Finite_state_machine.png)
            - Agenda based
            ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/DM_Agenda_based.png)
        - [Markov Decision Process](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec11.pdf)
            - Components: **Agent, Environment, State, Action, Reward**
            - Goal: the agent’s goal is to **maximize the cumulative rewards** (i.e., expected *discounted* (i.e., care more about the immediate reward over future rewards, discount rate γ) return of rewards)
            - **Policy**: what's the probability for an agent to select any **action** from a given state ?
            - **Value function**: how good is any given action or any given state for an agent ?
            - **Q-function**: input the state-atcion pair, output the **Q-value**. The letter **“Q”** is used to represent the **quality** of taking a given action in a given state.
            - **Bellman optimality equation**
                - ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/Bellman_optimality_equation.png)
                - For any state-action pair `(s,a)` at time `t`, the expected return is `R_(t+1)` (i.e. the expected reward we get from taking action `a` in state `s`) + the maximum expected discounted return that can be achieved from any possible next state-action pair.
            - **Q-learning**
                - It is used for learning the **optimal policy** by learning the optimal Q-values for each state-action pair in a Markov Decision Process
                - **Q-table**: store the Q-values for each state-action pair, the dimension is **(#states * #actions)**
                - Steps
                    - Initialize all Q-values in the Q-table to 0
                    - For each time-step in each episode:
                        - Choose an action (using the exploration-exploitation trade-off, i.e., **epsilon greedy** strategy)
                            - `if random_num > epsilon:` choose action via **exploitation**: choose the action with the highest Q-value for its current state
                            - `else:` choose action via **exploration**: randomly choosing action
                        - Update the Q-value function
                            - ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/Q-value_formula.png)
                            - where α is the **learning rate**, γ is the **discount rate**
            - **DQN**
                - Use a neural network to approximate the Q-function
                - **Input** the state, **output** the Q-values for each action that can be taken from that state, the **loss** is the gap between the output Q-values (by policy network) and the target Q-values (by target network) from Bellman equation
                - **Experience replay**: store the agent’s experiences at each time step called the **replay memory** data set, which stores the last `N` experiences. At time `t`, the agent's experience `e_t` is defined as this tuple: `e_t=(s_t, a_t, r_t+1, s_t+1)`
                - **Train** the network with **randomly choose** the samples in replay memory to **break the correlation between consecutive samples**
                - **Target network**
                    - Clone of the policy network, weights are frozen with policy network’s weights, and after `x` time steps, copy policy network's weights to target network
                    - The goal of target network is find the value of the `max` term in Bellman euqation to calculate the target Q-value
 
        ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/DM_Markov_decision_process.png)
        - [Neural network](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec12.pdf)
        ![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/DM_End_to_end.png)
        
  - NLG: convert action to utterance
      - **Action - template**
      - **Hybrid model** [8]
          - If the user dialogue actions can be found in the predefined sentence templates, the **template-based NLG ** is applied; otherwise, the utterance is generated by the **model-based NLG**
## 1.2 End-to-end
  - see report of [Dr.Che](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/NLPCC2018_Day3_Tencent_workshop_chewangxiang.pdf)

# 2. Non-task-oriented
## 2.1 Retrieval-based 
  - Given pre-defined responses, retrieval based model predict one response given current input context [2]
  - Single/Multi turn response matching
  - Two steps [3]:  
    - retrieval top-k response candidates by directly matching
    - reranking and give best candidate by incorporing context
  - **Multi-turn**
    - [Sequential matching network](https://arxiv.org/abs/1612.01627), and other [state-of-the-art models](https://mp.weixin.qq.com/s/yC8uYwti9Meyt83xkmbmcg)
  - [AIML](https://medium.com/pandorabots-blog/aiml-tutorial-creating-a-context-aware-multi-functional-chatbot-e5e82c027a6a) for pattern matching
  
## 2.2 Generative  
  - Generative new responses from scratch [2]
  - **Multi-turn**
    - [HRED](https://arxiv.org/pdf/1507.02221.pdf) transfered from [**query suggestion**](https://arxiv.org/pdf/1507.02221.pdf), see [industrail application](https://mp.weixin.qq.com/s/6dKticG2I2zqlxnZ3W0ZgQ)
    - [VHRED](https://arxiv.org/abs/1605.06069)
    - [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/pdf/1507.04808.pdf)
    - [Context-sensitive ACL 2018](http://aclweb.org/anthology/C18-1206)
  - **Avoid generic responses**
    - [Mutual information](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec12.pdf)
  - [Cs224s slides](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec12.pdf)
  
  
## 2.3 Hybrid 
  - [Alime](http://www.aclweb.org/anthology/P/P17/P17-2079.pdf)
## Video open courses
- [How to use question to increase Chatbot's interactivity ?](https://mp.weixin.qq.com/s/5yEcgzhEVx0mlBFerZNTgQ)


# Leading Researchers
- [Minlie Huang](http://coai.cs.tsinghua.edu.cn/hml/dataset/)
- [Rui Yan](http://www.ruiyan.me/)
- [Nan Duan](https://www.microsoft.com/en-us/research/people/nanduan/)

# Dataset
- [Ubuntu Dialogue Corpus v2.0](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
- [A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)
- [Dialog State Tracking Challenge](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/)
- [Douban Conversation Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection#douban-conversation-corpus)
- [2017第六届全国社会媒体处理大会之中文人机对话技术评测](https://github.com/HITlilingzhi/SMP2017ECDT-DATA)

# Implementation
- [wxpy](https://github.com/youfou/wxpy)
- [wxBot](https://github.com/liuwons/wxBot)
- [WeRoBot](https://github.com/offu/WeRoBot): weChat bot
- [Awesome-Chatbot](https://github.com/fendouai/Awesome-Chatbot) gives a list of public repositories about chatbot

# Products
- [Baidu UNIT](https://ai.baidu.com/unit/home)
- [IP梦工厂](https://ipd.gowild.cn/login)
- [图灵机器人](http://www.turingapi.com/)
- [小i机器人](http://www.xiaoi.com/)
- [海知智能](http://ruyi.ai/)
- [XiaoIce](https://www.msxiaobing.com/)
- [Pandorabots](https://home.pandorabots.com/home.html)
- [Wit.ai](https://wit.ai/)
- [Rasa](https://www.rasa.com/)
    - `tensorflow_embedding` in NLU from [StarSpace: Embed All The Things!](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/StarSpace-Embed%20_All_The_Things!.pdf)
- [DialogFlow](https://dialogflow.com/)
- [ChatterBot](https://github.com/gunthercox/ChatterBot)
- [Rokid](https://www.rokid.com/cn/)
![](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/materials_others/Rokid_frame.png)
- [DeepPavlov](https://github.com/deepmipt/DeepPavlov)


# References
[1] [A Survey on Dialogue Systems: Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)  
[2] Deep Learning for Chatbots [part 1](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/), [part 2](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)  
[3] https://c.m.163.com/news/l/180148.html?spssid=fba792c95ad60299db5435a91da37e10&spsw=1&spss=other.  
[4] [A Network-based End-to-End Trainable Task-oriented Dialogue System](http://mi.eng.cam.ac.uk/~sjy/papers/wgmv17.pdf), and [implementation](https://github.com/shawnwun/NNDIAL), Ali Xiaomi  
[5] [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](https://arxiv.org/pdf/1606.01269.pdf), [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://arxiv.org/pdf/1702.03274.pdf), and [implementation](https://github.com/voicy-ai/DialogStateTracking)  
[6] [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/pdf/1605.07683.pdf) from Facebook, and [implementation](https://github.com/vyraun/chatbot-MemN2N-tensorflow)  
[7] [Sequential Short-Text Classification with
Recurrent and Convolutional Neural Networks](https://arxiv.org/pdf/1603.03827.pdf)  
[8] [“Chitty-Chitty-Chat Bot”: Deep Learning for Conversational AI](https://www.ijcai.org/proceedings/2018/0778.pdf)  
[9] [End-to-End Task-Completion Neural Dialogue Systems](http://aclweb.org/anthology/I17-1074), [implementation](https://github.com/MiuLab/TC-Bot)

