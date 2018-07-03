# Dialog system categories
- Question answering
- Task-oriented (in closed domain)
- Non-task-oriented (i.e., chat bots, in open domain)

# Question answering
## Classification based
- In first step, classify query into big categories.

## Retrieval and reranking based
- Process query: word segmentation, stop-words removal, **expansion of synonym**.
- Retrieval
  - TFIDF
  - BM2.5
- Rerank
  - [SIF](https://github.com/gaoisbest/NLP-Projects/blob/master/Text%20similarity/SIF.py)
  - Word Mover's Distance
  - [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks, 2015](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf)
  - [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, 2017](https://arxiv.org/pdf/1705.02364.pdf)
  - ...
## Direct sentence vector matching
  - **Sentence vector quick search tool**: [Faiss](https://github.com/facebookresearch/faiss)

# Task-oriented
- Pipeline
  - NLU -> Dialog state tracking -> Policy learning -> NLG
  - NLU: maps the utterance into semantic slots
  - DST: estimates the goal at every turn
  - PL: generates next action, can be done by supervised learning or reinforcement learning
  - NLG: convert action to utterance
- End-to-end
  - ...
## Reinforcement learning
  - Entity: **agent** and **environment** 
  - Rule: agent takes **action** and environment gives **reward** and **state**
  - **Policy**: a rule that the agent should follow to select actions given the current state
# Non-task-oriented
- Retrieval  
  - Given pre-defined responses, retrieval based model predict one response given current input context [2].
  - Two steps [3]:  
    - retrieval top-k response candidates by directly matching
    - reranking and give best by incorporing context
- Generative  
  - Generative new responses from scratch [2].

# Implementation
- [ChatterBot](https://github.com/gunthercox/ChatterBot) supplies a framework for building chatbot, and [Awesome-Chatbot](https://github.com/fendouai/Awesome-Chatbot) gives a list of public repositories about chatbot.
- [wxpy](https://github.com/youfou/wxpy), [wxBot](https://github.com/liuwons/wxBot), [WeRoBot](https://github.com/offu/WeRoBot): weChat bot.

# References
[1] [A Survey on Dialogue Systems: Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf)  
[2] Deep Learning for Chatbots [part 1](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/), [part 2](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)  
[3] https://c.m.163.com/news/l/180148.html?spssid=fba792c95ad60299db5435a91da37e10&spsw=1&spss=other.  
[4] [A Network-based End-to-End Trainable Task-oriented Dialogue System](http://mi.eng.cam.ac.uk/~sjy/papers/wgmv17.pdf), and [implementation](https://github.com/shawnwun/NNDIAL), Ali Xiaomi  
[5] [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](https://arxiv.org/pdf/1606.01269.pdf), [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://arxiv.org/pdf/1702.03274.pdf), and [implementation](https://github.com/voicy-ai/DialogStateTracking)  
[6] [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/pdf/1605.07683.pdf) from Facebook, and [implementation](https://github.com/vyraun/chatbot-MemN2N-tensorflow)

