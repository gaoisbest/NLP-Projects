# Introduction
Deep semantic match has broad applications in **information retrieval**, **question answering**, **dialog system**, **paraphrase**, etc.

# Models
## Statistical models
- TFIDF
- BM25
- Query likelihood
- Jaccard distance

## Deep learning models
- **Interaction-based** is **perfered**
    - ARC-II
    - DeepMatch
    - MatchPyramid
    
- **Representation-based**
    - DSSM
    - CDSSM
    - ARC-I
- Implementation
    - [MatchZoo](https://github.com/NTMC-Community/MatchZoo)
    
# [Evaluation metrics](https://github.com/gaoisbest/NLP-Projects/blob/master/Information_retrieval/materials_papers/IR_EvaluationMetrics.pdf)
- Set retrieval
    - Precision
        - Proportion of retrieved documents that are relevant, `|retrieved & relevant| / |retrieved|`
    - Recall
        - Proportion of relevant documents that are retrieved, `|retrieved & relevant| / |relevant|`
    - F1
        - Harmonic mean of precision and recall
- Ranked retrieval
    - Precision@k
        - Proportion of top-k documents that are relevant
    - Recall@k
        - Proportion of relevant documents that are in the top-k
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/Information_retrieval/materials_papers/Precision_at_k_recall_at_k_example.png)
    
    - Precision-recall curve
    
    - Average precision
        - Average all precision@k values when recall is 1.0
        - **Mean Average Precision (MAP)**
            - Take mean of average precisions across a set of queries
        - Example: Number of relevant documents for thisquery is 10
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/Information_retrieval/materials_papers/Average_precision_part_1.png)
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/Information_retrieval/materials_papers/Average_precision_part_2.png)
    
    - Discounted Cumulative Gain (DCG)
        - Assumption: more than two levels of relevance
        - **Normalized Discounted Cumulative Gain (NDCG)**
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/Information_retrieval/materials_papers/Discounted_cumulative_gain.png)
    

# Tutorials
- [SIGIR 2018 Deep Learning for Matching in Search and Recommendation](https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf)
- [《搜索与推荐中的深度学习匹配》之搜索篇](https://zhuanlan.zhihu.com/p/38296950)
- [基于GRU和am-softmax的句子相似度模型](https://spaces.ac.cn/archives/5743)
- [Top competition solutions](https://github.com/Smilexuhc/Data-Competition-TopSolution)
- [问题对语义相似度计算-参赛总结](https://mp.weixin.qq.com/s?__biz=MzU4OTczNTg2OQ==&mid=2247484262&idx=1&sn=d5fb0df4e5aa641065c1b8c2df4142ef&chksm=fdc9b1b4cabe38a2103e5082ed2abccd7b375d79bafceec74576250d6a0dbb0c2e8f28a0ef81&mpshare=1&scene=23&srcid=0106iH7dvawP51FBLHUJ2Chb)

# Industrial applications
- [Sogou IR](https://mp.weixin.qq.com/s/9RFp4-2M0FSgD2A2LE7scA)
- [Baidu SimNet](https://github.com/baidu/AnyQ)
- [Yunzhisheng](https://mp.weixin.qq.com/s/TjM_8FqKYX8KMLFo4TtNtA)
- [Baidu AnyQ](https://github.com/baidu/AnyQ)

