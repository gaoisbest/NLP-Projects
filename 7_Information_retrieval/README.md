# Introduction
- Deep semantic match has broad applications in **information retrieval**, **question answering**, **dialog system**, **paraphrase**, etc.
- In production (e.g., retrieval-based chatbot), **retrieval** (e.g., Elasticsearch) **+ rerank** (e.g., deep matching model) is two steps to perform information retrieval


# Models
## Statistical models
- TFIDF
- [BM25 (BestMatch)](https://en.wikipedia.org/wiki/Okapi_BM25)
    - `k1` is term frequency saturation (词频饱和度), value between `[1.2, 2.0]`
    - `b` is field length normalization, value between `[0, 1]`, usually `0.75`
- Query likelihood
- [Jaccard](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50)
    - Size of intersection divided by size of union of two sets
    - Word duplication does not matter
    - 
    ```
    def jaccard_sim(str1, str2): 
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    ```

## Deep learning models
- **Interaction-based** is **perfered**
    - ARC-II
    - DeepMatch
    - MatchPyramid
    
- **Representation-based**
    - DSSM
    - CDSSM
    - ARC-I
- **[Model Review: A Deep Look into Neural Ranking Models for Information Retrieval](https://arxiv.org/pdf/1903.06902.pdf)**
- Implementation
    - [MatchZoo](https://github.com/NTMC-Community/MatchZoo)
    - [Annoy](https://github.com/spotify/annoy)
    - [Faiss](https://github.com/facebookresearch/faiss)
## Learning to Rank
- Use machine learning models to build ranking models
- Features
    - BM25
    - PageRank
    - Edit distance
- Categories
    - Pointwise
    - Pariwise
    - Listwise
- LambdaMART
# [Evaluation metrics](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_papers/IR_EvaluationMetrics.pdf)
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
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_papers/Precision_at_k_recall_at_k_example.png)
    
    - Precision-recall curve
    
    - Average precision
        - Average all precision@k values when recall is 1.0
        - **Mean Average Precision (MAP)**
            - Take mean of average precisions across a set of queries
        - Example: Number of relevant documents for thisquery is 10
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_papers/Average_precision_part_1.png)
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_papers/Average_precision_part_2.png)
    
    - Discounted Cumulative Gain (DCG)
        - Assumption: more than two levels of relevance
        - **Normalized Discounted Cumulative Gain (NDCG)**
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_papers/Discounted_cumulative_gain.png)

# Books
- [Learning to Rank for Information Retrieval](https://www.cda.cn/uploadfile/image/20151220/20151220115436_46293.pdf) by Tie-Yan Liu, 2015
- [Learning to Rank for Information Retrieval and Natural Language Processing](http://www.iro.umontreal.ca/~nie/IFT6255/Books/Learning-to-rank.pdf) by Hang Li, 2011

# Tutorials
- [SIGIR 2018 Deep Learning for Matching in Search and Recommendation](https://www.comp.nus.edu.sg/~xiangnan/sigir18-deep.pdf)
- [《搜索与推荐中的深度学习匹配》之搜索篇](https://zhuanlan.zhihu.com/p/38296950)
- [基于GRU和am-softmax的句子相似度模型](https://spaces.ac.cn/archives/5743)
- [Top competition solutions](https://github.com/Smilexuhc/Data-Competition-TopSolution)
- [问题对语义相似度计算-参赛总结](https://mp.weixin.qq.com/s?__biz=MzU4OTczNTg2OQ==&mid=2247484262&idx=1&sn=d5fb0df4e5aa641065c1b8c2df4142ef&chksm=fdc9b1b4cabe38a2103e5082ed2abccd7b375d79bafceec74576250d6a0dbb0c2e8f28a0ef81&mpshare=1&scene=23&srcid=0106iH7dvawP51FBLHUJ2Chb)
- [平安人寿-智能问答系统的探索与实践](https://www.bilibili.com/video/av46314741/), [pdf](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/materials_reports/%E5%B9%B3%E5%AE%89%E4%BA%BA%E5%AF%BF_%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F%E7%9A%84%E6%8E%A2%E7%B4%A2%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)

# Industrial applications
- [Sogou IR](https://mp.weixin.qq.com/s/9RFp4-2M0FSgD2A2LE7scA)
- [Baidu SimNet](https://github.com/baidu/AnyQ)
- [Yunzhisheng](https://mp.weixin.qq.com/s/TjM_8FqKYX8KMLFo4TtNtA)
- [Baidu AnyQ](https://github.com/baidu/AnyQ)

