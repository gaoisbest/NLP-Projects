# Question answering
- KBQA
- FAQ
- MRC
## 1. Knowledge Based Question Answering (KBQA)

## 2. Retrieval Based FAQ [1]
### 2.1 Classification based
- Fist build a knowledge base, which includes several big categories, each category contain same questions and a answer
- Classify query into big categories, and we can return the answer of this category
- Drawbacks: must train a new classifier when adding a new category

### 2.2 Retrieval and reranking based
- 2.2.1 Process query: word segmentation, stop-words removal, **expansion of synonym**.
- 2.2.2 Retrieval
  - TFIDF
  - BM2.5
- 2.2.3 Rerank
  - [SIF](https://github.com/gaoisbest/NLP-Projects/blob/master/Text%20similarity/SIF.py)
  - Word Mover's Distance
  - ...
- Drawbacks: if the correct answer is not recalled ? 
- Learning to rank
  - Pointwise
  - Pairwise
  - Listwise
  
### 2.3 Direct sentence vector matching
- Matching on all questions, do not have recall problem in 1.2
- [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks, 2015](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf)
- [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, 2017](https://arxiv.org/pdf/1705.02364.pdf)
- **Sentence vector quick search tool**: [Faiss](https://github.com/facebookresearch/faiss)

## 3. Machine Reading Comprehension (MRC)
Please see [here](https://github.com/gaoisbest/NLP-Projects/blob/master/Machine_reading_comprehension/README.md).


# References
- [1] [全球互联网架构大会-对话机器人的最佳实践-助理来也](https://github.com/gaoisbest/NLP-Projects/blob/master/Dialog%20system/%E5%85%A8%E7%90%83%E4%BA%92%E8%81%94%E7%BD%91%E6%9E%B6%E6%9E%84%E5%A4%A7%E4%BC%9A_%E5%AF%B9%E8%AF%9D%E6%9C%BA%E5%99%A8%E4%BA%BA%E7%9A%84%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5_%E5%8A%A9%E7%90%86%E6%9D%A5%E4%B9%9F.pdf)  
