# Outline
- Text classification models
    - [Sogou corpus](http://www.sogou.com/labs/resource/cs.php)
    - fastText
    - RNNs
    - CNNs
    - HAN
    - Discussion
- Sentiment analysis
- Unbalanced dataset

## 1. fastText
### Principle
- Averaged word (with n-grams) vectors + softmax [1].   
- Just like Continuous BOW model, where the **center word** is replaced by the **label** now.   
- In another view, fastText likes CNN configured with window size = `1` (unigram) or `n` (n-grams) and average pooling [2]. 

### Tricks
- Fast: [**hierarchical softmax**](https://becominghuman.ai/hierarchical-softmax-as-output-activation-function-in-neural-network-1d19089c4f49)
    - Huffman tree. 
    - Reduce computational complexity from `O(k*h)` to `O(log(k)*h)`, where `k` is the number of categories and `h` is the hidden dimension.
    - At test time, each node has a probability. DFS and track maximum probability give top-1 prediction. With *binary heep*, top-T predictions are calculated at the cost of `O(log(T))`.
- Accuracy: **n-grams** with *hashing trick*
    - Incorporate **word order** information. Higher order grams (i.e., bigrams, trigrams, n-grams=5) performs better.
    - 10M bins for bigrams, and 100M for n-grams [1].    
    - These n-grams in the same bin share the embedding vector [3].

### Future Direction
- Incorporate POS information [from Alibaba meeting].

### References
[1] [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
[2] https://www.zhihu.com/question/48345431  
[3] http://albertxiebnu.github.io/fasttext/


## 2. RNNs
- `LSTM_text_classification_version_1.ipynb`. See [Chinese notes](http://url.cn/5cLDOQI), [中文解读](http://url.cn/5cLDOQI).
- `LSTM_text_classification_version_2.ipynb`. See [Chinese notes](http://url.cn/5w5VbaI), [中文解读](http://url.cn/5w5VbaI).
- Concatenate **character** features and **word** features together to feed to FC.
- To be done: **LSTM + Attention, Bidirectional LSTM + Attention**

### 2.1 Multiplicative LSTM
[Openai's](https://blog.openai.com/unsupervised-sentiment-neuron/) work that finds a **sentiment neuron**. The model consists of unsupervised language model + Logistic regression with L1 regularization.  

### References
[1] https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/  
[2] http://tobiaslee.top/2017/08/29/Attention-based-LSTM-for-Text-Classification/  
[3] https://www.cloudsek.com/announcements/blog/hierarchical-attention-text-classification/  


## 3. CNNs
### 3.1 Plain CNNs
#### Principle
- Converting the sentence to a tensor of shape [height=seq_len, width=embedding_size, channels=1] with word embedding. 
- Convolution and max-pooling on the tensor.
- Fully-connected with softmax.

#### Model
##### Structure and Dimensions
- Input: `(batch_size, height=seq_length, width=embedding_size, channels=1)`. `tf.nn.embedding_lookup`, `tf.expand_dims`
- for f in filter_sizes:
    - Convolution `tf.nn.conv2d`
        - **Conv - add bias - ReLU**
        - Filter: `(filter_height=f, filter_width=embedding_size, in_channels=1, out_channels=num_filters)`
        - Output tensor: `(batch_size, seq_length-filter_size+1 (stride=1), 1, num_filters)`
    - Max-pool `tf.nn.max_pool`
        - ksize: `[1, seq_length-filter_size+1, 1, 1]`
        - Output tensor: `(batch_size, 1, 1, num_filters)` 
- Concatenate output tensor for each filter_size to `(batch_size, 1, 1, len(filter_sizes)*num_filters)` and `tf.reshape` to `(batch_size, len(filter_sizes)*num_filters)`
- FC1 with drop-out
    - `(batch_size, len(filter_sizes)*num_filters)`
- FC2
    - `(batch_size, num_classes)`
#### Implementation
- `CNN_text_classification.ipynb`. See [Chinese notes](http://url.cn/5kW61T4), [中文解读](http://url.cn/5kW61T4).

#### References
[1] http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/  
[2] https://github.com/gaussic/text-classification-cnn-rnn

### 3.2 CNNs with k-max pooling
- To do
### 3.3 DPCNN
- To do

## 4. [HAN](https://github.com/gaoisbest/NLP-Projects/blob/master/4_Text_classification/materials_papers/HAN.pdf)

#### References
[1] 

## 5. Discussion
- For long sentence, CNNs is better than RNNs [1].
- **Long-term dependency** is **not significant** for text classification problem [2].
- [Recursive NNs](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) incorporate **syntax** information.
- Tricks
    - Normalization
    - Dynamic max-pooling
- We can calling free api (i.e., ai.baidu.com) to build trainging data, see an [example](https://github.com/gaoisbest/NLP-Projects/blob/master/Text%20classification/Create_training_data_by_calling_free_baidu_api.py)
### References
[1] https://www.zhihu.com/question/41625896  
[2] https://hanxiao.github.io/2018/06/25/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/?from=timeline&isappinstalled=0  

# Tutorials
- [1] [Data grand competition defense](https://github.com/gaoisbest/NLP-Projects/blob/master/4_Text_classification/materials_papers/%E8%BE%BE%E8%A7%82%E5%88%86%E7%B1%BB%E7%AB%9E%E8%B5%9B%E7%AD%94%E8%BE%A9ppt.pdf)

# Sentiment analysis
## Tools
- [Baidu Senta](https://github.com/baidu/Senta)

# Unbalanced dataset
- Data
    - Up-sampling of minor class
        - Replication same samples
        - Synthesis by SMOTE
        - Data augmentation by replacing verb or adjective
    - Down-sampling of major class
- Model
    - Ensemble
        - Bagging and Boosting
    - One-class SVM
- Loss function
    - [Class_weight](https://blog.mimacom.com/text-classification/) is used as parameters to weight loss, **weight less frequent classes higher than very frequent classes**
    ```
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    ```
    - [Focal loss](https://ldzhangyx.github.io/2018/11/16/focal-loss/), [implementation 1](https://github.com/Hsuxu/FocalLoss-PyTorch), [implementation 2](https://blog.csdn.net/Umi_you/article/details/80982190)
    ```
    # multi-class
    fl = -(1-pt)^gamma * logpt
    ```
- Evaluation metric
    - Default decision threshold is not valid for imbalanced data
    - [AUC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) gives performance across the whole range of decision thresholds, ROC curves are typically used in **binary classification**

## Evaluation metric
- TPR (True Positive Rate) = # True positives / # positives = Recall = TP / (TP+FN)
- FPR (False Positive Rate) = # False Positives / # negatives = FP / (FP+TN)
- Precision =# True positives / # predicted positive = TP/(TP+FP)
- Recall = # True positives / # positives = TP / (TP+FN)
- Recall = True Positive Rate

### References
[1] https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/  
[2] https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/
