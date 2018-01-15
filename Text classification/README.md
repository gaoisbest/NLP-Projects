Text classificaton (i.e., fastText and lstm) based on [Sogou](http://www.sogou.com/labs/resource/cs.php) corpus.

## fastText

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


## Convolutional Neural Networks for text classification
### Principle
- Converting the sentence to a tensor of shape [height=seq_len, width=embedding_size, channels=1] with word embedding. 
- Convolution and max-pooling on the tensor.
- Fully-connected with softmax.

### Model
#### Structure and Dimensions
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

### References
[1] http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/  
[2] https://github.com/gaussic/text-classification-cnn-rnn
