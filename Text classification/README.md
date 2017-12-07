Text classificaton (i.e., fastText and lstm) based on [Sogou](http://www.sogou.com/labs/resource/cs.php) corpus.

## fastText

### Principle
Averaged word (with n-grams) vectors + softmax [1].  
Just like Continuous BOW model, where the **center word** is replaced by the **label** now.  
In another view, fastText likes CNN configured with window size = `1` (unigram) or `n` (n-grams) and average pooling [2]. 

### 'Fast' tricks
- Hierarchical softmax
    - Reduce computational complexity from `O(k*h)` to `O(log(k)*h)`, where `k` is the number of categories and `h` is the hidden dimension.
- n-grams with hashing
    - Incorporate **word order** information.

### Improvement
- Incorporate POS information [from Alibaba meeting].

References:  
[1] [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
[2] https://www.zhihu.com/question/48345431  

