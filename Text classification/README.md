Text classificaton (i.e., fastText and lstm) based on [Sogou](http://www.sogou.com/labs/resource/cs.php) corpus.

## fastText

### Principle
Averaged word (with n-grams) vectors + softmax.

### 'Fast' tricks
- Hierarchical softmax
    - Reduce computational complexity from `O(k*h)` to `O(log(k)*h)`, where `k` is the number of categories and `h` is the hidden dimension.

References:  
[1] [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
