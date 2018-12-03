# NLP-Projects
Natural Language Processing related projects, which includes concepts and srcipts about:
- Word2vec: `gensim`, `fastText` and `tensorflow` implementations. See [Chinese notes](http://url.cn/5PKmy7W), [中文解读](http://url.cn/5PKmy7W).
- Sentence2vec: `doc2vec`, `word2vec averaging` and `Smooth Inverse Frequency` implementations. See 
- Text_classification: `tensorflow LSTM` (See [Chinese notes 1](http://url.cn/5cLDOQI), [中文解读 1](http://url.cn/5cLDOQI) and [Chinese notes 2](http://url.cn/5w5VbaI), [中文解读 2](http://url.cn/5w5VbaI)) and `fastText` implementations. 
- Chinese_word_segmentation: `HMM Viterbi` implementations. See [Chinese notes](http://url.cn/5x4KR8u), [中文解读](http://url.cn/5x4KR8u).
- Sequence_labeling-NER: brands NER via bi-directional LSTM + CRF, `tensorflow` implementation. See [Chinese notes](http://url.cn/5fcC754), [中文解读](http://url.cn/5fcC754).
- Machine_reading_comprehension: [introduction](https://github.com/gaoisbest/NLP-Projects/blob/master/Machine_reading_comprehension/README.md), `BiDAF+ELMo` implementation.
- Knowledge_graph: [introduction](https://github.com/gaoisbest/NLP-Projects/blob/master/Knowledge_graph/README.md)

# DL best practices in NLP
## 1. Word embeddings
- Use **pre-trained embeddings** if available. 
- Embedding dimension is task-dependent
	- Smaller dimensionality (i.e., 100) works well for syntactic tasks (i.e., NER, POS tagging)
	- Larger dimensionality (i.e., 300) is useful for semantic tasks (i.e., sentiment analysis)

## 2. Depth
- 3 or 4 layer Bi-LSTMs (e.g. POS tagging, semantic role labelling). 
- 8 encoder and 8 decoder layers (e.g., Google's NMT)
- In most case, shallower model(i.e., 2 layers) is good enough.

## 3. Layer connections (for avoiding vanishing gradients)
- Highway layer
	- `h = t * a(WX+b) + (1-t) * X`, where `t=sigmoid(W_TX+b_T)` is called **transform** gate.
	- Application: language modelling and speech recognition.
	- Implementation: `tf.contrib.rnn.HighwayWrapper`
- Residual connection
	- `h = a(WX+b) + X`
	- Implementation: `tf.contrib.rnn.ResidualWrapper`
- Dense connection
	- `h_l = a(W[X_1, ..., X_l] + b)`
	- Application: multi-task learning
## 4. Dropout
- Batch normalization in CV likes dropout in NLP.
- Dropout rate of **0.5** is perferred.
- **Recurrent dropout** ([what's the difference between recurrent dropout and traditional dropout ?](https://stackoverflow.com/questions/47415036/tensorflow-how-to-use-variational-recurrent-dropout-correctly)) applies the same dropout mask across timesteps at layer *l*. Implementation: `tf.contrib.rnn.DropoutWrapper(variational_recurrent=True)`

## 5. LSTM tricks
- Treat initial state as variable [2]
```
# note: if here is LSTMCell, a bug appear: https://stackoverflow.com/questions/42947351/tensorflow-dynamic-rnn-typeerror-tensor-object-is-not-iterable
cell = tf.nn.rnn_cell.GRUCell(state_size)
init_state = tf.get_variable('init_state', [1, state_size], initializer=tf.constant_initializer(0.0))
# https://stackoverflow.com/questions/44486523/should-the-variables-of-the-initial-state-of-a-dynamic-rnn-among-the-inputs-of
init_state = tf.tile(init_state, [batch_size, 1])
```
- Gradients clipping
```
variables = tf.trainable_variables()
gradients = tf.gradients(ys=cost, xs=variables)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
optimize = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, variables), global_step=self.global_step)
```

## 6. Attention
The attention [review 1](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) and [review 2](https://zhuanlan.zhihu.com/p/31547842) summarize attention mechanism into several types:
- Additive vs Multiplicative attention
- Self attention
- Soft vs Hard attention
- Global vs Local attention

## Reference
[1] http://ruder.io/deep-learning-nlp-best-practices/  
[2] https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html  

# Awesome packages
## Chinese
- [pyltp](http://pyltp.readthedocs.io/zh_CN/develop/api.html)
- [HanLP](http://hanlp.linrunsoft.com/index.html)

## English
### 1. [Spacy](https://spacy.io)
### 2. [gensim](https://radimrehurek.com/gensim/)

# Awesome public apis
- [Baidu AI Open Platform](https://ai.baidu.com/)
- [Tencent AI Open Platform](https://ai.qq.com/)
- [Tencent NLP](http://nlp.qq.com/)
