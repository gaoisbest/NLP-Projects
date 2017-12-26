# NLP-Projects
Natural Language Processing related projects, which includes concepts and srcipts about:
- Word2vec: `gensim`, `fastText` and `tensorflow` implementations. See [Chinese notes](http://url.cn/5PKmy7W), [中文解读](http://url.cn/5PKmy7W).
- Text similarity: `gensim doc2vec` and `gensim word2vec averaging` implementations. 
- Text classification: `tensorflow LSTM` (See [Chinese notes 1](http://url.cn/5cLDOQI), [中文解读 1](http://url.cn/5cLDOQI) and [Chinese notes 2](http://url.cn/5w5VbaI), [中文解读 2](http://url.cn/5w5VbaI)) and `fastText` implementations. 
- Chinese word segmentation: `HMM Viterbi` implementations. See [Chinese notes](http://url.cn/5x4KR8u), [中文解读](http://url.cn/5x4KR8u).
- Sequence labeling - NER: brands NER via bi-directional LSTM + CRF, `tensorflow` implementation. See [Chinese notes](http://url.cn/5fcC754), [中文解读](http://url.cn/5fcC754).
- ..

## DL best practices in NLP
### 1. Word embeddings
- Use **pre-trained embeddings** if available. 
- Embedding dimension is task-dependent
	- Smaller dimensionality (i.e., 100) works well for syntactic tasks (i.e., NER, POS tagging)
	- Larger dimensionality (i.e., 300) is useful for semantic tasks (i.e., sentiment analysis)

### 2. Depth
- 3 or 4 layer Bi-LSTMs (e.g. POS tagging, semantic role labelling). 
- 8 encoder and 8 decoder layers (e.g., Google's NMT)
- In most case, shallower model(i.e., 2 layers) is good enough.

### 3. Layer connections (for avoiding vanishing gradients)
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
### 4. Dropout
  - Batch normalization in CV likes dropout in NLP.
  - Dropout rate of **0.5** is perferred.
  - **Recurrent dropout** ([what's the difference between recurrent dropout and traditional dropout ?](https://stackoverflow.com/questions/47415036/tensorflow-how-to-use-variational-recurrent-dropout-correctly)) applies the same dropout mask across timesteps at layer *l*. Implementation: `tf.contrib.rnn.DropoutWrapper(variational_recurrent=True)`

Reference:  
http://ruder.io/deep-learning-nlp-best-practices/
