# Introduction
**Named entity recognition (NER)**, **Part-of-Speeching tagging (POS)** and **Chinese word segmentation** are tasks of **sequence labeling**.  
Without deep learning, statistical **linear-chain conditional random fields (linear-chain CRF)** is the favorable solution. 
But it requires hand-crafted features. In deep learning fields, bi-directional LSTM + CRF gives the state-of-the-art performance, which is 
discussed in detail here.

# Model
### Model architecture
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Model_architecture.png)

The above edited image is from [1] shows the model architecture.  
**Embedding layer**: Word embedding (Word emb.) together with character representation (Char rep.) are as inputs to bi-directional LSTM layer.   
**Bi-directional LSTM layer**: Then the outputs of both forward LSTM and backward LSTM (which encodes the contextual word representation [2]) are concatenated as inputs to the CRF layer.  
**CRF layer**: And the CRF layer gives the final prediction.  

For embedding layer, **CNN** (to extract morphological information, such as prefix or suffix [3]) or **bi-directional LSTM** [4] can be used to obtain character representation. As to the performance, they have **no significant difference** [1]. See the image from [1] below:  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Character_representation.png)

### Model formula
For an input sequence `X`, which has `n` characters (i.e., ranges from `0` to `n-1`), and it has been padded with the `start` and `end` symbols.  
$$X = (start, x_{0}, x_{1}, x_{2}, ..., x_{n-1}, end)$$  

For the sequence predictions `y`, which also been padded with the `start_tag` and `end_tag`.  
$$y = (startTag, y_{0}, y_{1}, y_{2}, ..., y_{n-1}, endTag)$$  

Paper [4] define its score to be  
$$s(X, y) = \sum\limits_{i = -1}^{n} A_{yi, yi+1} + \sum\limits_{i=-1}^{n} P_{i, yi}$$  

`P` (shape of [n+2, k+2], 2 means the padded sequence and tags) is the score matrix output by the bi-directional LSTM. And $P_{i,j}$ is the score of the $j^{th}$ tag of $i^{th}$ character.  
`A` is the tag transition score matrix. And $A_{i,j}$ means the transition score from the tag $i$ to tag $j$.  

The softmax over all possible tag sequences gives the probability for the sequence $y$:  
$$p(y|X) = \frac{e^{s(X, y)}}{\sum\limits_{y_{tmp} \in Y_X} e^{s(X, y_{tmp})}}$$  

The goal is maximizing the log-probability of the correct tag sequence:  
$$\log p(y|X) = s(X,y) - \log\large{(}\sum\limits_{y_{tmp} \in Y_X} e^{s(X, y_{tmp})} \large{)}$$  

This can be solved by Forward-Backward (for computing probability) and Viterbi (for decoding best tags) algorithms.

**For Chinese NER**, paper [5] proposes that character embedding are composed of **pre-trained character embedding** and embedding that learned by bi-directional lstm on radical features (偏旁部首).

### Model parameters
- Word embeddings
- Tags transition matrix `A`
- Matrix `P` related parameters: recurrent units of bi-directional LSTM, linear projection weights

However, part of model parameters or hyper-parameters are more important, paper [1] gives a detailed assessment, and the results are:  
- Major impact
  - Pre-trained word embedding
  - Gradient clipping
- Minor impact
  - Number of bi-directional LSTM layer
  - Number of recurrent units, default 100 is preferred
  - Mini-batch size: 8 (for small datasets), 32 (for large datasets)
  - Tagging schema: IOBES

# Implementation
Here, I focus on **brands** NER. For shoes, li Ning (李宁), adidas, nike are brands. For jewelry, ross or 周生生 are brands. So, my goal is finding the brands mentioned in the sentence.  

For example, if the input sentence is 'I bought a Li Ning (李宁) hat yesterday', then the model will recognize Li Ning (李宁) as the brand.  

There are total eight categories here (i.e., shoe, digital, jewelry, bag, facial mask, watch, clothes, wine) and each category contains several brands.  

The initial source codes are from [6], the following shows the **key points** about the codes:
- The model input is the concatenation of character embedding and word segmentation embedding. How to incorporate word segmentation information ?
  - perform Chinese word segmentation.
  - encoding rule: BIES format (i.e., B:1, I:2, E:3, S:0).
  - For example: `u'我买了富士康手机'`-> `u'我 买 了 富士康 手机'` -> `[0, 0, 0, 1, 2, 3, 1, 3]`

- Dropout on character embedding
  - according to Section 4.3 in paper [4]. 
  - `rnn_inputs = tf.nn.dropout(x=char_word_embeddings, keep_prob=self.keep_prob, name='lstm_inputs_dropout')`

- How to get actual sequence length in one batch ? 
  - `seq_lengths = tf.cast(tf.reduce_sum(input_tensor=tf.sign(tf.abs(self.char_inputs)), axis=1), tf.int32)`

- How to initialize the pre-trained character embedding ? Please see `create_model` method in `train.py`.
  - `old_weights = sess.run(model.char_embeddings.read_value())`
  - `new_weights = load_pre_trained_embedding(old_weights)`
  - `sess.run(model.char_embeddings.assign(new_weights))`
  
- Outputs of bi-directional LSTM are concatenated into prejection layer.
  - `outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cells['forward'], cell_bw=rnn_cells['backward'], inputs=rnn_inputs, sequence_length=seq_lengths, dtype=tf.float32)`
  - `return tf.concat(outputs, axis=-1)`
- OOV problem
  - If the test character is not in `char_to_id` dictionary, then convert it to `<UNK>` symbol.
  - `[char_to_id[char] if char in char_to_id else char_to_id['<UNK>'] for char in line]`
- How to perpare mini-batch data ?
  - For minimum padding, the data should be sorted first.
  - see `BatchManager` method in `utils_train.py`

I **revised** the codes as follows:  
- Add `end_logits` relevant codes in `cost_layer` function in `model.py`, which gives a relative better F1 score. For more details, please see my answer to the [issue](https://github.com/zjy-ucas/ChineseNER/issues/10).
- Add `softmax` classifier besides `CRF` classifier. Although `CRF` classifier gives better results, the training speed of `softmax` classifier are faster.
- Add `tf.summary` and detailed code comments.
- Rearrange code files.

### Data
The brands NER training data are from crawled Weibo. Please see sample training, development and testing data in `data` folder for input format.

### Try to run
- First, `BrandsNERModel_Train.ipynb` is used to train the model.
- Second, `BrandsNERModel_Test.ipynb` is used to test the model.

### Example results
- `{'entities': [{'word': '特步', 'type': 'SHOE', 'start': 3, 'end': 5}], 'string': '我买了特步鞋'}`  
- `{'entities': [{'word': '一叶子', 'type': 'FACIAL_MASK', 'start': 0, 'end': 3}], 'string': '一叶子面膜真不错'}`

### To be solved problem
- Please sovle my [question](https://stackoverflow.com/questions/47129405/how-to-perform-named-entity-disambiguation) if you have the solution, thanks.

# References
[1] [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf) and [implementation](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf)  
[2] https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html  
[3] [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)  
[4] [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)  
[5] [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_20)  
[6] [ChineseNER](https://github.com/zjy-ucas/ChineseNER)
