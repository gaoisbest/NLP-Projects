# Introduction
**Named entity recognition (NER)**, **Part-of-Speeching tagging (POS)** and **Chinese word segmentation** are tasks of **sequence labeling**.  
Without deep learning, statistical **linear-chain conditional random fields (linear-chain CRF)** is the favorable solution. 
But it requires hand-crafted features. In deep learning fields, bi-directional LSTM + CRF gives the state-of-the-art performance, which is 
discussed in detail here.

# Model
### Model architecture
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Model_architecture.png)

The above image from [1] shows the model architecture. Word embedding (Word emb.) together with character representation (Char rep.) are as inputs to bi-directional LSTM layer. Then the outputs of both forward LSTM and backward LSTM (which encodes the contextual word representation [2]) are concatenated as inputs to the CRF layer. And the CRF layer gives the final prediction.  

CNN [3] or bi-directional LSTM [4] can be used to obtain character representation. See the image from [1] below:  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Character_representation.png)

### Model parameters
- Word embeddings
- Tags transition matrix `A`
- Matrix `P` related parameters: recurrent units of bi-directional LSTM, linear projection weights

# Implementation
Here, I focus on **brands** NER.  
For example, if the input sentence is 'I bought a LiNing hat yesterday', then the model will recognize LiNing as the brand.

# References
[1] [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf)
[2] https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
[3] [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)
[4] [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
