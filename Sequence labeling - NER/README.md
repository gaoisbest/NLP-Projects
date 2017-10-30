# Introduction
**Named entity recognition (NER)**, **Part-of-Speeching tagging (POS)** and **Chinese word segmentation** are tasks of **sequence labeling**. 
Here, I focus on **brands** NER.  
For example, if the input sentence is 'I bought a LiNing hat yesterday', then the model will recognize LiNing as the brand.

Without deep learning, statistical **linear-chain conditional random fields (linear-chain CRF)** is the favorable solution. 
But it requires hand-crafted features. In deep learning fields, bi-directional LSTM + CRF gives the state-of-the-art performance, which is 
discussed in detail here.

# Model
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Model_architecture.png)

The above image from [1] shows the model architecture. Word embedding (Word emb.) together with character representation (Char rep.) are as inputs to bi-directional LSTM layer. Then the outputs of both forward LSTM and backward LSTM are concatenated as inputs to the CRF layer. And the CRF layer gives the final prediction.  

CNN or bi-directional LSTM can be used to obtain character representation. See the image from [1] below:  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/Character_representation.png)



# References
[1] [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf)
