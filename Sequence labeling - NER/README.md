# Introduction
**Named entity recognition (NER)**, **Part-of-Speeching tagging (POS)** and **Chinese word segmentation** are tasks of **sequence labeling**. 
Here, I focus on **brands** NER.  
For example, if the input sentence is 'I bought a LiNing hat yesterday', then the model will recognize LiNing as the brand.

Without deep learning, statistical **linear-chain conditional random fields (linear-chain CRF)** is the favorable solution. 
But it requires hand-crafted features. In deep learning fields, Bi-directional LSTM + CRF gives the state-of-the-art performance.

# Model

