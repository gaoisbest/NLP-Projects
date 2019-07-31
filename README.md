# NLP-Projects
Natural Language Processing projects, which includes concepts and scripts about:
- [0_Word2vec](https://github.com/gaoisbest/NLP-Projects/blob/master/0_Word2vec/README.md)
    - `gensim`, `fastText` and `tensorflow` implementations. See [Chinese notes](http://url.cn/5PKmy7W), [中文解读](http://url.cn/5PKmy7W)

- [1_Sentence2vec](https://github.com/gaoisbest/NLP-Projects/blob/master/1_Sentence2vec/README.md)
    - `doc2vec`, `word2vec averaging` and `Smooth Inverse Frequency` implementations

- [2_Machine_reading_comprehension](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/README.md)

- [3_Dialog_system](https://github.com/gaoisbest/NLP-Projects/blob/master/3_Dialog_system/README.md)
    - Categories and components of dialog system

- [4_Text_classification](https://github.com/gaoisbest/NLP-Projects/blob/master/4_Text_classification/README.md)
    - `tensorflow LSTM` (See [Chinese notes 1](http://url.cn/5cLDOQI), [中文解读 1](http://url.cn/5cLDOQI) and [Chinese notes 2](http://url.cn/5w5VbaI), [中文解读 2](http://url.cn/5w5VbaI))
    - `fastText` implementation

- [5_Pretraining_LM](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/README.md)
    - Principle of ELMo, ULMFit, GPT, BERT, XLNet

- [6_Sequence_labeling](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/README.md)
    - [Chinese_word_segmentation](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/Chinese_word_segmentation/README.md)
        - `HMM Viterbi` implementations. See [Chinese notes](http://url.cn/5x4KR8u), [中文解读](http://url.cn/5x4KR8u)
    - [Named_Entity_Recognition](https://github.com/gaoisbest/NLP-Projects/tree/master/6_Sequence_labeling/Named_Entity_Recognition)
        - Brands NER via bi-directional LSTM + CRF, `tensorflow` implementation. See [Chinese notes](http://url.cn/5fcC754), [中文解读](http://url.cn/5fcC754)

- [7_Information_retrieval](https://github.com/gaoisbest/NLP-Projects/blob/master/7_Information_retrieval/README.md)

- [8_Information_extraction](https://github.com/gaoisbest/NLP-Projects/blob/master/8_Information_extraction/README.md)

- [9_Knowledge_graph](https://github.com/gaoisbest/NLP-Projects/blob/master/9_Knowledge_graph/README.md)

- [10_Text_generation](https://github.com/gaoisbest/NLP-Projects/blob/master/10_Text_generation/README.md)

- [11_Network_embedding](https://github.com/gaoisbest/NLP-Projects/blob/master/11_Network_embedding/README.md)


# Concepts
### 1. Attention
- Attention == **weighted averages**
- The attention [review 1](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) and [review 2](https://zhuanlan.zhihu.com/p/31547842) summarize attention mechanism into several types:
    - Additive vs Multiplicative attention
    - Self attention
    - Soft vs Hard attention
    - Global vs Local attention

### 2. CNNs, RNNs and Transformer
- **Parallelization** [1]
    - RNNs
        - Why not good ? 
	    - **Last step's output is input of current step**
	- Solutions
	    - **Simple Recurrent Units (SRU)**
	        - Perform parallelization on each hidden state neuron independently
	    - **Sliced RNNs**
	        - Separate sequences into windows, use RNNs in each window, use another RNNs above windows
	        - Same as CNNs
    - CNNs
        - Why good ?
	    - For different windows in one filter
	    - For different filters
	
- **Long-range dependency** [1]
    - CNNs
        - Why not good ?
	    - Single convolution can only caputure window-range dependency
	- [Solutions](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture11-convnets.pdf)
	    - Dilated CNNs
	    - Deep CNNs
	        - `N * [Convolution + skip-connection]`
	        - For example, window size=3 and sliding step=1, second convolution can cover 5 words (i.e., 1-2-3, 2-3-4, 3-4-5)
    - Transformer > RNNs > CNNs
- **Position** [1]
    - CNNs
        - Why not good ?
	    - Convolution preserves **relative-order** information, but **max-pooling discards them**
	- Solutions
	    - Discard max-pooling, use deep CNNs with skip-connections instead
	    - Add position embedding, just like in ConvS2S

    - [Transformer](https://github.com/gaoisbest/NLP-Projects/blob/master/Pretraining_LM/README.md#transformer)
        - Why not good ?
	    - In self-attention, one word attends to other words and generate the summarization vector without relative position information
    

- **Semantic features extraction** [2]
    - Transformer > CNNs == RNNs

### 3. Pattern of DL in NLP models [3]
- **Data**
    - Preprocess
        - [Sub-word segmentation](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46) to avoid OOV and reduce vocabulary size
            - [sentencepiece](https://github.com/google/sentencepiece)
    - Pre-training (e.g., ELMO, BERT)
    - [Multi-task learning](https://mp.weixin.qq.com/s/ulZBmyt_L-RgGEGhNxrHeQ)
    - Transfer learning, [ref_1](https://mp.weixin.qq.com/s/UJlmjFHWhnlXXJoRv4zkEQ), [ref_2](http://ruder.io/transfer-learning/)
        - Use source task/domain `S` to increase target task/domain `T`
	- If `S` has a zero/one/few instances, we call it zero-shot, one-shot, few-shot learning, respectively
    
- **Model**
    - Encoder
        - CNNs, RNNs, Transformer
    - Structure
        - Sequential, Tree, Graph
- **Learning** (change loss definition)
    - Adversarial learning
    - Reinforcement learning

#### References
- [1] [Review](https://zhuanlan.zhihu.com/p/54743941)
- [2] [Why self-attention? A targeted evaluation of neural machine translation architectures](http://aclweb.org/anthology/D18-1458)
- [3] [ACL 2019 oral](https://zhuanlan.zhihu.com/p/72725518?utm_source=wechat_timeline&utm_medium=social&utm_oi=35938507948032&wechatShare=1&s_r=0&from=timeline&isappinstalled=0)

# Awesome public apis
- [Baidu AI Open Platform](https://ai.baidu.com/)
- [Tencent AI Open Platform](https://ai.qq.com/)
- [Tencent NLP](http://nlp.qq.com/)

# Awesome packages
### Chinese
- [pyltp](http://pyltp.readthedocs.io/zh_CN/develop/api.html)
- [HanLP](http://hanlp.linrunsoft.com/index.html)

### English
- [Spacy](https://spacy.io)
- [gensim](https://radimrehurek.com/gensim/)
- [Install tensorflow with one line](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc): `conda install tensorflow-gpu`

# Future directions
- [Multi-task learning](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture17-multitask.pdf)
- [Self-training](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture20-future.pdf)
