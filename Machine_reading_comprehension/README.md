# Introduction
Human reading comprehension belongs to cognitive psychology field. Roughly, there are three different **comprehension types**, i.e., literal comprehension (字面理解), inference comprehension (推断理解) and critical comprehension (评价理解).  

For machine reading comprehension (mrc), [Deep read: A reading comprehension system](http://www.aclweb.org/anthology/P99-1042) in ACL 1999 gives the first study. [Towards the Machine Comprehension of Text: An Essay](https://pdfs.semanticscholar.org/a4dd/e51a7f3f0f731d2c27d64948706604792efa.pdf) by Microsoft gives a review. EMNLP 2014 best paper [Modeling Biological Processes for Reading Comprehension](http://www.aclweb.org/anthology/D14-1159) proposes **feature engineering based symbolic models**. After that, lots of **deep learning models** appear. [Tencent AI part 1](https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/) illustrates building blocks of mrc deep learning models. [Tencent AI  Part 2](https://hanxiao.github.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/?from=timeline&isappinstalled=0) proposes their new [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997). [bAbI](https://research.fb.com/downloads/babi/) datasets from Facebook gives the ai-complete concept. 


# Deep learning Models
- [QANet](https://arxiv.org/pdf/1804.09541.pdf)
- [S-Net](https://arxiv.org/pdf/1706.04815.pdf) from MSR for MS-MARCO
- [R-Net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) from MSR for MS-MARCO
- [V-Net](https://arxiv.org/abs/1805.02220) from Baidu NLP for MS-MARCO
- [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997)
- [FastQA](http://www.aclweb.org/anthology/K17-1028), [comment](http://www.shuang0420.com/2018/05/13/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Making%20Neural%20QA%20as%20Simple%20as%20Possible%20but%20not%20Simpler/)
- Match-LSTM
- [BiDAF](https://arxiv.org/pdf/1611.01603.pdf) from AllenNLP **baseline for MS-MARCO**
- [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/pdf/1710.10723.pdf) **baseline for HotpotQA**, [official code](https://github.com/allenai/document-qa) and [code by HotpotQA](https://github.com/hotpotqa/hotpot/blob/master/model.py)

- FusionNet
- Reinforced M-Reader
- Model reviews [part 1](https://mp.weixin.qq.com/s/V2HcHgmW-SfJDwzqydadoA) and [part 2](https://mp.weixin.qq.com/s/IahvlkiACOAjicX68teA0A)

# Dataset
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), extractive dataset
- [MS-MARCO](http://www.msmarco.org/), generative dataset
  - Differences from SQuAD
    - SQuAD constrains the answer to be an exact sub-span in the passage, while the answer in MS-MARCO may covers several passages
    - SQuAD only has one passage for a question, while MS-MARCO contains multiple passages
- [DuReader](http://ai.baidu.com/broad/subordinate?dataset=dureader)
- [HotpotQA](https://hotpotqa.github.io/)
- [TriviaQA](http://nlp.cs.washington.edu/triviaqa/)
- [CoQA](https://stanfordnlp.github.io/coqa/): A Conversational Question Answering Challenge
- [QuAC](http://quac.ai./): Question Answering in Context

# Materials
- [Dr. Lin from Naturali](https://mp.weixin.qq.com/s/6nAm1sJrAj3qqUAagwWutg)
- [paperweekly seminar](https://www.bilibili.com/video/av33802404/)

# Applications
- Learning to ask (i.e., neural questioner)
- Open domain QA
    - [DrQA](https://arxiv.org/pdf/1704.00051.pdf)
    - [DS-QA](http://aclweb.org/anthology/P18-1161)
    - [R^3](https://arxiv.org/pdf/1709.00023.pdf)

# Take-home messages
- **Context**
    - [Minimal context in ACL 2018](https://arxiv.org/pdf/1805.08092.pdf)
- **Model**
    - Make model as simple as possible, see [FastQA](http://www.aclweb.org/anthology/K17-1028)
    - Speed up using [QRNN](https://arxiv.org/pdf/1611.01576.pdf), [SRU](https://arxiv.org/pdf/1709.02755.pdf) or [Skim-RNN](Neural Speed Reading via Skim-RNN)
    
- **Transfer learning**
    - Word embedding: [GloVe is better than word2vec](https://arxiv.org/pdf/1703.00993.pdf)
    - Language model: [CoVe](https://arxiv.org/pdf/1708.00107.pdf), [ELMo](https://arxiv.org/pdf/1802.05365.pdf)


    
