# Introduction
Human reading comprehension belongs to cognitive psychology field. Roughly, there are three different **comprehension types**, i.e., literal comprehension (字面理解), inference comprehension (推断理解) and critical comprehension (评价理解).  

For machine reading comprehension (mrc), [Deep read: A reading comprehension system](http://www.aclweb.org/anthology/P99-1042) in ACL 1999 gives the first study. [Towards the Machine Comprehension of Text: An Essay](https://pdfs.semanticscholar.org/a4dd/e51a7f3f0f731d2c27d64948706604792efa.pdf) by Microsoft gives a review. EMNLP 2014 best paper [Modeling Biological Processes for Reading Comprehension](http://www.aclweb.org/anthology/D14-1159) proposes **feature engineering based symbolic models**. After that, lots of **deep learning models** appear. [Tencent AI part 1](https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/) illustrates building blocks of mrc deep learning models. [Tencent AI  Part 2](https://hanxiao.github.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/?from=timeline&isappinstalled=0) proposes their new [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997). [bAbI](https://research.fb.com/downloads/babi/) datasets from Facebook gives the ai-complete concept.  

[**MRC components**](https://github.com/gaoisbest/NLP-Projects/blob/master/Machine_reading_comprehension/materials_CCL2018/CCL2018_MRC.pdf):
- Passage
    - Single or multiple
- Question
    - Cloze or query
- Candidate
    - Multiple choice (e.x., [Co-Matching](http://aclweb.org/anthology/P18-2118), [HMA](https://arxiv.org/pdf/1803.05655.pdf)) and Opinion questions ([DuReader](https://arxiv.org/pdf/1711.05073.pdf))
- Answer
    - Extraction or generation


# Deep learning Models
- Extraction based (i.e., predicting the start and end positions)
    - [QANet](https://arxiv.org/pdf/1804.09541.pdf)
    - [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997)
    - [R-Net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) from MSR for MS-MARCO
    - [Match-LSTM](https://arxiv.org/pdf/1608.07905.pdf)
    
- Extraction-then-synthesis based
    - [S-Net](https://github.com/gaoisbest/NLP-Projects/blob/master/Machine_reading_comprehension/materials_papers/%20S-Net%20-%20From%20Answer%20Extraction%20to%20Answer%20Generation%20for%20Machine%20Reading%20Comprehension.pdf) from MSR for MS-MARCO
        - Step 1: **extracts evidence snippets** by matching question and passage via pointer network. Add **passage ranking** as an additional task to conduct multi-task learning.
        - Step 2: generate the answer by **synthesizing the passage, question and evidence snippets** via seq2seq. Evidence snippets are labeled as features.
        
    - [V-Net](https://arxiv.org/abs/1805.02220) from Baidu NLP for MS-MARCO
    
- To be classified...
  
    - [FastQA](http://www.aclweb.org/anthology/K17-1028), [comment](http://www.shuang0420.com/2018/05/13/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Making%20Neural%20QA%20as%20Simple%20as%20Possible%20but%20not%20Simpler/)
    
    - [BiDAF](https://arxiv.org/pdf/1611.01603.pdf) from AllenNLP **baseline for MS-MARCO**
    - [Documentqa](https://github.com/allenai/document-qa)
    - FusionNet
    - Reinforced M-Reader
    - Model reviews [part 1](https://mp.weixin.qq.com/s/V2HcHgmW-SfJDwzqydadoA) and [part 2](https://mp.weixin.qq.com/s/IahvlkiACOAjicX68teA0A)

# Dataset
- Multiple option
    - [MCTest](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/)
- Cloze
    - English
        - [CNN/Daily-Mail](https://arxiv.org/pdf/1506.03340.pdf)
        - [CBT](https://arxiv.org/pdf/1511.02301.pdf)
    - Chinese
        - [PeopleDaily/CFT](https://arxiv.org/pdf/1607.02250.pdf)
- Question answering
    - English
        - Extractive
            - Single-hop (i.e., single document) reasoning
                - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), extractive dataset
                    - For **unanswerable** questions
                        - [SQUADRUN Dataset](http://aclweb.org/anthology/P18-2124)
                        - [U-Net](https://arxiv.org/pdf/1810.06638.pdf)
                        - [Read + Verify](https://arxiv.org/pdf/1808.05759.pdf)
                        - [Answer Verifier](https://link.springer.com/chapter/10.1007/978-3-319-99495-6_8)
            - Multi-hop reasoning
                - [HotpotQA](https://hotpotqa.github.io/): A Dataset for Diverse, Explainable Multi-hop Question Answering
                    - Data source: English Wikipedia dump
                    - Provides **supporting facts** to support **explainable** reasoning
                    - Novel question type: **comparison question**, which contains **yes/no question**
                    - Baseline model: [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/pdf/1710.10723.pdf), [official code](https://github.com/allenai/document-qa) and [code by HotpotQA](https://github.com/hotpotqa/hotpot/blob/master/model.py)
                - [TriviaQA](http://nlp.cs.washington.edu/triviaqa/)
                - [SearchQA](https://arxiv.org/abs/1704.05179)
                - [CoQA](https://stanfordnlp.github.io/coqa/): A Conversational Question Answering Challenge
                - [QuAC](http://quac.ai./): Question Answering in Context
                - [AI2 Reasoning Challenge](http://data.allenai.org/arc/)
        - Generative
            - Multi-hop reasoning
                - [MS-MARCO](http://www.msmarco.org/)
                    - Data source: Bing queries. 10 passages per query. [Download link for v1 and v2](https://github.com/zlsh80826/MSMARCO/blob/master/data/download.py)
                    - Answers are human-generated
                    - Give passage candidates, and annotate which passage is correct
                    - Baseline model: [BiDAF](https://github.com/dfcf93/MSMARCOV2/tree/master/Q%2BA/BidafBaseline)
    - Chinese
        - [DuReader](http://ai.baidu.com/broad/subordinate?dataset=dureader)
            - Model: [V-Net](https://yizhong-wang.com/papers/acl2018-reading-slides.pdf)
        - [CMRC 2018](https://hfl-rc.github.io/cmrc2018/)

# Materials
- [Dr. Lin from Naturali](https://mp.weixin.qq.com/s/6nAm1sJrAj3qqUAagwWutg)
- [paperweekly seminar](https://www.bilibili.com/video/av33802404/)

# Leading companies
- [Zhuiyi](https://zhuiyi.ai/)
- [Naturali](https://www.naturali.io/)

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


    
