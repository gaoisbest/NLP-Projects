# Introduction
Human reading comprehension belongs to cognitive psychology field. Roughly, there are three different **comprehension types**, i.e., literal comprehension (字面理解), inference comprehension (推断理解) and critical comprehension (评价理解).  

For machine reading comprehension (mrc), [Deep read: A reading comprehension system](http://www.aclweb.org/anthology/P99-1042) in ACL 1999 gives the first study. [Towards the Machine Comprehension of Text: An Essay](https://pdfs.semanticscholar.org/a4dd/e51a7f3f0f731d2c27d64948706604792efa.pdf) by Microsoft gives a review. EMNLP 2014 best paper [Modeling Biological Processes for Reading Comprehension](http://www.aclweb.org/anthology/D14-1159) proposes **feature engineering based symbolic models**. After that, lots of **deep learning models** appear. [Tencent AI part 1](https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/) illustrates building blocks of mrc deep learning models. [Tencent AI  Part 2](https://hanxiao.github.io/2018/09/09/Dual-Ask-Answer-Network-for-Machine-Reading-Comprehension/?from=timeline&isappinstalled=0) proposes their new [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997). [bAbI](https://research.fb.com/downloads/babi/) datasets from Facebook gives the ai-complete concept. [Neural Machine Reading Comprehension: Methods and Trends](https://arxiv.org/abs/1907.01118) presents a new review about MRC.

[**MRC components**](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_CCL2018/CCL2018_MRC.pdf):
- Passage
    - Single or multiple
- Question
    - Cloze or query
- Candidate
    - Multiple choice (e.x., [Co-Matching](http://aclweb.org/anthology/P18-2118), [HMA](https://arxiv.org/pdf/1803.05655.pdf)) and Opinion questions ([DuReader](https://arxiv.org/pdf/1711.05073.pdf))
- Answer
    - Extraction or generation


# Deep learning Models
## Model list
- [BiDAF](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/BiDAF.pdf) from AllenNLP, baseline for MS-MARCO
    - Attention flow layer: **context-to-query attention** (i.e., which query words are most relevant to each context word, `softmax(row)`) and **query-to-context attention** (i.e., which context words have the closest similarity to one of the query word, `softmax(max(column))`), based on **similarity matrix**
    - Similarity function
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/BiDAF_similarity_function.png)
    - Model structure
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/BiDAF_model.png)
    - [Official implementation](https://allenai.github.io/bi-att-flow/)
    - [Model illustration](https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/)
    - [BiDAF + Self attention + ELMo](https://arxiv.org/pdf/1710.10723.pdf)
    
- [R-Net](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/R-Net.pdf) for MS-MARCO
    - Core layer 1: **gated (applied to passage word and attention-pooling of question) attention-based recurrent network** matches passage and question to obtain question-aware passage representation
    - Core layer 2: **self-matching** layer to aggregate the passage information
    - Model structure
![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/R-Net_model.png)
    - Implementation: [pytorch](https://github.com/HKUST-KnowComp/MnemonicReader/blob/master/r_net.py), [tensorflow](https://github.com/HKUST-KnowComp/R-Net)
- [S-Net](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/S-Net.pdf) for MS-MARCO
    - Step 1: **extracts evidence snippets** by matching question and passage via pointer network. Add **passage ranking** as an additional task to conduct multi-task learning.
    - Step 2: generate the answer by **synthesizing the passage, question and evidence snippets** via seq2seq. Evidence snippets are labeled as features.
    
- [QANet](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/QANet.pdf)
    - **Separable convolution** + **self-attention** (Each position as a query to match all positions as keys)
    - Data augmentation via backtranslation
    - Model structure
![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/QANet_model.png)
    - [Implementation](https://github.com/NLPLearn/QANet)
- [Multi-answer Multi-task](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/A_multi-answer_multi-task_framework_for_real-world_MRC.pdf)
    - Three loss for multiple answer span
        - Average loss
        - Weighted average loss
        - Minimum value of the loss
    - Combine passage ranking as multi-task learning
        - As answer span can occur in multiple passages, **pointwise sigmoid function** instead of softmax function is used
    - Minimum risk training
        - Direct optimize the evaluation metric instead of maximizing MLE
    - Prediction is only **single answer span**
        
- [Match-LSTM](https://arxiv.org/pdf/1608.07905.pdf)
- [U-Net](https://arxiv.org/pdf/1810.06638.pdf)
    - [Illustration](https://mp.weixin.qq.com/s/VmmWEJJXXGLaE5-mLMZbpQ)
- [Dual Ask-Answer Network](https://arxiv.org/abs/1809.01997)
- [Gated Self-Matching Networks](http://www.aclweb.org/anthology/P17-1018)
- [V-Net](https://arxiv.org/abs/1805.02220) from Baidu NLP for MS-MARCO
- [FastQA](http://www.aclweb.org/anthology/K17-1028), [comment](http://www.shuang0420.com/2018/05/13/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Making%20Neural%20QA%20as%20Simple%20as%20Possible%20but%20not%20Simpler/)
- [Documentqa](https://github.com/allenai/document-qa)
- Model reviews [part 1](https://mp.weixin.qq.com/s/V2HcHgmW-SfJDwzqydadoA) and [part 2](https://mp.weixin.qq.com/s/IahvlkiACOAjicX68teA0A)

## Model structure
- Embedding layer
    - Character-level embedding
        - 1D CNN (BiDAF)
        - Last hidden states of BiRNN (R-Net)
        - It is useful to **OOV** tokens
    - Word-level embedding
        - GloVe pre-trained embedding is used frequently
    - Features
        - [Binary and weighted *word-in-question* (without stop-words) feature (FastQA)](http://www.aclweb.org/anthology/K17-1028)
        - POS tag
        - Query type
- Encoding layer
    - Concatenation of forword and backword hidden states of BiRNN (BiDAF)
    - `[convolution-layer * # + self-attention layer + feed-forward layer]` (QANet)
- Context-query attention layer
    - Context and query similarity matrix (BiDAF, QANet)
- Model layer
    - BiRNN (BiDAF)
    - Gated attention-based recurrent network (R-Net)
    - Passage self-matching
    - `[convolution-layer * # + self-attention layer + feed-forward layer]` (QANet)
- Output layer
    - Direct output (BiDAF, QANet)
    - [Pointer network](http://fastml.com/introduction-to-pointer-networks/) (R-Net)
        - Simplify seq2seq mechanism
        - It only points at the **probability** of elements and get **a permutation of inputs**
        - Not all pointers is necessary, for mrc and summarization, for example, only two pointers is needed

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
                - [SQuAD 2.0](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/Know_What_You_Don%E2%80%99t_Know_Unanswerable_Questions_for_SQuAD.pdf), for **unanswerable** questions
                - [Google Natural Questions](https://ai.google.com/research/NaturalQuestions) contains short and long answers
                
            - Multi-hop reasoning
                - [HotpotQA](https://hotpotqa.github.io/): A Dataset for Diverse, Explainable Multi-hop Question Answering
                    - Data source: English Wikipedia dump
                    - Provides **supporting facts** to support **explainable** reasoning
                    - Novel question type: **comparison question**, which contains **yes/no question**
                    - Baseline model: [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/pdf/1710.10723.pdf), [official code](https://github.com/allenai/document-qa) and [code by HotpotQA](https://github.com/hotpotqa/hotpot/blob/master/model.py)
                - [TriviaQA](http://nlp.cs.washington.edu/triviaqa/)
                    - Has much longer context (2895 tokens per context on average) and may contain several paragraphs
                    - Much noisier than SQuAD due to the lack of human labeling
                    - Possible that the context is not related to the answer at all, as it is crawled by key words
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
            - Data source: real anonymized user queries, contains more than 300K questions, 1.4M evident documents and **human generated answers**.
            - Baseline model: [BiDAF and Match-LSTM](https://github.com/baidu/DuReader)
            - Advanced model: [V-Net](https://yizhong-wang.com/papers/acl2018-reading-slides.pdf)
        - [CMRC 2018](https://hfl-rc.github.io/cmrc2018/)
# Evaluation metrics
- Exact Match
    - Clean the text: remove a, an, the, whitespace, punctuation and lowercase
    - [Implementation](https://github.com/HKUST-KnowComp/R-Net/blob/master/util.py#L134)
- F1
    - It measures the **portion of overlap tokens** between the predicted answer and groundtruth
    - [Implementation](https://github.com/HKUST-KnowComp/R-Net/blob/master/util.py#L121)
- BLEU
- ROUGE-L

# In action
- [Sougou MRC Toolkit](https://github.com/sogou/SMRCToolkit), [paper](https://arxiv.org/pdf/1903.11848.pdf)
- [2019 Dureader competition](http://lic2019.ccf.org.cn/read)
- [2018 DuReader competition summary](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/2018_DuReader_top_solution_summary.pdf)
- [Naturali video version](https://mp.weixin.qq.com/s/6nAm1sJrAj3qqUAagwWutg), [text version](http://tech.163.com/18/0729/10/DNSH3QU600098IEO.html)
    - Data preprocess, [implementation](https://github.com/baidu/DuReader/blob/master/paddle/paragraph_extraction.py)
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/Naturali_data_preprocess.png)
    - Model
    ![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/Naturali_model.png)

- [Paperweekly seminar](https://www.bilibili.com/video/av33802404/)
- [Zhuiyi video](http://www.mooc.ai/course/596/learn?lessonid=2928), [text 1](https://mp.weixin.qq.com/s/EQsxkCwQW_gzlB3Ozd5_ZQ) and [text 2](https://mp.weixin.qq.com/s/GrEfupJdAUFSp2l-myA1Ig)
    - Data preprocess
        - Filter out query or answer in None
        - Context normalization, i.e., lowercase, punctuation
        - Answer length limit, context length limit (threshold is determined by statistics)
        - Data augmentation i.e., back-translation or similar QA data
        - Training data quality, e.g., same query type has different answer format, `1963 year`, `1990 year stop the usage`
    - Feature engineering
        - Query type
            - Who, when, where, how, number, why, how long
        - ELMo
            - Word level
    - Model (based on R-Net)
        - Embedding
            - ELMo only (without word2vec)
            - POS embedding
            - Query type embedding
            - Binary word-in-question feature
        - Encoding
            - Multi-layer BiGRU
        - Context-query attention
            - Gated-dropout (filtering useful message) for query
        - Prediction
            - Pointer network
            - Probability = start * stop
        ![](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/Zhuiyi_model.png)
    - Training
        - Born-Again Neural Network, teacher = student
            

# Applications
- Learning to ask (i.e., neural questioner)
- Open domain QA
    - [DrQA](https://arxiv.org/pdf/1704.00051.pdf), see [danqi's PHD thesis](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/Daiqi_stanford_PhD_thesis_MRC.pdf) for detailed information
    - [DS-QA](http://aclweb.org/anthology/P18-1161)
    - [R^3](https://arxiv.org/pdf/1709.00023.pdf)
- [Search](https://mp.weixin.qq.com/s/GrEfupJdAUFSp2l-myA1Ig)
    - Summarization -> KBQA -> MRC
    
    
# Take-home messages
- **Context**
    - [Minimal context in ACL 2018](https://arxiv.org/pdf/1805.08092.pdf)
- **Model**
    - Make model as simple as possible, see [FastQA](http://www.aclweb.org/anthology/K17-1028)
    - Speed up using [QRNN](https://arxiv.org/pdf/1611.01576.pdf), [SRU](https://arxiv.org/pdf/1709.02755.pdf) or [Skim-RNN](Neural Speed Reading via Skim-RNN)
- **Syntax integration**
    - Syntax features include POS tags, NER results, linearized PCFG tree tags cannot give additional benefits (from discussion of R-Net).
- **Transfer learning**
    - Word embedding: [GloVe is better than word2vec](https://arxiv.org/pdf/1703.00993.pdf)
    - Language model: [CoVe](https://arxiv.org/pdf/1708.00107.pdf), [ELMo](https://arxiv.org/pdf/1802.05365.pdf)
- **Unanswerable question type**
    - [Add padding position](https://github.com/gaoisbest/NLP-Projects/blob/master/2_Machine_reading_comprehension/materials_papers/I_Know_There_Is_No_Answer_Modeling_Answer_Validation_for_MRC.pdf)
    - [Trainable bias](https://arxiv.org/pdf/1706.04115.pdf)
    - [Two-Stage Synthesis Networks for Transfer Learning in Machine Comprehension](https://github.com/davidgolub/QuestionGeneration)
        - Based on SQuAD, train answer and question generation network `gen`
        - For new dataset, use `gen` to generate new question, answer datasets, then train new MRC model based on these datasets

    
