# Introduction
- Why work? Transfer learning from computer vision shows that **low-level features are general, task-free, can be shared; high-level features are task-dependent** [1]
- History: Statistical LM -> NNLM -> Word2vec (fixed word embeddings, has **polysemy** problem) -> [ELMo](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/ELMo.pdf) (**dynamic word embedding**) -> [ULMFiT](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/ULMFiT.pdf) -> [GPT](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/GPT.pdf) -> [BERT](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/BERT.pdf). Refer to review [1](https://mp.weixin.qq.com/s/A-PKyZcXwOz-2lL-hBmjsA), [2](https://zhuanlan.zhihu.com/p/49271699?utm_medium=social&utm_source=wechat_session&wechatShare=2&from=timeline&isappinstalled=0), [3](https://mp.weixin.qq.com/s/-mdHtQ55C05eSRZZP7DlOg), [4.1](https://mp.weixin.qq.com/s/Rd3-ypRYiJObi-e2JDeOjQ) and [4.2](https://mp.weixin.qq.com/s/7imMQ3GkD52xP7N4fqNPog) see their difference
- Two stages: unsupervised **pre-training** on large corpus, then supervised **feature-based** (e.g., ELMo, BERT) or **fine-tuning** (e.g., GPT, BERT) to downstream tasks
    - **Feature-based** strategy uses task-specific architecture that includes the pre-trained representations as additional features
    - **Fine-tuning** stragety introduces minimal task-specific parameters and is trained on downstream tasks by fine-tuning the pretrained parameters

# Models
## ELMo (Embeddings from Language Models)
### Principle
- **Model**: 2 layer forward LSTM + 2 layer backward LSTM
- **Pre-training objective**: bidirectional LM (i.e., concatenation of independently trained left-to-right and right-to-left LMs)
- **Features**: the feature of the word is **linear combination of all hidden states of biLM**
- ELMo is **deep contextualized** word representation, overcome the **polysemy** problem that word2vec (always fixed vectors given different context) has
- Lower biLSTM layer catches syntax (e.x., POS tagging), and higher biLSTM layer catches semantic (e.g., word sense disambiguation

### Implementation
[AllenNLP ELMo page](https://allennlp.org/elmo) gives a detailed explanation about ELMo. And [AllenNLp github page](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) describes how to use ELMo:
- Get contextual representations uses trained model
- Train a new model based on ELMo. Class `allennlp.modules.elmo.Elmo` calculates weighted representation
- Interactively. Class `allennlp.commands.elmo.ElmoEmbedder` return lstm hidden states for each word
- With existing allennlp model
    - Edit `bidaf.jsonnet` from [`/training_config/`](https://github.com/allenai/allennlp/tree/master/training_config)
    - Run `allennlp train training_config/bidaf.jsonnet -s output_model_file_path`
    - See [BiDAF](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_demos/bidaf.jsonnet) example
- **How to use** ? [Concatenate ELMos with context-independent word embeddings instead of replacing them](http://mlexplained.com/2018/06/15/paper-dissected-deep-contextualized-word-representations-explained/)
## ULMFiT
### Principle
- Three steps
    - step 1: general-domain LM pretraining
    - step 2: target task LM fine-tuning
        -  **Discriminative fine-tuning**: tune each layer (as different layer capture different information type) with different learning rate
        - **Gradual unfreezing**: first unfreeze the last layer and fine-tune all un-frozen layers for one epoch, then unfreeze the next lower frozen layer and repeat, until we fine-tune all layers until convergence at the last iteration
        - **Slanted triangular learning rates**: a short increase and a long decay period for learning rate
    - step 3: target task classifier fine-tuning
        - `fc1_y = ReLU(Dropout(Batch normalization(fc1(X))) -> Softmax(Dropout(Batch normalization(fc2(fc1_y)))`
        - `X = Concat(h_t, mean_pooling(H), max_pooling(H))`
### Implementation
- [Official page](http://nlp.fast.ai/category/classification.html)


## GPT (Generative Pre-training)
### Principle
- **Model**: multi-layer left-to-right (left-context-only) Transformer decoder
- **Pre-training objective**: LM
![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/GPT_fine-tuning.png)
### Implementation
- [Official page](https://github.com/openai/finetune-transformer-lm) gives code and model about GPT

## BERT (Bidirectional Encoder Representations from Transformers)
### Principle
- **Model**: multi-layer bidirectional Transformer encoder
- **Pre-training objective**
    - Masked language model (MLM, inspired by the Cloze task, prevent each token 'see itself' in multi-layer bidirectional context), for **one** sentence
    - Next sentence prediction (NSP), for **two** sentences
- **Input**: token embeddings + segment embeddings + position embeddings. First token is `[CLS]`, sentence pairs between sentences is `[SEP]`
- MLM
    - Disadvantages
        - Since no `[MASK]` in fine-tuning, there is a mismatch between pre-training and fine-tuning
        - Since only 15% token is sampled, the converge rate is slow
    - An example (15% token of whole training data is sampled)
        - `my dog is hairy` -> **80%** replace `hairy` with `[MASK]`, e.g., `my dog is hairy -> my dog is [MASK]`
        - `my dog is hairy` -> **10%** replace `hairy` with a random word, e.g., `my dog is hairy -> my dog is apple`
        - `my dog is hairy` -> **10%** keep unchanged, e.g., `my dog is hairy -> my dog is hairy`
- NSP
    - Reveal the relationship between two sentences is not directly captured by language model
    - 50% B is next of A, 50% B is a random sentence
    - An example
        - Input = `[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]`
        - Label = `IsNext`
        - Input = `[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]`
        - Label = `NotNext`
- Fine-tuning
    - Single sentence / sentence pair classification task
        - Input: final hidden state of Transformer encoder about `[CLS]`
        - New parameter: `W`
        - All of the parameters of BERT and `W` are fine-tuned jointly
    - Question answering
        - Input: final hidden state of Transformer encoder about all tokens
        - New parameter: `start vector` and `end vector`
        - Predicted span is `[max(softmax(dot product(token i hidden state, start vector))), max(softmax(dot product(token i hidden state, end vector)))`

![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/BERT_fine-tuning.png)
### Implementation
- [Official page](https://github.com/google-research/bert) gives pretrained models about BERT
    - Preprocess
        - Typically, the `Uncased` model is **better** unless that `case` information is important (e.g., NER or POS tagging)
        - `Variable-length` problem (`doc_stride`) of SQuAD context paragraphs
    - Tokenization
        - For Chinese, BERT use **character-based tokenization**
        - For all other languages, BERT use**WordPiece tokenization** 
        - Code `BasicTokenizer` in `tokenization.py` 
    - `Out-of-memory`
        - `max_seq_length`
        - `train_batch_size`
    
- [Naturali](https://www.jianshu.com/p/aa2eff7ec5c1) gives details about BERT fine-tune
- [bert-as-service](https://github.com/hanxiao/bert-as-service)
- [Illustrated bert](https://jalammar.github.io/illustrated-bert/)
- [pytorch bert](https://github.com/huggingface/pytorch-pretrained-BERT)

### Application
- QA [4]
    - Steps
        - step 1: convert long documents to passage, build inverted index
        - step 2: retrieval top k candidates passage by BM25 + RM3
        - step 3: input (query, passage) to BERT, output contains/not contains answer or answer 90i89i9uij4start/end position
    - Papers
        - End-to-End Open-Domain Question Answering with BERTserini
        - FAQ Retrieval using Query-Qestion Similarity and BERT-Based Query-Answer Relevance
        - A BERT Baseline for the Natural Questions
        - Data Augmentation for BERT Fine-Tuning in Open-Domain Question Answering
![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_demos/QA_bert.jpg)

- IR [4]
    - Papers
        - Simple Applications of BERT for Ad Hoc Document Retrieval [4.1]
        - Passage Re-ranking with BERT
        - Investigating the Successes and Failures of BERT for Passage Re-Ranking
    - Dealing with long document
        - Replace document with several sentences [4.1] with ASSUMPTION: query are related to some sentences in the document
- Dialogue system [4]
    - Single-turn
        - [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf)
    - Multi-turn
        - [Comparison of Transfer-Learning Approaches for Response Selection in Multi-Turn Conversations](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/Comparison_of_Transfer-Learning_Approaches_for_Response_Selection%20in_Multi-Turn_Conversations.pdf)
- Conclusion [4]
    - BERT is good at **sentence matching** (may be the NSP task during pre-training) and **deep semantic feature extracting** (e.g., QA, but not excel at shallow feature task such as classification and sequence labeling).
    - BERT is not good at **sequence generation**.
        



## Transformer
### Principle
- Principle: each layer has two **sub-layers** (i.e., multi-head self-attention and position-wise feedforward), the output of each sub-layer is `LayerNorm(x+Sublayer(x))`. Additional **positional embedding** is added to embedding.
- Blocks
    - **Encoder block**
        - Multi-head self-attention
            - A layer that helps the encoder look at other words in the input sentence as it encodes a specific word
            - Embedding with time singal = **Embeddings + Positional Encoding**
            - **One-head self-attention steps** (i.e., *scaled dot-product attention*)
                - Packing `n` word embeddings into a matrix `X`
                - Multiplying `X` by the weight matrices `W_Q`, `W_K`, `W_V` to generate `Q`, `K` and `V`
                - `Z = softmax(Q * K_T / sqrt(d_k)) * V`, where `d_k` is column number of `W_Q`, `Z` shape is `(n, d_k)`
            - **Multi-head self-attention steps**
                - First generating different `Z_0`, `Z_1`, ... by different `Q`, `K` and `V` via `W_Q`, `W_K`, `W_V`
                - Then generating `Z` by concatenating `Z_0`, `Z_1`, ... to `Z` and multiply with weight matrix `W_0`
                ![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_demos/transformer_multi-headed_self-attention.png)
            - Why **scaled** dot-product attention ?
                - Definition: `softmax(Q * K_T / sqtr(d_k)) * V`
                - `Q * K_T` may large for `d_k`, cause `softmax` returns `0` or `1`
        - Position-wise feed-forward
            - The exact same feed-forward network is independently applied to each position
            - Two linear transformation: `max(0, W1x + b1)W2 + b2`
        - Residuals
            - Add & Layer Normalization, i.e.m **`LayerNorm(X + Z)`**
            ![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_demos/transformer_residual_layer_norm.png)
            
    - **Decoder block**
        - Multi-head self-attention
            - It is only allowed to attend to earlier positions in the output sequence (This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation) .
        - Encoder-Decoder attention
            - Helps the decoder focus on relevant parts of the input sentence
            - The layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack
        - Position-wise feed-forward
        - [Three self-attentions](https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d)
![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_demos/Different_self-attention_in_encoder_decoder.png)
    - **Linear and softmax layer**
        - Linear projects the decoder output to the logits of vocabulary size
        - Softmax the logits and choose the index with largest probability
    
### Implementation
- [The annotated transformer](http://nlp.seas.harvard.edu/2018/04/01/attention.html)
- [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
- [pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [tensorflow](https://github.com/tensorflow/tensor2tensor)  

# BERT, GPT, ELMo comparison
![](https://github.com/gaoisbest/NLP-Projects/blob/master/5_Pretraining_LM/materials_papers/BERT_GPT_ELMo_comparison.png)

# References
- [1] https://zhuanlan.zhihu.com/p/49271699?utm_medium=social&utm_source=wechat_session&wechatShare=2&from=timeline&isappinstalled=0
- [2] [Transformer interpretation](https://www.jiqizhixin.com/articles/2018-01-10-20), [self attention](https://www.paperweekly.site/papers/notes/339)
- [3] [Bert时代的创新：Bert应用模式比较及其它](https://zhuanlan.zhihu.com/p/65470719)
- [4] [Bert时代的创新（应用篇）：Bert在NLP各领域的应用进展](https://zhuanlan.zhihu.com/p/68446772)
