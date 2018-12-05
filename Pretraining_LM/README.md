# Introduction [1]
- Step 1: pre-training, step 2: fine-tuning
- Transfer learning from computer vision shows that **low-level features can be shared and high-level features are task-dependent**, therefore we can use our own data to fine-tune the pre-trained model with **same model strcuture**.
- [History](https://zhuanlan.zhihu.com/p/49271699?utm_medium=social&utm_source=wechat_session&wechatShare=2&from=timeline&isappinstalled=0): NNLM -> Word2vec (cannot handle polysemy) -> ELMo (dynamic word embedding, biLM + biLSTM) -> ULMFiT (three steps) -> GPT (start fine-tune schema, uniLM + Transformer) -> BERT (biLM + Transformer)
- [BERT](https://arxiv.org/pdf/1810.04805.pdf), [ELMo](https://arxiv.org/abs/1802.05365), [ULMFiT](https://arxiv.org/abs/1801.06146) and [OpenAI GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) shows that **pretrained language models** can achieve state-of-the-art results on a wide range of NLP tasks. Refer to review [1](https://mp.weixin.qq.com/s/A-PKyZcXwOz-2lL-hBmjsA), [2](https://zhuanlan.zhihu.com/p/49271699?utm_medium=social&utm_source=wechat_session&wechatShare=2&from=timeline&isappinstalled=0) and [3](https://mp.weixin.qq.com/s/-mdHtQ55C05eSRZZP7DlOg) see their difference.
- Word vectors built by Word2vec or GloVe can only be used as initialization of **first layer** of deep networks.
- Gain **primal “knowledge” of the language structures** before a more specific supervised training step [4].

# How to use pretrained language models [1]
- Use the pre-trained language model as a **fixed feature extractor** and incorporate its representation as features into a randomly initialized model, as used in ELMo
- **Fine-tune the entire language model**, as done by ULMFiT. Fine-tuning approach is what is typically done in CV where either the top-most or several of the top layers are fine-tuned. 

# Examples
## Transformer
- Principle: belongs to **Encoder-Decoder** framework [3]
- Blocks
    - **Encoder block**
        - Multi-head self-attention
            - A layer that helps the encoder look at other words in the input sentence as it encodes a specific word
            - Embedding with time singal = Embeddings + Positional Encoding
        - Position-wise feed-forward
            - The exact same feed-forward network is independently applied to each position
        - Residuals
            - Add & Layer Normalization
    - **Decoder block**
        - Multi-head self-attention
            - It is only allowed to attend to earlier positions in the output sequence (This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation) .
        - Encoder-Decoder attention
            - Helps the decoder focus on relevant parts of the input sentence
            - The layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack
        - Position-wise feed-forward
    - **Linear and softmax layer**
        - Linear projects the decoder output to the logits of vocabulary size
        - Softmax the logits and choose the index with largest probability

- Implementation
    - [The annotated transformer](http://nlp.seas.harvard.edu/2018/04/01/attention.html)
    - [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
    - [pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
    - [tensorflow](https://github.com/tensorflow/tensor2tensor)  

## ELMo (Embeddings from Language Models)
### Principle
- Bidirectional language model
- Lower biLSTM layer catches syntax, and higher biLSTM layer catches semantic.
- **Feature-based pretraining**.

### Implementation
[AllenNLP ELMo page](https://allennlp.org/elmo) gives a detailed explanation about ELMo. And [AllenNLp github page](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) describes how to use ELMo:
- Get contextual representations uses trained model
- Train a new model based on ELMo. Class `allennlp.modules.elmo.Elmo` calculates weighted representation
- Interactively. Class `allennlp.commands.elmo.ElmoEmbedder` return lstm hidden states for each word
- With existing allennlp model
    - Edit `bidaf.jsonnet` from [`/training_config/`](https://github.com/allenai/allennlp/tree/master/training_config)
    - Run `allennlp train training_config/bidaf.jsonnet -s output_model_file_path`
    - See [BiDAF](https://github.com/gaoisbest/NLP-Projects/blob/master/Pretraining_LM/bidaf.jsonnet) example


## BERT (Bidirectional Encoder Representations from Transformers).  
### Principle
- Two strategies
    - Masked LM
    - Next Sentence Prediction
    - They are trained together, with the goal of minimizing the combined loss function of them [5]. 
- Bidirectional Transformer to language modeling
- BERT is basically a trained Transformer Encoder stack. **BERT base** has 12 encoding layers, 768 hidden units in feedforward-networks and 12 attention heads. **BERT large** has 24 encoding layers, 1024 hidden units in feedforward-networks and 16 attention heads.
- BERT’s goal is to generate a language model, only the encoder mechanism is necessary. 
- the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. 
- As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).
- When training language models, there is a challenge of defining a prediction goal. Many models predict the next word in a sequence (e.g. “The child came home from ___”), a directional approach which inherently limits context learning. To overcome this challenge, BERT uses two training strategies:
Masked LM (MLM)
Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `[MASK]` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. In technical terms, the prediction of the output words requires:

Adding a classification layer on top of the encoder output.
Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
Calculating the probability of each word in the vocabulary with softmax.
Next Sentence Prediction (NSP)
- the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. 

To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:

A `[CLS]` token is inserted at the beginning of the first sentence and a `[SEP]` token is inserted at the end of each sentence.
A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token embeddings with a vocabulary of 2.
A positional embedding is added to each token to indicate its position in the sequence. The concept and implementation of positional embedding are presented in the Transformer paper.

When training the BERT model, [5]


- **Model-based pretraining**.
### Implementation
- [Official page](https://github.com/google-research/bert) gives pretrained models about BERT
- [Naturali](https://www.jianshu.com/p/aa2eff7ec5c1) gives details about BERT fine-tune


## ULMFIT [2]
## OpenAI GPT

# References
- [1] [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
- [2] [blog](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
- [3] [Transformer illustration](https://jalammar.github.io/illustrated-transformer/), [Transformer interpretation](https://www.jiqizhixin.com/articles/2018-01-10-20), [self attention](https://www.paperweekly.site/papers/notes/339) 
- [4] https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada
- [5] https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/
- [6] https://jalammar.github.io/illustrated-bert/
