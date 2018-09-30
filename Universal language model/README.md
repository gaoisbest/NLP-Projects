# Introduction [1]
- [ELMo](https://arxiv.org/abs/1802.05365), [ULMFiT](https://arxiv.org/abs/1801.06146) and [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) shows that **pretrained language models** can achieve state-of-the-art results on a wide range of NLP tasks.
- Word vectors built by Word2vec or GloVe can only be used as initialization of **first layer** of deep networks.
- Gain **primal “knowledge” of the language structures** before a more specific supervised training step [4].

# How to use pretrained language models [1]
- Use the pre-trained language model as a **fixed feature extractor** and incorporate its representation as features into a randomly initialized model, as used in ELMo
- **Fine-tune the entire language model**, as done by ULMFiT. Fine-tuning approach is what is typically done in CV where either the top-most or several of the top layers are fine-tuned. 

# Examples
## ELMo
## ULMFIT [2]
## OpenAI transformer
### Transformer
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
    - [pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
    - [tensorflow](https://github.com/tensorflow/tensor2tensor)  

# References
- [1] [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
- [2] [blog](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
- [3] [Transformer illustration](https://jalammar.github.io/illustrated-transformer/), [Transformer interpretation](https://www.jiqizhixin.com/articles/2018-01-10-20), [self attention](https://www.paperweekly.site/papers/notes/339) 
- [4] https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada

