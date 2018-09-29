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
### Transformer [3]
- Belongs to **Encoder-Decoder** framework
- Encoder block
    - Multi-head self-attention
        - A layer that helps the encoder look at other words in the input sentence as it encodes a specific word
    - Position-wise feed-forward
        - The exact same feed-forward network is independently applied to each position
- Decoder block
    - Multi-head self-attention
    - Encoder-Decoder attention
        - Helps the decoder focus on relevant parts of the input sentence
    - Position-wise feed-forward


Implementation: [pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch), [tensorflow](https://github.com/Kyubyong/transformer)  
[self attention](https://www.paperweekly.site/papers/notes/339) 


# References
- [1] [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/)
- [2] [blog](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
- [3] [Transformer illustration](https://jalammar.github.io/illustrated-transformer/), [2.1](https://www.jiqizhixin.com/articles/2018-01-10-20), [2.2]()
- [4] https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada

