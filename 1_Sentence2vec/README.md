# Unsupervised sentence embedding models
- Word mover's distance: [paper](http://proceedings.mlr.press/v37/kusnerb15.pdf), [gensim example](https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html)
- [Extension of BOW based on word2vec](https://www.zhihu.com/question/29978268/answer/55338644): The sentence vector contains top 10 word vector similarity between each word of the sentence and the vocabulary.
- [LSTM + AutoEncoder](https://www.zhihu.com/question/29978268/answer/114093723)
- [SIF](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/SIF.py)
- [Mean word vectors](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/Word2vec_mean.py)
- [Doc2vec](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/Doc2vec_model.py)
- [LDA](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/LDA_model.py)
- [LSI](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/LSI_model.py)
- [Sent2vec](https://github.com/epfml/sent2vec)
- [Simhash](https://leons.im/posts/a-python-implementation-of-simhash-algorithm/)
- [Skip-Thought](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
- [Quick thoughts](https://arxiv.org/pdf/1803.02893.pdf)


# Sentence encoder example
- https://hanxiao.github.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/

# Distance
## Cosine distance
```
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

a = np.array([0, 1, 2, 3])
b = np.array([0, 1, 3, 3]) 
print 'scipy cosine similarity: {}, sklearn similarity: {}'.format(1 - cosine(a, b), cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])
# scipy cosine similarity: 0.981022943176, sklearn similarity: 0.981022943176

```
## Edit distance

## Jaro–Winkler distance

# Notes of paper: Distributed Representations of Sentences and Documents
- Two models are proposed: **PV-DM (Paragraph Vector-Distributed Memory) and PV-DBOW (Paragraph Vector-Distributed Bag of Words). PV-DM just likes CBOW in Word2vec and PV-DBOW just likes Skip-gram in Word2vec**. PV-DM is consistently better than PV-DBOW. 
- Model principles: during training, concatenate the paragraph vector with several word vectors from a paragraph and predict the following word in the given context. While paragraph vectors are unique among paragraphs, the word vectors are shared. At
prediction time, the paragraph vectors are inferred by fixing the word vectors and training the new paragraph vector until convergence.
- For PV-DM: using concatenation in PV-DM is often better than sum.

![PV-DM](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/PV-DM.png)

- For PV-DBOW:

![PV-DBOW](https://github.com/gaoisbest/NLP-Projects/blob/master/Sentence2vec/PV-DBOW.png)

- BOW features lose the ordering of the words and also ignore semantics of the words (Dot product of any two word vector is zero). Word vector concatenation reserve the word order.
- Weighted averaging of word vectors loses the word order in the same way as the standard bag-of-words models do.
- For long documents, bag-of-words models perform quite well.


