
# Notes of paper: Distributed Representations of Sentences and Documents
* Two models are proposed: **PV-DM (Paragraph Vector-Distributed Memory) and PV-DBOW (Paragraph Vector-Distributed Bag of Words). PV-DM just likes CBOW in Word2vec and PV-DBOW just likes Skip-gram in Word2vec**. PV-DM is consistently better than PV-DBOW. 
* Model principles: during training, concatenate the paragraph vector with several word vectors from a paragraph and predict the following word in the given context. While paragraph vectors are unique among paragraphs, the word vectors are shared. At
prediction time, the paragraph vectors are inferred by fixing the word vectors and training the new paragraph vector until convergence.
* For PV-DM: using concatenation in PV-DM is often better than sum.

![PV-DM](https://github.com/gaoisbest/NLP-Projects/blob/master/Text%20similarity/PV-DM.png)

* For PV-DBOW:

![PV-DBOW](https://github.com/gaoisbest/NLP-Projects/blob/master/Text%20similarity/PV-DBOW.png)

* BOW features lose the ordering of the words and also ignore semantics of the words (Dot product of any two word vector is zero). Word vector concatenation reserve the word order.
* Weighted averaging of word vectors loses the word order in the same way as the standard bag-of-words models do.
* For long documents, bag-of-words models perform quite well.

# Cosine similarity

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
