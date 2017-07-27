
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
