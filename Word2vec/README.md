# Example of word representations

```
Supposed corpus:

Document 1: I like drinking water. They also like drinking water.

Document 2: They like drinking coffee.
```
## One-hot vector
Represent every word as |V|*1 vector, with one denotes the word index, and others are all 0.

#### Pros and cons
* Dot product of any two word vectors is zero, therefore, one-hot vector can not tell the similarity.

## Count vector

|          | also |   coffee      |   drinking |       I        |     like      |     They     |     water    |
|   :---:  | :---:|     :---:     |    :---:   |        :---:   |    :---:      |     :---:    |     :---:    |
|Document 1| 1    |      0        |      2     |        1       |      2        |     1        |       2      |
|Document 2| 0    |      1        |      1     |        0       |      1        |      1       |     0        |

#### Pros and cons
* High dimension (#Vocabulary size * #Corpus size) and sparsity
* Ignore the word order 
* Filter low frequency words

## TF-IDF

`TF` = word count /  total word counts in the document. E.g., TF(water, Document 1) = 2/9.

`IDF = log(N/n)`, `N` the number of documents in the corpus, `n` is the number of documents that word appears. E.g., IDF(They) = log(2/2) = 0.


|          | also |   coffee      |   drinking |    I        |     like      |     They     |     water    |
|   :---:  | :---:|     :---:     |    :---:   |     :---:   |    :---:      |     :---:    |     :---:    |
|Document 1| 1/9*log(2/1)| 0      |2/9*log(2/2)=0|1/2*log(2/1)|2/9*log(2/2)=0 |1/9*log(2/2)=0|2/9*log(2/1)  |
|Document 2| 0    |1/4*log(2/1)   |1/4*log(2/1)|     0       | 1/4*log(2/1)  |  1/4*log(2/1)|     0        |



#### Pros and cons
* Penalising common words that appeared in the corpus.

## Co-occurence matrix


`Co-occurrence`: the number of times that two words co-occurence in a context window.

`Context Window`: composed of window size and direction. 

The results of window size **2** and **both** direction:

|          | also |   coffee      |   drinking |   I        |     like      |     They     |     water    |
|   :---:  | :---:|     :---:     |    :---:   |    :---:   |    :---:      |     :---:    |     :---:    |
|   also   |  0   |      0        |      1     |      0     |      1        |     1        |       1      |
|   coffee |  0   |      0        |      1     |      0     |      1        |      0       |     0        |
| drinking |  1   |      1        |      0     |     1      |      3        |      2       |     2        |
|   I      |  0   |      0        |      1     |      0     |      1        |      0       |     0        |
|   like   |  1   |      1        |      3     |     1      |      0        |      2       |     2        |
|   They   |  1   |      0        |      2     |     0      |      2        |      0       |     1        |
|   water  |  1   |      0        |      2     |      0     |      2        |      1       |     0        |

#### Pros and cons
* high dimension
* applying SVD


## Word2vec

### Distributional hypothesis
**Similar words have similar context.**

### Subsampling
Frequent words (such as the, and) are [subsampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) .
For each word in one sentence, it can be deleted or not according its frequency. And the hyper-parameter sampling rate (i.e., `sample` in [gensim](https://radimrehurek.com/gensim/models/word2vec.html) `Word2Vec`, default value is `1e-3`)


### Negative Sampling
Negative samples are selected proportional to its frequency (`f(w)^3/4`).


### Visualization tool: https://ronxin.github.io/wevi/


Reference:

http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes1.pdf
