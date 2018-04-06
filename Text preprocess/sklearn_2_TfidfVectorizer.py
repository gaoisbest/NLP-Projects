
from sklearn.feature_extraction.text import TfidfVectorizer

input_texts = ['你 真的 真的 很好 啊', '我 也 真的 很好 啊', '大家 真的 很好 啊']

tfidf_vec = TfidfVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
tfidf_vec.fit(input_texts)

# {'你': 1, '真的': 6, '很好': 4, '啊': 2, '我': 5, '也': 0, '大家': 3}
print(tfidf_vec.vocabulary_)

# [1.69314718 1.69314718 1.         1.69314718 1.         1.69314718 1.]
# the idf of words 啊 很好 很好 is 1.0
print(tfidf_vec.idf_)


# TfidfVectorizer = CounterVectorizer + TfidfTransformer