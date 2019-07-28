
from sklearn.feature_extraction.text import CountVectorizer

input_text = ['你 真的 真的 很好 啊']

count_vec_1 = CountVectorizer()
count_vec_1.fit(input_text)

# since defulat token_pattern=’(?u)\b\w\w+\b’, it will exclude one character words
print(count_vec_1.vocabulary_) # outputs: {'真的': 1, '很好': 0}

res_1 = count_vec_1.transform(input_text)
print(res_1.toarray()) # [[1 2]]


# change to token_pattern=u"(?u)\\b\\w+\\b" works !
# see https://stackoverflow.com/questions/33260505/countvectorizer-ignoring-i
count_vec_2 = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
count_vec_2.fit(input_text)
print(count_vec_2.vocabulary_) # outputs: {'你': 0, '真的': 3, '很好': 2, '啊': 1}
res_2 = count_vec_2.transform(input_text)
print(res_2.toarray()) # [[1 1 1 2]]