
# see https://github.com/facebookresearch/fastText/blob/master/tutorials/unsupervised-learning.md for more example 

# build model
./fasttext cbow -input word_vector_108000.cs -output cbow_hs_w2v_model -minCount 1 -dim 128 -ws 5 -loss hs -t 0.001 -epoch 5

#Read 15M word
#Number of words:  331438
#Number of labels: 0
#Progress: 100.0%  words/sec/thread: 61957  lr: 0.000000  loss: 7.671538  eta: 0h0m 

# test model
./fasttext nn cbow_hs_w2v_model.bin 
#Pre-computing word vectors... done.
Query word? 奔驰
奔驰路 0.729661
奥迪 0.653473
气哄哄 0.582002
雷克萨斯 0.549148
奔驰女 0.548206
骏马奔驰 0.53358
英菲尼迪 0.531917
国产奔驰 0.530568
宝马 0.520426
超跑车 0.502464