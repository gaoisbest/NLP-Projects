
# see https://github.com/facebookresearch/fastText/blob/master/tutorials/unsupervised-learning.md for more example 

# build model
./fasttext skipgram -input ./word_vector_108000.cs -output sg_ns_w2v_model -minCount 1 -dim 128 -ws 5 -loss ns -neg 5 -t 0.001 -epoch 5
Read 15M words
Number of words:  331438
Number of labels: 0
Progress: 100.0%  words/sec/thread: 27962  lr: 0.000000  loss: 1.231153  eta: 0h0m 

# test model 
./fasttext nn sg_ns_w2v_model.bin 
Pre-computing word vectors... done.
Query word? 奔驰
宝马 0.82771
梅赛德斯 0.824145
奔驰路 0.811303
奥迪 0.781083
梅赛 0.774891
捷豹 0.769684
路虎 0.754368
其肉 0.751838
宝马店 0.74773
雷克萨斯 0.746135
