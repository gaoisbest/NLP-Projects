
# Introduction

In NLP, hidden Markov model (HMM) can act as a tool for sequence tagging problems, such as chinese word segmentation,  part-of-speech tagging and named-entity-recognition.

Hidden Markov chain generate **hidden states**, and each hidden state generate one observation, which will lead to **observation sequence**.

# Model parameters
- Initial hidden state probability vector `Pi` (shape: [N*1])
- Hidden state transition probability matrix `A` (shape: (N*N))
- Observation sequence generation probability matrix `B`(shape: [N*M]).

# Hypothesis:
- Hidden state probability distribution at time `t` is only depends on hidden states at time `t-1`.
- Observation at time `t` is only created by hidden states at time `t`.


# Three problems
- Given model parameters (i.e, `Pi A B`), calculate the **probability of observation sequence**. SOLUTION: **Forward-backward**.
- Given model parameters (i.e, `Pi A B`) and observation sequence, calculate the **most posibble hidden state sequence**. This problem also called **decoding**. SOLUTION: **Viterbi**.
- Given observation sequences, learning model parameters (i.e., `Pi A B`). SOLUTION: **Baum-Walch**.

# Relationship of HMM and RNN
HMM has a strong Markov chain hypothesis, i.e., HMM only can model fixed size inputs. 

Despite of efficiency, RNN can model arbitary length of inputs in theory.

In my view, RNN is more suitable for dealing with big data.


# Action 1: Viterbi - Chinese word segmentation
## Hidden states
```
B (Begin), M (Middle), E (End), S (Single)
```

## Initial hidden state probability vector `Pi`

The probabilities are in `log` scale, and the values are from [jieba](https://github.com/fxsjy/jieba/blob/master/jieba/finalseg/prob_start.py).

|          | propability |
|   :---:  |  :---:      |
|   B      | -0.26268660809250016|
|   M      | -3.14e+100          |
|   E      | -3.14e+100          |
|   S      | -1.4652633398537678 |


## Hidden state transition probability matrix `A`
The probabilities are in `log` scale, and the values are from [jieba](https://github.com/fxsjy/jieba/blob/master/jieba/finalseg/prob_trans.py).

|          |         B         |      M            |           E        |          S        |
|   :---:  |     :---:         |     --:           |          :---:     |           :---:   |
|    B     |        -3.14e+100 |-0.916290731874155 | -0.510825623765990 |  -3.14e+100       |
|    M     |     -3.14e+100    |-1.2603623820268226|-0.33344856811948514|  -3.14e+100       |
|    E     |-0.5897149736854513| -3.14e+100        |       -3.14e+100   |-0.8085250474669937|
|    S     |-0.7211965654669841| -3.14e+100        |      -3.14e+100    |-0.6658631448798212|

## Observation sequence generation probability matrix `B` of each hidden states (top 3)
The probabilities are in `log` scale, and the values are from [jieba](https://github.com/fxsjy/jieba/blob/master/jieba/finalseg/prob_emit.p).

|          |      一     |      不     |      中     |
|   :---:  |  :---:      |      :---:  |     :---:   |
|   B      |-3.65449787504|-4.27426205594|-4.59674331528|


|          |     民      |      大    |      國     |
|   :---:      |   :---:     |    :---:    |    :---:    |
|   M    |-3.83658638317|   -3.84593626859|  -4.12462500647 |


|          |     國     |      国     |     個     |
|   :---:  |  :---:      |    :---:    |  :---:      |
|   E      |-4.50500612889|-4.50500612889|  -4.60184809489   |

|          |    的     |      了     |      是     |
|   :---:  |  :---:      |   :---:     |    :---:    |
|   S      |    -2.24017668006 |      -3.52336181351   |      -3.62656137689    |

## Viterbi codes
[here](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/HMM/HMM_viterbi_word_segmentor.py)
