# -*- coding: utf-8 -*-

import pickle
from operator import itemgetter
# Python version: 2.7

def viterbi(obs, hidden_states, pi, a, b):
    # shape: (number of chars of obs, number of hidden_states)
    Delta = [{}]
    # shape: (number of chars of obs, number of hidden_states)
    Psi = [{}]

    # initialization for t = 0
    for i in hidden_states:
        # For obs[0], if there is no corresponding hidden state, then give MIN_FLOAT
        Delta[0][i] = pi[i] + b[i].get(obs[0], MIN_FLOAT)
        Psi[0][i] = 0

    # for t = 1, 2, ..., len(obs)-1
    for t in range(1, len(obs)):
        Delta.append({}) # add new row
        Psi.append({}) # add new row
        for i in hidden_states:
            # only consider part of previous states
            prev_status_list = [(Delta[t-1][j] + a[j].get(i, MIN_FLOAT), j) for j in prev_status[i]]
            max_prob, max_state = max(prev_status_list, key=itemgetter(0))
            Delta[t][i] = max_prob + b[i].get(obs[t], MIN_FLOAT)
            Psi[t][i] = max_state

    # the final state should be in ['E', 'S']
    best_prob, best_state = max([(Delta[len(obs)-1][i], i) for i in 'ES'], key=itemgetter(0))

    best_path = [] # generate most possible hidden states in order
    best_path.append(best_state)
    for inv in range(len(obs)-1, 0, -1):
        best_state = Psi[inv][best_state]
        best_path.insert(0, best_state)
    return (best_prob, best_path)

def cut(sentence, best_path_list):
    begin, nexti = 0, 0
    for i, char in enumerate(sentence): # iterate best hidden states list to segment sentence
        pos = best_path_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]


if __name__ == '__main__':

    MIN_FLOAT = -3.14e100

    # Begin, End, Middle, Single
    hidden_states = 'BMES'

    # initial probability vector of hidden states
    initial_hidden_states_pro_vec = pickle.load(open('prob_start.p', 'r'))

    # transition probability matrix of hidden states
    hidden_state_trans_pro_mat = pickle.load(open('prob_trans.p', 'r'))

    # generation probability matrix of observation sequence
    observation_seq_gen_pro_mat = pickle.load(open('prob_emit.p', 'r'))

    # for generating example data
    for h_s in hidden_states:
        print('='*10 + '>' + h_s)
        top_three = sorted(observation_seq_gen_pro_mat[h_s].items(), key=itemgetter(1), reverse=True)[0:3]
        for w in top_three:
            print(w[0].encode('utf-8'))
            print(observation_seq_gen_pro_mat[h_s][w[0]])

    #  previous states at time t-1
    prev_status = {
        'B': 'ES',
        'M': 'MB',
        'S': 'SE',
        'E': 'BM'
    }

    for test_sentence in [u'我喜欢吃苹果', u'苹果喜欢吃我']:
        prob, best_path_list = viterbi(test_sentence, hidden_states, initial_hidden_states_pro_vec, hidden_state_trans_pro_mat, observation_seq_gen_pro_mat)
        print(' '.join(list(cut(test_sentence, best_path_list))))