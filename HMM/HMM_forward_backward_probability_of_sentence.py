# -*- coding: utf-8 -*-

import pickle
# Python version: 2.7

def forward(obs, hidden_states, pi, a, b):
    # shape: (number of chars of obs, number of hidden_states)
    alpha = [{}]

    # initialization for t = 0
    for i in hidden_states:
        # For obs[0], if there is no corresponding hidden state, then give MIN_FLOAT
        alpha[0][i] = pi[i] + b[i].get(obs[0], MIN_FLOAT)
        print('i:{}, b[i].get(obs[t], MIN_FLOAT):{}'.format(i, b[i].get(obs[0], MIN_FLOAT)))

    # for t = 1, 2, ..., len(obs)-1
    for t in range(1, len(obs)):
        alpha.append({}) # add new row
        print('='*100)
        print(obs[t])
        for i in hidden_states:
            # only consider part of previous hidden states
            prev_status_list = [alpha[t-1][j] + a[j].get(i, MIN_FLOAT) for j in prev_status[i]]
            alpha[t][i] = sum(prev_status_list) + b[i].get(obs[t], MIN_FLOAT)
            print('i:{}, b[i].get(obs[t], MIN_FLOAT):{}'.format(i, b[i].get(obs[t], MIN_FLOAT)))


    # hidden state at t = len(obs)-1 should be 'E' or 'S'
    prob = sum([alpha[len(obs)-1][i] for i in 'ES'])
    return prob, alpha

def backward(obs, hidden_states, pi, a, b):
    # shape: (number of chars of obs, number of hidden_states)
    beta = []
    for _ in range(len(obs)):
        beta.append({})

    # initialization for t = len(obs) - 1
    for i in hidden_states:
        beta[len(obs)-1][i] = 0 # since whole probabilities are in log scale, and log(1) = 0

    print(beta)

    # for t = len(obs)-2, len(obs)-3, ..., 0
    for t in range(len(obs)-2, -1, -1):
        for i in hidden_states:
            # only consider part of previous hidden states
            back_status_list = [a[i].get(j, MIN_FLOAT)+b[j].get(obs[t+1], MIN_FLOAT)+beta[t+1][j] for j in back_status[i]]
            beta[t][i] = sum(back_status_list)

    # hidden state at t = 0 should be 'B' or 'S'
    prob = sum([pi[i]+b[i].get(obs[0], MIN_FLOAT)+beta[0][i] for i in 'BS'])
    return prob, beta


if __name__ == '__main__':

    # initial value is -3.14e100, for illustration, let it equal to -1000 here
    MIN_FLOAT = -1000

    # Begin, End, Middle, Single
    hidden_states = 'BMES'

    # initial probability vector of hidden states
    initial_hidden_states_pro_vec = pickle.load(open('prob_start.p', 'r'))
    # for illustration, let it equal to -1000 here
    initial_hidden_states_pro_vec['M'] = MIN_FLOAT
    initial_hidden_states_pro_vec['E'] = MIN_FLOAT
    print(initial_hidden_states_pro_vec)

    # transition probability matrix of hidden states
    hidden_state_trans_pro_mat = pickle.load(open('prob_trans.p', 'r'))

    # generation probability matrix of observation sequence
    observation_seq_gen_pro_mat = pickle.load(open('prob_emit.p', 'r'))

    #  previous states at time t-1
    prev_status = {
        'B': 'ES',
        'M': 'MB',
        'S': 'SE',
        'E': 'BM'
    }

    # back states at time t+1
    back_status = {
        'B': 'ME',
        'M': 'ME',
        'S': 'SB',
        'E': 'BS'
    }

    for test_sentence in [u'我喜欢吃苹果']:#, u'苹果喜欢吃我']:#[[u'我', u'喜欢', u'吃', u'苹果'], [u'苹果', u'喜欢', u'吃', u'我']]: # ,
        print('='*100)
        print(' '.join(test_sentence).encode('utf-8'))
        forward_prob, alpha = forward(test_sentence, hidden_states, initial_hidden_states_pro_vec, hidden_state_trans_pro_mat, observation_seq_gen_pro_mat)
        backward_prob, beta = backward(test_sentence, hidden_states, initial_hidden_states_pro_vec, hidden_state_trans_pro_mat, observation_seq_gen_pro_mat)
        print('Forward probability:%f, backward probability:%f' % (forward_prob, backward_prob))
        print(alpha)
        print(beta)


        gamma = [{},{},{},{}]
        t = 3
        for i in hidden_states:
            all_states = [alpha[t][j]+beta[t][j] for j in hidden_states]
            gamma[t][i] = (alpha[t][i] + beta[t][i]) / sum(all_states)

        print(gamma)

        example_i = 'S'
        example_j = 'B'
        t = 0
        deno_sum = 0.0

        for i in hidden_states:
            for j in hidden_states:
                deno_sum += alpha[t][i] + hidden_state_trans_pro_mat[i].get(j, MIN_FLOAT) + observation_seq_gen_pro_mat[j].get(test_sentence[t+1], MIN_FLOAT) + beta[t+1][j]
        Xi_i_j = (alpha[t][example_i] + hidden_state_trans_pro_mat[example_i].get(example_j, MIN_FLOAT) + observation_seq_gen_pro_mat[example_j].get(test_sentence[t+1], MIN_FLOAT) + beta[t+1][example_j]) / deno_sum
        print('Xi_i_j:%f' % Xi_i_j)


