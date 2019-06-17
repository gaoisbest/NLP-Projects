# Probabilistic Graph Model
## 1. HMM
- **Generative**, Direct, Bayesian network
- `P(x_1, y_1, ..., x_n, y_n) = p(y_1)p(x_1|y_1)product[p(y_i|y_i-1)p(x_i|y_i)]`, where `x` is observed value and `y` is hidden variable

## 2. CRFs
- **Discriminative**, Undirect, Markov network
- `p(y|x) = 1/Z * exp(sum_j[sum_i[lambda_j * t_j[y_i+1, y_i, x, i]]] + sum_k[sum_i[miu_k * s_k[y_i, x, i]]])`, where `t_j[y_i+1, y_i, x, i]` is **transition feature function** defined on two neighbor position and `s_k[y_i, x, i]` is **status feature function** defined on position `i`

## 3. Approximate inference (近似推断)
### 3.1 Sampling
- Randomized approximation (随机近似)
- **Markov Chain Monte Carlo**
    - Build Markov chain with stationary distribution `p`
    - If the Markov chain runs long time and reach to stationary distribution `p`, then the samples `x` according to distribution `p`
#### 3.1.1 Metropolis-Hastings sampling
#### 3.1.2 Gibbs sampling

### 3.2. Variational inference
- Deterministic approximation (确定近似)
- Use known simple distribution to approximate to infered complex distribution

## 4. LDA
- **Generative**, Direct, Bayesian network
- One **document** has several **topics**, 
