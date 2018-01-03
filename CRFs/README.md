# Concepts
## Graphical Model
### Definition
It represents a complex distribution over many variables as a product of **local factors** on smaller subsets of variables.  
The **factorization** of the probability density corresponds to a set of **conditional independence** relationships satisfied by the distribution.
### Category
- Directed model
- Undirected model
### Examples
- Bayesian networks
- Markov random fields

## Generative model vs discriminative model
- Generative model: **p(y, x)**, **y generates x**.
  - Naive Bayes
  - HMM
- Discriminative model: **p(y|x)**.
  - Logistic regression
  - CRFs

## References
[1] An introduction to Conditional Random Fields
