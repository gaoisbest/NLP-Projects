# Concepts [1]
## 1. Graphical Model
### Definition
- It represents a complex distribution over many variables as a product of **local factors** on smaller subsets of variables.  
- The **factorization** of the probability density corresponds to a set of **conditional independence** (i.e., **Markov network**, see below) relationships satisfied by the distribution.
- **Markov network** are graphs over only **random variables**, rather than factors. It represents conditional independence (e.g., a and b are conditional independent if **p(a,b|c)=p(a|c)p(b|c)** or **p(a|b,c)=p(a|c)** [2]) realtionships in a multivariate distribution.

### Category
- Undirected model  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/CRFs/1_undirected_graphical_model.png)  
where  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/CRFs/1_undirected_graphical_model_Z.png)  
is called **partition function** and  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/CRFs/1_undirected_graphical_model_Psi.png)  
is a non-negative scalar, called **local function** or **local factor**.  
- Directed model  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/CRFs/2_graphical_model.png)  
where  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/CRFs/2_graphical_model_pai_s.png)  
is parents of `Ys`.

### Examples
- Bayesian networks
- Markov random fields

## 2. Generative model vs discriminative model
- Generative model: **p(y, x)**, **y generates x**.
  - Naive Bayes
  - HMM
- Discriminative model: **p(y|x)**.
  - Logistic regression
  - CRFs

## References
[1] An introduction to Conditional Random Fields  
[2] https://www.zhihu.com/question/51543503
