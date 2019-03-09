## 1. Graphical Model
### Definition
- It represents a complex distribution over many variables as a product of **local factors** on smaller subsets of variables.  
- The **factorization** of the probability density corresponds to a set of **conditional independence** (i.e., **Markov network**, see below) relationships satisfied by the distribution.
- **Markov network** are graphs over only **random variables**, rather than factors. It represents conditional independence (e.g., a and b are conditional independent if **p(a,b|c)=p(a|c)p(b|c)** or **p(a|b,c)=p(a|c)** [2]) realtionships in a multivariate distribution.
- **random field** refers to a particular distribution which defined by an undirected model.

### Category
- Undirected model  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/1_undirected_graphical_model.png)  
where  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/1_undirected_graphical_model_Z.png)  
is called **partition function** and  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/1_undirected_graphical_model_Psi.png)  
is a non-negative scalar, called **local function** or **local factor**.  
- Directed model  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/2_graphical_model.png)  
where  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/2_graphical_model_pai_s.png)  
is parents of `Ys`.

### Examples
- Bayesian networks
- Markov random fields

## 2. Generative model vs discriminative model
- Generative model: label **y** probabilistically **'generate'** feature **x**. **p(y, x) = p(y) * p(x|y)**.
  - Naive Bayes: features are independent.  
  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/3_Naive_bayes_formula.png)  
  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/3_Naive_bayes_directed_model.png)
  - HMM  
  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/4_HMM_formula.png)  
  which can be rewritten as  
  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/4_HMM_formula_another_way.png)  
  where  
  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/4_HMM_formula_another_way_parameter.png)  
    - Application: [Chinese word segmentation](https://github.com/gaoisbest/NLP-Projects/tree/master/6_Sequence_labeling/Chinese_word_segmentation/HMM_viterbi_word_segmentor.py), for more details, see [Chinese notes](http://url.cn/5TFdvty).
  - How to incorporate **interdependent** features into generative model ? 
    - Simplify independent output assumption.  ![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/4_HMM_interdependent_feature.png) 
- Discriminative model: from feature **x** to predict label **y**. **p(y|x)**. Make conditional independence assumption among **y**, but **not** among **x**.
  - Logistic regression
  - CRFs

- Both models are avoid modeling **p(x)**, which is difficulty that **x** often contains highly dependent feautres.  

![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/5_NB_LR_HMM_CRF.png)

## 3. Linear-chain CRFs
- Definition  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/6_linear_chain_CRFs.png)  
where  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/6_linear_chain_CRFs_Z.png)  
- Categories  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/6_linear_chain_CRFs_category_1.png)  
![](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/CRFs/6_linear_chain_CRFs_category_2_3.png)

## References
[1] An introduction to Conditional Random Fields  
[2] https://www.zhihu.com/question/51543503
