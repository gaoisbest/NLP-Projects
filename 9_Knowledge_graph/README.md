# Introduction
- Resource Description Framework (RDF)
  - Resource: object that has URI identification, such as web page, images, videos
  - Description: attributes, features and relations among resources
  - Framework: description model, language and syntax
  - Basic unit, triple pattern: Subject(主语) -- predicate(谓语) -- Object(宾语)
  - Hop: path from source entity to target entity
  - SPARQL: query language for RDF
    - Variable in RDF starts with '?' or '$'
- Compared with deep learning, knowledge graph provides **interpretable**.

# Principle
- **1. Data**
  - 1.1 Structured data (graph mapping)
    - E.g., relational database (i.e., Database2Rdf), open kg (i.e., linked data, graph mapping 图映射)
  - 1.2 Semi-structured data (wrapper)
    - E.g., web page, web table, Wikipedia infobox 
  - 1.3 Unstructured data (information extraction, often in closed domin)
    - E.g., natural language, images, video
- **2. Knowledge extraction from 1.3 Unstructured data**
  - **2.1 Entity extraction**
    - Value/number detection and recognition
    - [Running example](https://github.com/gaoisbest/NLP-Projects/blob/master/6_Sequence_labeling/Named_Entity_Recognition/README.md)
    - **Entity linking**
      - Definition: Find the entity (i.e., **entity mention** 实体指称项) in text and linking it to existing knowledge graph
      - Entity Disambiguation 实体消歧, page rank
      - Co-reference Resolution (CR) 共指消解
  - **2.2 Relation extraction**
      - **Input**: unstructured text, a group of entities. **Output**: a group of triplets, e.g., (First Entity, Second Entity, Relation Type)
      - Methods [1, 2]
        - **Pattern / rule matching**
          - Trigger word pattern
          - Dependency parsing pattern, verb is trigger word. [Running example](https://mp.weixin.qq.com/s/Q-WMYSTjGGxIMGNq-wfpRg)
        - **Supervised method**
          - [Running example](https://www.microsoft.com/developerblog/2016/09/13/training-a-classifier-for-relation-extraction-from-medical-literature/)
          - Two classifier
            - **Yes or no** classifier determine if there is a relation
            - **Relation** classifier determine the exact relation
        - **Semi-supervised method**
          - **Bootstrapping**
            - Example [1].
          - **Distant supervision**
            - **Input**: unstructured text, database contains known entity relations. **Output**; a set of labeled data
            - Combination of **bootstrapping** and **supervised**. Cannot find **new** relationship. Example [1].
            - Deep model: PCNN
  - **2.3 Event extraction**
    - Trigger word 触发词
    - Time 时间
    - Location 地点
    - **Event detection and tracking**
- ** 3.Knowledge fusion**
  - Entity alignment
    - Similar description
    - Similar attribute - value
    - Similar neighbor entities
  - Schema matching
  - Instance matching
- Knowledge representation learning
  - Convert entity and relationship into vectors
  - Application: link prediction (given S and P, predict O) or relation prediction, knowledge reasoning
  - Translation based Methods
    - **TransE**: head + relation = tail
      - Drawbacks
        - cannot process one-vs-multiple (一对多), multiple-vs-one (多对一) or multiple-vs-multiple (多对多) relationship
        - cannot process symmetric relationship
    - **TransH**
    - **TransR**
# Tools
- [Protege](https://protege.stanford.edu): edit ontology by hand, visualization, and reasoning
- [DeepDive](http://deepdive.stanford.edu/): relation extraction, [tutorial](http://www.openkg.cn/tool/cn-deepdive)

# Graph database
- [Nebula](https://nebula-graph.io/), used by MeiTuan

# Application
- Question answering
  - How to convert natural language to query language ?
- Searching
- Recommender system
- Event prediction
- Knowledge reasoning
- Financial

# Examples
- [OpenKG.CN](http://openkg.cn/) publishes most recent report.
- [CN-DBPedia](http://kw.fudan.edu.cn/apis/intro/) extract structured information from Baidu Baike.
- [Zhishi.me](http://zhishi.me/) ensembles Baidu Baike, Hudong Baike and Chinese Wikipedia. [Dump data](http://openkg.cn/dataset/zhishi-me-dump)
- [Agriculture Knowledge Graph](https://github.com/qq547276542/Agriculture_KnowledgeGraph)
- [Knowledge graph demo](https://github.com/Shuang0420/knowledge_graph_demo)
- [KBQA based on ES](http://openkg.cn/tool/elasticsearch-kbqa)

# Reference
[1] [NLP笔记-Relation Extraction](http://www.shuang0420.com/2017/04/10/NLP%E7%AC%94%E8%AE%B0%20-%20Relation%20Extraction/)  
[2] [知识抽取-实体及关系抽取](http://www.shuang0420.com/2018/09/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E5%AE%9E%E4%BD%93%E5%8F%8A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96/)

