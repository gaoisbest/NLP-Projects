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
    - [Running example](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/README.md)
    - **Entity linking**
      - Definition: Find the entity (i.e., **entity mention** 实体指称项) in text and linking it to existing knowledge graph
      - Entity Disambiguation 实体消歧, page rank
      - Co-reference Resolution (CR) 共指消解
  - **2.2 Relation extraction**
      - **Input**: unstructured text, a group of entities. **Output**: a group of triplets, e.g., (First Entity, Second Entity, Relation Type)
      - Methods
        - **Pattern / rule matching**
          - Trigger word pattern
          - Dependency parsing pattern, verb is trigger word. [Running example](https://mp.weixin.qq.com/s/Q-WMYSTjGGxIMGNq-wfpRg)
        - **Supervised method**
          - [Running example](https://www.microsoft.com/developerblog/2016/09/13/training-a-classifier-for-relation-extraction-from-medical-literature/)
        - **Semi-supervised method**
          - **Distant supervision**
            - **Input**: unstructured text, database contains known entity relations. **Output**; a set of labeled data
            - Cannot find **new** relationship
            - Deep model: PCNN
          - **Bootstrapping**
  - **2.3 Event extraction** (i.e., 多元关系抽取)
    - Trigger word 触发词
    - Time 时间
    - Location 地点
    - **Event detection and tracking**
- Knowledge fusion
  - Entity alignment
    - Similar description
    - Similar attribute - value
    - Similar neighbor entities
# Tools
- [Protege](https://protege.stanford.edu): edit ontology by hand, visualization, and reasoning
- [DeepDive](http://deepdive.stanford.edu/)
 
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


