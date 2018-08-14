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

# Pipeline
- From unstructured natural language
- Entity extraction -> relation extraction -> graph storage -> retrieval.
- Knowledge fusion
  - Entity alignment
    - Similar description
    - Similar attribute - value
    - Similar neighbor entities
# Tools
- [Protege](https://protege.stanford.edu): edit ontology by hand, visualization, and reasoning
 
# Application
- Question answering
  - How to convert natural language to query language ?
- Searching
- Recommender system
- Event prediction
- Knowledge reasoning

# Examples
- [OpenKG.CN](http://openkg.cn/) publishes most recent report.
