# Tutorial

**To search papers on IEEE, Arxiv or Paper with Code platforms:**
```python
import trendflow as tf
tf.search_papers('machine learning', 2018, 2022, 50, to_pandas=True)
```

**To analyze the research trends:**
```python
import trendflow as tf
result = tf.trends_analysis('machine learning', 50,2018,2022,platforms=['ieee','arxiv'])
ieee_clusters, ieee_articles = result['ieee']

print(ieee_articles) # all the papers returned from IEEE

print(ieee_clusters) # check the information of the clusters

print(ieee_clusters[0].top_5_keyphrases) # check the top-5 keyphrases of the first cluster
```
