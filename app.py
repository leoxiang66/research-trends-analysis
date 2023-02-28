if __name__ == '__main__':
    import trendflow as tf
    import pandas as pd
    clusters, articles = tf.trends_analysis('machine learning', 50,2018,2022,platforms=['IEEE'])['ieee']
    print(clusters)
    print(articles)