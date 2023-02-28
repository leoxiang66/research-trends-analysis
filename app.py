if __name__ == '__main__':
    import trendflow as tf
    import pandas as pd
    results = tf.search_papers('machine learning', 2018, 2022, 50, to_pandas=True)
    print(results)