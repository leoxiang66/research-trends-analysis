class Configuration:
    def __init__(self, plm:str, dimension_reduction:str,clustering:str,keywords_extraction:str):
        self.plm = plm
        self.dimension_reduction = dimension_reduction
        self.clustering = clustering
        self.keywords_extraction = keywords_extraction


class BaselineConfig(Configuration):
    def __init__(self):
        super().__init__('''all-mpnet-base-v2''', 'none', 'kmeans-euclidean', 'keyphrase-transformer')
