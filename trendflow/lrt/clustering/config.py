class Configuration:
    def __init__(self, plm:str, dimension_reduction:str,clustering:str,keywords_extraction:str):
        self.plm = plm
        self.dimension_reduction = dimension_reduction
        self.clustering = clustering
        self.keywords_extraction = keywords_extraction

    def to_dict(self) -> dict:
        ret = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                ret[key] = value.to_dict()
            else:
                ret[key] = value
        return ret

    def __str__(self):
        return self.to_dict().__str__()
    def __repr__(self):
        return self.to_dict().__repr__()


class BaselineConfig(Configuration):
    def __init__(self):
        super().__init__('''all-mpnet-base-v2''', 'none', 'kmeans-euclidean', 'keyphrase-transformer')
