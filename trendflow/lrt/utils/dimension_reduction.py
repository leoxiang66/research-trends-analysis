from sklearn.decomposition import PCA as pca


class BaseDimensionReduction:
    def dimension_reduction(self,X):
        raise NotImplementedError()

class PCA(BaseDimensionReduction):
    def __init__(self, n_components: int = 0.8, *args, **kwargs) -> None:
        super().__init__()
        self.pca = pca(n_components,*args,**kwargs)


    def dimension_reduction(self, X):
        self.pca.fit(X=X)
        print(f'>>> The reduced dimension is {self.pca.n_components_}.')
        return self.pca.transform(X)
