from typing import List
from .config import BaselineConfig, Configuration
from ..utils import __create_model__
from sklearn.preprocessing import StandardScaler
from .clusters import ClusterList
from unsupervised_learning.clustering import GaussianMixture, Silhouette

class ClusterPipeline:
    def __init__(self, config:Configuration = None):
        if config is None:
            self.__setup__(BaselineConfig())
        else:
            self.__setup__(config)

    def __setup__(self, config:Configuration):
        self.PTM = __create_model__(config.plm)
        self.dimension_reduction = __create_model__(config.dimension_reduction)
        self.clustering = __create_model__(config.clustering)
        self.keywords_extraction = __create_model__(config.keywords_extraction)

    def __1_generate_word_embeddings__(self, documents: List[str]):
        '''

        :param documents: a list of N strings:
        :return: np.ndarray: Nx384 (sentence-transformers)
        '''
        print(f'>>> start generating word embeddings...')
        print(f'>>> successfully generated word embeddings...')
        return self.PTM.encode(documents)

    def __2_dimenstion_reduction__(self, embeddings):
        '''

        :param embeddings: NxD
        :return: Nxd, d<<D
        '''
        if self.dimension_reduction is None:
            return embeddings
        print(f'>>> start dimension reduction...')
        embeddings = self.dimension_reduction.dimension_reduction(embeddings)
        print(f'>>> finished dimension reduction...')
        return embeddings

    def __3_clustering__(self, embeddings, return_cluster_centers = False, max_k: int =10, standarization = False):
        '''

        :param embeddings: Nxd
        :return:
        '''
        if self.clustering is None:
            return embeddings
        else:
            print(f'>>> start clustering...')

            ######## new: standarization ########
            if standarization:
                print(f'>>> start standardization...')
                scaler = StandardScaler()
                embeddings = scaler.fit_transform(embeddings)
                print(f'>>> finished standardization...')
            ######## new: standarization ########

            best_k_algo = Silhouette(GaussianMixture,2,max_k)
            best_k = best_k_algo.get_best_k(embeddings)
            print(f'>>> The best K is {best_k}.')

            labels, cluster_centers = self.clustering(embeddings, k=best_k)
            clusters = ClusterList(best_k)
            clusters.instantiate(labels)
            print(f'>>> finished clustering...')

            if return_cluster_centers:
                return clusters, cluster_centers
            return clusters

    def __4_keywords_extraction__(self, clusters: ClusterList, documents: List[str]):
        '''

        :param clusters: N documents
        :return: clusters, where each cluster has added keyphrases
        '''
        if self.keywords_extraction is None:
            return clusters
        else:
            print(f'>>> start keywords extraction')
            for cluster in clusters:
                doc_ids = cluster.get_elements()
                input_abstracts = [documents[i] for i in doc_ids] #[str]
                keyphrases = self.keywords_extraction(input_abstracts) #[{keys...}]
                cluster.add_keyphrase(keyphrases)
                # for doc_id in doc_ids:
                #     keyphrases = self.keywords_extraction(documents[doc_id])
                #     cluster.add_keyphrase(keyphrases)
            print(f'>>> finished keywords extraction')
            return clusters


    def __call__(self, documents: List[str], max_k:int, standarization = False):
        print(f'>>> pipeline starts...')
        x = self.__1_generate_word_embeddings__(documents)
        x = self.__2_dimenstion_reduction__(x)
        clusters = self.__3_clustering__(x,max_k=max_k,standarization=standarization)
        outputs = self.__4_keywords_extraction__(clusters, documents)
        print(f'>>> pipeline finished!\n')
        return outputs
