from .clustering import *
from typing import List
import textdistance as td
from .utils import UnionFind, ArticleList
from .academic_query import AcademicQuery
from .clustering.clusters import KeyphraseCount



class LiteratureResearchTool:
    def __init__(self, cluster_config: Configuration = None):
        self.literature_search = AcademicQuery
        self.cluster_pipeline = ClusterPipeline(cluster_config)


    def __postprocess_clusters__(self, clusters: ClusterList,query: str) ->ClusterList:
        '''
        add top-5 keyphrases to each cluster
        :param clusters:
        :return: clusters
        '''
        def condition(x: KeyphraseCount, y: KeyphraseCount):
            return td.ratcliff_obershelp(x.keyphrase, y.keyphrase) > 0.8

        def valid_keyphrase(x:KeyphraseCount):
            tmp = x.keyphrase
            return tmp is not None and tmp != '' and not tmp.isspace() and  len(tmp)!=1\
                 and  tmp != query


        for cluster in clusters:

            keyphrases = cluster.get_keyphrases() # [kc]
            keyphrases = list(filter(valid_keyphrase,keyphrases))
            unionfind = UnionFind(keyphrases, condition)
            unionfind.union_step()

            tmp = unionfind.get_unions() # dict(root_id = [kc])
            tmp = tmp.values() # [[kc]]
            # [[kc]] -> [ new kc] -> sorted
            tmp = [KeyphraseCount.reduce(x) for x in tmp]
            keyphrases = sorted(tmp,key= lambda x: x.count,reverse=True)[:5]
            keyphrases = [x.keyphrase for x in keyphrases]

            # keyphrases = sorted(list(unionfind.get_unions().values()), key=len, reverse=True)[:5]  # top-5 keyphrases: list
            # for i in keyphrases:
            #     tmp = '/'.join(i)
            #     cluster.top_5_keyphrases.append(tmp)
            cluster.top_5_keyphrases = keyphrases

        return clusters

    def __call__(self,
                 query: str,
                 num_papers: int,
                 start_year: int,
                 end_year: int,
                 max_k: int,
                 platforms: List[str] = ['IEEE', 'Arxiv', 'Paper with Code'],
                 loading_ctx_manager = None,
                 standardization = False
                 ):

        ret = dict()
        for platform in platforms:
            print(f'>>> Search on {platform}...')
            if loading_ctx_manager:
                with loading_ctx_manager():
                    clusters, articles = self.__platformPipeline__(platform,query,num_papers,start_year,end_year,max_k,standardization)
            else:
                clusters, articles = self.__platformPipeline__(platform, query, num_papers, start_year, end_year,max_k,standardization)

            clusters.sort()
            if platform == 'IEEE':
                ret['ieee'] = clusters,articles
            elif platform == 'Arxiv':
                ret['arxiv'] = clusters,articles
            elif platform == 'Paper with Code':
                ret ['paper_with_code'] = clusters,articles
        return ret


    def yield_(self,
            query: str,
            num_papers: int,
            start_year: int,
            end_year: int,
            max_k: int,
            platforms: List[str] = ['IEEE', 'Arxiv', 'Paper with Code'],
            loading_ctx_manager=None,
            standardization=False
    ):
        for platform in platforms:
            print(f'>>> Search on {platform}...')
            if loading_ctx_manager:
                with loading_ctx_manager():
                    clusters, articles = self.__platformPipeline__(platform,query,num_papers,start_year,end_year,max_k,standardization)
            else:
                clusters, articles = self.__platformPipeline__(platform, query, num_papers, start_year, end_year,max_k,standardization)

            clusters.sort()
            yield clusters,articles

    def __platformPipeline__(self,platforn_name:str,
                             query: str,
                             num_papers: int,
                             start_year: int,
                             end_year: int,
                             max_k: int,
                             standardization
                             ) -> (ClusterList,ArticleList):


        def ieee_process(
                query: str,
                num_papers: int,
                start_year: int,
                end_year: int,
        ):
            articles = ArticleList.parse_ieee_articles(
            self.literature_search.ieee(query, start_year, end_year, num_papers))  # ArticleList
            abstracts = articles.getAbstracts()  # List[str]
            clusters = self.cluster_pipeline(abstracts,max_k,standardization)
            clusters = self.__postprocess_clusters__(clusters,query)
            return clusters, articles

        def arxiv_process(
                query: str,
                num_papers: int,
        ):
            articles = ArticleList.parse_arxiv_articles(
            self.literature_search.arxiv(query, num_papers))  # ArticleList
            abstracts = articles.getAbstracts()  # List[str]
            clusters = self.cluster_pipeline(abstracts,max_k,standardization)
            clusters = self.__postprocess_clusters__(clusters,query)
            return clusters, articles


        def pwc_process(
                query: str,
                num_papers: int,
        ):
            articles = ArticleList.parse_pwc_articles(
            self.literature_search.paper_with_code(query, num_papers))  # ArticleList
            abstracts = articles.getAbstracts()  # List[str]
            clusters = self.cluster_pipeline(abstracts,max_k,standardization)
            clusters = self.__postprocess_clusters__(clusters,query)
            return clusters, articles

        if platforn_name == 'IEEE':
            return ieee_process(query,num_papers,start_year,end_year)
        elif platforn_name == 'Arxiv':
            return arxiv_process(query,num_papers)
        elif platforn_name == 'Paper with Code':
            return pwc_process(query,num_papers)
        else:
            raise RuntimeError('This platform is not supported. Please open an issue on the GitHub.')











