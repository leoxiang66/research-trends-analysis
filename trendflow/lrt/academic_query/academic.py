from requests_toolkit import ArxivQuery,IEEEQuery,PaperWithCodeQuery
from typing import List

class AcademicQuery:
    @classmethod
    def arxiv(cls,
              query: str,
              max_results: int = 50
              ) -> List[dict]:
        ret = ArxivQuery.query(query,'',0,max_results)
        if not isinstance(ret,list):
            return [ret]
        return ret

    @classmethod
    def ieee(cls,
             query: str,
             start_year: int,
             end_year: int,
             num_papers: int = 200
             ) -> List[dict]:
        IEEEQuery.__setup_api_key__('vpd9yy325enruv27zj2d353e')
        ret = IEEEQuery.query(query,start_year,end_year,num_papers)
        if not isinstance(ret,list):
            return [ret]
        return ret

    @classmethod
    def paper_with_code(cls,
                        query: str,
                        items_per_page = 50) ->List[dict]:
        ret = PaperWithCodeQuery.query(query, 1,items_per_page)
        if not isinstance(ret, list):
            return [ret]
        return ret