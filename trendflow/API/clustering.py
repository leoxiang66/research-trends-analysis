from ..lrt_instance import baseline_lrt

def trends_analysis(
        query:str,
        num_papers: int,
        start_year:int,
        end_year:int,
        platforms =['IEEE', 'Arxiv', 'Paper with Code']
                    ):

    return baseline_lrt(
        query,
        num_papers,
        start_year,
        end_year,
        platforms = platforms,
        max_k=10
    )