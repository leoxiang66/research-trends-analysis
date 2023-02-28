from ..lrt.academic_query import AcademicQuery
import pandas as pd

def search_papers(query:str,start_year:int,end_year:int,num_papers:int, to_pandas=False):
    ieee = AcademicQuery.ieee(query=query,start_year=start_year,end_year=end_year,num_papers=num_papers)
    arxiv = AcademicQuery.arxiv(query,num_papers)
    pwc = AcademicQuery.paper_with_code(query,num_papers)

    if to_pandas:
        return dict(
            ieee = pd.DataFrame(ieee),
            arxiv = pd.DataFrame(arxiv),
            paper_with_code = pd.DataFrame(pwc)
        )
    else:
        return dict(
            ieee = ieee,
            arxiv = arxiv,
            paper_with_code = pwc
        )
