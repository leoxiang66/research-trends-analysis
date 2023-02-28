from typing import List, Union, Optional
import pandas as pd
class Article:
    '''
    attributes:
    - title: str
    - authors: list of str
    - abstract: str
    - url: str
    - publication_year: int
    '''
    def __init__(self,
                 title: str,
                 authors: List[str],
                 abstract: str,
                 url: str,
                 publication_year: int
                 ) -> None:
        super().__init__()
        self.title = title
        self.authors = authors
        self.url = url
        self.publication_year = publication_year
        self.abstract = abstract.replace('\n',' ')
    def __str__(self):
        ret = ''
        ret +=self.title +'\n- '
        ret +=f"authors: {';'.join(self.authors)}" + '\n- '
        ret += f'''abstract: {self.abstract}''' + '\n- '
        ret += f'''url: {self.url}'''+ '\n- '
        ret += f'''publication year: {self.publication_year}'''+ '\n\n'

        return ret

    def getDict(self) -> dict:
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'url': self.url,
            'publication_year': self.publication_year
        }

class ArticleList:
    '''
    list of articles
    '''
    def __init__(self,articles:Optional[Union[Article, List[Article]]]=None) -> None:
        super().__init__()
        self.__list__ = [] # List[Article]

        if articles is not None:
            self.addArticles(articles)

    def addArticles(self, articles:Union[Article, List[Article]]):
        if isinstance(articles,Article):
            self.__list__.append(articles)
        elif isinstance(articles, list):
            self.__list__ += articles

    # subscriptable and slice-able
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__list__[idx]
        if isinstance(idx, slice):
            # return
            return self.__list__[0 if idx.start is None else idx.start: idx.stop: 0 if idx.step is None else idx.step]


    def __str__(self):
        ret = f'There are {len(self.__list__)} articles:\n'
        for id, article in enumerate(self.__list__):
            ret += f'{id+1}) '
            ret += f'{article}'

        return ret

    # return an iterator that can be used in for loop etc.
    def __iter__(self):
        return self.__list__.__iter__()

    def __len__(self):
        return len(self.__list__)

    def getDataFrame(self) ->pd.DataFrame:
        return pd.DataFrame(
            [x.getDict() for x in self.__list__]
        )


    @classmethod
    def parse_ieee_articles(cls,items: Union[dict, List[dict]]):
        if isinstance(items,dict):
            items = [items]

        ret = [
            Article(
                title=item['title'],
                authors=[x['full_name'] for x in item['authors']['authors']],
                abstract=item['abstract'],
                url=item['html_url'],
                publication_year=item['publication_year']
            )
            for item in items ] # List[Article]

        ret = ArticleList(ret)
        return ret

    @classmethod
    def parse_arxiv_articles(cls, items: Union[dict, List[dict]]):
        if isinstance(items, dict):
            items = [items]

        def __getAuthors__(item):
            if isinstance(item['author'],list):
                return [x['name'] for x in item['author']]
            else:
                return [item['author']['name']]

        ret = [
            Article(
                title=item['title'],
                authors=__getAuthors__(item),
                abstract=item['summary'],
                url=item['id'],
                publication_year=item['published'][:4]
            )
            for item in items]  # List[Article]

        ret = ArticleList(ret)
        return ret

    @classmethod
    def parse_pwc_articles(cls, items: Union[dict, List[dict]]):
        if isinstance(items, dict):
            items = [items]

        ret = [
            Article(
                title=item['title'],
                authors=item['authors'],
                abstract=item['abstract'],
                url=item['url_abs'],
                publication_year=item['published'][:4]
            )
            for item in items]  # List[Article]

        ret = ArticleList(ret)
        return ret

    def getAbstracts(self) -> List[str]:
        return [x.abstract for x in self.__list__]

    def getTitles(self) -> List[str]:
        return [x.title for x in self.__list__]

    def getArticles(self) -> List[Article]:
        return self.__list__

if __name__ == '__main__':
    item = [{'doi': '10.1109/COMPSAC51774.2021.00100',
 'title': 'Towards Developing An EMR in Mental Health Care for Children’s Mental Health Development among the Underserved Communities in USA',
 'publisher': 'IEEE',
 'isbn': '978-1-6654-2464-6',
 'issn': '0730-3157',
 'rank': 1,
 'authors': {'authors': [{'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088961521',
    'id': 37088961521,
    'full_name': 'Kazi Zawad Arefin',
    'author_order': 1},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962639',
    'id': 37088962639,
    'full_name': 'Kazi Shafiul Alam Shuvo',
    'author_order': 2},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088511010',
    'id': 37088511010,
    'full_name': 'Masud Rabbani',
    'author_order': 3},
   {'affiliation': 'Product Developer, Marquette Energy Analytics, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088961612',
    'id': 37088961612,
    'full_name': 'Peter Dobbs',
    'author_order': 4},
   {'affiliation': 'Next Step Clinic, Mental Health America of WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962516',
    'id': 37088962516,
    'full_name': 'Leah Jepson',
    'author_order': 5},
   {'affiliation': 'Next Step Clinic, Mental Health America of WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962336',
    'id': 37088962336,
    'full_name': 'Amy Leventhal',
    'author_order': 6},
   {'affiliation': 'Department of Psychology, Marquette University, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962101',
    'id': 37088962101,
    'full_name': 'Amy Vaughan Van Heeke',
    'author_order': 7},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37270354900',
    'id': 37270354900,
    'full_name': 'Sheikh Iqbal Ahamed',
    'author_order': 8}]},
 'access_type': 'LOCKED',
 'content_type': 'Conferences',
 'abstract': "Next Step Clinic (NSC) is a neighborhood-based mental clinic in Milwaukee in the USA for early identification and intervention of Autism spectrum disorder (ASD) children. NSC's primary goal is to serve the underserved families in that area with children aged 15 months to 10 years who have ASD symptoms free of cost. Our proposed and implemented Electronic Medical Records (NSC: EMR) has been developed for NSC. This paper describes the NSC: EMR's design specification and whole development process with the workflow control of this system in NSC. This NSC: EMR has been used to record the patient’s medical data and make appointments both physically or virtually. The integration of standardized psychological evaluation form has reduced the paperwork and physical storage burden for the family navigator. By deploying the system, the family navigator can increase their productivity from the screening to all intervention processes to deal with ASD children. Even in the lockdown time, due to the pandemic of COVID-19, about 84 ASD patients from the deprived family at that area got registered and took intervention through this NSC: EMR. The usability and cost-effective feature has already shown the potential of NSC: EMR, and it will be scaled to serve a large population in the USA and beyond.",
 'article_number': '9529808',
 'pdf_url': 'https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9529808',
 'html_url': 'https://ieeexplore.ieee.org/document/9529808/',
 'abstract_url': 'https://ieeexplore.ieee.org/document/9529808/',
 'publication_title': '2021 IEEE 45th Annual Computers, Software, and Applications Conference (COMPSAC)',
 'conference_location': 'Madrid, Spain',
 'conference_dates': '12-16 July 2021',
 'publication_number': 9529349,
 'is_number': 9529356,
 'publication_year': 2021,
 'publication_date': '12-16 July 2021',
 'start_page': '688',
 'end_page': '693',
 'citing_paper_count': 2,
 'citing_patent_count': 0,
 'index_terms': {'ieee_terms': {'terms': ['Pediatrics',
    'Pandemics',
    'Navigation',
    'Mental health',
    'Tools',
    'Software',
    'Information technology']},
  'author_terms': {'terms': ['Electronic medical record (EMR)',
    'Mental Health Care (MHC)',
    'Autism Spectrum Disorder (ASD)',
    'Health Information Technology (HIT)',
    'Mental Health Professional (MHP)']}},
 'isbn_formats': {'isbns': [{'format': 'Print on Demand(PoD) ISBN',
    'value': '978-1-6654-2464-6',
    'isbnType': 'New-2005'},
   {'format': 'Electronic ISBN',
    'value': '978-1-6654-2463-9',
    'isbnType': 'New-2005'}]}},{'doi': '10.1109/COMPSAC51774.2021.00100',
 'title': 'Towards Developing An EMR in Mental Health Care for Children’s Mental Health Development among the Underserved Communities in USA',
 'publisher': 'IEEE',
 'isbn': '978-1-6654-2464-6',
 'issn': '0730-3157',
 'rank': 1,
 'authors': {'authors': [{'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088961521',
    'id': 37088961521,
    'full_name': 'Kazi Zawad Arefin',
    'author_order': 1},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962639',
    'id': 37088962639,
    'full_name': 'Kazi Shafiul Alam Shuvo',
    'author_order': 2},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088511010',
    'id': 37088511010,
    'full_name': 'Masud Rabbani',
    'author_order': 3},
   {'affiliation': 'Product Developer, Marquette Energy Analytics, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088961612',
    'id': 37088961612,
    'full_name': 'Peter Dobbs',
    'author_order': 4},
   {'affiliation': 'Next Step Clinic, Mental Health America of WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962516',
    'id': 37088962516,
    'full_name': 'Leah Jepson',
    'author_order': 5},
   {'affiliation': 'Next Step Clinic, Mental Health America of WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962336',
    'id': 37088962336,
    'full_name': 'Amy Leventhal',
    'author_order': 6},
   {'affiliation': 'Department of Psychology, Marquette University, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37088962101',
    'id': 37088962101,
    'full_name': 'Amy Vaughan Van Heeke',
    'author_order': 7},
   {'affiliation': 'Department of Computer Science, Ubicomp Lab, Marquette University, Milwaukee, WI, USA',
    'authorUrl': 'https://ieeexplore.ieee.org/author/37270354900',
    'id': 37270354900,
    'full_name': 'Sheikh Iqbal Ahamed',
    'author_order': 8}]},
 'access_type': 'LOCKED',
 'content_type': 'Conferences',
 'abstract': "Next Step Clinic (NSC) is a neighborhood-based mental clinic in Milwaukee in the USA for early identification and intervention of Autism spectrum disorder (ASD) children. NSC's primary goal is to serve the underserved families in that area with children aged 15 months to 10 years who have ASD symptoms free of cost. Our proposed and implemented Electronic Medical Records (NSC: EMR) has been developed for NSC. This paper describes the NSC: EMR's design specification and whole development process with the workflow control of this system in NSC. This NSC: EMR has been used to record the patient’s medical data and make appointments both physically or virtually. The integration of standardized psychological evaluation form has reduced the paperwork and physical storage burden for the family navigator. By deploying the system, the family navigator can increase their productivity from the screening to all intervention processes to deal with ASD children. Even in the lockdown time, due to the pandemic of COVID-19, about 84 ASD patients from the deprived family at that area got registered and took intervention through this NSC: EMR. The usability and cost-effective feature has already shown the potential of NSC: EMR, and it will be scaled to serve a large population in the USA and beyond.",
 'article_number': '9529808',
 'pdf_url': 'https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9529808',
 'html_url': 'https://ieeexplore.ieee.org/document/9529808/',
 'abstract_url': 'https://ieeexplore.ieee.org/document/9529808/',
 'publication_title': '2021 IEEE 45th Annual Computers, Software, and Applications Conference (COMPSAC)',
 'conference_location': 'Madrid, Spain',
 'conference_dates': '12-16 July 2021',
 'publication_number': 9529349,
 'is_number': 9529356,
 'publication_year': 2021,
 'publication_date': '12-16 July 2021',
 'start_page': '688',
 'end_page': '693',
 'citing_paper_count': 2,
 'citing_patent_count': 0,
 'index_terms': {'ieee_terms': {'terms': ['Pediatrics',
    'Pandemics',
    'Navigation',
    'Mental health',
    'Tools',
    'Software',
    'Information technology']},
  'author_terms': {'terms': ['Electronic medical record (EMR)',
    'Mental Health Care (MHC)',
    'Autism Spectrum Disorder (ASD)',
    'Health Information Technology (HIT)',
    'Mental Health Professional (MHP)']}},
 'isbn_formats': {'isbns': [{'format': 'Print on Demand(PoD) ISBN',
    'value': '978-1-6654-2464-6',
    'isbnType': 'New-2005'},
   {'format': 'Electronic ISBN',
    'value': '978-1-6654-2463-9',
    'isbnType': 'New-2005'}]}}]
    ieee_articles = ArticleList.parse_ieee_articles(item)
    print(ieee_articles)

    item = [{'id': 'http://arxiv.org/abs/2106.08047v1',
 'updated': '2021-06-15T11:07:51Z',
 'published': '2021-06-15T11:07:51Z',
 'title': 'Comparisons of Australian Mental Health Distributions',
 'summary': 'Bayesian nonparametric estimates of Australian mental health distributions\nare obtained to assess how the mental health status of the population has\nchanged over time and to compare the mental health status of female/male and\nindigenous/non-indigenous population subgroups. First- and second-order\nstochastic dominance are used to compare distributions, with results presented\nin terms of the posterior probability of dominance and the posterior\nprobability of no dominance. Our results suggest mental health has deteriorated\nin recent years, that males mental health status is better than that of\nfemales, and non-indigenous health status is better than that of the indigenous\npopulation.',
 'author': [{'name': 'David Gunawan'},
  {'name': 'William Griffiths'},
  {'name': 'Duangkamon Chotikapanich'}],
 'link': [{'@href': 'http://arxiv.org/abs/2106.08047v1',
   '@rel': 'alternate',
   '@type': 'text/html'},
  {'@title': 'pdf',
   '@href': 'http://arxiv.org/pdf/2106.08047v1',
   '@rel': 'related',
   '@type': 'application/pdf'}],
 'arxiv:primary_category': {'@xmlns:arxiv': 'http://arxiv.org/schemas/atom',
  '@term': 'econ.EM',
  '@scheme': 'http://arxiv.org/schemas/atom'},
 'category': {'@term': 'econ.EM', '@scheme': 'http://arxiv.org/schemas/atom'}},
            {'id': 'http://arxiv.org/abs/2106.08047v1',
 'updated': '2021-06-15T11:07:51Z',
 'published': '2021-06-15T11:07:51Z',
 'title': 'Comparisons of Australian Mental Health Distributions',
 'summary': 'Bayesian nonparametric estimates of Australian mental health distributions\nare obtained to assess how the mental health status of the population has\nchanged over time and to compare the mental health status of female/male and\nindigenous/non-indigenous population subgroups. First- and second-order\nstochastic dominance are used to compare distributions, with results presented\nin terms of the posterior probability of dominance and the posterior\nprobability of no dominance. Our results suggest mental health has deteriorated\nin recent years, that males mental health status is better than that of\nfemales, and non-indigenous health status is better than that of the indigenous\npopulation.',
 'author': [{'name': 'David Gunawan'},
  {'name': 'William Griffiths'},
  {'name': 'Duangkamon Chotikapanich'}],
 'link': [{'@href': 'http://arxiv.org/abs/2106.08047v1',
   '@rel': 'alternate',
   '@type': 'text/html'},
  {'@title': 'pdf',
   '@href': 'http://arxiv.org/pdf/2106.08047v1',
   '@rel': 'related',
   '@type': 'application/pdf'}],
 'arxiv:primary_category': {'@xmlns:arxiv': 'http://arxiv.org/schemas/atom',
  '@term': 'econ.EM',
  '@scheme': 'http://arxiv.org/schemas/atom'},
 'category': {'@term': 'econ.EM', '@scheme': 'http://arxiv.org/schemas/atom'}}]

    arxiv_articles = ArticleList.parse_arxiv_articles(item)
    print(arxiv_articles)

    item = [{'id': 'smhd-a-large-scale-resource-for-exploring',
 'arxiv_id': '1806.05258',
 'nips_id': None,
 'url_abs': 'http://arxiv.org/abs/1806.05258v2',
 'url_pdf': 'http://arxiv.org/pdf/1806.05258v2.pdf',
 'title': 'SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions',
 'abstract': "Mental health is a significant and growing public health concern. As language\nusage can be leveraged to obtain crucial insights into mental health\nconditions, there is a need for large-scale, labeled, mental health-related\ndatasets of users who have been diagnosed with one or more of such conditions.\nIn this paper, we investigate the creation of high-precision patterns to\nidentify self-reported diagnoses of nine different mental health conditions,\nand obtain high-quality labeled data without the need for manual labelling. We\nintroduce the SMHD (Self-reported Mental Health Diagnoses) dataset and make it\navailable. SMHD is a novel large dataset of social media posts from users with\none or multiple mental health conditions along with matched control users. We\nexamine distinctions in users' language, as measured by linguistic and\npsychological variables. We further explore text classification methods to\nidentify individuals with mental conditions through their language.",
 'authors': ['Sean MacAvaney',
  'Bart Desmet',
  'Nazli Goharian',
  'Andrew Yates',
  'Luca Soldaini',
  'Arman Cohan'],
 'published': '2018-06-13',
 'conference': 'smhd-a-large-scale-resource-for-exploring-1',
 'conference_url_abs': 'https://aclanthology.org/C18-1126',
 'conference_url_pdf': 'https://aclanthology.org/C18-1126.pdf',
 'proceeding': 'coling-2018-8'},
            {'id': 'smhd-a-large-scale-resource-for-exploring',
             'arxiv_id': '1806.05258',
             'nips_id': None,
             'url_abs': 'http://arxiv.org/abs/1806.05258v2',
             'url_pdf': 'http://arxiv.org/pdf/1806.05258v2.pdf',
             'title': 'SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions',
             'abstract': "Mental health is a significant and growing public health concern. As language\nusage can be leveraged to obtain crucial insights into mental health\nconditions, there is a need for large-scale, labeled, mental health-related\ndatasets of users who have been diagnosed with one or more of such conditions.\nIn this paper, we investigate the creation of high-precision patterns to\nidentify self-reported diagnoses of nine different mental health conditions,\nand obtain high-quality labeled data without the need for manual labelling. We\nintroduce the SMHD (Self-reported Mental Health Diagnoses) dataset and make it\navailable. SMHD is a novel large dataset of social media posts from users with\none or multiple mental health conditions along with matched control users. We\nexamine distinctions in users' language, as measured by linguistic and\npsychological variables. We further explore text classification methods to\nidentify individuals with mental conditions through their language.",
             'authors': ['Sean MacAvaney',
                         'Bart Desmet',
                         'Nazli Goharian',
                         'Andrew Yates',
                         'Luca Soldaini',
                         'Arman Cohan'],
             'published': '2018-06-13',
             'conference': 'smhd-a-large-scale-resource-for-exploring-1',
             'conference_url_abs': 'https://aclanthology.org/C18-1126',
             'conference_url_pdf': 'https://aclanthology.org/C18-1126.pdf',
             'proceeding': 'coling-2018-8'}
            ]
    pwc_articles = ArticleList.parse_pwc_articles(item)
    print(pwc_articles)

    for i in ieee_articles:
        print(i)

    print(pwc_articles.getDataFrame())