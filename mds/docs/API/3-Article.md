# Article
This section introduces the class `Article` and `ArticleList`.

## Article
The class `Article` is an internal class for TrendFlow. It has following instance attributes:

- title (str): the title of the article
- authors (list of str): the authors of the article
- abstract (str): the abstract of the articles
- url (str): the url of the article
- publication_year (int): the publication year of the article

### Instance Functions
- `to_dict(self) -> dict`: return the dict form of this article


## ArticleList
The list representation of articles.

### Instance Functions
- `addArticles(self, articles:Union[Article, List[Article]])`: add article to the list
- `to_dataframe(self) ->pd.DataFrame`: convert to `pd.DataFrame` object 
- `getArticles(self) -> List[Article]:` return a list of the articles
- `getAbstracts(self) -> List[str]`: return a list of abstracts of the articles
- `getTitles(self) -> List[str]:`: return a list of titles of the articles

### Class Functions
- `parse_ieee_articles(cls,items: Union[dict, List[dict]])`: parse the search results from IEEE platform into a `ArticleList` object
- `parse_arxiv_articles(cls, items: Union[dict, List[dict]])`: parse the search results from Arxiv platform into a `ArticleList` object
- `parse_pwc_articles(cls, items: Union[dict, List[dict]])`: parse the search results from Paper with Code platform into a `ArticleList` object
