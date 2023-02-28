# Literature Search
Users can search the literature by keywords. We offer three literature platforms for searching: IEEE, Arxiv and Paper with Code.

## Function
```python
import trendflow as tf
tf.search_papers('machine learning', 2018, 2022, 50, to_pandas=True)
```
### Parameters
- query (string): keywords for searching literature
- start_year (int): start publication year of the papers
- end_year (int): end publication year of the papers
- num_papers (int): number of papers to search
- to_pandas (bool): if set to true, then the search results are converted to `pd.Dataframe`, otherwise a list of dicts

### Return
return a dict object containing 3 keys: `'ieee'`, `'arxiv'` and `'paper_with_code'`. The values for each key is a pd.Dataframe if `to_pandas` is True, otherwise a list of dicts.

**example output**
```
{'ieee':                                                 title  ...  conference_dates
0   IEEE Approved Draft Guide for Architectural Fr...  ...               NaN
1   IEEE Draft Guide for Architectural Framework a...  ...               NaN
2                Survey on lie group machine learning  ...               NaN
3   Machine learning-based distinction of left and...  ...     1-5 Nov. 2021
4   Machine-Learning Prediction of Informatics Stu...  ...   29-30 Jan. 2022
5   Imbalanced Data Classification Based on Extrem...  ...   15-18 July 2018
6   IEEE Draft Standard for Technical Framework an...  ...               NaN
7   IEEE Draft Standard for Technical Framework an...  ...               NaN
8   A machine learning approach to predict the res...  ...   25-27 Feb. 2022
9   Multimodal Representation Learning: Advances, ...  ...    7-10 July 2019
10  Unsupervised Machine Learning Methods for Arti...  ...     1-5 Nov. 2021
11  Predicting Creditworthiness of Smartphone User...  ...   29-30 Jan. 2022
12  Classification of Mobile Phone Price Dataset U...  ...   22-24 July 2022
13  Active Learning and Machine Teaching for Onlin...  ...   13-16 Dec. 2021
14  Fleet learning of thermal error compensation i...  ...   7-10 Sept. 2021
15  Machine Learning-Based Heart Disease Predictio...  ...   22-25 Aug. 2022
16  Sentiment Analysis of Covid19 Vaccines Tweets ...  ...    26-27 May 2022
17  Diagnosing Autism Spectrum Disorder Using Mach...  ...  15-17 Sept. 2021
18  Comparison of Different Machine Learning Algor...  ...    26-27 May 2022
19  Empirical Research on Multifactor Quantitative...  ...   22-24 July 2022
20  Fashion Images Classification using Machine Le...  ...      8-9 May 2022
21  Toward Machine Learning and Big Data Approache...  ...    9-11 Dec. 2019
22  A novel method for detecting disk filtration a...  ...               NaN
23  Machine Learning Opportunities In Cloud Comput...  ...   26-28 Nov. 2018
24  A Review on Machine Learning Styles in Compute...  ...               NaN
25  Traffic Prediction for Intelligent Transportat...  ...     7-8 Feb. 2020
26  Cyberattacks Predictions Workflow using Machin...  ...   16-17 Dec. 2021
27  Comparison Of Different Machine Learning Metho...  ...     5-7 Aug. 2022
28  A Multi-source Based Healthcare Method for Hea...  ...   14-14 Nov. 2021
29  Precise Medical Diagnosis For Brain Tumor Dete...  ...   12-13 Nov. 2022
30  What are they Researching? Examining Industry-...  ...   17-20 Dec. 2018
31  Fuzzt Set-Based Kernel Extreme Learning Machin...  ...     4-5 Dec. 2021
32  An Overview of Machine Learning Techniques for...  ...     8-9 Oct. 2022
33  IEEE Standard for Technical Framework and Requ...  ...               NaN
34  Crab Molting Identification using Machine Lear...  ...   29-30 Jan. 2022
35  Virus Prediction Using Machine Learning Techni...  ...  25-26 March 2022
36  Review on evaluation techniques for better stu...  ...  28-30 April 2021
37  Classifying Quality of Web Services Using Mach...  ...   28-29 Dec. 2021
38  Improve the Accuracy of Students Admission at ...  ...    1-3 March 2022
39  Using Electronic Health Records and Machine Le...  ...     3-7 Dec. 2018
40  Machine Learning for Efficient Assessment and ...  ...   23-24 Oct. 2018
41  Machine learning-based recommendation trust mo...  ...     6-8 Dec. 2018
42  Content-Based Recommendation Using Machine Lea...  ...   25-28 Oct. 2021
43  Reliability Analysis and Optimization of Compu...  ...   19-21 June 2022
44  Research on Radio Frequency Finerprint Licaliz...  ...     3-5 Dec. 2021
45  Provide an Improved Model for Detecting Persia...  ...    11-12 May 2022
46  Quantum Computing and Quantum Machine Learning...  ...     8-9 Oct. 2022
47  Development of Machine-Learning Algorithms for...  ...     1-5 Nov. 2021
48  Broken Rotor Bars Fault Detection in Induction...  ...     6-10 May 2022
49      The Top 10 Risks of Machine Learning Security  ...               NaN

[50 rows x 34 columns], 'arxiv':                                    id  ...                                  arxiv:journal_ref
0   http://arxiv.org/abs/1909.03550v1  ...                                                NaN
1   http://arxiv.org/abs/1811.04422v1  ...                                                NaN
2   http://arxiv.org/abs/1707.04849v1  ...                                                NaN
3   http://arxiv.org/abs/1909.09246v1  ...                                                NaN
4   http://arxiv.org/abs/2301.09753v1  ...                                                NaN
5    http://arxiv.org/abs/0904.3664v1  ...                                                NaN
6   http://arxiv.org/abs/2012.04105v1  ...                                                NaN
7   http://arxiv.org/abs/2204.07492v2  ...  {'@xmlns:arxiv': 'http://arxiv.org/schemas/ato...
8   http://arxiv.org/abs/1911.06612v1  ...                                                NaN
9   http://arxiv.org/abs/1909.01866v1  ...  {'@xmlns:arxiv': 'http://arxiv.org/schemas/ato...
10  http://arxiv.org/abs/1903.08801v1  ...                                                NaN
11  http://arxiv.org/abs/1907.08908v1  ...                                                NaN
12  http://arxiv.org/abs/1707.09562v3  ...                                                NaN
13  http://arxiv.org/abs/2108.07915v1  ...                                                NaN
14  http://arxiv.org/abs/2206.07090v1  ...                                                NaN
15  http://arxiv.org/abs/1507.02188v1  ...                                                NaN
16   http://arxiv.org/abs/1212.2686v1  ...                                                NaN
17  http://arxiv.org/abs/2001.04942v2  ...                                                NaN
18  http://arxiv.org/abs/1607.02450v2  ...                                                NaN
19  http://arxiv.org/abs/2007.01503v1  ...                                                NaN
20  http://arxiv.org/abs/1906.06821v2  ...                                                NaN
21  http://arxiv.org/abs/1911.00776v1  ...                                                NaN
22  http://arxiv.org/abs/2201.01288v1  ...                                                NaN
23  http://arxiv.org/abs/2011.11819v1  ...                                                NaN
24  http://arxiv.org/abs/2004.00993v2  ...                                                NaN
25  http://arxiv.org/abs/2009.11087v1  ...                                                NaN
26  http://arxiv.org/abs/2003.05155v2  ...                                                NaN
27  http://arxiv.org/abs/1706.08001v1  ...                                                NaN
28   http://arxiv.org/abs/1207.4676v2  ...                                                NaN
29  http://arxiv.org/abs/1603.02185v1  ...                                                NaN
30  http://arxiv.org/abs/1910.12387v2  ...                                                NaN
31  http://arxiv.org/abs/2007.05479v1  ...                                                NaN
32  http://arxiv.org/abs/2007.14206v1  ...                                                NaN
33  http://arxiv.org/abs/1908.04710v3  ...  {'@xmlns:arxiv': 'http://arxiv.org/schemas/ato...
34  http://arxiv.org/abs/2002.12364v1  ...  {'@xmlns:arxiv': 'http://arxiv.org/schemas/ato...
35  http://arxiv.org/abs/2001.09608v1  ...                                                NaN
36  http://arxiv.org/abs/1509.00913v3  ...                                                NaN
37  http://arxiv.org/abs/2110.12773v1  ...                                                NaN
38  http://arxiv.org/abs/1607.01400v1  ...  {'@xmlns:arxiv': 'http://arxiv.org/schemas/ato...
39  http://arxiv.org/abs/2202.10564v1  ...                                                NaN
40  http://arxiv.org/abs/1510.00633v1  ...                                                NaN
41  http://arxiv.org/abs/1802.03830v1  ...                                                NaN
42  http://arxiv.org/abs/2106.07032v1  ...                                                NaN
43  http://arxiv.org/abs/1612.04858v1  ...                                                NaN
44  http://arxiv.org/abs/1702.08608v2  ...                                                NaN
45  http://arxiv.org/abs/1705.07538v2  ...                                                NaN
46  http://arxiv.org/abs/1808.00033v3  ...                                                NaN
47  http://arxiv.org/abs/1911.08587v1  ...                                                NaN
48  http://arxiv.org/abs/2007.01977v1  ...                                                NaN
49  http://arxiv.org/abs/2007.07981v1  ...                                                NaN

[50 rows x 12 columns], 'paper_with_code':                                                 id  ...       proceeding
0     snap-ml-a-hierarchical-framework-for-machine  ...  neurips-2018-12
1        a-novel-hybrid-machine-learning-model-for  ...             None
2            orthogonal-machine-learning-power-and  ...      icml-2018-7
3     on-machine-learning-and-structure-for-mobile  ...             None
4     data-driven-decentralized-optimal-power-flow  ...             None
5       interpretable-machine-learning-for-privacy  ...             None
6    a-machine-learning-item-recommendation-system  ...             None
7           plug-in-regularized-estimation-of-high  ...             None
8              static-malware-detection-subterfuge  ...             None
9        two-use-cases-of-machine-learning-for-sdn  ...             None
10     ml-fv-heartsuit-a-survey-on-the-application  ...             None
11       can-machine-learning-identify-interesting  ...             None
12           a-hybrid-econometric-machine-learning  ...             None
13                machine-learning-cicy-threefolds  ...             None
14        machine-learning-based-colon-deformation  ...             None
15    residual-unfairness-in-fair-machine-learning  ...      icml-2018-7
16          online-adaptive-machine-learning-based  ...             None
17          reduced-order-modeling-through-machine  ...             None
18           opportunities-in-machine-learning-for  ...             None
19          a-machine-learning-framework-for-stock  ...             None
20        machine-learning-for-yield-curve-feature  ...             None
21         scikit-learn-machine-learning-in-python  ...             None
22        analysis-of-dawnbench-a-time-to-accuracy  ...             None
23    bindsnet-a-machine-learning-oriented-spiking  ...             None
24            learning-a-code-machine-learning-for  ...             None
25             ml-leaks-model-and-data-independent  ...             None
26          explaining-explanations-an-overview-of  ...             None
27       learning-from-exemplars-and-prototypes-in  ...             None
28     a-comparison-of-machine-learning-algorithms  ...             None
29    deploying-customized-data-representation-and  ...             None
30         a-guide-to-constraining-effective-field  ...             None
31      constraining-effective-field-theories-with  ...             None
32          interpreting-deep-learning-the-machine  ...             None
33        defending-against-machine-learning-model  ...             None
34        grader-variability-and-the-importance-of  ...             None
35             predictive-performance-modeling-for  ...             None
36        a-progressive-batching-l-bfgs-method-for  ...      icml-2018-7
37      currency-exchange-prediction-using-machine  ...             None
38   towards-computational-fluorescence-microscopy  ...             None
39      machine-learning-for-prediction-of-extreme  ...             None
40      on-formalizing-fairness-in-prediction-with  ...             None
41       intensive-preprocessing-of-kdd-cup-99-for  ...             None
42   model-based-pricing-for-machine-learning-in-a  ...             None
43             qunatification-of-metabolites-in-mr  ...             None
44  corpus-conversion-service-a-machine-learning-1  ...             None
45   machine-learning-inference-of-fluid-variables  ...             None
46    wikipedia-for-smart-machines-and-double-deep  ...             None
47         the-marginal-value-of-adaptive-gradient  ...  neurips-2017-12
48       geomstats-a-python-package-for-riemannian  ...      iclr-2019-5
49     the-roles-of-supervised-machine-learning-in  ...             None

[50 rows x 13 columns]}
```