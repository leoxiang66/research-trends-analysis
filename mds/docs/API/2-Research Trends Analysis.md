# Research Trends Analysis
Based on the query entered by the user, TrendFlow

1. search query on the documentation platform
2. clustering the returned documents
3. generate keywords for each clustering for research trends

## Function
```python
import trendflow as tf
result = tf.trends_analysis('machine learning', 50,2018,2022,platforms=['ieee','arxiv'])
ieee_clusters, ieee_articles = result['ieee']
```

### Parameters
- query (str): keywords for searching literature,
- num_papers (int): number of papers to search,
- start_year (int): start publication year of the papers,
- end_year (int): end publication year of the papers,
- platforms (list) : what platforms to search on, defaut = ['ieee', 'arxiv', 'paper_with_code']

### Return
return a dict object containing keys `'ieee'`, `'arxiv'` and `'paper_with_code'`. The values for each key is a tuple of `ClusterList` and `ArticleList` objects

**example output**
```
>>> Search on IEEE...
>>> pipeline starts...
>>> start generating word embeddings...
>>> successfully generated word embeddings...
>>> start clustering...
>>> The best K is 9.
>>> finished clustering...
>>> start keywords extraction
>>> finished keywords extraction
>>> pipeline finished!

There are 9 clusters:
cluster 0 contains: [2, 9, 13, 20, 24, 25, 32, 34, 40, 45, 46].
cluster 1 contains: [11, 15, 16, 17, 27, 35, 48].
cluster 2 contains: [4, 8, 19, 21, 30, 36, 38].
cluster 3 contains: [0, 1, 6, 7, 33, 49].
cluster 4 contains: [12, 23, 37, 42, 44].
cluster 5 contains: [3, 10, 28, 39, 47].
cluster 6 contains: [18, 22, 26, 41, 43].
cluster 7 contains: [5, 29, 31].
cluster 8 contains: [14].

There are 50 articles:
1) IEEE Approved Draft Guide for Architectural Framework and Application of Federated Machine Learning
- authors: 
- abstract: Federated machine learning defines a machine learning framework that allows a collective model to be constructed from data that is distributed across repositories owned by different organizations or devices. A blueprint for data usage and model building across organizations and devices while meeting applicable privacy, security and regulatory requirements is provided in this guide. It defines the architectural framework and application guidelines for federated machine learning, including description and definition of federated machine learning; the categories federated machine learning and the application scenarios to which each category applies; performance evaluation of federated machine learning; and associated regulatory requirements.
- url: https://ieeexplore.ieee.org/document/9154804/
- publication year: 2020

2) IEEE Draft Guide for Architectural Framework and Application of Federated Machine Learning
- authors: 
- abstract: Federated machine learning defines a machine learning framework that allows a collective model to be constructed from data that is distributed across repositories owned by different organizations or devices. A blueprint for data usage and model building across organizations and devices while meeting applicable privacy, security and regulatory requirements is provided in this guide. It defines the architectural framework and application guidelines for federated machine learning, including description and definition of federated machine learning; the categories federated machine learning and the application scenarios to which each category applies; performance evaluation of federated machine learning; and associated regulatory requirements.
- url: https://ieeexplore.ieee.org/document/9134988/
- publication year: 2020

3) Survey on lie group machine learning
- authors: Mei Lu;Fanzhang Li
- abstract: Lie group machine learning is recognized as the theoretical basis of brain intelligence, brain learning, higher machine learning, and higher artificial intelligence. Sample sets of Lie group matrices are widely available in practical applications. Lie group learning is a vibrant field of increasing importance and extraordinary potential and thus needs to be developed further. This study aims to provide a comprehensive survey on recent advances in Lie group machine learning. We introduce Lie group machine learning techniques in three major categories: supervised Lie group machine learning, semisupervised Lie group machine learning, and unsupervised Lie group machine learning. In addition, we introduce the special application of Lie group machine learning in image processing. This work covers the following techniques: Lie group machine learning model, Lie group subspace orbit generation learning, symplectic group learning, quantum group learning, Lie group fiber bundle learning, Lie group cover learning, Lie group deep structure learning, Lie group semisupervised learning, Lie group kernel learning, tensor learning, frame bundle connection learning, spectral estimation learning, Finsler geometric learning, homology boundary learning, category representation learning, and neuromorphic synergy learning. Overall, this survey aims to provide an insightful overview of state-of-the-art development in the field of Lie group machine learning. It will enable researchers to comprehensively understand the state of the field, identify the most appropriate tools for particular applications, and identify directions for future research.
- url: https://ieeexplore.ieee.org/document/9259191/
- publication year: 2020

4) Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data
- authors: Martin Ullrich;Arne Küderle;Luca Reggi;Andrea Cereatti;Bjoern M. Eskofier;Felix Kluge
- abstract: Digital gait measures derived from wearable inertial sensors have been shown to support the treatment of patients with motor impairments. From a technical perspective, the detection of left and right initial foot contacts (ICs) is essential for the computation of stride-by-stride outcome measures including gait asymmetry. However, in a majority of studies only one sensor close to the center of mass is used, complicating the assignment of detected ICs to the respective foot. Therefore, we developed an algorithm including supervised machine learning (ML) models for the robust classification of left and right ICs using multiple features from the gyroscope located at the lower back. The approach was tested on a data set including 40 participants (ten healthy controls, ten hemiparetic, ten Parkinson’s disease, and ten Huntington’s disease patients) and reached an accuracy of 96.3% for the overall data set and up to 100.0% for the Parkinson’s sub data set. These results were compared to a state-of-the-art algorithm. The ML approaches outperformed this traditional algorithm in all subgroups. Our study contributes to an improved classification of left and right ICs in inertial sensor signals recorded at the lower back and thus enables a reliable computation of clinically relevant mobility measures.
- url: https://ieeexplore.ieee.org/document/9630653/
- publication year: 2021

5) Machine-Learning Prediction of Informatics Students Interest to the MBKM Program: A Study Case in Universitas Pembangunan Jaya
- authors: Nur Uddin;Safitri Jaya;Edi Purwanto;Ananda Arta Dwi Putra;Muhammad Wildan Fadhilah;Abimanyu Luthfi Rizq Ramadhan
- abstract: This paper presents a prediction model of student interest to join in the MBKM (Merdeka Belajar Kampus Merdeka) program. The MBKM is a new learning program launched by the Indonesian ministry of Education and Culture to improve the quality and competency of the students. This program offers a freedom to the students in accomplishing their study. Since this is a new program, knowing the students interest is very important in preparation, implementation, and improvement of the program. The students interest can be known through a survey, but this is time consuming and expensive. While a survey is difficult to be done, a prediction would be an alternative solution to know the student interest. Machine learning is applied to predict the students interest by implementing support vector machine (SVM) as the learning algorithm. The machine learning is built using a dataset that was obtained through a survey to the students at the Department of Informatics, Universitas Pembangunan Jaya (UPJ). The result shows that the machine learning was able to predict the student interest with accuracy up to 89.29%.
- url: https://ieeexplore.ieee.org/document/9743125/
- publication year: 2022

6) Imbalanced Data Classification Based on Extreme Learning Machine Autoencoder
- authors: Chu Shen;Su-Fang Zhang;Jun-Hai Zhai;Ding-Sheng Luo;Jun-Fen Chen
- abstract: In practice, there are many imbalanced data classification problems, for example, spam filtering, credit card fraud detection and software defect prediction etc. it is important in theory as well as in application for investigating the problem of imbalanced data classification. In order to deal with this problem, based on extreme learning machine autoencoder, this paper proposed an approach for addressing the problem of binary imbalanced data classification. The proposed method includes 3 steps. (1) the positive instances are used as seeds, new samples are generated for increasing the number of positive instances by extreme learning machine autoencoder, the generated new samples are similar with the positive instances but not same. (2) step (1) is repeated several times, and a balanced data set is obtained. (3) a classifier is trained with the balanced data set and used to classify unseen samples. The experimental results demonstrate that the proposed approach is feasible and effective.
- url: https://ieeexplore.ieee.org/document/8526934/
- publication year: 2018

7) IEEE Draft Standard for Technical Framework and Requirements of Trusted Execution Environment based Shared Machine Learning
- authors: 
- abstract: The framework and architecture for machine learning in which a model is trained using encrypted data that has been aggregated from multiple sources and is processed by a trusted third party are defined in this standard. Functional components, workflows, security requirements, technical requirements, and protocols are specified in this standard.
- url: https://ieeexplore.ieee.org/document/9363102/
- publication year: 2021

8) IEEE Draft Standard for Technical Framework and Requirements of Trusted Execution Environment based Shared Machine Learning
- authors: 
- abstract: The framework and architecture for machine learning in which a model is trained using encrypted data that has been aggregated from multiple sources and is processed by a trusted third party are defined in this standard. Functional components, workflows, security requirements, technical requirements, and protocols are specified in this standard.
- url: https://ieeexplore.ieee.org/document/9462600/
- publication year: 2021

9) A machine learning approach to predict the result of League of Legends
- authors: Qiyuan Shen
- abstract: Nowadays, the MOBA game is the game type with the most audiences and players around the world. Recently, the League of Legends has become an official sport as an e-sport among 37 events in the 2022 Asia Games held in Hangzhou. As the development in the e-sport, analytical skills are also involved in this field. The topic of this research is to use the machine learning approach to analyze the data of the League of Legends and make a prediction about the result of the game. In this research, the method of machine learning is applied to the dataset which records the first 10 minutes in diamond-ranked games. Several popular machine learning (AdaBoost, GradientBoost, RandomForest, ExtraTree, SVM, Naïve Bayes, KNN, LogisticRegression, and DecisionTree) are applied to test the performance by cross-validation. Then several algorithms that outperform others are selected to make a voting classifier to predict the game result. The accuracy of the voting classifier is 72.68%.
- url: https://ieeexplore.ieee.org/document/9763608/
- publication year: 2022

10) Multimodal Representation Learning: Advances, Trends and Challenges
- authors: Su-Fang Zhang;Jun-Hai Zhai;Bo-Jun Xie;Yan Zhan;Xin Wang
- abstract: Representation learning is the base and crucial for consequential tasks, such as classification, regression, and recognition. The goal of representation learning is to automatically learning good features with deep models. Multimodal representation learning is a special representation learning, which automatically learns good features from multiple modalities, and these modalities are not independent, there are correlations and associations among modalities. Furthermore, multimodal data are usually heterogeneous. Due to the characteristics, multimodal representation learning poses many difficulties: how to combine multimodal data from heterogeneous sources; how to jointly learning features from multimodal data; how to effectively describe the correlations and associations, etc. These difficulties triggered great interest of researchers along with the upsurge of deep learning, many deep multimodal learning methods have been proposed by different researchers. In this paper, we present an overview of deep multimodal learning, especially the approaches proposed within the last decades. We provide potential readers with advances, trends and challenges, which can be very helpful to researchers in the field of machine, especially for the ones engaging in the study of multimodal deep machine learning.
- url: https://ieeexplore.ieee.org/document/8949228/
- publication year: 2019

11) Unsupervised Machine Learning Methods for Artifact Removal in Electrodermal Activity
- authors: Sandya Subramanian;Bryan Tseng;Riccardo Barbieri;Emery N Brown
- abstract: Artifact detection and removal is a crucial step in all data preprocessing pipelines for physiological time series data, especially when collected outside of controlled experimental settings. The fact that such artifact is often readily identifiable by eye suggests that unsupervised machine learning algorithms may be a promising option that do not require manually labeled training datasets. Existing methods are often heuristic-based, not generalizable, or developed for controlled experimental settings with less artifact. In this study, we test the ability of three such unsupervised learning algorithms, isolation forests, 1-class support vector machine, and K-nearest neighbor distance, to remove heavy cautery-related artifact from electrodermal activity (EDA) data collected while six subjects underwent surgery. We first defined 12 features for each halfsecond window as inputs to the unsupervised learning methods. For each subject, we compared the best performing unsupervised learning method to four other existing methods for EDA artifact removal. For all six subjects, the unsupervised learning method was the only one successful at fully removing the artifact. This approach can easily be expanded to other modalities of physiological data in complex settings.Clinical Relevance— Robust artifact detection methods allow for the use of diverse physiological data even in complex clinical settings to inform diagnostic and therapeutic decisions.
- url: https://ieeexplore.ieee.org/document/9630535/
- publication year: 2021

12) Predicting Creditworthiness of Smartphone Users in Indonesia during the COVID-19 pandemic using Machine Learning
- authors: R R Kartika Winahyu;Maman Somantri;Oky Dwi Nurhayati
- abstract: In this research work, we attempted to predict the creditworthiness of smartphone users in Indonesia during the COVID-19 pandemic using machine learning. Principal Component Analysis (PCA) and Kmeans algorithms are used for the prediction of creditworthiness with the used a dataset of 1050 respondents consisting of twelve questions to smartphone users in Indonesia during the COVID-19 pandemic. The four different classification algorithms (Logistic Regression, Support Vector Machine, Decision Tree, and Naive Bayes) were tested to classify the creditworthiness of smartphone users in Indonesia. The tests carried out included testing for accuracy, precision, recall, F1-score, and Area Under Curve Receiver Operating Characteristics (AUCROC) assesment. Logistic Regression algorithm shows the perfect performances whereas Naïve Bayes (NB) shows the least. The results of this research also provide new knowledge about the influential and non-influential variables based on the twelve questions conducted to the respondents of smartphone users in Indonesia during the COVID-19 pandemic.
- url: https://ieeexplore.ieee.org/document/9742831/
- publication year: 2022

13) Classification of Mobile Phone Price Dataset Using Machine Learning Algorithms
- authors: Ningyuan Hu
- abstract: With the development of technology, mobile phones are an indispensable part of human life. Factors such as brand, internal memory, wifi, battery power, camera and availability of 4G are now modifying consumers' decisions on buying mobile phones. But people fail to link those factors with the price of mobile phones; in this case, this paper is aimed to figure out the problem by using machine learning algorithms like Support Vector Machine, Decision Tree, K Nearest Neighbors and Naive Bayes to train the mobile phone dataset before making predictions of the price level. We used appropriate algorithms to predict smartphone prices based on accuracy, precision, recall and F1 score. This not only helps customers have a better choice on the mobile phone but also gives advice to businesses selling mobile phones that the way to set reasonable prices with the different features they offer. This idea of predicting prices level will give support to customers to choose mobile phones wisely in the future. The result illustrates that among the 4 classifiers, SVM returns to the most desirable performance with 94.8% of accuracy, 97.3 of F1 score (without feature selection) and 95.5% of accuracy, 97.7% of F1 score (with feature selection).
- url: https://ieeexplore.ieee.org/document/9882236/
- publication year: 2022

14) Active Learning and Machine Teaching for Online Learning: A Study of Attention and Labelling Cost
- authors: Agnes Tegen;Paul Davidsson;Jan A. Persson
- abstract: Interactive Machine Learning (ML) has the potential to lower the manual labelling effort needed, as well as increase classification performance by incorporating a human-in-the loop component. However, the assumptions made regarding the interactive behaviour of the human in experiments are often not realistic. Active learning typically treats the human as a passive, but always correct, participant. Machine teaching provides a more proactive role for the human, but generally assumes that the human is constantly monitoring the learning process. In this paper, we present an interactive online framework and perform experiments to compare active learning, machine teaching and combined approaches. We study not only the classification performance, but also the effort (to label samples) and attention (to monitor the ML system) required of the human. Results from experiments show that a combined approach generally performs better with less effort compared to active learning and machine teaching. With regards to attention, the best performing strategy varied depending on the problem setup.
- url: https://ieeexplore.ieee.org/document/9680069/
- publication year: 2021

15) Fleet learning of thermal error compensation in machine tools
- authors: Fabian Stoop;Josef Mayr;Clemens Sulz;Friedrich Bleicher;Konrad Wegener
- abstract: Thermal error compensation of machine tools promotes sustainable production. The thermal adaptive learning control (TALC) and machine learning approaches are the required enabling principals. Fleet learnings are key resources to develop sustainable machine tool fleets in terms of thermally induced machine tool error. The target is to integrate each machine tool of the fleet in a learning network. Federated learning with a central cloud server and dedicated edge computing on the one hand keeps the independence of each individual machine tool high and on the other hand leverages the learning of the entire fleet. The outlined concept is based on the TALC, combined with a machine agnostic and machine specific characterization and communication. The proposed system is validated with environmental measurements for two machine tools of the same type, one situated at ETH Zurich and the other one at TU Wien.
- url: https://ieeexplore.ieee.org/document/9613231/
- publication year: 2021

16) Machine Learning-Based Heart Disease Prediction: A Study for Home Personalized Care
- authors: Goutam Kumar Sahoo;Keerthana Kanike;Santos Kumar Das;Poonam Singh
- abstract: This study develops a framework for personalized care to tackle heart disease risk using an at-home system. The machine learning models used to predict heart disease are Logistic Regression, K - Nearest Neighbor, Support Vector Machine, Naive Bayes, Decision Tree, Random Forest and XG Boost. Timely and efficient detection of heart disease plays an important role in health care. It is essential to detect cardiovascular disease (CVD) at the earliest, consult a specialist doctor before the severity of the disease and start medication. The performance of the proposed model was assessed using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. Compared to all machine learning algorithms, the Random Forest algorithm shows a better performance accuracy score of 90.16%. The best model may evaluate patient fitness rather than routine hospital visits. The proposed work will reduce the burden on hospitals and help hospitals reach only critical patients.
- url: https://ieeexplore.ieee.org/document/9943373/
- publication year: 2022

17) Sentiment Analysis of Covid19 Vaccines Tweets Using NLP and Machine Learning Classifiers
- authors: Amarjeet Rawat;Himani Maheshwari;Manisha Khanduja;Rajiv Kumar;Minakshi Memoria;Sanjeev Kumar
- abstract: Sentiment Analysis (SA) is an approach for detecting subjective information such as thoughts, outlooks, reactions, and emotional state. The majority of previous SA work treats it as a text-classification problem that requires labelled input to train the model. However, obtaining a tagged dataset is difficult. We will have to do it by hand the majority of the time. Another concern is that the absence of sufficient cross-domain portability creates challenging situation to reuse same-labelled data across applications. As a result, we will have to manually classify data for each domain. This research work applies sentiment analysis to evaluate the entire vaccine twitter dataset. The work involves the lexicon analysis using NLP libraries like neattext, textblob and multi class classification using BERT. This word evaluates and compares the results of the machine learning algorithms.
- url: https://ieeexplore.ieee.org/document/9850629/
- publication year: 2022

18) Diagnosing Autism Spectrum Disorder Using Machine Learning Techniques
- authors: Hidayet Takçı;Saliha Yeşilyurt
- abstract: Autism is a generalized pervasive developmental disorder that can be characterized by language and communication disorders. Screening tests are often used to diagnose such a disorder; however, they are usually time-consuming and costly tests. In recent years, machine learning methods have been frequently utilized for this purpose due to their performance and efficiency. This paper employs the most eight prominent machine learning algorithms and presents an empirical evaluation of their performances in diagnosing autism disorder on four different benchmark datasets, which are up-to-date and originate from the QCHAT, AQ-10-child, and AQ-10-adult screening tests. In doing so, we also utilize precision, sensitivity, specificity, and classification accuracy metrics to scrutinize their performances. According to the experimental results, the best outcomes are obtained with C-SVC, a classifier based on a support vector machine. More importantly, in terms of C-SVC performance metrics even lead to 100% in all datasets. Multivariate logistic regression has been taken second place. On the other hand, the lowest results are obtained with the C4.5 algorithm, a decision tree-based algorithm.
- url: https://ieeexplore.ieee.org/document/9558975/
- publication year: 2021

19) Comparison of Different Machine Learning Algorithms Based on Intrusion Detection System
- authors: Utkarsh Dixit;Suman Bhatia;Pramod Bhatia
- abstract: An IDS is a system that helps in detecting any kind of doubtful activity on a computer network. It is capable of identifying suspicious activities at both the levels i.e. locally at the system level and in transit at the network level. Since, the system does not have its own dataset as a result it is inefficient in identifying unknown attacks. In order to overcome this inefficiency, we make use of ML. ML assists in analysing and categorizing attacks on diverse datasets. In this study, the efficacy of eight machine learning algorithms based on KDD CUP99 is assessed. Based on our implementation and analysis, amongst the eight Algorithms considered here, Support Vector Machine (SVM), Random Forest (RF) and Decision Tree (DT) have the highest testing accuracy of which got SVM does have the highest accuracy
- url: https://ieeexplore.ieee.org/document/9850515/
- publication year: 2022

20) Empirical Research on Multifactor Quantitative Stock Selection Strategy Based on Machine Learning
- authors: Chengzhao Zhang;Huiyue Tang
- abstract: In this paper, stock selection strategy design based on machine learning and multi-factor analysis is a research hotspot in quantitative investment field. Four machine learning algorithms including support vector machine, gradient lifting regression, random forest and linear regression are used to predict the rise and fall of stocks by taking stock fundamentals as input variables. The portfolio strategy is constructed on this basis. Finally, the stock selection strategy is further optimized. The empirical results show that the multifactor quantitative stock selection strategy has a good stock selection effect, and yield performance under the support vector machine algorithm is the best. With the increase of the number of factors, there is an inverse relationship between the fitting degree and the yield under various algorithms.
- url: https://ieeexplore.ieee.org/document/9882240/
- publication year: 2022

21) Fashion Images Classification using Machine Learning, Deep Learning and Transfer Learning Models
- authors: Bougareche Samia;Zehani Soraya;Mimi Malika
- abstract: Fashion is the way we present ourselves which mainly focuses on vision, has attracted great interest from computer vision researchers. It is generally used to search fashion products in online shopping malls to know the descriptive information of the product. The main objectives of our paper is to use deep learning (DL) and machine learning (ML) methods to correctly identify and categorize clothing images. In this work, we used ML algorithms (support vector machines (SVM), K-Nearest Neirghbors (KNN), Decision tree (DT), Random Forest (RF)), DL algorithms (Convolutionnal Neurals Network (CNN), AlexNet, GoogleNet, LeNet, LeNet5) and the transfer learning using a pretrained models (VGG16, MobileNet and RestNet50). We trained and tested our models online using google colaboratory with Tensorflow/Keras and Scikit-Learn libraries that support deep learning and machine learning in Python. The main metric used in our study to evaluate the performance of ML and DL algorithms is the accuracy and matrix confusion. The best result for the ML models is obtained with the use of ANN (88.71%) and for the DL models is obtained for the GoogleNet architecture (93.75%). The results obtained showed that the number of epochs and the depth of the network have an effect in obtaining the best results.
- url: https://ieeexplore.ieee.org/document/9786364/
- publication year: 2022

22) Toward Machine Learning and Big Data Approaches for Learning Analytics
- authors: Prasanth Sai Gouripeddi;Ramkiran Gouripeddi;Sai Preeti Gouripeddi
- abstract: There is a paradigm shift in education due to online learning approaches and virtual learning environments. Machine learning methods have been used in a limited manner previously for learning analytics. These models can predict learning outcomes and enable understanding relationships between various learning variables. The data required for such predictions are usually complex with multiple relationships. In this paper, we use Support Vector Regression and Graph representation on the Open University Learning Analytics Dataset to provide a view into the use of machine learning methods and graph databases in creating predictive models for bettering the learning approaches.
- url: https://ieeexplore.ieee.org/document/8983761/
- publication year: 2019

23) A novel method for detecting disk filtration attacks via the various machine learning algorithms
- authors: Weijun Zhu;Mingliang Xu
- abstract: Disk Filtration (DF) Malware can attack air-gapped computers. However, none of the existing technique can detect DF attacks. To address this problem, a method for detecting the DF attacks based on the fourteen Machine Learning (ML) algorithms is proposed in this paper. First, we collect a number of data about Power Spectral Density (PSD) and frequency of the sound wave from the Hard Disk Drive (HDD). Second, the corresponding machine learning models are trained respectively using the collected data. Third, the trained ML models are employed to detect whether a DF attack occurs or not respectively, if given pair of values of PSD and frequency are input. The experimental results show that the max accuracy of detection is greater than or equal to 99.4%.
- url: https://ieeexplore.ieee.org/document/9089181/
- publication year: 2020

24) Machine Learning Opportunities In Cloud Computing Data Center Management for 5G Services
- authors: Fabio López-Pires;Benjamín Barán
- abstract: Emerging paradigms associated with cloud computing operations are considered to serve as a basis for integrating 5G components and protocols. In the context of resource management for cloud computing data centers, several research challenges could be addressed through state-of-the-art machine learning techniques. This paper presents identified opportunities on improving critical resource management decisions, analyzing the potential of applying machine learning to solve these relevant problems, mainly in two-phase optimization schemes for virtual machine placement (VMP). Potential directions for future research are also presented.
- url: https://ieeexplore.ieee.org/document/8597920/
- publication year: 2018

25) A Review on Machine Learning Styles in Computer Vision—Techniques and Future Directions
- authors: Supriya V. Mahadevkar;Bharti Khemani;Shruti Patil;Ketan Kotecha;Deepali R. Vora;Ajith Abraham;Lubna Abdelkareim Gabralla
- abstract: Computer applications have considerably shifted from single data processing to machine learning in recent years due to the accessibility and availability of massive volumes of data obtained through the internet and various sources. Machine learning is automating human assistance by training an algorithm on relevant data. Supervised, Unsupervised, and Reinforcement Learning are the three fundamental categories of machine learning techniques. In this paper, we have discussed the different learning styles used in the field of Computer vision, Deep Learning, Neural networks, and machine learning. Some of the most recent applications of machine learning in computer vision include object identification, object classification, and extracting usable information from images, graphic documents, and videos. Some machine learning techniques frequently include zero-shot learning, active learning, contrastive learning, self-supervised learning, life-long learning, semi-supervised learning, ensemble learning, sequential learning, and multi-view learning used in computer vision until now. There is a lack of systematic reviews about all learning styles. This paper presents literature analysis of how different machine learning styles evolved in the field of Artificial Intelligence (AI) for computer vision. This research examines and evaluates machine learning applications in computer vision and future forecasting. This paper will be helpful for researchers working with learning styles as it gives a deep insight into future directions.
- url: https://ieeexplore.ieee.org/document/9903420/
- publication year: 2022

26) Traffic Prediction for Intelligent Transportation System using Machine Learning
- authors: Gaurav Meena;Deepanjali Sharma;Mehul Mahrishi
- abstract: This paper aims to develop a tool for predicting accurate and timely traffic flow Information. Traffic Environment involves everything that can affect the traffic flowing on the road, whether it's traffic signals, accidents, rallies, even repairing of roads that can cause a jam. If we have prior information which is very near approximate about all the above and many more daily life situations which can affect traffic then, a driver or rider can make an informed decision. Also, it helps in the future of autonomous vehicles. In the current decades, traffic data have been generating exponentially, and we have moved towards the big data concepts for transportation. Available prediction methods for traffic flow use some traffic prediction models and are still unsatisfactory to handle real-world applications. This fact inspired us to work on the traffic flow forecast problem build on the traffic data and models.It is cumbersome to forecast the traffic flow accurately because the data available for the transportation system is insanely huge. In this work, we planned to use machine learning, genetic, soft computing, and deep learning algorithms to analyse the big-data for the transportation system with much-reduced complexity. Also, Image Processing algorithms are involved in traffic sign recognition, which eventually helps for the right training of autonomous vehicles.
- url: https://ieeexplore.ieee.org/document/9091758/
- publication year: 2020

27) Cyberattacks Predictions Workflow using Machine Learning
- authors: Carlos Eduardo Barrera Pérez;Jairo E. Serrano;Juan Carlos Martinez-Santos
- abstract: This research aims to validate the effectiveness of a machine learning model composed of three classifiers: decision tree, logistic regression, and support vector machines. Through the design of a workflow, we demonstrate the effectiveness of the model. First, we execute a network attack, and then monitoring, processing, storage, visualization, and data transfer tools are implemented to create the most realistic environment possible and obtain more accurate predictions.
- url: https://ieeexplore.ieee.org/document/9690527/
- publication year: 2021

28) Comparison Of Different Machine Learning Methods Applied To Obesity Classification
- authors: Zhenghao He
- abstract: Estimation for obesity levels is always an important topic in medical field since it can provide useful guidance for people that would like to lose weight or keep fit. The article tries to find a model that can predict obesity and provides people with the information of how to avoid overweight. To be more specific, this article applied dimension reduction to the data set to simplify the data and tried to Figure out a most decisive feature of obesity through Principal Component Analysis (PCA) based on the data set. The article also used some machine learning methods like Support Vector Machine (SVM), Decision Tree to do prediction of obesity and wanted to find the major reason of obesity. In addition, the article uses Artificial Neural Network (ANN) to do prediction which has more powerful feature extraction ability to do this. Finally, the article found that family history of obesity is the most decisive feature, and it may because of obesity may be greatly affected by genes or the family eating diet may have great influence. And both ANN and Decision tree’s accuracy of prediction is higher than 90%.
- url: https://ieeexplore.ieee.org/document/9943193/
- publication year: 2022

29) A Multi-source Based Healthcare Method for Heart Disease Prediction by Machine Learning
- authors: Shuying Shen
- abstract: Accurate prediction of heart disease can save thousands of lives and de-crease health care cost significantly. In order to increase prediction accuracy-cy, we need to analyze data from multiple sources. However, current prediction methods based on machine learning do not consider the benefit of multiple sources. In this article, we combine four sensors with the electronic medical records (EMR), and perform feature extraction, preprocessing, feature fusion to predict heart disease by the support vector machines (SVM) and the convolutional neural network (CNN). The four sensors, including the medical sensor, the activity sensor, the sleeping sensor, and the emotion sensor use feature extraction techniques that are tailored for each sensor, considering their characteristics. Through analysis, it is demonstrated that the proposed method can increase the accuracy of heart disease prediction.
- url: https://ieeexplore.ieee.org/document/9706793/
- publication year: 2021

30) Precise Medical Diagnosis For Brain Tumor Detection and Data Sample Imbalance Analysis using Enhanced Kernel Extreme Learning Machine Model with Deep Belief Network Compared to Extreme Machine Learning
- authors: Vangireddy Vishnu Vardhan Reddy;Uma Priyadarsini P. S;Saroj Kumar Tiwari
- abstract: To identify the brain tumor according to the categorial identification by using the symptoms. Materials and Methods: To identify brain tumors using Kernel Extreme Learning Machine with improved accuracy over Extreme Machine Learning. The total number of samples that are evaluated on the proposed methodology is 10 in each of 2 groups. Results: The proposed hybrid Kernel Extreme Learning Machine approach gives accuracy 93.31% which is significantly better in classification when compared to Extreme Machine Learning which has less accuracy 81.91% and level of significance is 0.01 $(\mathrm{p} < 0.05)$. Conclusion: Identifying brain tumor was achieved significantly better by using a novel functional glioma innovation as Kernel Extreme Learning Machine compared to Extreme Machine Learning.
- url: https://ieeexplore.ieee.org/document/10022465/
- publication year: 2022

31) What are they Researching? Examining Industry-Based Doctoral Dissertation Research through the Lens of Machine Learning
- authors: Ion C. Freeman;Ashley J. Haigler;Suzanna E. Schmeelk;Lisa R. Ellrodt;Tonya L. Fields
- abstract: This paper examines industry-based doctoral dissertation research in a professional computing doctoral program for full time working professionals through the lens of different machine learning algorithms to understand topics explored by full time working industry professionals. This research paper examines machine learning algorithms and the IBM Watson Discovery machine learning tool to categorize dissertation research topics defended at Pace University. The research provides insights into differences in machine learning algorithm categorization using natural language processing.
- url: https://ieeexplore.ieee.org/document/8614242/
- publication year: 2018

32) Fuzzt Set-Based Kernel Extreme Learning Machine Autoencoder for Multi-Label Classification
- authors: Qingshuo Zhang;Eric C. C. Tsang;Meng Hu;Qiang He;Degang Chen
- abstract: The multi-label learning algorithm based on an extreme learning machine has the advantage of high efficiency and generalization ability, but its classification ability is weak due to ignoring the correlation between features and labels. Accordingly, in this paper, the fuzzy set-based kernel extreme learning machine autoencoder for multi-label classification (KELM-AE-fuzzy) is proposed. Firstly, the correlation between features and labels is analyzed based on fuzzy set theory, and the correlation label membership matrix and label completion matrix are constructed. Then, the kernel extreme learning machine autoencoder is used to fuse the correlation label membership matrix with the original feature space and generate the reconstructed feature space. Eventually, kernel extreme learning machine (KELM) is used as a classifier, where the label matrix is used with the label completion matrix. Comparative experiments on several multi-label datasets demonstrate that KELM-AE-fuzzy outperforms other multi-label algorithms, and the effectiveness of the proposed algorithm is verified.
- url: https://ieeexplore.ieee.org/document/9737260/
- publication year: 2021

33) An Overview of Machine Learning Techniques for Evaluation of Pavement Condition
- authors: Aradhana Chavan;Sunil Pimplikar;Ashlesha Deshmukh
- abstract: Pavement management systems play a vital role in the development of a country as it is a very important part of the economy. Maintaining a good quality of the road is the key duty of the road authorities. They require methods for pavement-related data collection and analysis to evaluate its condition. Machine learning (ML) methods can be utilized for defect classification from an image, defect recognition and segmentation in the assessment of pavement distress. This paper presents an overview of the machine learning techniques used to analyze pavement condition data. Moreover, information collection methods and pavement condition indices are also studied from the point of view of ML algorithms. Future research directions are also presented by highlighting the limitations of using ML techniques for the assessment of pavement conditions.
- url: https://ieeexplore.ieee.org/document/9989164/
- publication year: 2022

34) IEEE Standard for Technical Framework and Requirements of Trusted Execution Environment based Shared Machine Learning
- authors: 
- abstract: The framework and architecture for machine learning in which a model is trained using encrypted data that has been aggregated from multiple sources and is processed by a trusted third party are defined in this standard. Functional components, workflows, security requirements, technical requirements, and protocols are specified in this standard.
- url: https://ieeexplore.ieee.org/document/9586768/
- publication year: 2021

35) Crab Molting Identification using Machine Learning Classifiers
- authors: Runal Rezkiawan Baharuddin;Muhammad Niswar;Amil Ahmad Ilham;Shigeru Kashihara
- abstract: Soft-shell crab is an export product in which foreign demand is much higher than production. The production of soft-shell crabs done by selecting the crabs just prior to molting and placing them in a box until the molting occurs. Molting is a natural process of shedding the shell when crabs respond to the lack of growth space within its shell. Shortly after molting, the new crab shells are still very soft and will be hardened in a few hours after the crabs absorb calcium from water. Farmer must harvest the crab while the crabs’ shell is soft. This study investigates the initial identification of crab molting using machine learning classifier. We collected 1060 image datasets of crab molting and we divide data into 1000 training data and 60 testing data. We use three machine learning classifiers, namely K-Nearest Neighbors (k-NN), Support Vector Machine (SVM), and the Random Forest Classifier (RFC). This study aims to compare and determine the best classification algorithm to be used for crab’s molting identification. The experimental results show that, KNN is the best classification algorithm for initial identification of crab’s molting.
- url: https://ieeexplore.ieee.org/document/9743136/
- publication year: 2022

36) Virus Prediction Using Machine Learning Techniques
- authors: Jerin Reji;R Satheesh Kumar
- abstract: In biological aspects, a virus is a microorganism that is smaller in size and can replicate within a host organism. A virus can affect a variety of living organisms like animals and plants. The genetic codes can be DNA or RNA. A virus cannot replicate itself. It needs a host organism for replication. A human being, an animal, or a plant can be this host organism. This replication can cause various effects in living organisms. These effects can lead to various diseases. Viruses are different in their biological structure and can affect some regions in the animal body. So the task of detecting all types of viruses, in the same way, is not possible. Thus, there is a need for different detection techniques for different viruses. Various viruses, diseases, and machine learning algorithms for detecting these diseases are covered in this study. For the review process in this study, ten different articles were chosen. In creating machine learning models for disease diagnosis, each of these publications uses a variety of machine learning algorithms and feature selection methods. Support vector machine (SVM), Linear model (LM), Linear regression (LR), K-Nearest Neighbors (KNN), Artificial neural network (ANN), K-means, and other machine learning approaches are utilized in these publications. Various feature selection approaches and machine learning methods are used in these articles. Because of that, the suggested model's accuracy varies. This research can be used to blend different machine learning algorithms into one or to gain a better understanding of viruses and diseases.
- url: https://ieeexplore.ieee.org/document/9785020/
- publication year: 2022

37) Review on evaluation techniques for better student learning outcomes using machine learning
- authors: Pooja Rana;Lovi Raj Gupta;Mithilesh Kumar Dubey;Gulshan Kumar
- abstract: The paper represents review on student learning outcomes on the basis of various evaluation parameters which plays an important role in an education system. Student learning outcomes along with other attributes are taken into consideration like learner factor, learner engagement, learning strategies use, teacher experience, motivational beliefs and technology in learning etc. With the help of examination and evaluation we can measure student learning outcome. Classification Algorithms like Decision Tree, Naïve Bayes and Support Vector Machine can help us to classify student’s performance. This classifier helps in tracking student performance. With the use of machine learning techniques we are trying to identify whether learning outcome is achieved or not. Students learning evaluation should be done on regular basis so that true learning outcomes can be measure. Once learning outcome is evaluated on regular basis, its aggregation should be done to sum up the learning outcome of course.
- url: https://ieeexplore.ieee.org/document/9445294/
- publication year: 2021

38) Classifying Quality of Web Services Using Machine Learning Classification and Cross Validation Techniques
- authors: Noor Al-Huda Hamed Olewy;Ameer Kadhim Hadi
- abstract: The growing amount of online services provided through the Internet is continually increasing. As a result, consumers are finding it more difficult to choose the proper service among a huge number of functionally comparable candidate services. It is unrealistic to inspect every web service for its quality value since it consumes a lot of resources. As a result, the subject of Web quality of service prediction has gotten a lot of attention in recent years. Using machine learning techniques, the present work suggests a model for the classification of the quality of web services by using cross validation techniques. Four algorithms of classification machine learning are applied in this work: Logistic Regression, Random Forest (DF), Support Vector Machine (SVM) and Neural Network (NN). When comparing the results, it was discovered that the Random Forest had the best accuracy. The cross validation, person correlation and normalization techniques are used in this work to compare them with the result of the algorithms. After choosing the best algorithm, a web service is created for the forecast of quality using Azure Machine Learning studio.
- url: https://ieeexplore.ieee.org/document/9773416/
- publication year: 2021

39) Improve the Accuracy of Students Admission at Universities Using Machine Learning Techniques
- authors: Basem Assiri;Mohammed Bashraheel;Ala Alsuri
- abstract: The advancement of technology contributes in the development of many field of life. One of the major fields to focus on is the field of higher education. Actually, Saudi's universities provide free education to the students, so large number of students apply to the universities. In response to that, universities usually maintain admission policies. Universities' admission policies and procedures focus on students Grade Point Average in high school (GPAH), General Aptitude Test (GAT) and Achievement Test (AT). In fact, guiding students to the suitable major improves students' achievements and success. This paper studies the admission criteria for universities in Saudi Arabia. This paper investigates the hidden details that lies behind students' GP AH, GAT and AT. Those details influence the process of students' major selection at universities. Indeed, this research uses machine learning models to include more features such as the grades of high school courses to predict the suitable majors for the students. We use K-Nearest Neighbor (KNN), Decision Tree (DT) and Support Vector Machine (SVM) to classify students into suitable majors. This process enhances the enrollments of applicants in appropriate majors. Furthermore, the experiments show that KNN gives the highest accuracy rate as it reaches 100%, while DT's accuracy rate is 81 % and SVM's accuracy rate is 75%.
- url: https://ieeexplore.ieee.org/document/9736349/
- publication year: 2022

40) Using Electronic Health Records and Machine Learning to Make Medical-Related Predictions from Non-Medical Data
- authors: Stavros Pitoglou;Yiannis Koumpouros;Athanasios Anastasiou
- abstract: Objectives: Administrative HIS (Hospital Information System) and EHR (Electronic Health Record) data are characterized by lower privacy sensitivity, thus easier portability and handling, as well as higher information quality. In this paper we test the hypothesis that the application of machine learning techniques on data of this nature can be used to address prediction/forecasting problems in the Health IT domain. The novelty of this approach consists in that medical data (test results, diagnoses, doctors' notes etc.) are not included in the predictors' dataset. Moreover, there is limited need for separation of patient cohorts based on specific health conditions. Methods: We experiment with the prediction of the probability of early readmission at the time of a patient's discharge. We extract real HIS data and perform data processing techniques. We then apply a series of machine learning algorithms (Logistic Regression, Support Vector Machine, Gaussian Naïve Bayes, K-Nearest Neighbors and Deep Multilayer Neural Network) and measure the performance of the emergent models. Results: All applied methods performed well above random guessing, even with minimal hyper-parameter tuning. Conclusions: Given that the experiments provide evidence in favor of the underlying hypothesis, future experimentation on more fine-tuned (thus more robust) models could result in applications suited for productive environments.
- url: https://ieeexplore.ieee.org/document/8614004/
- publication year: 2018

41) Machine Learning for Efficient Assessment and Prediction of Human Performance in Collaborative Learning Environments
- authors: Pravin Chopade;Saad M Khan;David Edwards;Alina von Davier
- abstract: The objective of this work is to propose a machine learning-based methodology system architecture and algorithms to find patterns of learning, interaction, and relationship and effective assessment for a complex system involving massive data that could be obtained from a proposed collaborative learning environment (CLE). Collaborative learning may take place between dyads or larger team members to find solutions for real-time events or problems, and to discuss concepts or interactions during situational judgment tasks (SJT). Modeling a collaborative, networked system that involves multimodal data presents many challenges. This paper focuses on proposing a Machine Learning - (ML)-based system architecture to promote understanding of the behaviors, group dynamics, and interactions in the CLE. Our framework integrates techniques from computational psychometrics (CP) and deep learning models that include the utilization of convolutional neural networks (CNNs) for feature extraction, skill identification, and pattern recognition. Our framework also identifies the behavioral components at a micro level, and can help us model behaviors of a group involved in learning.
- url: https://ieeexplore.ieee.org/document/8574203/
- publication year: 2018

42) Machine learning-based recommendation trust model for machine-to-machine communication
- authors: Elvin Eziama;Luz M.S Jaimes;Agajo James;Kenneth Sorle Nwizege;Ali Balador;Kemal Tepe
- abstract: The Machine Type Communication Devices (MTCDs) are usually based on Internet Protocol (IP), which can cause billions of connected objects to be part of the Internet. The enormous amount of data coming from these devices are quite heterogeneous in nature, which can lead to security issues, such as injection attacks, ballot stuffing, and bad mouthing. Consequently, this work considers machine learning trust evaluation as an effective and accurate option for solving the issues associate with security threats. In this paper, a comparative analysis is carried out with five different machine learning approaches: Naive Bayes (NB), Decision Tree (DT), Linear and Radial Support Vector Machine (SVM), KNearest Neighbor (KNN), and Random Forest (RF). As a critical element of the research, the recommendations consider different Machine-to-Machine (M2M) communication nodes with regard to their ability to identify malicious and honest information. To validate the performances of these models, two trust computation measures were used: Receiver Operating Characteristics (ROCs), Precision and Recall. The malicious data was formulated in Matlab. A scenario was created where 50% of the information were modified to be malicious. The malicious nodes were varied in the ranges of 10%, 20%, 30%, 40%, and the results were carefully analyzed.
- url: https://ieeexplore.ieee.org/document/8705147/
- publication year: 2018

43) Content-Based Recommendation Using Machine Learning
- authors: Yifan Tai;Zhenyu Sun;Zixuan Yao
- abstract: Currently, the user profile based online recommender system has become a hit both in research and engineering domain. Accurately capturing users' profile is the key of recommendation. Recently, lots of researches on user profile extraction have been launched, including content-based recommendation. To better capture users' profiles, a three-step profiling method is adopted in this work. (1) Purchase item prediction is made based on Logistic Regression. (2) Purchase category prediction is made based on support vector machine (SVM), and (3) User's rating prediction is made based on convolutional neural network (CNN) and Long Short-Term Memory (LSTM). This work outperformed the baseline model on the user dataset collected from Amazon. So, in conclusion, the work has the ability of giving reasonable recommendation for users who would like to purchase online. In the future, the video signal processing techniques will also be taken under consideration to capture users' face expression for better recommendation.
- url: https://ieeexplore.ieee.org/document/9596525/
- publication year: 2021

44) Reliability Analysis and Optimization of Computer Communication Network Based on Machine Learning Algorithm
- authors: Dai-xiong Liu
- abstract: Machine learning is to find laws from observed data and use these laws to predict future data or unobservable data. Network measurement and routing optimization strategy are critical components in the routing optimization problem. Due to the continuous progress of information technology, computer information technology is widely used in various fields, so its security and reliability will be paid more and more attention. The unsupervised learning classification is carried out through the fast density clustering algorithm to classify the importance of nodes, which can be effectively applied to the important evaluation of communication network nodes and support the planning of the communication network. Given the progress of communication technology, optical fiber technology and computer internet technology, the network's functions have been strengthened daily, and the research on the reliability of computer communication networks has been promoted to develop in depth. Furthermore, optimization theory can realize the bandwidth allocation of a communication network. The important is that computer communication network reliability based on machine learning algorithm has great economic value, social value and social benefit.
- url: https://ieeexplore.ieee.org/document/9968321/
- publication year: 2022

45) Research on Radio Frequency Finerprint Licalization based on Machine Learning
- authors: Zhaoyu Chen
- abstract: With the proliferation of the internet and mobile devices, location-based service (LBS) has become an indispensable part of everyday activities. Moreover, as a lot of indoor activities are conducted within concrete buildings with dense obstacles, localization methods that are able to provide accurate information with efficiency in such complex environments, is key to successful application of LBS. A variety of positioning technologies have been developed over the years. This paper has investigated and compared various machine learning methods for the prediction of locations based on RSS data. It introduces the recent development of RSS technology in indoor localization and further investigates the application of machine learning methods for location prediction. WIFI-based RSS methods address the challenges in indoor localization where GPS and sensor networks failed to solve. Machine learning models which predict location or coordinates generally achieve high accuracy. However, the choice of specific models depends on the environment setup. The application of RSS methods addresses the difficulties in localization in environments with obstacles. While significant improvement in accuracy can be achieved by machine learning techniques, the computational cost is still controllable with customization in environmental and device setup. The cost-benefit suggests further research in this area will be beneficial and potentially profitable for industrialization.
- url: https://ieeexplore.ieee.org/document/9730949/
- publication year: 2021

46) Provide an Improved Model for Detecting Persian SMS Spam by Integrating Deep Learning and Machine Learning Models
- authors: Roya Khorashadizadeh;Somayyeh Jafarali Jassbi;Alireza Yari
- abstract: Spam is an example of unwanted content sent by unknown users and causing problems for mobile phone users. Disadvantages of spam include the inconvenience to the user, the loss of network traffic, the imposition of a calculation fee, the occupation of the physical space of the mobile phone, the misuse and fraud of the recipient. For this reason, the automatic detection of annoying text messages can be fundamental. Also, recognizing intelligently generated text messages is a challenge. Nevertheless, the current methods in this field face obstacles, such as the lack of appropriate Persian datasets. Experiences have shown that approaches based on deep and combined learning have better results in uncovering the annoying text messages. Accordingly, this study has attempted to provide an efficient method for detecting SMS spam by integrating machine learning classification algorithms and deep learning models. In the proposed method, after performing preprocessing on our collected dataset, two convolutional neural network layers and one LSTM layer and a fully connected layer are applied to extract the features are applied on the data which forms the deep learning part of the proposed method. The Support vector machine then utilizes the extracted information and features to perform the final classification, which is a part of the Machine Learning methods. The results show that the proposed model implements better than other algorithms and 97. 7% accuracy was achieved.
- url: https://ieeexplore.ieee.org/document/9786238/
- publication year: 2022

47) Quantum Computing and Quantum Machine Learning Classification – A Survey
- authors: P Kuppusamy;Nnvs Yaswanth Kumar;Jyotsna Dontireddy;Celestine Iwendi
- abstract: The rapid development of machine learning technology leads to make the devices in the industries working autonomously. However, the growth of sensors in the industries leads to produce vast data that is utilized by machine learning algorithms to improve the autonomous devices’ performances. However, classical ML algorithms and hardware systems cannot process large data to meet real-time problems. Hence, the researchers have developed Quantum Computing hardware systems and Quantum Machine learning algorithms to speed up the process. This research work presented the review of quantum computing mechanisms and QML algorithms that are applied to classify the images. This work demonstrated the performance comparison of various QML algorithms. It showed that the images are classified using various QML algorithms faster than classical ML algorithms in terms of time.
- url: https://ieeexplore.ieee.org/document/9989137/
- publication year: 2022

48) Development of Machine-Learning Algorithms for Recognition of Subjects’ Upper Limb Movement Intention Using Electroencephalogram Signals
- authors: Fatima Al-Khuzaei;Leen Al Homoud;Dana Alyafei;Reza Tafreshi;Md Ferdous Wahid
- abstract: This study aims to classify rest and upper limb movements execution and intention using electroencephalogram (EEG) signals by developing machine-learning (ML) algorithms. Five different MLs are implemented, including k-Nearest Neighbor (KNN), Linear Discriminant Analysis (LDA), Naïve Bayes (NB), Support Vector Machine (SVM), and Random Forest (RF). The EEG data from fifteen healthy subjects during motor execution (ME) and motor imagination (MI) are preprocessed with Independent Component Analysis (ICA) to reduce eye-blinking associated artifacts. A sliding window technique varying from 1 s to 2 s is used to segment the signals. The majority voting (MV) strategy is employed during the post-processing stage. The results show that the application of ICA increases the accuracy of MI up to 6%, which is improved further by 1-2% using the MV (p<0.05). However, the improvement in the accuracies is more significant in MI (>5%) than in ME (<1%), indicating a more significant influence of eye-blinking artifacts in the EEG signals during MI than ME. Among the MLs, both RF and SVM consistently produced better accuracies in both ME and MI. Using RF, the 2 s window size produced the highest accuracies in both ME and MI than the smaller window sizes.
- url: https://ieeexplore.ieee.org/document/9629781/
- publication year: 2021

49) Broken Rotor Bars Fault Detection in Induction Machine Using Machine Learning Algorithms
- authors: Saddam Bensaoucha;Sandrine Moreau;Sid Ahmed Bessedik;Aissa Ameur
- abstract: This paper aims to diagnose the Broken Rotor Bars (BRBs) fault in a three-phase induction machine using seven Machine Learning Algorithms (MLAs), which are respectively Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes algorithm (NB), Decision Tree (DT), Random Forest (RF), Discriminant Analysis (DA) and Extreme Learning Machine (ELM). The extracted features by the application of the Fast Fourier Transform (FFT) on Hilbert Modules (HM) are used as inputs to train the used MLAs. To evaluate the performance of these algorithms, we use several predefined models for each algorithm. The obtained results show that three algorithms (SVM, KNN, and ELM) gave a high performance with an accuracy of 100%.
- url: https://ieeexplore.ieee.org/document/9955744/
- publication year: 2022

50) The Top 10 Risks of Machine Learning Security
- authors: Gary McGraw;Richie Bonett;Victor Shepardson;Harold Figueroa
- abstract: Our recent architectural risk analysis of machine learning systems identified 78 particular risks associated with nine specific components found in most machine learning systems. In this article, we describe and discuss the 10 most important security risks of those 78.
- url: https://ieeexplore.ieee.org/document/9107290/
- publication year: 2020
```