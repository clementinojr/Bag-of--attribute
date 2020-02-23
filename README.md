# Bag-of-Attribute#

**Attention**: Bag-of-Attribute **not** a commercial method. It is designed for educational and demonstration purposes **only**.


### 1. Minimum requirements ###

* Python 3.
* PostgreSQL.


### 2. Using Bag-of-Attribute ###

**2.1 Organization Directories and files **

```
.Bag-of-attribute
    ├── Experiments
│   ├── Original Experiments
│   │   ├── Hierarchical Clustering
│   │   │   ├── WithNull
│   │   │   │   ├── hierarchial_clustering_w_null.py
│   │   │   │   ├── RESULTADO_K_FOLD_10   Grupo_ 244.csv
│   │   │   │   ├── RESULTADO_K_FOLD_1   Grupo_ 1.csv
│   │   │   │   ├── RESULTADO_K_FOLD_2   Grupo_ 542.csv
│   │   │   │   ├── RESULTADO_K_FOLD_3   Grupo_ 534.csv
│   │   │   │   ├── RESULTADO_K_FOLD_4   Grupo_ 568.csv
│   │   │   │   ├── RESULTADO_K_FOLD_5   Grupo_ 633.csv
│   │   │   │   ├── RESULTADO_K_FOLD_6   Grupo_ 262.csv
│   │   │   │   ├── RESULTADO_K_FOLD_7   Grupo_ 454.csv
│   │   │   │   ├── RESULTADO_K_FOLD_8   Grupo_ 206.csv
│   │   │   │   ├── RESULTADO_K_FOLD_9   Grupo_ 669.csv
│   │   │   │   ├── Shilhoutte_KFold_10.csv
│   │   │   │   ├── Shilhoutte_KFold_1.csv
│   │   │   │   ├── Shilhoutte_KFold_2.csv
│   │   │   │   ├── Shilhoutte_KFold_3.csv
│   │   │   │   ├── Shilhoutte_KFold_4.csv
│   │   │   │   ├── Shilhoutte_KFold_5.csv
│   │   │   │   ├── Shilhoutte_KFold_6.csv
│   │   │   │   ├── Shilhoutte_KFold_7.csv
│   │   │   │   ├── Shilhoutte_KFold_8.csv
│   │   │   │   ├── Shilhoutte_KFold_9.csv
│   │   │   │   └── TESTE.txt
│   │   │   └── WithOutNull
│   │   │       ├── hierarchial_clustering_wOut_null.py
│   │   │       ├── RESULTADO_K_FOLD_10   Grupo_ 5.csv
│   │   │       ├── RESULTADO_K_FOLD_1   Grupo_ 164.csv
│   │   │       ├── RESULTADO_K_FOLD_2   Grupo_ 326.csv
│   │   │       ├── RESULTADO_K_FOLD_3   Grupo_ 14.csv
│   │   │       ├── RESULTADO_K_FOLD_4   Grupo_ 22.csv
│   │   │       ├── RESULTADO_K_FOLD_5   Grupo_ 47.csv
│   │   │       ├── RESULTADO_K_FOLD_6   Grupo_ 5.csv
│   │   │       ├── RESULTADO_K_FOLD_7   Grupo_ 23.csv
│   │   │       ├── RESULTADO_K_FOLD_8   Grupo_ 185.csv
│   │   │       ├── RESULTADO_K_FOLD_9   Grupo_ 54.csv
│   │   │       ├── Shilhoutte_KFold_10.csv
│   │   │       ├── Shilhoutte_KFold_1.csv
│   │   │       ├── Shilhoutte_KFold_2.csv
│   │   │       ├── Shilhoutte_KFold_3.csv
│   │   │       ├── Shilhoutte_KFold_4.csv
│   │   │       ├── Shilhoutte_KFold_5.csv
│   │   │       ├── Shilhoutte_KFold_6.csv
│   │   │       ├── Shilhoutte_KFold_7.csv
│   │   │       ├── Shilhoutte_KFold_8.csv
│   │   │       ├── Shilhoutte_KFold_9.csv
│   │   │       └── TESTE.txt
│   │   └── SK Means
│   │       ├── Withnull
│   │       │   ├── __pycache__
│   │       │   │   └── skmeans.cpython-36.pyc
│   │       │   ├── RESULTADO_K_FOLD_10   Grupo_ 32.csv
│   │       │   ├── RESULTADO_K_FOLD_1   Grupo_ 36.csv
│   │       │   ├── RESULTADO_K_FOLD_2   Grupo_ 57.csv
│   │       │   ├── RESULTADO_K_FOLD_3   Grupo_ 91.csv
│   │       │   ├── RESULTADO_K_FOLD_4   Grupo_ 112.csv
│   │       │   ├── RESULTADO_K_FOLD_5   Grupo_ 50.csv
│   │       │   ├── RESULTADO_K_FOLD_6   Grupo_ 62.csv
│   │       │   ├── RESULTADO_K_FOLD_7   Grupo_ 73.csv
│   │       │   ├── RESULTADO_K_FOLD_8   Grupo_ 188.csv
│   │       │   ├── RESULTADO_K_FOLD_9   Grupo_ 79.csv
│   │       │   ├── Shilhoutte_KFold_10.csv
│   │       │   ├── Shilhoutte_KFold_1.csv
│   │       │   ├── Shilhoutte_KFold_2.csv
│   │       │   ├── Shilhoutte_KFold_3.csv
│   │       │   ├── Shilhoutte_KFold_4.csv
│   │       │   ├── Shilhoutte_KFold_5.csv
│   │       │   ├── Shilhoutte_KFold_6.csv
│   │       │   ├── Shilhoutte_KFold_7.csv
│   │       │   ├── Shilhoutte_KFold_8.csv
│   │       │   ├── Shilhoutte_KFold_9.csv
│   │       │   ├── skmeans.py
│   │       │   ├── sk_means_w_null.py
│   │       │   └── TESTE.txt
│   │       └── WithOutNull
│   │           ├── RESULTADO_K_FOLD_10   Grupo_ 93.csv
│   │           ├── RESULTADO_K_FOLD_1   Grupo_ 131.csv
│   │           ├── RESULTADO_K_FOLD_2   Grupo_ 189.csv
│   │           ├── RESULTADO_K_FOLD_3   Grupo_ 121.csv
│   │           ├── RESULTADO_K_FOLD_4   Grupo_ 78.csv
│   │           ├── RESULTADO_K_FOLD_5   Grupo_ 72.csv
│   │           ├── RESULTADO_K_FOLD_6   Grupo_ 72.csv
│   │           ├── RESULTADO_K_FOLD_7   Grupo_ 80.csv
│   │           ├── RESULTADO_K_FOLD_8   Grupo_ 55.csv
│   │           ├── RESULTADO_K_FOLD_9   Grupo_ 72.csv
│   │           ├── Shilhoutte_KFold_10.csv
│   │           ├── Shilhoutte_KFold_1.csv
│   │           ├── Shilhoutte_KFold_2.csv
│   │           ├── Shilhoutte_KFold_3.csv
│   │           ├── Shilhoutte_KFold_4.csv
│   │           ├── Shilhoutte_KFold_5.csv
│   │           ├── Shilhoutte_KFold_6.csv
│   │           ├── Shilhoutte_KFold_7.csv
│   │           ├── Shilhoutte_KFold_8.csv
│   │           ├── Shilhoutte_KFold_9.csv
│   │           ├── skmeans.py
│   │           ├── sk_means_wOut_null.py
│   │           └── TESTE.txt
│   └── SelectCohort
│       ├── cohort_by_period_with_null.sql
│       └── cohort_by_period_without_null.sql
├── Generic Method
│   ├── GenericMethods.py
│   └── skmeans.py
├── GenericMethods.py
├── Info_SQL_and_Methods.txt
└── README.md

11 directories, 98 files

```

**2.2 Running experience**


List of packages required by J-EDA:
  * --
  * -- 
  * -- 
  * --
  * --
  * --
  * --




### References ####

[1] Dalianis Hercules, Hassel Martin, Henriksson Aron, Skeppstedt Maria.Stockholm  epr  corpus:  A  clinical database  used  to  improve  health  care   //  Swedish  Lan-guage Technology Conference. 2012. 17–18.

[2] Funkner Anastasia A., Yakovlev Aleksey N., Kovalchuk Sergey V.Towards  evolutionary  discovery  of  typical clinical pathways in electronic health records  // Procedia Computer  Science.  2017.  119.  234  –  244.   6th  International Young Scientist Conference on Computational Science, YSC 2017, 01-03 November 2017, Kotka, Finland.

[3]Galetsi Panagiota, Katsaliaki Korina, Kumar Sameer.Big  data  analytics  in  health  sector:  Theoretical  framework,  techniques  and  prospects  //  International  Journalof Information Management. 2020. 50. 206–216.

[4] Garcelon Nicolas, Neuraz Antoine, Benoit Vincent, Salomon Rémi, Kracker Sven, Suarez Felipe, Bahi-BuissonNadia,  HadjRabia  Smail,  Fischer  Alain,  MunnichArnold, others.   Finding  patients  using  similarity  measures in a rare diseases-oriented clinical data warehouse:Dr.  Warehouse  and  the  needle  in  the  needle  stack    //Journal of biomedical informatics. 2017. 73. 51–61.

[5] Gudivada V. N., Baeza-Yates R., Raghavan V. V.BigData:  Promises  and  Problems   //  Computer.  Mar  2015.48, 3. 20–23.

[6] Gupta R., Singhal A., Sai Sabitha A.Comparative Study of Clustering Algorithms by Conducting a District Level Analysis of Malnutrition  // 2018 8th International Conference on Cloud Computing, Data Science Engineering(Confluence). Jan 2018. 280–286.

[7] Hirano S., Iwata H., Kimura T., Tsumoto S.Clinical Pathway Generation from Order Histories and Discharge Summaries    //  2019  International  Conference  on  DataMining Workshops (ICDMW). Nov 2019. 333–340.

[8] Hoang Khanh Hung, Ho Tu Bao.  Learning  and  recommending  treatments  using  electronic  medical  records  //Knowledge-Based Systems. 2019. 181. 104788.


[9] Huang Zhengxing, Dong Wei, Ji Lei, Gan Chenxi,Lu Xudong, Duan Huilong.  Discovery  of  clinical  path-way  patterns  from  event  logs  using  probabilistic  topicmodels  //  Journal  of  Biomedical  Informatics.  2014.  47.39 – 57.


[10] Huang Zhengxing, Lu Xudong, Duan Huilong.Onmining clinical pathway patterns from medical behaviors//  Artificial  Intelligence  in  Medicine.  2012.  56,  1.  35  –50.


[11] Lima Daniel M, Rodrigues-Jr Jose F, Traina Agma JM,Pires Fabio A, Gutierrez Marco A.   Transforming  twodecades of ePR data to OMOP CDM for clinical research// Stud Health Technol Inform. 2019. 264. 233–237.


[12] Lin Yu-Kai, Lin Mingfeng, Chen Hsinchun. Do ElectronicHealth  Records  Affect  Quality  of  Care?  Evidence  from the HITECH Act // Information Systems Research. 2019.30, 1. 306–318.


[13]Lismont Jasmien, Janssens Anne-Sophie, OdnoletkovaIrina, Broucke Seppe vanden, Caron Filip, VanthienenJan.  A guide for the application of analytics on health-care processes: A dynamic view on patient pathways  //Computers  in  Biology  and  Medicine.  2016.  77.  125  –134.

[14] Salton G., Lesk M. E.Computer Evaluation of Indexingand Text Processing  // J. ACM. I 1968. 15, 1. 8–36.

[15] Usino Wendi, Prabuwono Anton Satria, Allehaibi KhalidHamed S., Bramantoro Arif, A Hasniaty, Amaldi Wahyu.Document Similarity Detection using K-Means and Cosine  Distance//  International  Journal  of  Advanced Computer Science and Applications. 2019. 10, 2.

[16] Yang Jun, Jiang Yu-Gang, Hauptmann Alexander G.,Ngo Chong-Wah.   Evaluating  Bag-of-visual-words  Representations  in  Scene  Classification    //  Proceedings  ofthe International Workshop on Workshop on Multimedia Information Retrieval. New York, NY, USA: ACM, 2007.197–206.  (MIR ’07).

[17] Zhao Jing, Papapetrou Panagiotis, Asker Lars, BostromHenrik.Learning  from  heterogeneous  temporal  datain  electronic  health  records    //  Journal  of  Biomedical Informatics. 2017. 65. 105 – 119