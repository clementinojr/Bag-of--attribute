# Bag-of-Attribute Representation (BOAR) #

**Attention**: Bag-of-Attribute **not** a commercial method. It is designed for educational and demonstration purposes **only**.

### 1. Brief introduction ###
The repository is organized as follows:
  * [Original Experiments:](https://github.com/JuniorClementino/Bag-of--attribute/tree/master/Experiments/OriginalExperiments) contains the base files used to implement and obtain the results. The folders, files, and codes are available in a procedural way to facilitate the understanding of the results.
  * [Generic Method:](https://github.com/JuniorClementino/Bag-of--attribute/tree/master/GenericMethod) contains codes in a generic way to facilitate the understanding and applicability of the method.
  * [Select Cohort:](https://github.com/JuniorClementino/Bag-of--attribute/tree/master/SelectCohort) contains the functions used to select the cohorts and the data retrieved after searching for each one.


### 2. Minimum requirements ###

* Python 3.
* PostgreSQL.

### 3. Required Library ###
  * json
  * re
  * pandas 
  * numpy 
  * sklearn.metrics *
  * sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  * sklearn.cluster import AgglomerativeClustering
  * time import gmtime, strftime
  * datetime import datetime
  * skmeans import SKMeans (in file)


### 4. Explain with code Bag-of-Attribute ###

 ![Main](./workflow_BoA.png)

**(1) Cohort selector:** module was implemented as a function in the {*Structured Query Language*} (SQL) language to select all data referring to hospital admissions started within a time interval, which is a previously configured parameter. The two function utilized in this work can be find in Folder **"Select Cohort"**: [cohort_by_period_with_null.sql](https://github.com/JuniorClementino/Bag-of--attribute/blob/master/SelectCohort/cohort_by_period_with_null.sql) and [cohort_by_period_without_null.sql](https://github.com/JuniorClementino/Bag-of--attribute/blob/master/SelectCohort/cohort_by_period_without_null.sql).


**(2) Result of query:** The data returned from the **"Select Cohort"** is made available in the files: [DataBaseWithNull.csv](https://github.com/JuniorClementino/Bag-of--attribute/blob/master/SelectCohort/DataBaseWithNull.csv) and [DataBaseWithOutNull.csv](https://github.com/JuniorClementino/Bag-of--attribute/blob/master/SelectCohort/DataBaseWithOutNull.csv) The first refers to **cohort results** that contain occurrences with null values and the second occurrences that do not have null values.


**(3) Builder:** In this step, the functions of the algorithm used in each step of the proposal will show. The functions can be found in: [GenericMethods.py](https://github.com/JuniorClementino/Bag-of--attribute/blob/master/GenericMethod/GenericMethods.py).


  ***(3.a):* Receiving JSON file from "Select cohort" and reading the data in CSV format and entering the desired attributes.**
  
  * **read_principal_file:**

```python
def read_principal_file(filename):
    df_initial= read_csv('df','filename')
    df_initial = df_initial.head(number_rows)
    return df_initial
```

  *  **search_att_name:**
  
  ```python
  def gettin_name_att_characteris(df_initial)
    arr = []
    count=0
    for i, row in df_initial.iterrows():
        obj = json.loads(row['internacao_json'])
        string_concept = ''
    
        string_concept+=obj['visit_concept_name']+'&'
        if(obj['procedimentos'] != None):
            for procedure_name in obj['procedimentos']:
                nome_do_procedimento = str(procedure_name['procedure_ocurrence_concept_name'])
                string_concept += nome_do_procedimento + "&"
           
        else:
            count+=1
        arr.append(string_concept)
    return arr
  ```
------------------------------------------------------------
 
  ***(3.b):* After selecting the attributes, it is necessary to transform to standardize their values. For this, apply the regular expression function. Then, a list is formed in the form of a dictionary to count the frequency of attributes.**
  * **put_id_nameoccurence:**
```python
  #--------------------------------------Getting "id" + occurrence name and creating auxiliar Dataframe-----------------------#
def put_id_nameoccurence(df_initial):
    count=0
    df_id_name_ocorrence = pd.DataFrame()
    lista_concep_name_ocorrencia=[]
    id_visit=[]
    cod_ocorrencia =[]
    for i, row in df_initial.iterrows():  
        obj = json.loads(row['internacao_json'])
   
        if(obj['ocorrencias'] != None):
            for procedure_name in obj['ocorrencias']:
                nome_do_procedimento = str(procedure_name['condition_ocurrence_concept_name'])
                lista_concep_name_ocorrencia.append(nome_do_procedimento)
                codigo_oco= str(procedure_name['condition_concept_id'])
                cod_ocorrencia.append(codigo_oco)
                id_visit.append(row['visit_id'])         
        
        else:
            lista_concep_name_ocorrencia.append("NULO")
            cod_ocorrencia.append("00000000000")
            id_visit.append(row['visit_id'] )
       
    list_of_tuples = list(zip(id_visit, cod_ocorrencia,lista_concep_name_ocorrencia))
    df_id_nome_concept = pd.DataFrame(list_of_tuples, columns = ['ID_visita','Cod_Ocorrencia' ,'Nome_Ocorrencia']) 

    return df_id_nome_concept
```

  * **regular_expression:**
  ```python
  #------------------------------------Pre-processing- Regular Expression - number and letter only-----------------------------# 
  def regular_expression(arr):
    input1 = arr
    arr = [x.lower() for x in input1] 
    arr = [re.sub(' +', ' ', elem) for elem in arr]
    arr = [re.sub("[^A-Za-z\d\&]", "_", elem) for elem in arr]                                   
    #----------- Pre-processing - To work only as a single word, changing separator to the TF-IDF___ method --------------------#
    arr = list(map(lambda s: s.replace('&' , ' '), arr))
    x = arr
    return x
  ```
----
  ***(3.c):* In this function, in addition to calculating the TF IDF, the values of the characteristic vectors are normalized.**
  * **calculate_tfidf:**
  ```python
  #---------------------------------------- Calling TF-IDF method, turning text into count---------------------------------------#
  def calculate_tfidf(x):
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(arr)
    df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    return df_tfidf
  ```
---
 
  ***(3.d):* Transform TF-IDF values to a dataframe: which can be used by different methods.**
  * **add_id_after_tfidf:**
  ```python
    ##-----------------------------Add "id" in dataframe, after realizing Tf–idf---------------------------------------------------#
    def add_id_after_tfidf(dataframe_tfid, df_initial):
      array_id2 =df_initial['visit_id'].values
      df_complete = dataframe_tfid.head(5530)
      df_complete.insert(loc=0, column='ID', value=array_id2)
    return df_complete
  ```
  ---


**4)Representation Result: The final result can be loaded into the memory of the execution program or saved in a CSV file.**
 
### 4. Organization Directories and files ###
```
.Bag-of-attribute
├── ./Experiments
│   └── ./Experiments/OriginalExperiments
│       ├── ./Experiments/OriginalExperiments/Hierarchical Clustering
│       │   ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithNull
│       │   │   ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithNull/hierarchial_clustering_w_null.py
│       │   │   ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithNull/Result_name_occurrences.txt
│       │   │   ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithNull/Results_K_Group
│       │   │   └── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithNull/Results_Shilhoutte_K
│       │   └── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithOutNull
│       │       ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithOutNull/hierarchial_clustering_wOut_null.py
│       │       ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithOutNull/Result_name_occurrences.txt
│       │       ├── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithOutNull/Results_K_Group
│       │       └── ./Experiments/OriginalExperiments/Hierarchical Clustering/WithOutNull/Results_Shilhoutte_K
│       └── ./Experiments/OriginalExperiments/SKMeans
│           ├── ./Experiments/OriginalExperiments/SKMeans/Withnull
│           │   ├── ./Experiments/OriginalExperiments/SKMeans/Withnull/Result_name_occurrences.txt
│           │   ├── ./Experiments/OriginalExperiments/SKMeans/Withnull/Results_K_Group
│           │   ├── ./Experiments/OriginalExperiments/SKMeans/Withnull/Results_Shilhoutte_K
│           │   ├── ./Experiments/OriginalExperiments/SKMeans/Withnull/skmeans.py
│           │   └── ./Experiments/OriginalExperiments/SKMeans/Withnull/sk_means_w_null.py
│           └── ./Experiments/OriginalExperiments/SKMeans/WithOutNull
│               ├── ./Experiments/OriginalExperiments/SKMeans/WithOutNull/Result_name_occurrences.txt
│               ├── ./Experiments/OriginalExperiments/SKMeans/WithOutNull/Results_K_Group
│               ├── ./Experiments/OriginalExperiments/SKMeans/WithOutNull/Results_Shilhoutte_K
│               ├── ./Experiments/OriginalExperiments/SKMeans/WithOutNull/skmeans.py
│               └── ./Experiments/OriginalExperiments/SKMeans/WithOutNull/sk_means_wOut_null.py
├── ./GenericMethod
│   ├── ./GenericMethod/GenericMethods.py
│   └── ./GenericMethod/skmeans.py
├── ./Info_SQL_and_Methods.txt
├── ./README.md
├── ./SelectCohort
│   ├── ./SelectCohort/cohort_by_period_with_null.sql
│   └── ./SelectCohort/cohort_by_period_without_null.sql
└── ./workflow_BoA.png

18 directories, 17 files

```
### Acknowledgement ###
The authors would like to thank Brazilian Coordination of Superior Level Staff Improvement (CAPES), grant PROEX-11357281/M;  the Sao Paulo Research Foundation (FAPESP), grants  2016/17078-0, 2018/06228-7, 2019/04660-1, 2018/06074-0; and the National Council for Scientific and Technological Development (CNPq).



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


### Contact ###

This work is part of my program master's degree. You can contact me writing to [juniorclementino@usp.br](https://jrclementino.netlify.com/).
