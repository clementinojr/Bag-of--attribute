#!/usr/bin/env python
# coding: utf-8
import json
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from time import gmtime, strftime
from datetime import datetime
from skmeans import SKMeans

#Reading file provide SQL
def read_csv(name_df,path):
    name_df = pd.read_csv(path+".csv",index_col=0)
    return name_df

#deleting columns that have no interest
def delete_column_df(df_name,*args):
    for name_col in args:
        df_name = df_name.drop(columns=[name_col])
    return df_name 

def save_file_csv(name_df, name_file):
    name_df.to_csv(namefile+"csv")
    print("File saved")

#function used to search for attributes in columns
def search_att_name(columns_name_json,attribute_json,concept_name_omop):
    list_attribute_name = []
    for i, row in df.iterrows():
        obj = json.loads(row[columns_name_json])
        if(obj[attribute_json] != None):
            for procedure_name in obj[attribute_json]:
                list_attribute_name.append(procedure_name[concept_name_omop])          
        else:    
            continue
    
    return list_attribute_name

#exclude repeated attributes from the list
def exclude_repeated_att(list_att):
    update_list = list(dict.fromkeys(list_att))
    return update_list

def CountFrequency(my_list,count,v): 
    nome_ocorrencia = []
    qnt_ocorrencia = []
    freq_ocorrencia =[]
    
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
      
    for key, value in freq.items():
        len_list = len(my_list)
        freq_oc= (value/len_list)*100
        
        nome_ocorrencia.append(key)
        qnt_ocorrencia.append(value)
        freq_ocorrencia.append(freq_oc)
        
    list_of_tuples = list(zip(nome_ocorrencia, qnt_ocorrencia,freq_ocorrencia))
    df_ocorrencias_ordenadas = pd.DataFrame(list_of_tuples, columns = ['nome_ocorrencia','qnt_ocorrencia' ,'freq_ocorrencia'])
    df_ocorrencias_ordenadas.sort_values('freq_ocorrencia', inplace=True, ascending=False)
    df_ocorrencias_ordenadas.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/SemNulo/RESULTADO_K_FOLD:'+str(contador)+ "   Grupo: "+str(v)+".csv")
    
    for o in df_ocorrencias_ordenadas.itertuples():
        f.write("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(o.nome_ocorrencia, o.qnt_ocorrencia,o.freq_ocorrencia) +"\r\n") 
        
def compare_k_AggClustering(k_list, X,name_method):
    # to find the best k number of clusters
    #X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:
        print("The current K-test is:",p)
        print("Time Start: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        if name_method == "hierarchical_clustering":
            labels = hierarchical_clustering(X,p)

        if name_method == "sk_means":
            labels = sk(X,p)

        # The higher (up to 1) the better
        s = round(metrics.silhouette_score(X, labels), 4)

        silhouette_list.append(s)

    # The higher (up to 1) the better
    key = silhouette_list.index(max(silhouette_list))

    k = k_list.__getitem__(key)

    print("Best silhouette =", max(silhouette_list), " for k=", k)
    f= open("TESTE.txt","a+")
    f.write("Best silhouette" +str(max(silhouette_list))+" for k="+ str(k)+"\r\n") 
    f.close()
    
    return k

#---- Optimize id search ------#
def binary_tree(df, inicio, fim, x):
    if inicio > fim:
        return -1
        
    meio = int((inicio+fim)/2)
    
    if df.iloc[meio,0] == x:
         return df.iloc[meio,1]
    elif df.iloc[meio,0] > x:
        return binary_tree(df,inicio, meio-1,x)
    else:
        return binary_tree(df,meio+1, fim,x)

#---Function to search for occurencce name
def add_name_occ(df_com_labels, df_com):
    arrayValores=[]
    arrayNomes=[]
    df_com_labels_e_name_occ = df_com_labels
    nome_do_procedimento=''

    for m in df_com_labels.itertuples():
    
        t = binary_tree(df_com,0,df_com.shape[0]-1,m[2])
    
        obj = json.loads(t)
        visit_id = int((obj['visit_occurrence_id']))
          
        if(obj['ocorrencias'] != None):
            for procedure_name in obj['ocorrencias']:
                nome_do_procedimento = str(procedure_name['condition_ocurrence_concept_name'])
                arrayValores.append(m[1:])
                arrayNomes.append(nome_do_procedimento)        
        else:
            arrayValores.append(m[1:])
            arrayNomes.append(nome_do_procedimento) 
    
    df_final = pd.DataFrame(arrayValores, columns = list(df_com_labels.columns)) 
    df_final.insert(loc=0, column='Nome Ocorrencia', value=arrayNomes)

    return df_final
        
#-----------------------------------------------------BEGIN--------------------------------------------------#        
def read_principal_file(filename):
    df_initial= read_csv('df','filename')
    df_initial = df_initial.head(5530)
    return df_initial

def sort_for_improve_search_id (df_initial,namecolum_sort):
    # Recive dataframe 
    # Return dataframe sort for improve search in content recovery
    df_original_sort = df_initial
    df_original.sort_values('namecolum_sort', inplace=True, ascending=True ,kind='mergesort')
    return df_original_sort



#-----------------------Pre-processing - Each procedure will be an attribute in the characteristic vector  --------#
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


#----------------------------------Getting "id" + occurrence name and creating auxiliar Dataframe-------------#
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


#-------------------Pre-processing- Regular Expression - number and letter only------------# 
def regular_expression(arr):
    input1 = arr
    arr = [x.lower() for x in input1] 
    arr = [re.sub(' +', ' ', elem) for elem in arr]
    arr = [re.sub("[^A-Za-z\d\&]", "_", elem) for elem in arr]                                   
    #--------- Pre-processing---- To work only as a single word, changing separator to the TF-IDF___ method -------------#
    arr = list(map(lambda s: s.replace('&' , ' '), arr))
    x = arr
    return x

#---- Calling TF-IDF method, turning text into count------
def calculate_tfidf(x):
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(arr)
    df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    return df_tfidf

####-----------------Add "id" in dataframe, after realizing Tf–idf-------------------------###
def add_id_after_tfidf(dataframe_tfid, df_initial):
    array_id2 =df_initial['visit_id'].values
    df_complete = dataframe_tfid.head(5530)
    df_complete.insert(loc=0, column='ID', value=array_id2)
    return df_complete


#Dividing the Data Frame Into 10 from K-foldAdd#
#-- Recive parameters Dataframe
# --- return one list from Dataframes
def divid_dataframeK_fold(df_complete)
ten_per_cent = int(dataframe.shape[0]/10)
inc =0
dfs =[]
aux = ten_per_cent -1
stop = int(dataframe.shape[0]+ten_per_cent)
for i in range(ten_per_cent, stop, ten_per_cent):
    df = "df"+str(i) 
    df =df_complete.loc[inc:aux+inc,]
    inc+=ten_per_cent
    i+=ten_per_cent
    dfs.append(df)
return dfs


def hierarchical_clustering(X,number_c):
    clusterer = AgglomerativeClustering(n_clusters=numero_k_test, linkage="average",affinity='cosine')
    clusterer.fit(test_np[:,1:])
    labels=clusterer.labels_
    return labels
        

def sk_means(X,number_c):
    labels = []
    kmeans_inst = SKMeans(number_c,iters=300)
    kmeans_inst.fit(X)
    labels=kmeans_inst.get_labels()
    return labels


def crosValidatidor_and_methods_clusterring(list_dataframes,name_method):
#----- Complete Method K --- FOLDS---#
    labels=[]
    permanente = dfs
    outra = dfs
    aux_dfs = permanente
    aux_dfs2 = permanente
    array_treino = []
    array_teste =[]
    contador =0
    for i in range(0,10):
   
        tmp=list(aux_dfs)
    
        if(i!=9):
            frames = [tmp[i], tmp[i+1]]
        else:
            frames = [tmp[i], tmp[0]]
        
        train = pd.concat(frames)
    
        if(i!=9):
            tmp.pop(i+1)
            tmp.pop(i)
        else:
            tmp.pop(i)
            tmp.pop(0)
    
        test = pd.concat(tmp)

        test_np = np.array(test)
        train_np = np.array(train)
    
    
        f= open("TESTE.txt","a+")
        f.write("________________________________________________________________"  +"\r\n") 
        f.write("Start time:"+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"\r\n") 
        f.write("k-FOLD:" +str(i+1)+"\r\n") 
        f.close()
    
        list_met=[]
        List_k=list(range(75,756, 75))
    
        print("Realizing Silhouette "+ str(i+1)+"\r\n")    
        numero_k_test = compare_k_AggClustering(List_k,train_np[ :,1:],name_method)
        f= open("TESTE.txt","a+")
        f.write("k-Teste:" +str(numero_k_test)+"\r\n") 
        f.close()
    
    
        print("Loading Clustering....")
        print("Star time: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        # Deciding what's method will be used
        if name_method == "hierarchical_clustering":
            labels = hierarchical_clustering(test_np[:,1:],numero_k_test)

        if name_method == "sk_means":
            labels = sk(test_np[:,1:],numero_k_test)

        id_ =test_np[:,0]
        
        #------------------------------------------------------------------#

        list_of_tuples = list(zip(id_,labels))
        new_df = pd.DataFrame(list_of_tuples, columns = ['ID','Labels']) 
    
    
        #-------------------------------Add all Labels into Complete Dataframe-----------------#
        complete_labels = test
        complete_labels.insert(loc=0, column='Labels', value=labels)
        complete_labels.sort_values('ID', inplace=True, ascending=True)
        df_in_file = add_name_occ(complete_labels,df_original)
        df_in_file.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/SemNulo/DataframeCompleto_Labels/DF_K_FOLD_T :'+str(contador)+".csv",index=False)
        #-------------------------------_--------------------------------------------------------_-----------------#
        
        
        s = metrics.silhouette_samples(test_np[:,1:], labels)
        join_s_labels = list(zip(labels, s))
        contador+=1
        df_silueta = pd.DataFrame(join_s_labels, columns = ['Labels','Silhouette_samples']) 
        df_silueta.to_csv("Shilhoutte_KFold_"+str(contador)+".csv")
        df_silueta.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/SemNulo/Shilhoutte_KFold_'+str(contador)+".csv")
        df_silueta.drop(df_silueta.index, inplace=True)
    
        #Performing the Data Frame that has the Occurrence Id and the Labels that were predicted
        for v in new_df['Labels'].mode():
            grupo_fre = v
            qnt_f = new_df['Labels'].value_counts().max()
            porc_f = (qnt_f/4424)*100 
            f= open("TESTE.txt","a+")
            f.write("The groups more frequerequentnte:\r\n")
            f.write("The Group  " +str(grupo_fre)+" include "+str(porc_f)+"%"+ " from test" +"\r\n") 
        
        for v in new_df['Labels'].mode():
            list_d =[]
            for m in new_df.itertuples():
                if m.Labels == v:
                    for n in df_id_nome_concept.itertuples():
                        if n.ID_visita == m.ID:
                            list_d.append(n.Nome_Ocorrencia)
            CountFrequency(list_d,count,v)    
    
        print("Final Time: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.close()
        return "Finsih"
    
   
    
