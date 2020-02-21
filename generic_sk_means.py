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
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from time import gmtime, strftime
from datetime import datetime
from skmeans import SKMeans

# In[55]:


#Reading file provide SQL
def read_csv(name_df,path):
    name_df = pd.read_csv(path+".csv",index_col=0)
    return name_df

def save_file_csv(name_df, name_file):
    name_df.to_csv(namefile+"csv")
    print("File saved")


# In[58]:


#function used to search for attributes in columns
def search_att_name(columns_name_json,attribute_json,concept_name_omop):
    list_attribute_name = []
    for i, row in df.iterrows():
        obj = json.loads(row[columns_name_json])
    #print(obj)
        if(obj[attribute_json] != None):
            for procedure_name in obj[attribute_json]:
            #print(procedure_name['procedure_ocurrence_concept_name'])
                list_attribute_name.append(procedure_name[concept_name_omop])
            
        else:    
            continue
    
    return list_attribute_name


#exclude repeated attributes from the list
def exclude_repeated_att(list_att):
    update_list = list(dict.fromkeys(list_att))
    return update_list




#function to join the list of attributes
def join_list(*args):
    for lists_ in args:
        final_list += lists_
    return  final_list   


# In[62]:


#function to fill dataframe, first argument: dataframe name and other lists
def fill_columns_data_frame(name_df_fill,*args):
    name_df_fill=pd.DataFrame()
    for j in args:
        for i in j:
            name_df_fill[i] = ""
    return name_df_fill       
    


def CountFrequency(my_list,count,v): 
    nome_ocorrencia = []
    qnt_ocorrencia = []
    freq_ocorrencia =[]
    # Creating an empty dictionary  
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
    df_ocorrencias_ordenadas.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/ComNulo/RESULTADO_K_FOLD:'+str(contador)+ "   Grupo: "+str(v)+".csv")
    #print(df_ocorrencias_ordenadas)
    
    for o in df_ocorrencias_ordenadas.itertuples():
        f.write("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(o.nome_ocorrencia, o.qnt_ocorrencia,o.freq_ocorrencia) +"\r\n") 



def compare_k_AggClustering(k_list, X):
    # to find the best k number of clusters
    #X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:
        print("O K teste da vez é o K:",p)
        print("Time Start: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        kmeans_inst = SKMeans(p,iters=300)
        kmeans_inst.fit(X)
        labels=kmeans_inst.get_labels()
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

def binary_tree(df, inicio, fim, x):
    if inicio > fim:
        print(x)
        return -1       
    meio = int((inicio+fim)/2)
    if df.iloc[meio,0] == x:
         return df.iloc[meio,1]
    elif df.iloc[meio,0] > x:
        return binary_tree(df,inicio, meio-1,x)
    else:
        return binary_tree(df,meio+1, fim,x)
    

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


        
df= read_csv('df','data')
df = df.head(45720)#-----------------
df_original = df
df_original.sort_values('visit_id', inplace=True, ascending=True ,kind='mergesort')
df_a = df
array_id2 =df_a['visit_id'].values


# In[73]:


#-----------------------Pre-processing - Cada procedimento será um atributo na tabela  --------# 
arr = []
arr_t=[]
for i, row in df.iterrows():
    obj = json.loads(row['internacao_json'])
    string_concept = ''
    arr_t.append(obj['visit_concept_name'])
    string_concept+=obj['visit_concept_name']+'&'
    #id = str(obj['visit_occurrence_id'])
    if(obj['procedimentos'] != None):
        for procedure_name in obj['procedimentos']:
            nome_do_procedimento = str(procedure_name['procedure_ocurrence_concept_name'])
            string_concept += nome_do_procedimento + "&"
            arr_t.append(nome_do_procedimento)
    else:
    arr.append(string_concept)


#----------------------------------Pegar id + nome das ocorrencias(AUX no Mapeamento Final)-------------#
#--------LEMBRAR de concatenar --------#string_conc_oco= ""
#df_id_name_ocorrence = pd.DataFrame(['ID_visit','Nome_Procedimento'],[0,0])
df_id_name_ocorrence = pd.DataFrame()
lista_concep_name_ocorrencia=[]
id_visit=[]
cod_ocorrencia =[]
for i, row in df.iterrows():
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
        
#----------------------------------Criando para o MAPEAMENTO Final----------#
list_of_tuples = list(zip(id_visit, cod_ocorrencia,lista_concep_name_ocorrencia))
df_id_nome_concept = pd.DataFrame(list_of_tuples, columns = ['ID_visita','Cod_Ocorrencia' ,'Nome_Ocorrencia']) 


#-------------------Pre-processing- Manter apenas Letra e numero------------# 
input = arr_t
arr_t = [x.lower() for x in input] 
input1 = arr
arr = [x.lower() for x in input1] 
arr_t = exclude_repeated_att(arr_t)
arr = [re.sub(' +', ' ', elem) for elem in arr]
arr = [re.sub("[^A-Za-z\d\&]", "_", elem) for elem in arr]

#---------------------- Pre-processing---- para lidar apenas como uma unica palavra, trocando sepador para o metodo IDFID___
arr = list(map(lambda s: s.replace('&' , ' '), arr))

#---- Criando e chamando método TF-IDF, tranformando texto em contagen------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(arr)

#---- Normalizando e criando um novo dataframe ---- Com VALORES  TF-IDF
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(arr)
df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())

#Colocando o id no Dataframe 
df_complete = df_tfidf.head(45720)#----------------------------------------------------------------------MUDAR

####-----------------------IMPORTANTE add ID APOS TDFID--------------------------###
df_complete.insert(loc=0, column='ID', value=array_id2)
df_complete_labels = df_complete

#Dividindo o DataFrame Em 10 e colocando em um lista de dataframe--# #-------------------------MUDA   
inc =0
dfs =[]
for i in range(4572, 50292, 4572):
    df = "df"+str(i) 
    df =df_complete.loc[inc:4571+inc,]
    inc+=4572
    print(df.shape)
    #print ("----------------------------------------------------------------")
    i+=4572
    dfs.append(df)

#----- Metodo Completo --- K --- FOLDS---#
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
    f.write("Tempo inicio:"+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"\r\n") 
    f.write("k-FOLD:" +str(i+1)+"\r\n") 
    f.close()
    
    list_met=[]
    List_k=list(range(75,756, 75))
    
    print("Realizando Shilueta "+ str(i+1)+"\r\n")    
    numero_k_test = compare_k_AggClustering(List_k,train_np[ :,1:])
    f= open("TESTE.txt","a+")
    f.write("k-Teste:" +str(numero_k_test)+"\r\n") 
    f.close()
    
    print("Loading Clustering....")
    print("Tempo inicio: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


    #--------------------------- SKMeans --------KNN w/ metric = cosine"-------##
    kmeans_inst = SKMeans(numero_k_test,iters=300)
    kmeans_inst.fit(test_np[:,1:])
    labels=[]
    labels=kmeans_inst.get_labels()
     #----------------------------------------------------------------------##
    
    id_ =test_np[:,0]

    list_of_tuples = list(zip(id_,labels))
    new_df = pd.DataFrame(list_of_tuples, columns = ['ID','Labels']) 
    
     #-------------------------------_Adicionando todos LAbels no Dataframe completo_-----------------#
    complete_labels = test
    complete_labels.insert(loc=0, column='Labels', value=labels)
    complete_labels.sort_values('Labels', inplace=True, ascending=False)
    
    df_in_file = add_name_occ(complete_labels,df_original)
    df_in_file.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/ComNulo/DataframeCompleto_Labels/DF_K_FOLD_T :'+str(contador)+".csv",index=False)
    
    #-------------------------------------------------#-------------------------#------------------------------------------------------------
    
     s = metrics.silhouette_samples(test_np[:,1:], labels)
     join_s_labels = list(zip(labels, s))
     contador+=1
     df_silueta = pd.DataFrame(join_s_labels, columns = ['Labels','Silhouette_samples']) 
     df_silueta.to_csv("Shilhoutte_KFold_"+str(contador)+".csv")
     df_silueta.to_csv(r'/home/oscar/ModeloCompleto/SkMeans/ComNulo/Shilhoutte_KFold_'+str(contador)+".csv")
     df_silueta.drop(df_silueta.index, inplace=True)
    

     #Percorrendo o Data Frame que possui o Id da Ocurrencia e os Labels que foram previstos
     for v in new_df['Labels'].mode():
         print("FAZENDO A MODAcd ")
         grupo_fre = v
         qnt_f = new_df['Labels'].value_counts().max()
         porc_f = (qnt_f/36576)*100 #--------------------------------------------------MUDAR
         f= open("TESTE.txt","a+")
         f.write("O(s) grupo(s) mais frequente:\r\n")
         f.write("Grupo  " +str(grupo_fre)+" abrange "+str(porc_f)+"%"+ " do teste" +"\r\n") 
        
        
    
     for v in new_df['Labels'].mode():
         list_d =[]
         for m in new_df.itertuples():
             if m.Labels == v:
                 for n in df_id_nome_concept.itertuples():
                     if n.ID_visita == m.ID:
                         list_d.append(n.Nome_Ocorrencia)
         CountFrequency(list_d,count,v)    
    
     print("Tempo FIM: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
     f.close()
    
    print("Tempo fim: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
   
    