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


# In[55]:


#Reading file provide SQL
def read_csv(name_df,path):
    name_df = pd.read_csv(path+".csv",index_col=0)
    return name_df
#df = pd.read_csv("data-1579021640655.csv") 





#Pre-processing

#deleting columns that have no interest
def delete_column_df(df_name,*args):
    for name_col in args:
        df_name = df_name.drop(columns=[name_col])
    return df_name 


# In[57]:


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


# In[59]:


#lista_=search_attribute_name('internacao_json','ocorrencias','condition_ocurrence_concept_name')


# In[60]:


#exclude repeated attributes from the list
def exclude_repeated_att(list_att):
    update_list = list(dict.fromkeys(list_att))
    return update_list


# In[61]:


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
    
    


# In[63]:


def CountFrequency(my_list,count,v): 
    nome_ocorrencia = []
    qnt_ocorrencia = []
    freq_ocorrencia =[]
    
    print("ENTROUUUUUUUUUUUU COUNT")
    
    #f= open("TESTE.txt","a+")  
    # Creating an empty dictionary  
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
      
    for key, value in freq.items():
        len_list = len(my_list)
        freq_oc= (value/len_list)*100
        
        #print ("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(key, value,freq_oc)) #-------------------------------------teSTE
        nome_ocorrencia.append(key)
        qnt_ocorrencia.append(value)
        freq_ocorrencia.append(freq_oc)
        
        
    print("VAI entrar no sort")    
    list_of_tuples = list(zip(nome_ocorrencia, qnt_ocorrencia,freq_ocorrencia))
    df_ocorrencias_ordenadas = pd.DataFrame(list_of_tuples, columns = ['nome_ocorrencia','qnt_ocorrencia' ,'freq_ocorrencia'])
    df_ocorrencias_ordenadas.sort_values('freq_ocorrencia', inplace=True, ascending=False)
    df_ocorrencias_ordenadas.to_csv(r'/home/oscar/ModeloCompleto/KmeansAgglo/ComNulo/RESULTADO_K_FOLD:'+str(contador)+ "   Grupo: "+str(v)+".csv")
    #print(df_ocorrencias_ordenadas)
    
    for o in df_ocorrencias_ordenadas.itertuples():
        f.write("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(o.nome_ocorrencia, o.qnt_ocorrencia,o.freq_ocorrencia) +"\r\n") 

# lista_ = exclude_repeated_att(lista_)
# fill_columns_data_frame('meu_data',lista_)
    


# In[65]:


# #function to fill dataframe, first argument: dataframe name and other lists. For each desired attribute:
# #- Arguments:
#     #--- columns_name_json = column name of the dataframe that is the json
#     #--- id_att = visit occurrence id
#     #--- attribute_json = json attribute / variable name
#     #--- concept_name_omop = name of concept / omop that will be used
    
# def fill_values_do_count_att (columns_name_json,id_att,attribute_json,concept_name_omop ):
#     for i, row in df.iterrows():
#         obj = json.loads(row[columns_name_json])
#         qnt = i
#         #print("Quantas vezes", i)
#         #print(obj)
#         id = str(obj[id_att])
#         if(obj[attribute_json] != None):
#             for procedure_name in obj[attribute_json]:
#                 #print(procedure_name['procedure_ocurrence_concept_name'])
#                 nome_do_procedimento = str(procedure_name[concept_name_omop]) 
#                 for col in new_df.columns:
#                     col= str(col)
#                     if nome_do_procedimento == col:
#                         #print(nome_do_procedimento + "-----------"   + col + "      Match" )
#                         new_df.at[i, 'id'] = id
#                         #df_test2.loc[df_test2.index[1], col] = "oi"
#                         new_df.at[i, col] = "OK"
#                         #df_test2 = df_test2.append({col : '1'}, index=i)
#                     else:
#                         #print("Não Deu 1" +nome_do_procedimento,col)
#                         #df_test2=df_test2.at[i, col] = "Vazio"
#                         new_df.at[i, 'id'] = id
#                         continue
            
#         else:
#             new_df.at[i, 'id'] = id
#             #print("Não Deu 2")
#             #df_test2=df_test2.at[i, col] = "Sem procedimento"
#             continue
#    return new_df         


# In[66]:


#join two or more dataframe
#- args = name of data frames
def join_df(df_1,*args):
    df_final_join=pd.DataFrame()
    for df in args:
        df_final_join= pd.concat([df_final_join,df], axis=1, sort=False)
    return  df_final_join   


# In[67]:


#delete column or columns from a data frame
#   --argument:
#    --dataframe name 
#    --columns names
def delete_column_df(df_name,*args):
    for name_col in args:
        df_name = df_name.drop(columns=[name_col])
    return df_name        




#Calculate idf:
#------argument: an array / numpy

# term frequency
def calculte_idf(name_array):
    smooth_idf = True
    norm_idf = True
    N = name_array.shape[0]
    tf = np.array([name_array[i, :] / np.sum(name_array, axis=1)[i] for i in range(N)])

    # inverse documents frequency
    df = np.count_nonzero(name_array, axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1  if smooth_idf else np.log( N / df )
    
    #Normalize 
    tfidf = normalize(tf*idf) if norm_idf else tf*idf
    name_df = pd.DataFrame(tfidf, columns=list(df3.columns))
    return name_df
    


# In[69]:


def normalize_and_create_df(idf):
    #Normalize 
    tfidf = normalize(tf*idf) if norm_idf else tf*idf
    name_df = pd.DataFrame(tfidf, columns=list(df3.columns))
    return name_df


# In[70]:


#Transform dataframe to array_np to calculate idf
def df_to_array_np(name_df):
    name_array=name_df.to_numpy().astype(np.float)
    return name_array


# In[71]:


def compare_k_AggClustering(k_list, X):
    # to find the best k number of clusters
    #X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:
        print("O K teste da vez é o K:",p)
        print("Time Start: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        clusterer = AgglomerativeClustering(n_clusters=p, linkage="average",affinity='cosine')
        clusterer.fit(X)
        #clusterer = AgglomerativeClustering(n_clusters=p, linkage="average",affinity='cosine')
        #clusterer.fit(X)
        # The higher (up to 1) the better
        s = round(metrics.silhouette_score(X, clusterer.labels_), 4)

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
        return -1
        
    meio = int((inicio+fim)/2)
    #print(df.iloc[meio,0])
    
    
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
    #df_com_labels_e_name_occ.drop(df.index, inplace=True)
    nome_do_procedimento=''
    
    
    for m in df_com_labels.itertuples():
        #print(m[2])
        
        t = binary_tree(df_com,0,df_com.shape[0]-1,m[2])

        obj = json.loads(t)
        visit_id = int((obj['visit_occurrence_id']))
#         print(obj)
          
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


# In[72]:


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
count=0
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
        count+=1
    arr.append(string_concept)


# In[74]:


#----------------------------------Pegar id + nome das ocorrencias(AUX no Mapeamento Final)-------------#
count=0
linhas =0
sera =0
laco_for=0
#--------LEMBRAR de concatenar --------#string_conc_oco= ""
#df_id_name_ocorrence = pd.DataFrame(['ID_visit','Nome_Procedimento'],[0,0])
df_id_name_ocorrence = pd.DataFrame()
lista_concep_name_ocorrencia=[]
id_visit=[]
cod_ocorrencia =[]
for i, row in df.iterrows():
    linhas+=1
    
    obj = json.loads(row['internacao_json'])
   
    if(obj['ocorrencias'] != None):
        laco_for +=1
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
        
        count+=1
#----------------------------------Criando para o MAPEAMENTO Final----------#
list_of_tuples = list(zip(id_visit, cod_ocorrencia,lista_concep_name_ocorrencia))
df_id_nome_concept = pd.DataFrame(list_of_tuples, columns = ['ID_visita','Cod_Ocorrencia' ,'Nome_Ocorrencia']) 


#-------------------Pre-processing- Manter apenas Letra e numero------------# 
input = arr_t
arr_t = [x.lower() for x in input] 

input1 = arr
arr = [x.lower() for x in input1] 

arr_t = exclude_repeated_att(arr_t)
#arr_t
#print(len(arr_t))

arr = [re.sub(' +', ' ', elem) for elem in arr]
arr = [re.sub("[^A-Za-z\d\&]", "_", elem) for elem in arr]





#---------------------- Pre-processing---- para lidar apenas como uma unica palavra, trocando sepador para o metodo IDFID___
arr = list(map(lambda s: s.replace('&' , ' '), arr))
#print(len(arr))


# In[77]:



#---- Criando e chamando método TF-IDF, tranformando texto em contagen------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(arr)


# In[78]:


#---- Normalizando e criando um novo dataframe ---- Com VALORES  TF-IDF
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(arr)
df_tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
print(df_tfidf)


# In[79]:


#Colocando o id no Dataframe 
df_complete = df_tfidf.head(45720)#----------------------------------------------------------------------MUDAR

#print(df_complete.shape)

####-----------------------IMPORTANTE add ID APOS TDFID--------------------------###
df_complete.insert(loc=0, column='ID', value=array_id2)
# In[81]:


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

# In[ ]:


#----- Metodo Completo --- K --- FOLDS---#
permanente = dfs
outra = dfs
aux_dfs = permanente
aux_dfs2 = permanente
array_treino = []
array_teste =[]
contador =0
for i in range(0,10):
    #df_treino =[]
    #array_teste =[]
    #array_teste.append(aux_dfs[i].values)
   # array_teste.append(aux_dfs[i+1].values)
#    print("-----------------------------------")
#    print(i)
#    print(i+1)

    tmp=list(aux_dfs)
    
    if(i!=9):
        frames = [tmp[i], tmp[i+1]]
    else:
        frames = [tmp[i], tmp[0]]
        

    train = pd.concat(frames)
     

    #print(train.shape)
    #break
    
    if(i!=9):
        tmp.pop(i+1)
        tmp.pop(i)
    else:
        tmp.pop(i)
        tmp.pop(0)
    
     
    
    

    test = pd.concat(tmp)
    
    
    #print(test.shape)
    #break
    
    
    
    test_np = np.array(test)
    train_np = np.array(train)
    #tudo = np.array(df_result_t)
    
    f= open("TESTE.txt","a+")
    f.write("________________________________________________________________"  +"\r\n") 
    f.write("Tempo inicio:"+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"\r\n") 
    f.write("k-FOLD:" +str(i+1)+"\r\n") 
    f.close()
    
    list_met=[]
    List_k=list(range(75,756, 75))
    
#    print(List_k)
    
    
    #for j in : # Tenho que mudar de 2 a qnt linha
    #    list_met.append(j)
    print("Realizando Shilueta "+ str(i+1)+"\r\n")    
    numero_k_test = compare_k_AggClustering(List_k,train_np[ :,1:])
    #numero_k_test = int((4572/41148)*numero_k)
    #numero_k_test =10
    f= open("TESTE.txt","a+")
    f.write("k-Teste:" +str(numero_k_test)+"\r\n") 
    f.close()
    
    
    
    print("Loading Clustering....")
#    print("Tempo inicio: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    print("Tempo inicio: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    clusterer = AgglomerativeClustering(n_clusters=numero_k_test, linkage="average",affinity='cosine')#------------------------------MUDAR
    clusterer.fit(test_np[:,1:])

    #--------------------------- KMEANS"-------##
    
    labels=[]
    labels=clusterer.labels_
     #----------------------------------------------------------------------##
    #print(oi[0,:])
    #id_ =list(oi)
    id_ =test_np[:,0]
    #print(oi[0,0])
    #print()
    #print("_______________________________")
    
    list_of_tuples = list(zip(id_,labels))
    new_df = pd.DataFrame(list_of_tuples, columns = ['ID','Labels']) 
    
     #-------------------------------_Adicionando todos LAbels no Dataframe completo_-----------------#
    complete_labels = test
    complete_labels.insert(loc=0, column='Labels', value=labels)
    complete_labels.sort_values('Labels', inplace=True, ascending=False)
    #print(complete_labels)
    df_in_file = add_name_occ(complete_labels,df_original)
    df_in_file.to_csv(r'/home/oscar/ModeloCompleto/KmeansAgglo/ComNulo/DataframeCompleto_Labels/DF_K_FOLD_T :'+str(contador)+".csv",index=False)
    
    #-------------------------------------------------#-------------------------#------------------------------------------------------------
    
    
    s = metrics.silhouette_samples(test_np[:,1:], labels)
    join_s_labels = list(zip(labels, s))
    contador+=1
    df_silueta = pd.DataFrame(join_s_labels, columns = ['Labels','Silhouette_samples']) 
    df_silueta.to_csv("Shilhoutte_KFold_"+str(contador)+".csv")
    df_silueta.to_csv(r'/home/oscar/ModeloCompleto/KmeansAgglo/ComNulo/Shilhoutte_KFold_'+str(contador)+".csv")
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
    
   






