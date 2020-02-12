import pandas as pd
# def CountFrequency(my_list): 
#     nome_ocorrencia = []
#     qnt_ocorrencia = []
#     freq_ocorrencia =[]
    
#     #print("ENTROUUUUUUUUUUUU COUNT")
    
#     #f= open("TESTE.txt","a+")  
#     # Creating an empty dictionary  
#     freq = {} 
#     for items in my_list: 
#         freq[items] = my_list.count(items) 
      
#     for key, value in freq.items():
#         len_list = len(my_list)
#         freq_oc= (value/len_list)*100
        
#         print ("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(key, value,freq_oc)) #-------------------------------------teSTE
#         nome_ocorrencia.append(key)
#         qnt_ocorrencia.append(value)
#         freq_ocorrencia.append(freq_oc)
        
        
# #    print("VAI entrar no sort")    
#     list_of_tuples = list(zip(nome_ocorrencia, qnt_ocorrencia,freq_ocorrencia))
#     df_ocorrencias_ordenadas = pd.DataFrame(list_of_tuples, columns = ['nome_ocorrencia','qnt_ocorrencia' ,'freq_ocorrencia'])
#     df_ocorrencias_ordenadas.sort_values('freq_ocorrencia', inplace=True, ascending=False)
#     print(df_ocorrencias_ordenadas)
#     f= open("TESTE_Bruno.txt","a+")
#     for o in df_ocorrencias_ordenadas.itertuples():
#         f.write("Nome Ocorrencia:  %s || Quantidade de Ocorrência %d :  Porcentagem da ocorrência: %f "%(o.nome_ocorrencia, o.qnt_ocorrencia,o.freq_ocorrencia) +"\r\n")
#     f.close()
        

# # 10 elementos
# # 3 Brunos
# # 6 Juninho
# # 1 Christian
# lista=["Bruno", "Juninho", "Juninho", "Bruno", "Juninho", "Juninho", "Bruno", "Juninho", "Juninho", "Christian"]

# CountFrequency(lista)

# ------------------------------------------------------------------------------------------------------------------------------------






######################################## BKP da linha 550 até a 580

#     #Percorrendo o Data Frame que possui o Id da Ocurrencia e os Labels que foram previstos
#     for v in new_df['Labels'].mode():
#         print("FAZENDO A MODAcd ")
#         grupo_fre = v
#         qnt_f = new_df['Labels'].value_counts().max()
#         porc_f = (qnt_f/36576)*100 #--------------------------------------------------MUDAR
#         f= open("TESTE.txt","a+")
#         f.write("O Grupo  " +str(grupo_fre)+" abrange "+str(porc_f)+"%"+ " do teste" +"\r\n") 
        
        
#         list_d =[]
        
#        #new_df :Data frame columns = ['ID','Labels']) 
#         print("Busca e frenquencia Ocorrencia")
#         print("Tempo inicio: "+ str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
#         for m in new_df.itertuples():
#             obj = m.Labels
#             obj_Id =m.ID
            
#             for n in df_id_nome_concept.itertuples():
#                 obj2 = n.ID_visita
#                 if obj2 == obj_Id:
#                     list_d.append(n.Nome_Ocorrencia)
        
#     L= open("list_d.txt","a+")
#     L.write(list_d)
#     L.close()
#     CountFrequency(list_d)
    
#     f.close()
    
    
# ---------------------------------------------------------------------------------------------------



new_df= pd.DataFrame([(11,1),(11,1),(11,1),(11,1),(11,1),(11,1),(22,2),(22,2),(22,2),(22,2)], columns = ['ID','Labels']) 


for v in new_df['Labels'].mode():
#     print(v)
    grupo_fre = v
    qnt_f = new_df['Labels'].value_counts().max() #Pega o valor máximo que representa o valor que as modas possuem
#     print(qnt_f)
    porc_f = (qnt_f/10)*100
#     print(porc_f)
    for v in new_df['Labels'].mode():
        for m in new_df.itertuples():
            if m.Labels == v:
                print(v, m)
#             if m.Labels == v:
                