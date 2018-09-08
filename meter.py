# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')

def is_ponta(x):
    x = int(x.split(' ')[1].split(':')[0])
    if x >=19 and x <=22:
        return 'ponta'
    else:
        return 'nponta'
    
def main(dataset):
    
    dataset = dataset[dataset['vab'] < 28]
    #Le o arquivo .csv para montar o dataset
    
    for col in ['ca' , 'cb' , 'cc']: 
    
        dataset[col] = dataset[col].map(lambda x: float(x.replace(',','.')))
        #transforma valores de corrente STR em Float
        
    dataset = dataset[dataset['ca'] < 36] #Exclui valores de corrente maior que 36 A [Ruido do sistema]
    dataset['pot'] = ((dataset.vab+dataset.vbc+dataset.vca)/3)*((dataset.ca+dataset.cb+dataset.cc)/3)*np.sqrt(3) 
    #Calc potencia instantanea do sistema
    
    consumo_total = np.trapz(dataset['pot'], dx=(1/60))
    # Calc consumo com integrando numericamente a potencia
    
    dataset['hora'] = dataset['data'].map(lambda x: is_ponta(x)) 
    dataset['hour'] = dataset['data'].map(lambda x: x.split(' ')[1]+'h')
    #Classifca o dataset com dentro ou fora de ponta
    
    dponta = dataset[dataset['hora']=='ponta']
    fponta = dataset[dataset['hora']=='nponta']
    #Separa o dataset total em 2 [Dentro e fora da ponta]
    
    consumo_dentro_ponta = np.trapz(dponta['pot'], dx=(1/60))
    consumo_fora_ponta = np.trapz(fponta['pot'], dx=(1/60))
    #Calcula com integral numerica os consumos dentro e fora da ponta
    
    dataset.drop_duplicates('data', inplace=True) # Deleta datas repetidas
    queda = dataset[dataset['ca']==0] #Transforma o dataset em um DF binario [1= tem energia e 0 = sem energia]
    queda = np.array(dataset[['vab','vbc','vca','ca' , 'cb', 'cc', 'consumo']]==0, dtype=np.uint8)*255
    #Transforma o dataset em array para tratar como imagem
    
    lx , contorno , lx = cv2.findContours(queda.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    qtd_quedas = len(contorno)
    #aplica um detector de objetos para contar as quedas de energia
    
    dataset.reset_index(drop=True, inplace=True)
    #reseta os indices do dataframe
    
    dataset['vmediag'] = (dataset['vab']*dataset['vbc']*dataset['vca'])**(1/3)
    dataset['cmediag'] = (dataset['ca']*dataset['cb']*dataset['cc'])**(1/3)
    group = dataset['hour vmediag cmediag pot'.split(' ')].groupby('hour')
    group = group.mean()
    #Calcula a média das grandezas nos minutos do dia

    dataset['dv'] = np.abs(np.gradient(dataset['vmediag']))*1000
    #Calcula a derivada numerica da média geometrica das tensoes das fases
    desvio = dataset[(dataset['dv'] > 500) & (dataset['cmediag'] > 0)]
    #Encontra as variações maiores que 500V e exclui as quedas de tensão
    fluatacao = list(desvio['data'].unique())
    #Dias onde ocorreram flutuação

    callback = {
    'consumo': round(consumo_total*0.001, 1),
    'consumo_dentro_ponta': round(consumo_dentro_ponta*0.001, 1),
    'consumo_fora_ponta': round(consumo_fora_ponta*0.001, 1),
    'demanda_dentro_ponta': round(dponta['pot'].max(), 1),
    'demanda_fora_ponta': round(fponta['pot'].max(), 1),
    'tensao_ab': round(dataset['vab'].mean(), 3),
    'desvio_ab': round(dataset['vab'].std()*1000, 0),
    'tensao_bc': round(dataset['vbc'].mean(), 3),
    'desvio_bc': round(dataset['vbc'].std()*1000, 0),
    'tensao_ca': round(dataset['vca'].mean(), 3),
    'desvio_ca': round(dataset['vca'].std()*1000, 0),
    'corrente_a': round(dataset['ca'].mean(), 1),
    'corrente_b': round(dataset['cb'].mean(), 1),
    'corrente_c': round(dataset['cc'].mean(), 1),
    'desvio_a': round(dataset['ca'].std(), 1),
    'desvio_b': round(dataset['cb'].std(), 1),
    'desvio_c': round(dataset['cc'].std(), 1),
    'quedas_total': qtd_quedas,
    'quedas': [],
    'chart_dataset': {
        'data': list(group.index),
        'tensao': list(round(group['vmediag'], 3)),
        'corrente': list(round(group['cmediag'], 1)),
        'potencia': list(round(group['pot'],0))
    },
    'vtcd': fluatacao

    }
    #Monta um dicionatrio com informações do dataset
    
    for c in contorno:
    
        queda = {
            'data': 0,
            'duracao': 0,
            'motivo': 0,        
        }
        x, y, w, h = cv2.boundingRect(c)
        queda['data'] = dataset['data'][y].split(' ')[0]
        queda['duracao'] = h
    
        if w > 3:
            queda['motivo'] = 'Externa'
        else:
            queda['motivo'] = 'Interna'
        callback['quedas'].append(queda)
    #Adiciona ao dicionario informações sobre as quedas
    
    return callback 
    #retorna o dicionario

def data_charts(dataset, dia):

    dataset = dataset[dataset['vab'] < 28]
    #Le o arquivo .csv para montar o dataset
    
    for col in ['ca' , 'cb' , 'cc']: 
    
        dataset[col] = dataset[col].map(lambda x: float(x.replace(',','.')))
        #transforma valores de corrente STR em Float
        
    dataset = dataset[dataset['ca'] < 36] #Exclui valores de corrente maior que 36 A [Ruido do sistema]

    dataset['vmediag'] = (dataset['vab']*dataset['vbc']*dataset['vca'])**(1/3)
    dataset['cmediag'] = (dataset['ca']*dataset['cb']*dataset['cc'])**(1/3)
    #Calcula média geométrica entre as tensoes de fase e correntes

    dataset['dia'] = dataset['data'].map(lambda x: x.split(' ')[0])
    dataset['hour'] = dataset['data'].map(lambda x: x.split(' ')[1])
    dataset = dataset[dataset['dia'] == dia]
    #filtra dataset de acordo com dia passado

    return {
        'labels': list(dataset['hour']),
        'corrente': list(round(dataset['cmediag'], 1)),
        'tensao': list(round(dataset['vmediag'], 3))
    }


