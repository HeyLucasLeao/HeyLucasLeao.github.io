import pandas as pd
import urllib.request
from os import listdir
from datetime import datetime
import csv
import numpy as np
from tabula import read_pdf
import requests
from datetime import timedelta
from time import sleep

PATH_PDF = r'../raspagem_dos_boletins_diarios/relatorios'
PATH_CSV = r'../raspagem_dos_boletins_diarios/raw_csvs'

def atualizar_csvs():
    
    data_csv = pd.to_datetime(data, format="%d_%m_%y_")
    data_csv = str(data_csv)[2:10]
    
    def change_rows(x):
        dici_csv = {'REDE PÚBLICA': 'Rede Publica',
        'Cardíaco': 'Cardiaco',
        'REDE PRIVADA': 'Rede Privada',
        'TOTAL': 'Total'}
        if x in dici_csv.keys():
            return dici_csv[x]
        return x
    
    def download_file(url):
        response = urllib.request.urlopen(url)
        data_download = pd.to_datetime(data, format="%d_%m_%y_")
        data_download = str(data_download)[2:10]
        
        for files in listdir(PATH_PDF):
            if data_download in files[files.index('_') + 1:files.index('.')]:
                return
            else:
                continue
        path = r'C:\Users\heylu\Documents\github\HeyLucasLeao.github.io\raspagem_dos_boletins_diarios\relatorios'

        with open(path + '\\' + f'relatorio_{data_download}.pdf', mode='wb') as file:
            file.write(response.read())
            
    for files in listdir(PATH_PDF):
            if data_csv in files[files.index('_') + 1:files.index('.')]:
                return
            else:
                continue
                
    download_file(link)
    
    atualizacao_de_csvs = taxa_de_ocupacao.copy()
    atualizacao_de_csvs['unidade'] = atualizacao_de_csvs['unidade'].apply(change_rows)
    
    normalizados = atualizacao_de_csvs.copy()
    
    
    atualizacao_de_csvs = atualizacao_de_csvs.T

    atualizacao_de_csvs.insert(loc=0, 
    column='Data', 
    value=data_csv)

    
    for files in listdir(PATH_CSV):
        with open(PATH_CSV + '\\' + files, 'a+', newline='') as f:
            writer = csv.writer(f)
            dados = np.array(atualizacao_de_csvs.loc[[files[:files.index('.')]]]).ravel()
            writer.writerow(dados)
            
    for file_name in listdir(PATH_CSV):
        df = pd.read_csv(PATH_CSV + "\\" + file_name, index_col='Data')
        for col in df.columns:
            for i in range(len(df[col])):
                if isinstance(df[col].iloc[i], str):
                    df[col].iloc[i] = df[col].iloc[i][:-1]
                    df[col].iloc[i] = df[col].iloc[i].replace(',', '.')
                    df[col].iloc[i] = float(df[col].iloc[i])
                    df[col].iloc[i] = round(df[col].iloc[i] / 100, 2)
                    df[col].iloc[i] = "{:.2f}".format(df[col].iloc[i])
        df.to_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs' + "/"+ file_name)

data = str(datetime.now())[2:10]
data = data.split('-')
data.reverse()
data = [x + "_" for x in data]
data = "".join(data)
data_url_acento = data
data_url_sem_acento = data
AREA = [265.022,
413.384,
373.253,
829.774]

while True:
    try:
        link = f'https://www.fvs.am.gov.br/media/publicacao/{data_url_acento}BOLETIM_DI%C3%81RIO_DE_CASOS_COVID-19.pdf'
        response = requests.get(link)
        response.raise_for_status()
        break
    except requests.HTTPError:
        data_url_acento = pd.to_datetime(data_url_acento, format="%d_%m_%y_")
        data_url_acento = data_url_acento - timedelta(1)
        data_url_acento = str(data_url_acento)[2:10]
        data_url_acento = data_url_acento.split('-')
        data_url_acento.reverse()
        data_url_acento = [x + "_" for x in data_url_acento]
        data_url_acento = "".join(data_url_acento)
        continue

while True:
    try:
        link = f'https://www.fvs.am.gov.br/media/publicacao/{data_url_sem_acento}BOLETIM_DIARIO_DE_CASOS_COVID-19.pdf'
        response = requests.get(link)
        response.raise_for_status()
        break
    except requests.HTTPError:
        data_url_sem_acento = pd.to_datetime(data_url_sem_acento, format="%d_%m_%y_")
        data_url_sem_acento = data_url_sem_acento - timedelta(1)
        data_url_sem_acento = str(data_url_sem_acento)[2:10]
        data_url_sem_acento = data_url_sem_acento.split('-')
        data_url_sem_acento.reverse()
        data_url_sem_acento = [x + "_" for x in data_url_sem_acento]
        data_url_sem_acento = "".join(data_url_sem_acento)
        continue
        
if pd.to_datetime(data_url_acento, format="%d_%m_%y_") > pd.to_datetime(data_url_sem_acento, format="%d_%m_%y_"):
    taxa_de_ocupacao = read_pdf(f'https://www.fvs.am.gov.br/media/publicacao/{data_url_acento}BOLETIM_DI%C3%81RIO_DE_CASOS_COVID-19.pdf', pages=2, area=AREA, stream=True)[0]
    link = f'https://www.fvs.am.gov.br/media/publicacao/{data_url_acento}BOLETIM_DI%C3%81RIO_DE_CASOS_COVID-19.pdf'
    data = data_url_acento
else:
    taxa_de_ocupacao = read_pdf(f'https://www.fvs.am.gov.br/media/publicacao/{data_url_sem_acento}BOLETIM_DIARIO_DE_CASOS_COVID-19.pdf', pages=2, area=AREA, stream=True)[0]
    link = r'http://www.fvs.am.gov.br/media/publicacao/{data_url_sem_acento}BOLETIM_DIARIO_DE_CASOS_COVID-19.pdf'
    data = data_url_sem_acento

taxa_de_ocupacao.drop(index=[0, 1, 8],columns=['Unnamed: 5'], inplace=True)

taxa_de_ocupacao.rename(columns={'Unnamed: 0': 'unidade',
                                 'Unnamed: 1': 'uti_geral',
                                 'Unnamed: 2': 'uti_covid-19',
                                'Unnamed: 3': 'leitos_clinicos_geral',
                                'TAXA DE OCUPAÇÃO EM MANAUS': 'leitos_clinicos_covid-19',
                                'Unnamed: 4': 'sala_vermelha_geral',
                                'Unnamed: 6': 'sala_vermelha_covid-19'}, inplace=True)

#taxa_de_ocupacao['uti_geral'] = [x.split()[-1] for x in taxa_de_ocupacao['unidade']]
#taxa_de_ocupacao['unidade'] = [" ".join(x.split()[:-1]) for x in taxa_de_ocupacao['unidade']]
taxa_de_ocupacao['uti_covid-19'] = taxa_de_ocupacao['leitos_clinicos_geral']
taxa_de_ocupacao['leitos_clinicos_geral'] = [x.split()[:-1][0] for x in taxa_de_ocupacao['leitos_clinicos_covid-19']]
taxa_de_ocupacao['leitos_clinicos_covid-19'] = [x.split()[-1] for x in taxa_de_ocupacao['leitos_clinicos_covid-19']]

if len(taxa_de_ocupacao.columns) == 7 and taxa_de_ocupacao.isnull().sum().sum() == 0:
    atualizar_csvs()
    print('CSVs atualizados.')
else:
    print('Taxa não extraída corretamente, favor verificar.')
    sleep(10)