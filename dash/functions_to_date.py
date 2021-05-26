from datetime import datetime
from datetime import timedelta
from calendar import month_abbr
import pandas as pd
import numpy as np

def dias():
    dois_meses_seguinte = str(datetime.now() + timedelta(weeks=8))[:10]
    dias = np.arange('2020-04-01', dois_meses_seguinte, dtype='datetime64[M]')
    dias = [str(x) + '-01' for x in dias]
    dias = pd.to_datetime(dias)
    return dias

#Ajuste de eixo X para gráficos de períodos mais curtos

def traduzir_eixo_x(data, janela_inicial, steps):
    #### Ajusta tickvals e ticktext baseado em semanais e em PT-BR
    
    dici = {'Jan': 'Jan',
                'Feb': 'Fev',
                'Mar': 'Mar',
                'Apr': 'Abr', 
                'May': 'Maio',
            'Jun': 'Jun',
            'Jul': 'Jul',
            'Aug': 'Ago',
            'Sep': 'Set',
            'Oct': 'Out', 
            'Nov': 'Nov',
            'Dec': 'Dez'}    
    ##conversão para string e ajuste para 7 dias
        
    data_str = [str(x) for x in data.values]
    data_str = [x[:x.index('T')] for x in data_str]
    data_str = data_str[janela_inicial:]
    tickvals = []
        
    for i in range(0, len(data_str), steps):
        tickvals.append(data_str[i])
    #separação para tradução de mês por meio de lib
        
    dias = [x[-2:] for x in tickvals]
    anos_sigla = [str(x)[:4] for x in tickvals]
    mes_int = [int(str(x)[5:7]) for x in tickvals]
    mes_sigla_ingles = [month_abbr[x] for x in mes_int]
    mes_sigla_portugues = [dici[x] for x in mes_sigla_ingles]
    mes_ano = meses_anos('2020-04-01')
    mes_ano = [" ".join((x, y)) for x,y in zip(mes_sigla_portugues, anos_sigla)]
    ticktext = [" ".join((x,y)) for x,y in zip (dias, mes_ano)]
        
    #retorna tickvals e ticktext
        
    return tickvals, ticktext

def epocas_festivas():
    epocas_festivas = pd.read_csv(r'../epocas_festivas/epocas_festivas.csv')
    epocas_festivas['date'] = ['2020-' + x for x in epocas_festivas['date'] if '2020-' not in x] #feito manualmente, necessário otimizar
    feriados_ano_atual = [(x.replace('2020-','2021-'),y) for x,y in zip(epocas_festivas['date'], epocas_festivas['name']) if '2021-' not in x]
    feriados_ano_atual = pd.DataFrame(feriados_ano_atual, columns=['date', 'name'])
    epocas_festivas = pd.concat([epocas_festivas, feriados_ano_atual], ignore_index=True)
    epocas_festivas['date'] = pd.to_datetime(epocas_festivas['date'])
    return epocas_festivas
#Ajuste de eixo X para gráficos de períodos mais longos

def meses_anos(data):
    
    dici = {'Jan': 'Jan',
            'Feb': 'Fev',
            'Mar': 'Mar',
            'Apr': 'Abr', 
            'May': 'Maio',
        'Jun': 'Jun',
        'Jul': 'Jul',
        'Aug': 'Ago',
        'Sep': 'Set',
        'Oct': 'Out', 
        'Nov': 'Nov',
        'Dec': 'Dez'}
    dois_meses_seguintes = str(datetime.now() + timedelta(weeks=8))[:10]
    
    dias = np.arange(data, dois_meses_seguintes, dtype='datetime64[M]')
    anos_sigla = [str(x)[:4] for x in dias]
    mes_int = [int(str(x)[5:7]) for x in dias]
    mes_sigla_ingles = [month_abbr[x] for x in mes_int]
    mes_sigla_portugues = [dici[x] for x in mes_sigla_ingles]
    
    meses_anos = [" ".join((x, y)) for x,y in zip(mes_sigla_portugues, anos_sigla)]

    return meses_anos

dias_traduzidos = {'Monday': 'Segunda', 
                    'Tuesday': 'Terça',
                    'Wednesday': 'Quarta',
                    'Thursday': 'Quinta',
                    'Friday': 'Sexta',
                    'Saturday': 'Sábado',
                    'Sunday': 'Domingo'}