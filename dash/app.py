#!/usr/bin/env python
# coding: utf-8

# # Relatório sobre Covid-19 com foco no Estado do Amazonas

# ###### Notícias da Fundação de Vigilância em Saúde do Amazonas (FVS-AM): https://share.streamlit.io/heylucasleao/noticias-fvs-am/main

import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.filters.hp_filter import hpfilter
import geopandas as gpd
from plotly.subplots import make_subplots
from functions_to_date import dias, epocas_festivas, meses_anos, traduzir_eixo_x, dias_traduzidos

from lightgbm import LGBMRegressor
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.compose import make_reduction


from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_table

import warnings
warnings.filterwarnings("ignore")

PATH_PDF = r'../raspagem_dos_boletins_diarios/relatorios'
PATH_CSV = r'../raspagem_dos_boletins_diarios/raw_csvs'
GLOBAL_TEMPLATE = 'plotly_dark'
MAPBOX_TOKEN = 'pk.eyJ1IjoiaGV5bHVjYXNsZWFvIiwiYSI6ImNrcDRqeW1yNzAzaHIycHNiZWo1bHBqeGkifQ.kw4Buc12S2g1CTZ7KhXwYA'
rgb = 'rgb(34, 34, 34)'
red = '#812929'
url = 'https://github.com/wcota/covid19br/blob/master/cases-brazil-cities-time.csv.gz?raw=true'
df = pd.read_csv(url, compression='gzip')
df['date'] = pd.to_datetime(df['date'])


df.drop('ibgeID', axis=1, inplace=True)
df.drop('epi_week',axis=1, inplace=True)
df.drop('cod_RegiaoDeSaude',axis=1, inplace=True)
df.drop('name_RegiaoDeSaude',axis=1, inplace=True)
df.drop('country', axis=1, inplace=True)


gjson_estados_brasileiros = gpd.read_file(r"https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")
gjson_municipios_amazonas = gpd.read_file(r"https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-13-mun.json")
gjson_estados_brasileiros.set_index('id', inplace=True)
gjson_municipios_amazonas.set_index('id', inplace=True)
dici = dict([(x,y) for x,y in zip(gjson_estados_brasileiros['sigla'], gjson_estados_brasileiros['name'])])


df = df.query("city != 'TOTAL'")
df['city'] = [x for x in df['city'] if x[:28] != 'CASO SEM LOCALIZAÇÃO DEFINIDA']
last_info = df['last_info_date'].sort_values(ascending=False).reset_index().iloc[0][1]
df.sort_values('totalCases', ascending=False,inplace=True)

total_de_casos_amazonas = df.query("state == 'AM'").groupby('date').sum()
total_de_casos_amazonas.reset_index(inplace=True)
total_de_casos_amazonas['dia_da_semana'] = total_de_casos_amazonas['date'].dt.day_name()
total_de_casos_amazonas['dia_da_semana'] = total_de_casos_amazonas['dia_da_semana'].map(dias_traduzidos)
total_de_casos_amazonas['media_movel_novos_casos'] = total_de_casos_amazonas['newCases'].ewm(span=7).mean().round()
total_de_casos_amazonas['media_movel_novos_casos'].fillna(value=0, inplace=True)

df_total_10_maiores_cidades = df.copy()
df_total_10_maiores_cidades = df_total_10_maiores_cidades.groupby('city').max().sort_values('totalCases', ascending=False).head(10)
df_total_10_maiores_cidades.reset_index(inplace=True)
df_total_10_maiores_cidades.sort_values('city', inplace=True)

total_por_estado = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
total_por_estado = total_por_estado.query(f"date == '{last_info}' and state != 'TOTAL'")
total_por_estado['name'] = [dici[x] for x in total_por_estado['state']]
total_casos_e_mortes_por_estado = total_por_estado
total_casos_e_mortes_por_estado.reset_index(inplace=True)
total_casos_e_mortes_por_estado['id'] = [(i + 1) for i in range(len(total_casos_e_mortes_por_estado['state']))]
epocas_festivas = pd.merge(epocas_festivas(), 
                        total_de_casos_amazonas[['date','newCases', 'newDeaths']], 
                           on='date',
                           how='left').sort_values('date').dropna().reset_index()
df_am = df.query("state == 'AM' and city != 'CASO SEM LOCALIZAÇÃO DEFINIDA/AM'")
df_am = df_am.query(f"date == '{last_info}'")
dici = dict([(x,y) for x,y in zip(gjson_municipios_amazonas.name, gjson_municipios_amazonas.index)])
df_am['city'] = [x[:x.index('/')] for x in df_am['city']]
df_am['id'] = [dici[x] for x in df_am['city']]

total_de_casos_amazonas_por_mes = total_de_casos_amazonas.set_index('date').groupby(pd.Grouper(freq='M')).sum()[['newDeaths','newCases']]
total_de_casos_amazonas_por_mes.reset_index(inplace=True)
total_de_casos_amazonas_por_mes['taxa_de_letalidade'] = round(total_de_casos_amazonas_por_mes['newDeaths']/total_de_casos_amazonas_por_mes['newCases'] * 100, 2)

total_de_casos_brasil = df.groupby('date').sum()
total_de_casos_brasil.reset_index(inplace=True)

#Criação de variaveis

total_de_casos_amazonas['crescimento_novos_casos'] = (total_de_casos_amazonas['newCases'].diff() / total_de_casos_amazonas['newCases'].rolling(7).mean()) * 100
total_de_casos_amazonas['crescimento_novos_obitos'] = (total_de_casos_amazonas['newDeaths'].diff() / total_de_casos_amazonas['newCases'].rolling(7).mean()) * 100
crescimento_percentual = pd.merge(total_de_casos_amazonas[['date','crescimento_novos_casos']], 
                                  total_de_casos_amazonas[['date', 'crescimento_novos_obitos']], 
                                  on='date', how='left')
crescimento_de_casos = crescimento_percentual[['date', 'crescimento_novos_casos']]
crescimento_de_casos.insert(1,column= 'variavel',value='crescimento_novos_casos')
crescimento_de_casos.rename(columns={'crescimento_novos_casos': 'valor'}, inplace=True)

crescimento_de_obitos = crescimento_percentual[['date', 'crescimento_novos_obitos']]
crescimento_de_obitos.insert(1,column= 'variavel',value='crescimento_novos_obitos')
crescimento_de_obitos.rename(columns={'crescimento_novos_obitos': 'valor'}, inplace=True)

crescimento = pd.concat([crescimento_de_casos, crescimento_de_obitos])

crescimento.sort_values(['date','variavel'], inplace=True)
dici = {'crescimento_novos_obitos': "Óbitos", 'crescimento_novos_casos': 'Casos'}
crescimento['variavel'] = crescimento['variavel'].apply(lambda x: dici[x])

media_casos_por_dia_da_semana = total_de_casos_amazonas.groupby('dia_da_semana')[['newCases', 'newDeaths', 'crescimento_novos_casos']].mean().round().reindex(['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'])
media_casos_por_dia_da_semana['newDeaths'] = [round(x) for x in media_casos_por_dia_da_semana['newDeaths']]
media_casos_por_dia_da_semana['newCases'] = [round(x) for x in media_casos_por_dia_da_semana['newCases']]
media_casos_por_dia_da_semana['crescimento_novos_casos'] = [round(x) for x in media_casos_por_dia_da_semana['crescimento_novos_casos']]
media_casos_por_dia_da_semana.rename(columns={'newDeaths': 'Novos Óbitos',
                                             'newCases': 'Novos Casos',
                                             'crescimento': 'Crescimento'}, inplace=True)
media_casos_por_dia_da_semana.rename_axis('Dia da Semana', inplace=True)

def dados_apresentaveis(x):
    x = round(x)
    x ="{:,}".format(x)
    x = x.replace(',','.')
    return x


def to_zero(x):
    if x < 0:
        x = 0
    return x

_, trend_newCases = hpfilter(total_de_casos_amazonas['newCases'])
trend_newCases = trend_newCases.apply(to_zero)
_, trend_newDeaths = hpfilter(total_de_casos_amazonas['newDeaths'])
trend_newDeaths = trend_newDeaths.apply(to_zero).round()

def show_figure1():
    fig = make_subplots(subplot_titles=('10 Cidades com os Maiores Casos & Óbitos Registrados por COVID-19', 
                                        'Relação de Casos & Óbitos por 100k Habitantes por Estado'),
                        rows=1, cols=2)

    ### Configuração de gráfico a esquerda
    fig.add_trace(go.Scatter(x = df_total_10_maiores_cidades['city'], 
                             y= df_total_10_maiores_cidades['totalCases'],
                             mode='markers',
                     marker=dict(color = df_total_10_maiores_cidades['totalCases'],
                                colorscale= ['cyan', 'crimson'],
                                size=df_total_10_maiores_cidades['totalCases'],
                                sizemode='area',
                                sizeref=110,
                                showscale=False),
                             text=df_total_10_maiores_cidades['deaths'],
                             name='',
                             customdata=df_total_10_maiores_cidades['last_info_date'],
                            hovertemplate='<br>%{x}<br>' + 
                             '<br>N.º de Casos: %{y:,2f}<br>' + 
                             'N.º de Óbitos: %{text:,2f}<br>' +
                            'Data: %{customdata}'), row=1, col=1) 

    fig.update_xaxes(title_text='Cidades', row=1, col=1)
    fig.update_yaxes(title_text='N.º de Casos', row=1, col=1)
    fig['data'][0]['showlegend']=False

    ### Configuração de gráfico a direita

    for col in total_por_estado['state'].values:
        df = total_por_estado[total_por_estado['state'] == col]
        fig.add_trace(go.Scatter(x = df['totalCases_per_100k_inhabitants'], 
                     y = df['deaths_per_100k_inhabitants'], 
                     mode='markers',
                    name=df['name'].values[0],
                    marker=dict(
                                size=df['totalCases'],
                                sizemode='area',
                                sizeref=80*5,
                                showscale=False,
                               opacity=0.50),
                    text=df['name'],
                    customdata=df[['totalCases', 'deaths']],
                    hoverlabel=dict(namelength=0),
                    hovertemplate='<br>%{text}<br>' + 
                    '<br>Casos por 100k Habitantes: %{x:,f}<br>' + 
                    'Óbitos por 100k Habitantes: %{y:,f}' +
                    '<br>N.º de Casos: %{customdata[0]:,2f}' + 
                    '<br>N.º de Óbitos: %{customdata[1]:,2f}<br>'), row=1, col=2)

    fig.update_yaxes(type="log", row=1,col=2)

    fig.update_traces(marker=dict(color='#597386'), row=1, col=2)

    fig.update_xaxes(title_text='Óbitos por 100k Habitantes', row=1, col=2)
    fig.update_yaxes(title_text='Casos por 100k Habitantes', row=1, col=2)

    ###Global

    fig.update_layout(height = 800, width = 1600, separators=",.", font=dict(size=12), template=GLOBAL_TEMPLATE, paper_bgcolor=rgb, plot_bgcolor=rgb)
    
    return fig

def tables():
        total_por_estado_tabela = total_por_estado.copy()
        total_por_estado_tabela.drop(columns = ['city',
                                            'country',
                                                'epi_week',
                                                'totalCasesMS',
                                                'deathsMS',
                                                'tests_per_100k_inhabitants',
                                                'tests',
                                                'suspects',
                                                'recovered',
                                                'deaths_by_totalCases',
                                                'date'], axis=1, inplace=True)
        total_por_estado_tabela.set_index('state', inplace=True)
        total_por_estado_tabela.sort_values('deaths_per_100k_inhabitants', ascending=False, inplace=True)



        total_por_estado_tabela['newCases'] = total_por_estado_tabela['newCases'].apply(dados_apresentaveis)
        total_por_estado_tabela['newDeaths'] = total_por_estado_tabela['newDeaths'].apply(dados_apresentaveis)
        total_por_estado_tabela['deaths'] = total_por_estado_tabela['deaths'].apply(dados_apresentaveis)
        total_por_estado_tabela['totalCases'] = total_por_estado_tabela['totalCases'].apply(dados_apresentaveis)
        total_por_estado_tabela['totalCases_per_100k_inhabitants'] = total_por_estado_tabela['totalCases_per_100k_inhabitants'].apply(dados_apresentaveis)
        total_por_estado_tabela['deaths_per_100k_inhabitants'] = total_por_estado_tabela['deaths_per_100k_inhabitants'].apply(dados_apresentaveis)
        total_por_estado_tabela['vaccinated'] = total_por_estado_tabela['vaccinated'].apply(dados_apresentaveis)
        total_por_estado_tabela['vaccinated_per_100k_inhabitants'] = total_por_estado_tabela['vaccinated_per_100k_inhabitants'].apply(dados_apresentaveis)
        total_por_estado_tabela.reset_index(inplace=True)

        total_por_estado_tabela = total_por_estado_tabela.rename(columns={'name': 'Estado',
                                                                        'state': 'Sigla', 
                                                                        'deaths': "Total de Óbitos", 
                                                                        'totalCases': 'Total de Casos', 
                                                                        'deaths_per_100k_inhabitants': 
                                                                        'Óbitos por 100k Habitantes', 
                                                                        'totalCases_per_100k_inhabitants': 
                                                                        'Total de Casos por 100k Habitantes', 
                                                                        'newCases': 'Novos Casos', 
                                                                        'newDeaths': 'Novos Óbitos', 
                                                                        'vaccinated': "Vacinados", 
                                                                        'vaccinated_per_100k_inhabitants': "Vacinados por 100k Habitantes"})


        total_por_estado_tabela.index = np.arange(1, len(total_por_estado_tabela) + 1)
        total_por_estado_tabela = total_por_estado_tabela[['Estado', 
                                                        'Sigla', 
                                                        'Novos Casos', 
                                                        'Novos Óbitos', 
                                                        'Total de Casos', 
                                                        'Total de Óbitos', 
                                                        'Total de Casos por 100k Habitantes', 
                                                        'Óbitos por 100k Habitantes', 
                                                        'Vacinados', 
                                                        'Vacinados por 100k Habitantes']]
        total_por_estado_tabela.drop(columns=['Novos Casos', 'Novos Óbitos', 'Total de Casos'], axis=1, inplace=True)



        df_am_tabela = df.query(f"state == 'AM' and city != 'CASO SEM LOCALIZAÇÃO DEFINIDA/AM' and date == '{last_info}'")
        df_am_tabela = df_am_tabela[['city', 'newDeaths', 'deaths', 'deaths_per_100k_inhabitants', 'totalCases_per_100k_inhabitants', 'deaths_by_totalCases']]
        df_am_tabela['city'] = [x[:x.index('/')] for x in df_am_tabela['city']]
        df_am_tabela['newDeaths'] = df_am_tabela['newDeaths'].apply(dados_apresentaveis)
        df_am_tabela['deaths'] = df_am_tabela['deaths'].apply(dados_apresentaveis)
        df_am_tabela['deaths_per_100k_inhabitants'] = df_am_tabela['deaths_per_100k_inhabitants'].apply(dados_apresentaveis)
        df_am_tabela['totalCases_per_100k_inhabitants'] = df_am_tabela['totalCases_per_100k_inhabitants'].apply(dados_apresentaveis)





        df_am_tabela['deaths_by_totalCases'] = df_am_tabela['deaths_by_totalCases'].apply(lambda x: x * 100)
        df_am_tabela['deaths_by_totalCases'] = df_am_tabela['deaths_by_totalCases'].apply(lambda x: round(x, 2))
        df_am_tabela['deaths_by_totalCases'] = df_am_tabela['deaths_by_totalCases'].apply(lambda x: "{:,.2f}".format(x))
        df_am_tabela['deaths_by_totalCases'] = df_am_tabela['deaths_by_totalCases'].apply(lambda x: x + " %")
        df_am_tabela['deaths_by_totalCases'] = df_am_tabela['deaths_by_totalCases'].apply(lambda x: x.replace('.', ','))
        df_am_tabela.sort_values('deaths_by_totalCases', inplace=True, ascending=False)


        df_am_tabela.drop('newDeaths', axis=1, inplace=True)
        df_am_tabela.rename(columns={
        'city': 'Cidade', 
        'deaths': 'Total de Óbitos', 
        'deaths_per_100k_inhabitants': 'Óbitos por 100k Habitantes', 
        'totalCases_per_100k_inhabitants': 'Total de Casos por 100k Habitantes',
        'deaths_by_totalCases': 'Percentual de Óbitos por Total de Casos'}, inplace=True)

        df_am_tabela.index = np.arange(1, len(df_am_tabela) + 1)

        return total_por_estado_tabela, df_am_tabela

def show_figure2():
    fig = px.choropleth_mapbox(data_frame=total_casos_e_mortes_por_estado, 
                               locations= 'id', 
                               geojson=gjson_estados_brasileiros, 
                               color = 'deaths', 
                               hover_name = 'name',
                               hover_data={'id': False, 'newCases': ":,2f", 'totalCases': ":,2f", 'newDeaths': ":2,f", 'deaths': ":,2f"}, 
                               center={'lat': -15, 'lon':-54}, 
                               zoom = 3.35, 
                               mapbox_style="carto-positron", 
                               color_continuous_scale=px.colors.sequential.Reds, 
                               opacity = 0.95,
                               labels={"totalCases": "N.º de Casos", "city": "Cidade", "deaths": "N.º de Óbitos", "newCases": "Novos Casos", "newDeaths": "Novos Óbitos"})

    fig.update_layout(width=800, height=800, separators=",.", template=GLOBAL_TEMPLATE, paper_bgcolor=rgb, plot_bgcolor=rgb)
    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=MAPBOX_TOKEN)
    fig.update_coloraxes(showscale=False)
    return fig

def show_figure3():
    fig = px.choropleth_mapbox(data_frame=df_am, 
                               locations= 'id', 
                               geojson=gjson_municipios_amazonas, 
                               color = 'deaths', 
                               hover_name = 'city',
                               hover_data={'id': False, 
                                           'newCases': ":,2f", 
                                           'totalCases': ":,2f", 
                                           'newDeaths': ":,2f", 
                                           'deaths': ":,2f"}, 
                               center={'lat': -5, 'lon':-65}, 
                               zoom = 4.60, 
                               mapbox_style="carto-positron",
                               range_color=(0, 500),
                               color_continuous_scale=px.colors.sequential.YlOrRd, 
                               opacity = 0.95,
                               labels={"totalCases": "N.º de Casos", 
                                       "city": "Cidade", 
                                       "deaths": "N.º de Óbitos", 
                                       "newCases": "Novos Casos", 
                                       "newDeaths": "Novos Óbitos"})

    fig.update_coloraxes(showscale=False)
    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=MAPBOX_TOKEN)
    fig.update_layout(width=800, height=800, separators=",.", template=GLOBAL_TEMPLATE, paper_bgcolor=rgb, plot_bgcolor=rgb)
    
    return fig


def show_figure4():
    fig = make_subplots(subplot_titles=('Casos & Óbitos',
                                       'Taxa de letalidade (CFR)'), 
                        rows=1, cols=2)
    
    fig.add_trace(go.Bar(
                 x=total_de_casos_amazonas_por_mes['date'], 
                 y=total_de_casos_amazonas_por_mes['newCases'], 
                marker=dict(color=total_de_casos_amazonas_por_mes['newDeaths'],
                           colorscale=px.colors.sequential.Reds),
                text=total_de_casos_amazonas_por_mes['newDeaths'],
                hovertemplate=
                             '<br>N.º de Casos: %{y:,2f}<br>' + 
                             'N.º de Óbitos: %{text:,2f}<br>',
                name=''
    ), row=1, col=1)


    fig.update_yaxes(title_text='N.º de Casos', row=1, col=1)

    
    ###Gráfico a direita
    
    fig.add_trace(go.Bar(
                name='',
                 x=total_de_casos_amazonas_por_mes['date'], 
                 y=total_de_casos_amazonas_por_mes['taxa_de_letalidade'], 
                marker=dict(color=total_de_casos_amazonas_por_mes['taxa_de_letalidade'],
                colorscale=px.colors.sequential.Brwnyl),
                hovertemplate=
                             '<br>N.º de Casos: %{text:,2f}<br>' + 
                             'N.º de Óbitos: %{customdata:,2f}<br>',
                customdata=total_de_casos_amazonas_por_mes['newDeaths'],
                text=total_de_casos_amazonas_por_mes['newCases']
    ), row=1, col=2)
    
    fig.update_traces(texttemplate="%{y} %", textposition= 'outside', row=1, col=2)
    fig.update_yaxes(title_text='Percentual (%)', row=1, col=2)
    
    ###Global
    
    fig.update_layout(separators=",.",
                      height= 800, 
                      width = 1600, 

                    hovermode='x',
                     showlegend=False,
                     template=GLOBAL_TEMPLATE,
                     paper_bgcolor=rgb,
                    plot_bgcolor=rgb)
    
    fig.update_xaxes(title_text='Data', 
                     tickformat= '%y/%m/%d',           
                     tickvals=dias(),     
                     ticktext=meses_anos('2020-03-31'))
    return fig


def show_figure5():
    fig = make_subplots(subplot_titles=('Brasil',
                                       'Amazonas'),
                        rows=1, 
                        cols=2)

    ##Gráfico a esquerda
    fig.add_trace(go.Bar(x=total_de_casos_brasil['date'], 
                     y=total_de_casos_brasil['totalCases'],
                                 text=total_de_casos_brasil['deaths'],
                                 name='',
                                 customdata=total_de_casos_brasil[['newCases', 'newDeaths']],
                                hovertemplate=
                                 '<br>N.º de Casos: %{y:,2f}<br>' + 
                                 'N.º de Óbitos: %{text:,2f}<br>' +
                                'Novos Casos: %{customdata[0]}<br>' + 
                                'Novos Óbitos: %{customdata[1]}'), row=1, col=1)

    ##Gráfico a direita
    fig.add_trace(go.Bar(x=total_de_casos_amazonas['date'], 
                     y=total_de_casos_amazonas['totalCases'],
                                 text=total_de_casos_amazonas['deaths'],
                                 name='',
                                 customdata=total_de_casos_amazonas[['newCases', 'newDeaths']],
                                hovertemplate=
                                 '<br>N.º de Casos: %{y:,2f}<br>' + 
                                 'N.º de Óbitos: %{text:,2f}<br>' +
                                'Novos Casos: %{customdata[0]}<br>' + 
                                'Novos Óbitos: %{customdata[1]}'), row=1, col=2)
    
    #Global
    fig.update_layout(
                        height= 800, 
                        width = 1600, 
                        separators=",.", 
                        hovermode='x',
                        template=GLOBAL_TEMPLATE,
                        paper_bgcolor=rgb,
                        plot_bgcolor=rgb)

    fig.update_traces(marker=dict(color='#597386'), showlegend=False)

    fig.update_xaxes(title_text='Data',
                     tickformat= '%y/%m/%d',   
                     tickvals=dias(),   
                     ticktext=meses_anos('2020-04-01'))
    fig.update_yaxes(title_text='N.º de Casos')
    return fig

def show_figure6():
    fig = px.bar(data_frame=total_de_casos_amazonas, 
                 x='date', 
                 y='newCases', 
                 hover_data={"newCases": ":,2f", 'newDeaths': ":,2f",'date': False}, 
                 labels={"newCases": "Novos Casos", "date": 'Data', 'newDeaths': 'Novos Óbitos'},
                 color = 'newCases', 
                 opacity= 0.75)

    fig.add_trace(go.Scatter(x=total_de_casos_amazonas['date'],
                             y=trend_newCases.round(), 
                             line=dict(color=red, width=1), 
                             name="Holt-Winters (SEHW) - Casos",
                             mode='lines', 
                             hoverinfo="y", 
                             showlegend=False, 
                             hovertemplate="%{y}"))


    fig.add_trace(go.Scatter(x=total_de_casos_amazonas['date'], 
                             y=trend_newDeaths, 
                             line=dict(color=red, width=1), 
                             name="Holt-Winters (SEHW) - Óbitos",
                             mode='lines',
                             hoverinfo='y' , 
                             showlegend=False, 
                             hovertemplate="%{y}"))

    fig.add_trace(go.Scatter(x=epocas_festivas['date'], 
                             y=epocas_festivas['newCases'], 
                             line=dict(color='steelblue', width=0.01),  
                             hovertemplate=epocas_festivas['name'],
                             mode='markers',
                             showlegend=False,
                            name='Feriado'))

    fig.update_layout(height= 800, 
                      width = 1600, 
                      separators=",.", 
                      paper_bgcolor=rgb,
                        plot_bgcolor=rgb,
                      xaxis=dict(tickformat= '%y/%m/%d', 
                                 tickvals=dias(), 
                                 ticktext=meses_anos('2020-04-01')))

    fig.update_layout(hovermode='x', template=GLOBAL_TEMPLATE)
    
    return fig



def show_figure7():
    fig = px.bar(data_frame = total_de_casos_amazonas, x='date', y='newDeaths', color='newDeaths', labels={'newDeaths': 'Óbitos', 'date': 'Data'})

    fig.update_traces(hovertemplate="%{y}", name='Óbitos')
    fig.add_trace(go.Scatter(x=total_de_casos_amazonas['date'], 
                             y=trend_newDeaths, 
                             line=dict(color=red, width=1), 
                             name="Holt-Winters (SEHW) - Óbitos",
                             mode='lines',
                             hoverinfo='y' , 
                             showlegend=False, 
                             hovertemplate="%{y}",
                             line_shape= 'spline'))

    fig.add_trace(go.Scatter(x=epocas_festivas['date'], 
                             y=epocas_festivas['newDeaths'], 
                             line=dict(color='steelblue', width=0.01),  
                             hovertemplate=epocas_festivas['name'],
                             mode='markers',
                             showlegend=False,
                            name='Feriado'))

    fig.update_layout(template=GLOBAL_TEMPLATE,
                        height= 800, 
                      width = 1600, 
                      hovermode='x',
                      separators=",.", 
                      xaxis=dict(tickformat= '%y/%m/%d', 
                                 tickvals=dias(), 
                                 ticktext=meses_anos('2020-04-01')),
                                paper_bgcolor=rgb,
                                plot_bgcolor=rgb)
    return fig



def show_figure8():    
    ###Criação de Gráfico
    fig = px.bar(crescimento.tail(14).round(2), 
                  y='valor',
                  x='date', 
                  color = 'valor',
                  labels = {'valor': 'Percentual (%)', 'date': 'Data', 'percentual': "Percentual (%)"},
                color_continuous_scale=['mediumaquamarine', 'maroon'],
                width=800, height=800,facet_col='variavel')


    fig.update_traces(hovertemplate="%{y} %")
    fig.update_layout(hovermode='x', 
                      separators=",.",
                      template=GLOBAL_TEMPLATE,
                      paper_bgcolor=rgb,
                     plot_bgcolor=rgb)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    fig.add_hline(y=0)
    
    
    tickvals, ticktext = traduzir_eixo_x(crescimento['date'].tail(14), 0, 4)
    
    ticktext = [x[:-4] for x in ticktext]
    
    fig.update_xaxes(tickformat= '%y/%m/%d', 
                     tickvals=tickvals, 
                     ticktext=ticktext)
    fig.update_yaxes(matches=None)
    return fig

def show_figure9():
    fig = px.bar(media_casos_por_dia_da_semana, 
                 y='Novos Casos', 
                 x=media_casos_por_dia_da_semana.index, 
                 color=media_casos_por_dia_da_semana.index, 
                 width= 400)

    fig.update_layout(template=GLOBAL_TEMPLATE,
                      width = 470, 
                      height=600,
                      separators=",.", 
                      xaxis={'tickangle': 35}, 
                      font=dict(size=11),
                     hovermode='x',
                     paper_bgcolor=rgb,
                     plot_bgcolor=rgb)

    fig.update_traces(hovertemplate="%{y}")
    
    return fig

def show_figure10():
    
    df_uti_geral = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/uti_geral.csv')
    df_uti_geral.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_uti_geral.insert(0, column='Unidade', value='UTI (Geral)')
    
    
    df_uti_covid = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/uti_covid-19.csv')
    df_uti_covid.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_uti_covid.insert(0, column='Unidade', value='UTI (Covid-19)')
    
    
    df_leitos_clinicos_geral = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/leitos_clinicos_geral.csv')
    df_leitos_clinicos_geral.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_leitos_clinicos_geral.insert(0, column='Unidade', value='Leitos Clínicos (Geral)')
    
    
    df_leitos_clinicos_covid = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/leitos_clinicos_covid-19.csv')
    df_leitos_clinicos_covid.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_leitos_clinicos_covid.insert(0, column='Unidade', value='Leitos Clínicos (Covid-19)')
    
    
    df_sala_vermelha_geral = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/sala_vermelha_geral.csv')
    df_sala_vermelha_geral.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_sala_vermelha_geral.insert(0, column='Unidade', value='Sala Vermelha (Geral)')
    
    
    df_sala_vermelha_covid = pd.read_csv(r'../raspagem_dos_boletins_diarios/normalized_csvs/sala_vermelha_covid-19.csv')
    df_sala_vermelha_covid.rename(columns = {'Rede Publica': 'Rede Pública',
    'Adulto (total)':'Adulto (Total)',
    'Oncologico': 'Oncológico',
    'Cardiaco': 'Cardíaco'}, inplace=True)
    df_sala_vermelha_covid.insert(0, column='Unidade', value='Sala Vermelha (Covid-19)')
    
    df = pd.concat([df_uti_geral, 
                    df_uti_covid, 
                    df_leitos_clinicos_geral, 
                    df_leitos_clinicos_covid, 
                    df_sala_vermelha_geral, 
                    df_sala_vermelha_covid])
    
    df.iloc[:,2:] = df.iloc[:,2:] * 100
    
    df['Data'] = ['20' + x for x in df['Data']]
    df['Data'] = pd.to_datetime(df['Data'])
    
    fig = px.line(data_frame = df, 
              x='Data', 
              y=df.drop(columns=['Unidade', 'Data']).columns,  
              facet_row='Unidade',
             height=1600, 
              width= 1600,
             labels={'value': 'Porcentagem (%)',
                    'Rede Publica': 'Rede Pública',
                    'Oncologico': 'Oncológico',
                    'Cardiaco': 'Cardíaco',
                    'variable': 'Setor'})


    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1], textangle=45))

    fig.update_layout(hovermode='x', template=GLOBAL_TEMPLATE, paper_bgcolor=rgb, plot_bgcolor=rgb)

    fig.update_traces(hovertemplate="%{y} %")

    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(matches=None)
    
    tickvals, ticktext = traduzir_eixo_x(df['Data'], 0, 14)
    
    ticktext = [x[:-4] for x in ticktext]
    
    fig.update_xaxes(tickformat= '%y/%m/%d', 
                     tickvals=tickvals, 
                     ticktext=ticktext,
                    tickangle=35)
    return fig

fh = np.arange(1, 14 + 1)
y = pd.Series(data=trend_newCases.values, index=total_de_casos_amazonas.date)
y.index.freq = 'D'
model = LGBMRegressor(random_state=4,
    learning_rate = 0.04591301953670739, 
    num_leaves = 45, 
    min_child_samples = 1, 
    subsample = 0.05,
    colsample_bytree = 0.9828905761860228,
    subsample_freq=1,
    n_estimators=685)
reg = make_reduction(estimator=model, window_length=14)
cv = ExpandingWindowSplitter(initial_window=60)
cross_val = evaluate(forecaster=reg, y=y, cv=cv, strategy="refit", return_data=True)
reg.fit(y)
y_pred = reg.predict(fh).round()

def show_figure11():
    fig = go.Figure()
    
    
    fig.add_trace(go.Bar(
                 x=total_de_casos_amazonas['date'].tail(30), 
                 y=total_de_casos_amazonas['newCases'].tail(30),
                hoverinfo='skip'))

    fig.update_traces(marker_color='gray')

    fig.add_trace(go.Scatter(x=total_de_casos_amazonas['date'].tail(30),
                             y=trend_newCases.round().tail(30), 
                             line=dict(color=red, width=1), 
                             name="Holt-Winters (SEHW) - Casos",
                             mode='lines', 
                             hoverinfo="y", 
                             showlegend=False, 
                             hovertemplate="%{y}",
                             fillcolor='Gray'))

    fig.add_trace(go.Scatter(x=y_pred.index, 
                             y=y_pred.values, 
                             line=dict(color=red, width=1), 
                             name=f"Predição por LightGBM",
                             mode='lines+markers',
                             hoverinfo='y' , 
                             showlegend=False, 
                             hovertemplate="%{y}",
                             opacity= 0.75))

    fig.update_layout(template=GLOBAL_TEMPLATE,
                    showlegend=False,
                    hovermode='x',
                      height= 400, 
                      width = 800, 
                      separators=",.", 
                        font=dict(size=11),
                     title='Predição de Tendência de Casos',
                     paper_bgcolor=rgb,
                     plot_bgcolor=rgb)
    
    tickvals, ticktext = traduzir_eixo_x(total_de_casos_amazonas['date'].tail(30), 6, 7)
    tickvals_pred, ticktext_pred = traduzir_eixo_x(y_pred.index, 4, 7)
    
    tickvals.extend(tickvals_pred)
    ticktext.extend(ticktext_pred)
    
    ticktext = [x[:-4] for x in ticktext]
    
    fig.update_xaxes(tickformat= '%y/%m/%d', 
                     tickvals=tickvals, 
                     ticktext=ticktext)
    
    smape = (cross_val['test_sMAPE'].mean() * 100).round(2)
    smape = str(smape)
    smape = smape.replace('.', ',')

    return fig, smape


ranking_nacional, ranking_municipal = tables()

app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY])

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "absolute",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem"
}

pred, smape = show_figure11()

sidebar = html.Div([
        html.H2("Relatório Amazonas", className="display-4"),
        html.Hr(),
        html.P(
            "Tópicos", className="lead"
        ),
        dbc.Nav(
                    [
        dbc.NavLink('Home', href='/', active='exact'),
        dbc.NavLink("N.º Diário de Casos & Óbitos no Amazonas", href='/pagina-1', active='exact'),
        dbc.NavLink('Casos & Óbitos por Estado & Munícipio', href='/pagina-2', active='exact'),
        dbc.NavLink('Relação Casos & Óbitos por Estado', href='/pagina-3', active='exact'),
        dbc.NavLink("Frequência Mensal no Amazonas", href='/pagina-4', active='exact'),
        dbc.NavLink('Crescimento dos Últimos 07 dias', href='/pagina-5', active='exact'),
        dbc.NavLink('Taxa de Ocupação de Hospital na Capital', href='/pagina-6', active='exact'),
        dbc.NavLink('Predição de Tendência', href='/pagina-7', active='exact'),
        dbc.NavLink("Quadro Evolutivo de Casos", href='/pagina-8', active='exact'),
        dbc.NavLink('Ranking Nacional de Óbitos por 100k', href='/pagina-9', active='exact'),
        dbc.NavLink('Ranking Municipal de Óbitos por 100k', href='/pagina-10', active='exact'),
        dbc.NavLink('Média de Registros por Dia da Semana', href='/pagina-11', active='exact'),
                    ], 
                    vertical=True, 
                    pills=True
                )
            ],
            style=SIDEBAR_STYLE,
)

content = html.Div(id='page-content', children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id='url'),
    sidebar,
    content
])

@app.callback(
    Output("page-content", 'children'),
    [Input("url", "pathname")]
)

def render_page_content(pathname):
    if pathname == "/":
        return [
           html.H1('Dados sobre Covid-19 com Foco no Estado do Amazonas', style={'textAlign': 'center'}),
           dcc.Markdown("""
           \n
           \n
           Projeto pessoal para fins educativos com finalidade de uma análise extensa, crítica e exploratória dos dados de Covid-19 dentro do estado do Amazonas. Para isso,
           utilizo um [banco de dados](https://github.com/wcota/covid19br) junto a informações da Secretaria Estado de Saúde do Amazonas.""", style={'textAlign': 'justify-all'}),
           dcc.Markdown("""
           Notícias da Fundação de Vigilância em Saúde do Amazonas (FVS-AM): https://share.streamlit.io/heylucasleao/noticias-fvs-am/main \n
           Covid19map: https://www.covid19map.com.br/case_map \n
           Repositório deste projeto: https://github.com/HeyLucasLeao/HeyLucasLeao.github.io \n
           Contato: https://t.me/heylucasleao \n
            LinkedIn: https://www.linkedin.com/in/lucas-le%C3%A3o-698a49206/""", style={'textAlign': 'justify-all'})
        ]
    elif pathname == "/pagina-1":
        return [
           dcc.Graph(figure=show_figure6()), dcc.Graph(figure=show_figure7())
        ]
    elif pathname == "/pagina-2":
        return [
            dcc.Graph(figure=show_figure2()), dcc.Graph(figure=show_figure3())
        ]
    elif pathname == "/pagina-3":
        return [
            dcc.Graph(figure=show_figure1()),
            dcc.Markdown('###### Coloração de gráfico referente a intensidade. Tamanho de bolha referente ao número de casos.')
        ]
    elif pathname == "/pagina-4":
        return [
            dcc.Graph(figure=show_figure4()),
            dcc.Markdown('###### Colorações relativas a: (1). N.º de Óbitos; (2). Intensidade')
        ]
    elif pathname == "/pagina-5":
        return [
            dcc.Graph(figure=show_figure8())
        ]
    elif pathname == "/pagina-6":
        return [
            dcc.Graph(figure=show_figure10())
        ]
    elif pathname == "/pagina-7":
        return [
            dcc.Graph(figure=pred),
            dcc.Markdown(f'###### Média de erro percentual absoluto médio simétrico: {smape} %')
        ]
    elif pathname == "/pagina-8":
        return [
            dcc.Graph(figure=show_figure5())
        ]
    elif pathname == "/pagina-9":
        return [
            dash_table.DataTable(id='dt1',
            columns=[{"name": i, "id": i} for i in ranking_nacional.columns],
            data=ranking_nacional.to_dict('records'),
            style_cell={'textAlign': 'left', 
                        'padding': '5px',
                        'backgroundColor': 'rgb(34, 34, 34)',
                        },
            style_header={
                'backgroundColor': 'rgb(34, 34, 34)',
                'fontWeight': 'bold',
                'color': "#fff"
            },
            style_as_list_view=True,
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(50, 50, 50)'
                }],
            )
        ]
    elif pathname == "/pagina-10":
        return [
            dash_table.DataTable(id='dt2',
            columns=[{"name": i, "id": i} for i in ranking_municipal.columns],
            data=ranking_municipal.to_dict('records'),
            style_cell={'textAlign': 'left', 
                        'padding': '5px',
                        'backgroundColor': 'rgb(34, 34, 34)',
                        },
            style_header={
                'backgroundColor': 'rgb(34, 34, 34)',
                'fontWeight': 'bold',
                'color': "#fff"
            },
            style_as_list_view=True,
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(50, 50, 50)'
                }]
                )
        ]
    elif pathname == "/pagina-11":
        return [
            dcc.Graph(figure=show_figure9())
        ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
if __name__ == '__main__':
    app.run_server(debug=True)