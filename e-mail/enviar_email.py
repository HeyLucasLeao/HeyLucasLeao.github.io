import smtplib
from email.message import EmailMessage
from os import environ
from os import system
import pandas as pd
from time import sleep
import datetime as dt
from os import replace

from enderecos import emails
from estrutura import mensagem
from conversor_ipynb_para_html import converter
from subprocess import call

EMAIL_PASSWORD = environ.get("EMAIL_PASS")
EMAIL_ADDRESS = environ.get("EMAIL_USER")
x = 0

while True:
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities.csv'
        df = pd.read_csv(url)
        data = str(dt.datetime.now())[:10]
        df_estado = pd.read_csv(
            'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
        df_estado_soma_casos = df_estado.query(
            f"state == 'AM' and date == '{data}'")['newCases'].values[0]
        soma_casos = df.query(f"state == 'AM' and last_info_date == '{data}' and city != 'CASO SEM LOCALIZAÇÃO DEFINIDA/AM'")[
            'newCases'].sum()
    except:
        x += 1
        for i in reversed(range(30)):
            print(f"Dados incompletos.")
            print(f"Tentativa nº: {x}")
            print(
                f"Tempo restante para a próxima tentativa: {i + 1} minuto(s).")
            sleep(60)
            system('cls')
        continue
    if soma_casos > 0 and df_estado_soma_casos > 0:
        try:
            converter()
        except TimeoutError:
            raise SystemExit(0)

        replace(r'C:\Users\lucas\Documents\Programação\Projeto COVID-19\ipynb\relatorio.html',
                r'C:\Users\lucas\Documents\Programação\Projeto COVID-19\index.html')

        with open(r'C:\Users\lucas\Documents\GitHub\HeyLucasLeao.github.io\push_automatico\upar_dados.py', "r") as f:
            print('Atualizando dados...')
            exec(f.read())

        print('Push feito com sucesso.')
        sleep(30)

        with open(r'C:\Users\lucas\Documents\Programação\Projeto COVID-19\index.html', "rb") as f:
            file_data = f.read()
            file_name = f"Relatório COVID-19 BRA & AM: {data}.html"

        msg = EmailMessage()
        msg['Subject'] = f"Relatório de COVID-19 BRA & AM: {data}"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = 'ilolt.llol@gmail.com'
        msg.set_content(mensagem)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("E-mail enviado com sucesso. \nProcesso finalizado.")
        sleep(10)
        break
    else:
        x += 1
        for i in reversed(range(30)):
            print(f"Dados incompletos.")
            print(f"Tentativa nº: {x}")
            print(
                f"Tempo restante para a próxima tentativa: {i + 1} minuto(s).")
            sleep(60)
            system('cls')
        continue
