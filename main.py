import pandas as pd
import datetime as dt
from os import system
from time import sleep
from subprocess import Popen
from datetime import timedelta

x = 0


"""verifica caso haja dados atualizados dentro do dataset. 
Caso seja verdade, inicia-se abertura dos outros algoritmos."""

while True:
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities.csv'
        df = pd.read_csv(url)
        data = str(dt.datetime.now() - timedelta(1))[:10] 
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
            print('Atualizando csvs...')
            from subprocess import Popen
            Popen.wait(Popen('conda run -n covid-19-am python updating.py',
             shell=True, 
             cwd=r'./raspagem_dos_boletins_diarios'), 
                timeout=360)
        except TimeoutError:
            raise SystemExit(0)

        with open(r'./push_automatico/upar_dados.py', "r") as f:
            exec(f.read())

        with open(r'./e-mail/enviar_email.py', "r") as f:
            exec(f.read())
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


print("Processo finalizado.")
