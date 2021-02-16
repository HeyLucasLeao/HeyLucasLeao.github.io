data = str(dt.datetime.now())[:10]
df_estado = pd.read_csv(
    'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
df_estado_soma_casos = df_estado.query(
    f"state == 'AM' and date == '{data}'")['newCases'][1]