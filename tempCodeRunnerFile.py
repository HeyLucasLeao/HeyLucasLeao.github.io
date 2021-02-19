with open(PREFIX_PATH + r'\push_automatico\upar_dados.py', "r") as f:
            print('Atualizando dados...')
            exec(f.read())