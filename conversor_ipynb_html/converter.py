from subprocess import Popen
from os import replace
from os import environ
from time import sleep

PREFIX_PATH = environ.get('REPO_PATH')
"""conversão do notebook para HTML, sem quaisquer código, 
facilitando visualização de resultado para leitores não-programadores."""

def converter():
    print('Convertendo ipynb...')
    Popen.wait(Popen('jupyter nbconvert --to html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=360 relatorio.ipynb --no-input',
                     cwd=PREFIX_PATH + r'\ipynb', shell=True), timeout=360)

    replace(PREFIX_PATH + r'\ipynb\relatorio.html',
            PREFIX_PATH + r'\index.html')


converter()
print('Jupyter Notebook convertido, index.html criado.')
sleep(10)
