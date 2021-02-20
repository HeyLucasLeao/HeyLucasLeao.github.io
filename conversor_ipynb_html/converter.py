from subprocess import Popen
from os import replace
from os import environ
from time import sleep

PREFIX_PATH = environ.get('REPO_PATH')


def converter():
    Popen.wait(Popen('jupyter nbconvert --to html --ExecutePreprocessor.enabled=True relatorio.ipynb --no-input',
                     cwd=PREFIX_PATH + r'\ipynb'), timeout=360)

    replace(PREFIX_PATH + r'\ipynb\relatorio.html',
            PREFIX_PATH + r'\index.html')


converter()
