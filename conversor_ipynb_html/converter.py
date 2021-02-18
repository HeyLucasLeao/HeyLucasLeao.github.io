from subprocess import Popen
from os import replace


def converter():
    Popen.wait(Popen('jupyter nbconvert --to html --ExecutePreprocessor.enabled=True relatorio.ipynb --no-input',
                     cwd=r'HeyLucasLeao.github.io\ipynb'), timeout=360)

    replace(r'HeyLucasLeao.github.io\ipynb\relatorio.html',
            r'HeyLucasLeao.github.io\index.html')


converter()
