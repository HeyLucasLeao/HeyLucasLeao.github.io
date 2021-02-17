from subprocess import Popen


def converter():
    Popen.wait(Popen('jupyter nbconvert --to html --ExecutePreprocessor.enabled=True relatorio.ipynb --no-input',
                     cwd=r'C:\Users\lucas\Documents\GitHub\HeyLucasLeao.github.io\ipynb'), timeout=360)
