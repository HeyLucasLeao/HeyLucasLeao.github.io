from subprocess import Popen


def converter():
    Popen.wait(Popen('jupyter nbconvert --to html --ExecutePreprocessor.enabled=True relatorio.ipynb --no-input',
                     cwd=r'C:\Users\lucas\Documents\Programação\Projeto COVID-19\ipynb'), timeout=360)
