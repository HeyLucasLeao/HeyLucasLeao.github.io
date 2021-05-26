from git import Repo
from os import environ
from time import sleep
from subprocess import Popen

"""inicia um push direto ao repo."""

user = environ.get('GITHUB_USER')
password = environ.get('GITHUB_PASSWORD')
PATH = environ.get('REPO_PATH')
remote = f"https://{user}:{password}@github.com:HeyLucasLeao/HeyLucasLeao.github.io.git"


def git_push():
    try:
        print('Atualizando dados...')
        repo = Repo(
            path=PATH)
        repo.git.add(PATH, update=True)
        repo.index.commit(f"BOT")
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Erro durante tentativa de push.')

def heroku():
    print('Atualizando heroku...')
    Popen.wait(Popen('git push heroku master',
                     cwd=PATH, shell=True), 
                     timeout=360)

git_push()
print('Push feito com sucesso.')
heroku()
print('Atualização feita com sucesso.')
sleep(15)
