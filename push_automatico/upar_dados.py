from git import Repo
from os import environ
import datetime as dt

user = environ.get('GITHUB_USER')
password = environ.get('GITHUB_PASSWORD')
PATH = environ.get('REPO_PATH')
remote = f"https://{user}:{password}@github.com:HeyLucasLeao/HeyLucasLeao.github.io.git"


def git_push():
    try:
        repo = Repo(
            path=PATH)
        repo.git.add(PATH, update=True)
        repo.index.commit(f"Relatório {dt.datetime.now()}")
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Erro durante tentativa de push.')


git_push()
