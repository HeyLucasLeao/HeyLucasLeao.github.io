from git import Repo
from os import environ
import datetime as dt

user = environ.get('GITHUB_USER')
password = environ.get('GITHUB_PASSWORD')
remote = f"https://{user}:{password}@github.com:HeyLucasLeao/HeyLucasLeao.github.io.git"

repo = Repo(path=r'C:\Users\lucas\Documents\GitHub\HeyLucasLeao.github.io')
repo.git.add(r'C:\Users\lucas\Documents\GitHub\HeyLucasLeao.github.io')
repo.index.commit("Atualização de dados")
origin = repo.remote(name='origin')
origin.push()
