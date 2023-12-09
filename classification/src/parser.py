import requests
from bs4 import BeautifulSoup as bfs
import os
import shutil
import logging
import pandas as pd
import time
import aiohttp

# строка в формате д.мм.гггг_чч:мм
time_now = f'{time.localtime()[2]}-{time.localtime()[1]}-{time.localtime()[0]}_{time.localtime()[3]}-{time.localtime()[4]}-{time.localtime()[5]}'

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    filename=f"classification/logs/parser/{time_now}.log",
    format="%(levelname)s: %(message)s",
    encoding="utf8"
)

MAIN_PAGE = "https://www.kommersant.ru"
ARCHIVE_PAGE = "https://www.kommersant.ru/archive"

def connect (url: str) -> requests.Response:
    attempts = 0
    response = None
    code = 0
    while attempts <= 5 and code != 200:
        try: 
            response = requests.get(url)
            code = response.status_code
            logging.info(f'Успешное подключение к {url} || код = {code}')
        except:
            logging.warning(f'Подключение к {url} вернуло код {code}. Попытка подключения {attempts}/5')
        attempts+=1
    
    return response if code == 200 else None

def get_categories () -> dict:
    print(connect(ARCHIVE_PAGE))
    response = connect(ARCHIVE_PAGE)
    
    if response != None:
        categories = {}
        
        categories_resset = bfs(response.text, "html.parser").find_all("li", {"class": "archive_section__item"})
        i = 1
        for tag in categories_resset:
            if i <= 14:
                categories[tag.get_text().strip()] = f"https://www.kommersant.ru{tag.select('.no_decor')[0].get('href')}"
                i+=1
            else:
                break
        
        return categories
    else: return None

