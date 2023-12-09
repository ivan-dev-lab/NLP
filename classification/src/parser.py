import requests
from bs4 import BeautifulSoup as bfs
import os
import shutil
import logging
import pandas as pd
import time
import asyncio

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    filename=f"classification/logs/parser.log",
    format="%(asctime)s:%(levelname)s:%(message)s",
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


# TODO: сделать более аккуратный код
# TODO: покрыть логами
async def get_docs_url (name: str, url: str, days_ago: int=0) -> dict:
        Urls = {}   
        response = connect(url)

        if days_ago > 0:
            link_prev = f'{MAIN_PAGE}/{bfs(response.text, "html.parser").find("a", {"class": "ui-button ui-button--standart ui-nav ui-nav--prev"}).get("href")}'

            links = []
            for day in range(1, days_ago+1):
                response_prev = connect(link_prev)
                link_prev = f'{MAIN_PAGE}/{bfs(response_prev.text, "html.parser").find("a", {"class": "ui-button ui-button--standart ui-nav ui-nav--prev"}).get("href")}'
                links.append(link_prev)
            
            links_in = []
            for link_0 in links:
                response_link = connect(link_0)

                links_resset = bfs(response_link.text, "html.parser").find_all("a", {"class": "uho__link uho__link--overlay"})
                links = []

                for link_1 in links_resset:
                    links_in.append(f"{MAIN_PAGE}/{link_1.get('href')}")

            Urls[name] = links_in               
        else:
            links_resset = bfs(response.text, "html.parser").find_all("a", {"class": "uho__link uho__link--overlay"})
            links = []

            for link in links_resset:
                links.append(f"{MAIN_PAGE}/{link.get('href')}")

            Urls[name] = links
    
        return Urls


async def parser ():
    tasks = []
    loop = asyncio.get_event_loop()
    categories = get_categories()

    for name, url in categories.items():
        logging.info(f'берутся ссылки из категории = {name} || url = {url}')
        tasks.append(loop.create_task( get_docs_url(name, url, days_ago=2) ))

    docs = await asyncio.gather(*tasks)
    
    

asyncio.run(parser ())