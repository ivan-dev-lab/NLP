import requests
from bs4 import BeautifulSoup as bfs
import os
import logging
import pandas as pd
import asyncio

logging.basicConfig(
    level=logging.INFO,
    filemode="w+",
    filename=f"classification/logs/run.log",
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

async def get_docs_url (name: str, url: str, days_ago: int=0) -> dict:
        Urls = {}   
        response = connect(url)

        if days_ago > 0:
            link_prev = f'{MAIN_PAGE}/{bfs(response.text, "html.parser").find("a", {"class": "ui-button ui-button--standart ui-nav ui-nav--prev"}).get("href")}'

            # список ссылок по категории name за определенное количество дней
            # на страницах в dayly_links находится список новостей за определенную дату
            dayly_links = []
            for day in range(1, days_ago+1):
                response_prev = connect(link_prev)
                link_prev = f'{MAIN_PAGE}/{bfs(response_prev.text, "html.parser").find("a", {"class": "ui-button ui-button--standart ui-nav ui-nav--prev"}).get("href")}'
                logging.info(f'берется список ссылок за дней назад = {day} || категория = {name} || link = {link_prev}')
                dayly_links.append(link_prev)

            # список с новостями за определенный день из dayly_links
            docs_links = []
            for link in dayly_links:
                response_link = connect(link)

                links_resset = bfs(response_link.text, "html.parser").find_all("a", {"class": "uho__link uho__link--overlay"})
                links = []

                for link in links_resset:
                    logging.info(f'берутся ссылки || категория = {name} || doc = {MAIN_PAGE}/{link.get("href")}')
                    docs_links.append(f"{MAIN_PAGE}/{link.get('href')}")
                    
            
            Urls[name] = docs_links  
                        
        else:
            links_resset = bfs(response.text, "html.parser").find_all("a", {"class": "uho__link uho__link--overlay"})
            links = []

            for link in links_resset:
                links.append(f"{MAIN_PAGE}/{link.get('href')}")
                logging.info(f'берутся ссылки || категория = {name} || doc = {MAIN_PAGE}/{link.get("href")}')

            Urls[name] = links

        return Urls

async def get_text_from_doc (name: str, url: str) -> dict:
    text_dict = {
        "Topic": name,
        "Text": ''
    }
    text_list = []

    response = connect(url)
    soup = bfs(response.text, "html.parser")

    title = soup.find("h1", {"class": "doc_header__name js-search-mark"}).text.strip() + "."
    text_list.append(title)

    texts_from_doc = soup.find_all("p", {"class": "doc__text"})

    for text in texts_from_doc:
        text_list.append(text.text)
    
    text_dict["Text"] = "".join(text_list)

    return text_dict

async def parser (save_path: str, days_ago: int=0):
    tasks_categories = []
    loop_categories = asyncio.get_event_loop()
    categories = get_categories()

    for name, url in categories.items():
        logging.info(f'берутся ссылки из категории = {name} || url = {url}')
        tasks_categories.append(loop_categories.create_task( get_docs_url(name, url, days_ago) ))

    links_dict = await asyncio.gather(*tasks_categories)

    tasks_docs = []
    loop_docs = asyncio.get_event_loop()
    for doc_dict in links_dict:
        for name, links in doc_dict.items():
            for link in links:
                logging.info(f'берется текст из категории = {name} || {link}')
                tasks_docs.append(loop_docs.create_task( get_text_from_doc(name, link) ))

    docs_dict = await asyncio.gather(*tasks_docs)
    
    result_data = []
    for block in docs_dict: result_data.append({'Topic': block['Topic'], 'Text': block['Text']})

    data = pd.DataFrame(data=result_data).sample(frac=1).reset_index(drop=True)
    data.to_csv(save_path)

    logging.info(f'процесс парсинга завершен, данные сохранены в {save_path}')