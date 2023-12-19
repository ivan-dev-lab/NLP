import pandas as pd
import nltk
import asyncio
import string
import re
import pymorphy3
import logging

logging.basicConfig (
    filename='classification/logs/preprocess.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    encoding='utf8'
)

async def clean_text (text: str) -> str:
    cleaned_text = []
    stopwords = nltk.corpus.stopwords.words('russian')
    punctuation = string.punctuation + '«»'
    tokenized_text = nltk.word_tokenize(text)

    for token in tokenized_text:
        if not(token in stopwords or token in punctuation):
            cleaned_token = re.sub(r'[\W\d]', '', token).lower()
            if cleaned_token: cleaned_text.append(cleaned_token)
            # любая не пустая строка - True

    return ' '.join(cleaned_text)

async def lemmatize (text) -> str:
    lemmatized_text = []
    # из-за ассинхронности text становится корутиной, поэтому необходимо использовать text.result() для получения строки
    tokenized_text = nltk.word_tokenize(text.result())
    morph = pymorphy3.MorphAnalyzer(lang='ru')
    
    for token in tokenized_text:
        parsed_word = morph.parse(token)
        tag = parsed_word[0].tag.POS # POS - часть речи

        if tag == "NOUN" or tag == 'VERB' or tag == 'INFN' or tag == 'ADJS' or tag == 'ADJF': 
            lemmatized_text.append(parsed_word[0].normal_form)

    return ' '.join(lemmatized_text)   

# docs_path - путь до файла csv с текстами
async def preprocess (docs_path: str, need_save: bool=False, save_path: str='data/preprocessed_docs.csv') -> pd.DataFrame:
    docs_df = pd.read_csv(docs_path, index_col=[0])
    tasks = []
    loop = asyncio.get_event_loop()

    for index,text in enumerate(docs_df['Text']):
        cleaned_text_task = loop.create_task(clean_text(text))
        logging.info(f'очищен текст {index}/{len(docs_df["Text"])} || {(index*100)/len(docs_df["Text"])}')
        lemmatize_task = loop.create_task(lemmatize(cleaned_text_task))
        logging.info(f'лемматизирован текст {index}/{len(docs_df["Text"])} || {(index*100)/len(docs_df["Text"])}')
        tasks.append(lemmatize_task)

    preprocessed_texts = await asyncio.gather(*tasks)
    
    docs_df['Text'] = preprocessed_texts

    if need_save:
        if save_path != docs_path: docs_df.to_csv(save_path)
        else: raise ValueError(f'save_path не должен совпадать с docs_path ({save_path} = {docs_path})')

    logging.info(f'процесс предобработки завершен, данные сохранены в {save_path}')
    return docs_df