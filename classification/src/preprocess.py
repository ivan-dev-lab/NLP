import pandas as pd
import nltk
import asyncio
import string
import re
import pymorphy3


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

async def lemmatize (text: str) -> str:
    lemmatized_text = []
    tokenized_text = nltk.word_tokenize(text)
    morph = pymorphy3.MorphAnalyzer(lang='ru')
    
    for token in tokenized_text:
        parsed_word = morph.parse(token)
        tag = parsed_word[0].tag.POS

        if tag == "NOUN" or tag == 'VERB' or tag == 'INFN' or tag == 'ADJS' or tag == 'ADJF': 
            lemmatized_text.append(parsed_word[0].normal_form)

    return ' '.join(lemmatized_text)   