import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
import numpy as np
import tensorflow as tf
import logging
import time
import asyncio
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.keras_model import create_model
from src.preprocess import preprocess
from src.parser import parser




logger = logging.getLogger(__name__)

if len(logger.handlers) == 0:
    logging.basicConfig(
        level=logging.INFO,
        filemode="w+",
        filename=f"classification/logs/run.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        encoding="utf8"
    )

# обучение моделей вынесено в отдельные ассинхронные функции, чтобы в последствие они обучались одновременно
async def train_sklearn_models (models_info: dict, data: dict) -> list:
    sklearn_models = models_info['sklearn_models']
    sklearn_models_path = models_info['sklearn_models_path']

    vect_x_train = data['vect_x_train']
    y_train = data['y_train']
    vect_x_test = data['vect_x_test']
    y_test = data['y_test']
    
    sklearn_results = []

    if len(sklearn_models) > 0 and len(sklearn_models) == len(sklearn_models_path):
        logger.info(f'началось обучение моделей sklearn ({len(sklearn_models)} шт.)')
        
        for index, model in enumerate(sklearn_models):
            current_result = {'model': str, 'accuracy': float}

            # регулярное выражение для получения названия модели из пути для сохранения
            # 'models/DecisionTreeClassifier.pkl' -> DecisionTreeClassifier
            model_name = ''.join([match.group()[1:len(match.group())-1] for match in re.finditer(r"/[A-Z]\w*.", sklearn_models_path[index])])

            current_result['model'] = model_name

            logger.info(f'сейчас обучается модель sklearn {current_result["model"]}')

            model = sklearn_models[index]()
            model.fit(vect_x_train, y_train)

            y_pred = model.predict(vect_x_test)
            score = accuracy_score(y_test, y_pred)
            current_result['accuracy'] = score

            sklearn_results.append(current_result)

            logger.info(f'результат обучения модели sklearn {current_result["model"]} = {current_result["accuracy"]}')

            try:
                with open(sklearn_models_path[index], mode='wb+') as model_file: 
                    pickle.dump(model, model_file)
                    logger.info(f'модель sklearn {current_result["model"]} успешно сохранена {sklearn_models_path[index]}')
            
            except Exception as error:
                logger.error(f'произошла ошибка при сохранении модели sklearn {current_result["model"]} в {sklearn_models_path[index]}:\n{error}')
    else:
        logger.warning(f'длина списка с моделями sklearn не соотвествует длине списка с путями для их сохранения\n{len(sklearn_models)} != {len(sklearn_models_path)}')

    return sklearn_results

async def train_keras_models (models_info: dict, data: dict) -> list:
    keras_models = models_info['keras_models']
    keras_models_path = models_info['keras_models_path']

    vect_x_train = data['vect_x_train']
    y_train = data['y_train']
    vect_x_test = data['vect_x_test']
    y_test = data['y_test']

    keras_results = []

    if len(keras_models) > 0 and len(keras_models) == len(keras_models_path):
        logger.info(f'началось обучение моделей keras ({len(keras_models)} шт.)')

        # vect_x_train, vect_x_test - разреженные представления данных, поэтому сначала их надо представить в виде плотных данных, затем в виде SparseTensor, а затем применить к ним tf.sparse.reorder
        try:
            vect_x_train_dense = vect_x_train.toarray()
            logger.info('разреженные данные vect_x_train представлены в виде плотных данных vect_x_train_dense')
            vect_x_train_sparse = tf.sparse.from_dense(vect_x_train_dense)
            logger.info('плотные данные vect_x_train_dense представлены как SparseTensor')
            vect_x_train_reordered = tf.sparse.reorder(vect_x_train_sparse)
            logger.info('применен tf.sparse.reorder к данным в виде SparseTensor vect_x_train_sparse')
        except Exception as error:
            logger.error(f'Произошла ошибка при преобразовании данных vect_x_train: {error}')

        try:
            vect_x_test_dense = vect_x_test.toarray()
            logger.info('разреженные данные vect_x_test представлены в виде плотных данных vect_x_test_dense')
            vect_x_test_sparse = tf.sparse.from_dense(vect_x_test_dense)
            logger.info('плотные данные vect_x_test_dense представлены как SparseTensor')
            vect_x_test_reordered = tf.sparse.reorder(vect_x_test_sparse)
            logger.info('применен tf.sparse.reorder к данным в виде SparseTensor vect_x_test_sparse')
        except Exception as error:
            logger.error(f'Произошла ошибка при преобразовании данных vect_x_test: {error}')

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        input_shape = (vect_x_test.shape[1], y_train.shape[1])

        for index, keras_model in enumerate(keras_models):
            current_result = {'model': 'str', 'accuracy': float}

            model_name = ''.join([match.group()[1:len(match.group())-1] for match in re.finditer(r"/[A-Z]\w*.", keras_models_path[index])])
            
            current_result['model'] = model_name

            logger.info(f'сейчас обучается модель keras {current_result["model"]}')

            # keras_model - функция создания модели keras
            model = keras_model(input_shape)

            model.fit(x=vect_x_train_reordered, y=y_train, batch_size=32, epochs=4)
            accuracy = model.evaluate(x=vect_x_test_reordered, y=y_test, batch_size=32)[1]

            current_result['accuracy'] = accuracy
            
            logger.info(f'результат обучения модели keras {current_result["model"]} = {current_result["accuracy"]}')

            keras_results.append(current_result)

            try:
                with open(keras_models_path[index], mode='wb+') as model_file: 
                    pickle.dump(model, model_file)
                    logger.info(f'модель keras {current_result["model"]} успешно сохранена {keras_models_path[index]}')
            except Exception as error:
                logger.error(f'произошла ошибка при сохранении модели keras {current_result["model"]} в {keras_models_path[index]}:\n{error}')

    else: 
        logger.warning(f'длина списка с моделями keras не соотвествует длине списка с путями для их сохранения\n{len(keras_models)} != {len(keras_models_path)}')

    return keras_results


# data_path - путь к уже предобработанным данным
# models - словарь, содержащий функции вызова sklearn моделей и функцию создания модели keras 
# структура словаря models: список с моделями sklearn 'sklearn_models', список путей для сохранения - 'sklearn_models_path'
# структура словаря models: список с моделью/моделями keras 'keras_models', список путей для сохранения - 'keras_models_path'
# функция возвращает результаты обучения моделей
async def train (data_path: str, models: dict,  limit=None) -> dict:
    models_results = {'sklearn_results': [], 'keras_results': []}
    if limit: preprocessed_data = pd.read_csv(data_path, index_col=[0]).iloc[0:limit]
    else: preprocessed_data = pd.read_csv(data_path, index_col=[0])
    
    one_hot_encoded_labels = pd.get_dummies(preprocessed_data['Topic'], prefix='Topic')
    encoded_data = pd.concat([preprocessed_data, one_hot_encoded_labels], axis=1).replace({True: 1, False: 0})
    
    X = encoded_data['Text']
    y = encoded_data.iloc[:, 2:]
    
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    logger.info(f'данные разделены на тренировочные и контрольные наборы.\nСоотношение: обучающий набор = {round((x_train.shape[0]*100)/X.shape[0],2)}% || контрольный набор = {round((x_test.shape[0]*100)/X.shape[0],2)}%\nРазмерности данных:\nx_train = {x_train.shape}\nx_test = {x_test.shape}\ny_train = {y_train.shape}\ny_test = {y_test.shape}')

    try:
        vectorizer = TfidfVectorizer()
        vect_x_train = vectorizer.fit_transform(raw_documents=x_train)
        vect_x_test = vectorizer.transform(raw_documents=x_test)

        logger.info(f'данные векторизованы\nРазмерности:\nx_train = {vect_x_train.shape}\nx_test = {vect_x_test.shape}')
    except Exception as error:
        logger.error(f'Произошла ошибка при векторизации данных: {error}')


    sklearn_models_info = {'sklearn_models': models['sklearn_models'], 'sklearn_models_path': models['sklearn_models_path'] }
    keras_models_info = {'keras_models': models['keras_models'], 'keras_models_path': models['keras_models_path'] }
    
    data = {'vect_x_train': vect_x_train, 'y_train': y_train, 'vect_x_test': vect_x_test, 'y_test': y_test}

    models_results_tasks = asyncio.gather(
        train_sklearn_models(sklearn_models_info, data),
        train_keras_models(keras_models_info, data)
    )
    models_results['sklearn_results'], models_results['keras_results'] = await models_results_tasks

    try:
        with open('classification/models/TF-IDF.pkl', mode='wb+') as vectorizer_file: 
            pickle.dump(vectorizer, vectorizer_file)
            logger.info('векторизатор TF-IDF успешно сохранен в classification/models/TF-IDF.pkl')
    except Exception as error:
        logger.error(f'произошла ошибка при сохранении векторизатора TF-IDF в classification/models/TF-IDF.pkl:\n{error}')

    logger.info(f'результаты обучения:\n{models_results}')
    
    return models_results


async def main ():
    start_time = time.time()
    # await parser('data/test.csv', days_ago=0),
    # await preprocess('data/test.csv', 'data/preprocessed_test.csv')

    keras_models = [create_model]
    keras_models_path = ['classification/models/keras/FullyConnected_Keras.pkl']

    sklearn_models = [ DecisionTreeClassifier, ExtraTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier]
    sklearn_models_path = ['classification/models/sklearn/DecisionTreeClassifier.pkl', 'classification/models/sklearn/ExtraTreeClassifier.pkl', 'classification/models/sklearn/RandomForestClassifier.pkl', 'classification/models/sklearn/ExtraTreesClassifier.pkl', 'classification/models/sklearn/KNeighborsClassifier.pkl']

    models = {}
    models['keras_models'] = keras_models
    models['keras_models_path'] = keras_models_path

    models['sklearn_models'] = sklearn_models
    models['sklearn_models_path'] = sklearn_models_path


    results = await train('data/preprocessed_docs.csv', models, limit=None)

    work_time = time.time() - start_time
    logger.info(f'время, затраченное на работу программы = {work_time} секунд')

# TODO: написать коллбеки для моделей keras
# TODO: написать подбор параметров GridSearchCV
# TODO: написать другие архитектуры для моделей keras
# TODO: написать модель на pytorch

asyncio.run(main())