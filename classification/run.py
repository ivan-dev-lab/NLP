import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
import numpy as np
import tensorflow as tf
from src.keras_model import create_model

# data_path - путь к уже предобработанным данным
# models - словарь, содержащий функции вызова sklearn моделей и функцию создания модели keras 
# структура словаря models: список с моделями sklearn 'sklearn_models', список путей для сохранения - 'sklearn_models_path'
# структура словаря models: список с моделью/моделями keras 'keras_models', список путей для сохранения - 'keras_models_path'
# функция возвращает результаты обучения моделей
def train (data_path: str, models: dict,  limit=None) -> dict:
    models_results = {'sklearn_results': [], 'keras_results': []}
    if limit: preprocessed_data = pd.read_csv(data_path, index_col=[0]).iloc[0:limit]
    else: preprocessed_data = pd.read_csv(data_path, index_col=[0])
    
    one_hot_encoded_labels = pd.get_dummies(preprocessed_data['Topic'], prefix='Topic')
    encoded_data = pd.concat([preprocessed_data, one_hot_encoded_labels], axis=1).replace({True: 1, False: 0})
    
    X = encoded_data['Text']
    y = encoded_data.iloc[:, 2:]
    
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    vect_x_train = vectorizer.fit_transform(raw_documents=x_train)
    vect_x_test = vectorizer.transform(raw_documents=x_test)

    sklearn_results = []
    if models.get('sklearn_models') != None:
        sklearn_models = models['sklearn_models']
        sklearn_models_path = models['sklearn_models_path']
        
        if len(sklearn_models) > 0 and len(sklearn_models) == len(sklearn_models_path):
            for index, model in enumerate(sklearn_models):
                current_result = {'model': str, 'accuracy': float}

                # регулярное выражение для получения названия модели из пути для сохранения
                # 'models/DecisionTreeClassifier.pkl' -> DecisionTreeClassifier
                model_name = ''.join([match.group()[1:len(match.group())-1] for match in re.finditer(r"/[A-Z]\w*.", sklearn_models_path[index])])

                current_result['model'] = model_name

                model = sklearn_models[index]()
                model.fit(vect_x_train, y_train)

                y_pred = model.predict(vect_x_test)
                score = accuracy_score(y_test, y_pred)
                current_result['accuracy'] = score

                sklearn_results.append(current_result)

                with open(sklearn_models_path[index], mode='wb+') as model_file: 
                    pickle.dump(model, model_file)

            models_results['sklearn_results'] = sklearn_results

    keras_results = []
    if models.get('keras_models') != None:
        keras_models = models['keras_models'] 
        keras_models_path = models['keras_models_path'] 
        if len(keras_models) > 0 and len(keras_models) == len(keras_models_path):

            # vect_x_train, vect_x_test - разреженные представления данных, поэтому сначала их надо представить в виде плотных данных, затем в виде SparseTensor, а затем применить к ним tf.sparse.reorder
            vect_x_train_dense = vect_x_train.toarray()
            vect_x_train_sparse = tf.sparse.from_dense(vect_x_train_dense)
            vect_x_train_reordered = tf.sparse.reorder(vect_x_train_sparse)

            vect_x_test_dense = vect_x_test.toarray()
            vect_x_test_sparse = tf.sparse.from_dense(vect_x_test_dense)
            vect_x_test_reordered = tf.sparse.reorder(vect_x_test_sparse)

            y_train = np.array(y_train)
            y_test = np.array(y_test)

            input_shape = (vect_x_test.shape[1], y_train.shape[1])

            for index, keras_model in enumerate(keras_models):
                current_result = {'model': 'str', 'accuracy': float}

                model_name = ''.join([match.group()[1:len(match.group())-1] for match in re.finditer(r"/[A-Z]\w*.", keras_models_path[index])])
                
                current_result['model'] = model_name

                # keras_model - функция создания модели keras
                model = keras_model(input_shape)

                model.fit(x=vect_x_train_reordered, y=y_train, batch_size=32, epochs=4)
                accuracy = model.evaluate(x=vect_x_test_reordered, y=y_test, batch_size=32)[1]

                current_result['accuracy'] = accuracy
                
                keras_results.append(current_result)

                with open(keras_models_path[index], mode='wb+') as model_file: pickle.dump(model, model_file)

            models_results['keras_results'] = keras_results

    with open('classification/models/TF-IDF.pkl', mode='wb+') as vectorizer_file: 
        pickle.dump(vectorizer, vectorizer_file)
    
    return models_results


keras_models = [create_model]
keras_models_path = ['classification/models/keras_models/FullyConnected_Keras.pkl']
models = {}
models['keras_models'] = keras_models
models['keras_models_path'] = keras_models_path

results = train('data/preprocessed_docs.csv', models, limit=None)
print(results)

# TODO: сделать обучение моделей ассинхронным
# TODO: написать коллбеки для моделей keras
# TODO: написать подбор параметров GridSearchCV
# TODO: написать другие архитектуры для моделей keras
# TODO: написать модель на pytorch
