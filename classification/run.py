import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
from src.keras_model import create_model

# data_path - путь к уже предобработанным данным
# models - словарь, содержащий функции вызова sklearn моделей и функцию создания модели keras 
# структура словаря models: список с моделями sklearn 'sklearn_models', список путей для сохранения - 'sklearn_models_path'
# структура словаря models: список с моделью/моделями keras 'keras_models', список путей для сохранения - 'keras_models_path'
# функция возвращает результаты обучения моделей
def train (data_path: str, models: dict,  limit=None) -> dict:
    models_results = {'sklearn_results': list, }
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
    print(vect_x_train.shape, y_train.shape, vect_x_test.shape, y_test.shape)

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

    if models.get('keras_models') != None:
        keras_models = models['keras_models'] 
        keras_models_path = models['keras_models_path'] 
        if len(keras_models) > 0 and len(keras_models) == len(keras_models_path):
            input_shape = (vect_x_test.shape[1], y_train.shape[1])
            for keras_model in keras_models:
                model = keras_model(input_shape)

                
                model.fit(x=vect_x_train, y=y_train, batch_size=32, epochs=10)
                # BUG: ValueError: Failed to find data adapter that can handle input: <class 'NoneType'>, <class 'NoneType'>

    with open('classification/models/TF-IDF.pkl', mode='wb+') as vectorizer_file: 
        pickle.dump(vectorizer, vectorizer_file)
    
    return models_results


keras_models = [create_model]
keras_models_path = ['classification/models/keras_models/model.h5']
models = {}
models['keras_models'] = keras_models
models['keras_models_path'] = keras_models_path

train('data/preprocessed_docs.csv', models, limit=10)