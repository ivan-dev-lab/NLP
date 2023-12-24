import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
import time
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# data_path - путь к уже предобработанным данным
# models_path - пути для сохранения моделей
# функция возвращает результаты обучения моделей
def train (data_path: str, sklearn_models: list=[], models_path: list=[],  limit=None) -> dict:
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
    if len(sklearn_models) > 0 and len(sklearn_models) == len(models_path):
        for index, model in enumerate(sklearn_models):
            current_result = {'model': str, 'accuracy': float}

            # регулярное выражение для получения названия модели из пути для сохранения
            # 'models/DecisionTreeClassifier.pkl' -> DecisionTreeClassifier
            model_name = ''.join([match.group()[1:len(match.group())-1] for match in re.finditer(r"/[A-Z]\w*.", models_path[index])])

            current_result['model'] = model_name
            
            model = sklearn_models[index]()
            model.fit(vect_x_train, y_train)

            y_pred = model.predict(vect_x_test)
            score = accuracy_score(y_test, y_pred)
            current_result['accuracy'] = score

            sklearn_results.append(current_result)

            with open(models_path[index], mode='wb+') as model_file: 
                pickle.dump(model, model_file)

        models_results['sklearn_results'] = sklearn_results

    with open('classification/models/TF-IDF.pkl', mode='wb+') as vectorizer_file: 
        pickle.dump(vectorizer, vectorizer_file)
    
    return models_results


sklearn_models = [DecisionTreeClassifier, ExtraTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier]
models_path = ['classification/models/DecisionTreeClassifier.pkl', 'classification/models/ExtraTreeClassifier.pkl', 'classification/models/RandomForestClassifier.pkl', 'classification/models/ExtraTreesClassifier.pkl', 'classification/models/KNeighborsClassifier.pkl']

start_time = time.time()
result = train('data/preprocessed_docs.csv', sklearn_models, models_path)

with open('classification/models/results.txt', mode='a+', encoding='utf8') as result_file:
    result_file.write('результаты для обработанного текста с глаголами и существительными\n')
    for group, results in result.items():
        result_file.write(f'{group}:\n')
        for models in results:
            result_file.write(f'{results["model"]}: {results["accuracy"]}\n')

                
        end_time = time.time()-start_time
        result_file.write(f'время на обучение - {end_time}')
                
# TODO: посмотреть как изменятся показатели модели если лемматищировать все слова