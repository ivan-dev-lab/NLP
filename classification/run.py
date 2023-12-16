import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# data_path - путь к уже предобработанным данным
# models_path - пути для сохранения моделей
# функция возвращает результаты обучения моделей
def train (data_path: str, limit: int, models_path: [], sklearn_models: list=[]) -> dict:
    preprocessed_data = pd.read_csv('data/preprocessed_docs.csv', index_col=[0]).iloc[0:limit]
    
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
            current_result['model'] = ...

            model = sklearn_models[index]
            model.fit(vect_x_train, y_train)

            y_pred = model.predict(vect_x_test)
            score = accuracy_score(y_test, y_pred)
            current_result['accuracy'] = score

            sklearn_results.append(current_result)



train ('data/preprocessed_docs.csv', [], 10)