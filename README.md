# NLP - учебный проект по обработке естественного языка

## Структура
В проекте представлено:
- Классификация текста на категории ([`classification/`](https://github.com/ivan-dev-lab/NLP/tree/dev?tab=readme-ov-file#%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F-%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%B0-%D0%BD%D0%B0-%D0%BA%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D0%B8))


### Классификация текста на категории:
#### Данные
Текста для обучения были взяты с архива новостного сайта [https://www.kommersant.ru](https://www.kommersant.ru)
****
#### Предобработка
В процессе предобработки текста были использованы библиотеки для работы с естественным языком `nltk` и `pymorphy3`. В ходе предобработки из текста убираются все слова, кроме имен существительных, глаголов, и т.п (см. файл `classification/src/preprocess.py`)
****
#### Обучение моделей
На данный момент **(28.12.2023)** обучено:
- 5 моделей машинного обучения из библиотеки `scikit-learn` ( [результаты обучения](https://github.com/ivan-dev-lab/NLP/tree/dev?tab=readme-ov-file#scikit-learn) )
- 1 нейронная сеть `Keras` ( [результаты обучения](https://github.com/ivan-dev-lab/NLP/tree/dev?tab=readme-ov-file#keras) )
****
#### Результаты обучения
##### scikit-learn:
- `KNeighborsClassifier` = **0.6306954436450839**
- `DecisionTreeClassifier` = 0.5059952038369304
- `ExtraTreeClassifier` = 0.3501199040767386
- `ExtraTreesClassifier` = 0.29136690647482016
- `RandomForestClassifier` = 0.25059952038369304
****
##### Keras
- `Полносвязная нейронная сеть` = **0.7601918578147888**

( [архитектуры нейронных сетей Keras](https://github.com/ivan-dev-lab/NLP/tree/dev?tab=readme-ov-file#%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D1%8B-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85-%D1%81%D0%B5%D1%82%D0%B5%D0%B9-keras) )
****

#### Архитектуры нейронных сетей Keras
- `Полносвязная нейронная сеть`

| Layer (type)           | Output Shape  | Param #     |
|------------------------|---------------|-------------|
| dense                  | (None, 512)    | 24,144,384  |
| dropout                | (None, 512)    | 0           |
| dense_1                | (None, 256)    | 131,328     |
| dropout_1              | (None, 256)    | 0           |
| dense_2                | (None, 128)    | 32,896      |
| dense_3                | (None, 14)     | 1,806       |

- Total params: **24,310,414 (92.74 MB)**
- Trainable params: **24,310,414 (92.74 MB)**
- Non-trainable params: **0 (0.00 Byte)**
****

## Автор проекта
**Улановский Иван:**
- **[GitHub](https://github.com/ivan-dev-lab)**
- **[Telegram](https://t.me/ivan_ne_chik06)**