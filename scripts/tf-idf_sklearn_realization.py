import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

train_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\train_data.csv"
)
test_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\test_data.csv"
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
X_train = train_df["text"]
X_test = test_df["text"]
y_train = train_df["class"]
y_test = test_df["class"]

vectorizer = TfidfVectorizer()
model = LogisticRegression()

logging.info("Векторизация текста")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

logging.info("Обучение и оценка модели")
model.fit(X_train, y_train)
y_preds = model.predict(X_test)

logging.info("Метрики классификации")
print(classification_report(y_test, y_preds))

logging.info("Сохранение моделей")
output_vectorizer_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\vectorizer_start_sklearn_realization.pkl")
output_model_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\lr_start_sklearn_realization.pkl")

with open(output_vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

with open(output_model_path, "wb") as file:
    pickle.dump(model, file)

logging.info("Сохранение моделей завершено")

#               precision    recall  f1-score   support
#
#            0       0.82      0.93      0.87      3737
#            1       0.91      0.83      0.87      2337
#            2       0.76      0.88      0.81      1484
#            3       0.82      0.43      0.56      1159
#
#     accuracy                           0.83      8717
#    macro avg       0.83      0.77      0.78      8717
# weighted avg       0.83      0.83      0.82      8717
