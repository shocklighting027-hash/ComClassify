import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pathlib import Path
import logging
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

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__solver": ["lbfgs", "saga"],
}

logger.info("Загрузка данных")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
X_train = train_df["text"]
X_test = test_df["text"]
y_train = train_df["class"]
y_test = test_df["class"]
logger.info("Train: %d, test: %d", len(train_df), len(test_df))

pipe = Pipeline([("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000))])

logger.info("Подбор гиперпараметров")
grid_search = GridSearchCV(
    pipe, param_grid=param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
)
grid_search.fit(X_train, y_train)
logger.info(
    "Лучшие параметры: %s, f1_weighted: %.4f",
    grid_search.best_params_,
    grid_search.best_score_,
)

logger.info("Оценка модели")
y_preds = grid_search.predict(X_test)

logger.info("Метрики классификации")
print(classification_report(y_test, y_preds))

output_vectorizer_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\vectorizer_start_sklearn_adv.pkl")
output_model_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\lr_start_sklearn_adv.pkl")

with open(output_vectorizer_path, "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["tfidf"], file)

with open(output_model_path, "wb") as file:
    pickle.dump(grid_search.best_estimator_.named_steps["clf"], file)


#               precision    recall  f1-score   support
#
#            0       0.84      0.93      0.88      3737
#            1       0.91      0.84      0.87      2337
#            2       0.77      0.87      0.82      1484
#            3       0.77      0.49      0.60      1159
#
#     accuracy                           0.84      8717
#    macro avg       0.82      0.78      0.79      8717
# weighted avg       0.84      0.84      0.83      8717
