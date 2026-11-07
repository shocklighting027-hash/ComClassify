import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path
from models import TFIDFVectorizer, LogisticRegression
import pickle

english_stop_words = {
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all',
    'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
    'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became',
    'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind',
    'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom',
    'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry',
    'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg',
    'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
    'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
    'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty',
    'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has',
    'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
    'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred',
    'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its',
    'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many',
    'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
    'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never',
    'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
    'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem',
    'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side',
    'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something',
    'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than',
    'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
    'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin',
    'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus',
    'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un',
    'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were',
    'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither',
    'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
    'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

train_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\train_data.csv")
test_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\test_data.csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
X_train, X_test = df_train["text"], df_test["text"]
y_train, y_test = df_train["class"].to_numpy(), df_test["class"].to_numpy()

BATCH_SIZE = 32
EPOCHS = 100
LR = 0.1

vectorizer = TFIDFVectorizer(
    stop_words=english_stop_words, batch_size=BATCH_SIZE).fit(X_train)
logger.info("Словарь: %d слов", len(vectorizer.vocab))

model = LogisticRegression(n_classes=len(np.unique(y_train)))
# Целочисленное деление, округление вверх
n_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE

with logging_redirect_tqdm():
    for epoch in range(1, EPOCHS + 1):
        loss_sum = 0.0
        batches = tqdm(vectorizer.transform(X_train), total=n_batches,
                       desc=f"Эпоха {epoch}/{EPOCHS}", leave=False)
        for i, batch_matrix in enumerate(batches):
            y_batch = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            loss_sum += model.train_on_batch(batch_matrix, y_batch, lr=LR)
            batches.set_postfix(loss=f"{loss_sum / (i + 1):.4f}")
        logger.info("Эпоха %d/%d: loss %.4f", epoch,
                    EPOCHS, loss_sum / n_batches)

    predictions = np.concatenate(
        [model.predict(batch) for batch in tqdm(
            vectorizer.transform(X_test), desc="Тест", leave=False)]
    )
    logger.info("test macro-f1: %.4f",
                f1_score(y_test, predictions, average="macro"))
    logger.info("\n%s", classification_report(
        y_test, predictions, zero_division=0))

output_vectorizer_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\vectorizer_start_personal_realization.pkl")
output_model_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\models\lr_start_personal_realization.pkl")

with open(output_vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

with open(output_model_path, "wb") as file:
    pickle.dump(model, file)


#               precision    recall  f1-score   support
#
#            0       0.71      0.96      0.82      3737
#            1       0.92      0.80      0.85      2337
#            2       0.87      0.60      0.71      1484
#            3       0.78      0.41      0.54      1159
#
#     accuracy                           0.78      8717
#    macro avg       0.82      0.69      0.73      8717
# weighted avg       0.80      0.78      0.77      8717
