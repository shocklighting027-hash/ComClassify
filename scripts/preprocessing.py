import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing_data import train_test_split

logging.basicConfig(level=logging.INFO)

input_file_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\interim\data_read.csv"
)

df = pd.read_csv(input_file_path)
text_label = df[["Комментарии", "Эмоциональная окраска"]].copy()
text_label.columns = ["text", "class"]
logging.info(
    "Удаление пробелов, приведение к нижниму регистру, удаление несуществующих значений"
)
text_label["class"] = text_label["class"].str.strip().str.lower()
text_label = text_label.dropna(subset=["text", "class"])

points = {"негативная": 0, "позитивная": 1, "мусор": 2, "нейтральная": 3}

class_counts = text_label["class"].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(
    x=text_label["class"].unique(),
    y=text_label["class"].value_counts(),
    hue=text_label["class"].unique(),
)
plt.title("Кол-во комментариев разных классов")
plt.xlabel("Окрас")
plt.ylabel("Кол-во")
plt.savefig("analyze.svg")
text_label["class"] = text_label["class"].apply(lambda x: points[x])

logging.info("Разделение данных на тренировочные и тестовые")
X = text_label["text"]
y = text_label["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=True)
train_data = pd.DataFrame({"text": X_train, "class": y_train})
test_data = pd.DataFrame({"text": X_test, "class": y_test})
output_train_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\train_data.csv"
)
output_test_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\output\test_data.csv"
)
train_data.to_csv(output_train_path, index=False)
test_data.to_csv(output_test_path, index=False)
