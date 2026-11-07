# ComClassify

ComClassify — сервис для классификации текстовых комментариев по эмоциональной окраске (негативная, позитивная, мусор, нейтральная).

## Структура проекта

- `scripts/read_file.py` — чтение исходного Excel-файла `data/raw/data_raw.xlsx` и сохранение в CSV.
- `scripts/preprocessing.py` — предобработка и разметка данных.
- `scripts/preprocessing_data.py` — дополнительная предобработка данных.
- `scripts/models.py` — собственные реализации: TF-IDF векторизатор, логистическая регрессия, стратифицированное разбиение выборки.
- `scripts/tf-idf_personal_realization.py` — обучение на собственных реализациях из `models.py`.
- `scripts/tf-idf_sklearn_realization.py` — обучение baseline на scikit-learn.
- `scripts/tf_idf_adv.py` — обучение на scikit-learn с подбором параметров.
- `scripts/backend.py` — FastAPI сервер для обработки запросов классификации.
- `scripts/database.py` — подключение к PostgreSQL через SQLAlchemy: таблица `predictions` и создание таблиц при старте.
- `scripts/frontend.py` — Streamlit веб-интерфейс для ввода текста и получения результата.
- `notebooks/BERT.ipynb` — дообучение BERT-моделей (Hugging Face).
- `notebooks/models/bert_checkpoints/checkpoint-2036/`, `checkpoint-2545/` — промежуточные чекпоинты обучения BERT.
- `models/bert_best/`, `models/bert_best_2/` — дообученные BERT-модели с токенизатором.
- `models/vectorizer_start_personal_realization.pkl`, `models/lr_start_personal_realization.pkl` — векторизатор и модель собственной реализации.
- `models/vectorizer_start_sklearn_realization.pkl`, `models/lr_start_sklearn_realization.pkl` — векторизатор и модель scikit-learn.
- `models/vectorizer_start_sklearn_adv.pkl`, `models/lr_start_sklearn_adv.pkl` — векторизатор и модель scikit-learn с подбором параметров.
- `data/raw/data_raw.xlsx` — исходные данные.
- `data/interim/data_read.csv` — промежуточный CSV после чтения Excel.
- `data/output/train_data.csv`, `data/output/test_data.csv` — обучающая и тестовая выборки.
- `analyze.svg` — иллюстрация анализа данных.
