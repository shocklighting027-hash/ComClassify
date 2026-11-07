import numpy as np


class LogisticRegression:

    """Модель для логистической регрессии, поддерживающая многоклассовую классификацию с использованием softmax."""

    def __init__(self, weights=None, bias=None, n_classes=None):
        self.weights = weights
        self.bias = bias
        self.n_classes = n_classes

    def _softmax(self, logits) -> np.ndarray:
        # для численной стабильности
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _initialize_weights(self, n_features) -> np.ndarray:
        self.weights = np.zeros((n_features, self.n_classes), dtype=np.float32)
        self.bias = np.zeros(self.n_classes, dtype=np.float32)

    def _compute_loss(self, y: np.ndarray, y_preds: np.ndarray):
        n_samples = y.shape[0]
        y_preds = np.clip(y_preds, 1e-15, None)
        loss = -1 / n_samples * np.sum(np.sum(y * np.log(y_preds), axis=1))
        return loss

    def _requires_grad(self, X, y, y_preds):
        n_samples = y.shape[0]
        dw = X.T @ (y_preds - y) / n_samples
        db = 1 / n_samples * np.sum(y_preds - y, axis=0)
        return dw, db

    def train_on_batch(self, X_batch, y_batch, lr=0.05):
        if self.weights is None:
            self._initialize_weights(X_batch.shape[1])
        y_train_onehot = np.eye(self.n_classes, dtype=np.float32)[
            y_batch.astype(int)]
        y_preds = self._softmax(X_batch @ self.weights + self.bias)
        loss = self._compute_loss(y_train_onehot, y_preds)
        dw, db = self._requires_grad(X_batch, y_train_onehot, y_preds)
        self.weights = self.weights - lr * dw
        self.bias = self.bias - lr * db
        return loss  # Возвращаем значение потерь для текущего батча

    def predict_proba(self, X):
        proba = self._softmax(X @ self.weights + self.bias)
        return proba

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class BinaryLogisticRegression:

    def __init__(self, weights=None, bias=None):
        self.weights = weights
        self.bias = bias

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _initialize_weights(self, n_features) -> np.ndarray:
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

    def _compute_loss(self, y: np.ndarray, y_preds: np.ndarray):
        n_samples = y.shape[0]
        y_preds = np.clip(y_preds, 1e-15, 1 - 1e-15)
        loss = -(1 / n_samples * (y @ np.log(y_preds) +
                 (1 - y) @ np.log(1 - y_preds)))
        return loss

    def _requires_grad(self, weights, X, y, y_preds):
        n_samples = y.shape[0]
        dw = 1 / n_samples * X.T @ (y_preds - y)
        db = 1 / n_samples * np.sum(y_preds - y)
        return dw, db

    def train_on_batch(self, X_batch, y_batch, lr=0.05):
        if self.weights is None:
            self._initialize_weights(X_batch.shape[1])
        y_preds = self._sigmoid(X_batch @ self.weights + self.bias)
        loss = self._compute_loss(y_batch, y_preds)
        dw, db = self._requires_grad(self.weights, X_batch, y_batch, y_preds)
        self.weights = self.weights - lr * dw
        self.bias = self.bias - lr * db
        return loss  # Возвращаем значение потерь для текущего батча

    def predict_proba(self, X):
        proba = self._sigmoid(X @ self.weights + self.bias)
        return proba

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)


class TFIDFVectorizer:
    """Векторизация текста методом TF-IDF"""

    def __init__(self, stop_words=None, batch_size=32, idf_vector=None):
        self.stop_words = stop_words if stop_words else set()
        self.vocab = {}
        self.idf_table = {}
        self.batch_size = batch_size
        self.idf_vector = idf_vector

    def _create_vocab(self, texts) -> dict:
        unique_words = set()
        vocab = {}

        for text in texts:
            words = text.split()
            for word in words:
                if word in self.stop_words:
                    continue
                unique_words.add(word)

        for uniq in unique_words:
            vocab[uniq] = len(vocab)

        return vocab

    def fit(self, texts) -> np.ndarray:
        """Обучает векторизатор на текстах и возвращает TF-IDF матрицу."""

        self.vocab = self._create_vocab(texts)
        self.idf_table = {idx: 0 for idx in self.vocab.values()}
        preprocessed_texts = [set(text.split()) for text in texts]
        document_count_for_word = {idx: 0 for idx in self.vocab.values()}

        for text in preprocessed_texts:
            for word in text:
                if word in self.vocab:
                    document_count_for_word[self.vocab[word]] += 1

        for word, count in document_count_for_word.items():
            self.idf_table[word] = np.log(len(texts) / count)

        self.idf_vector = np.array(
            [self.idf_table[idx] for idx in range(len(self.vocab))], dtype=np.float32
        )

        return self

    def transform(self, texts):
        """Преобразует новые тексты в TF-IDF векторы на основе уже обученного словаря и IDF."""

        for n in range(0, len(texts), self.batch_size):
            batch_texts = texts[n: n + self.batch_size]
            batch_matrix = np.zeros(
                (len(batch_texts), len(self.vocab)), dtype="float32"
            )
            for text_idx, text in enumerate(batch_texts):
                words = [word for word in text.split(
                ) if word not in self.stop_words]
                if words:
                    for word in words:
                        if word in self.vocab:
                            batch_matrix[text_idx,
                                         self.vocab[word]] += 1 / len(words)

            batch_matrix *= self.idf_vector  # Применяем IDF к TF

            yield batch_matrix
