import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess(filename):
    """Загрузка и предобработка текста"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        return re.findall(r"\w+|[^\w\s]", text)
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None


def create_sequences(tokens, tokenizer, seq_length=15):
    """Создание обучающих последовательностей"""
    if not tokens or len(tokens) <= seq_length:
        print("Недостаточно данных для создания последовательностей")
        return None, None

    sequences = []
    for i in range(seq_length, len(tokens)):
        sequences.append(' '.join(tokens[i - seq_length:i + 1]))

    seq = tokenizer.texts_to_sequences(sequences)
    if not seq:
        print("Не удалось создать последовательности")
        return None, None

    seq = pad_sequences(seq, maxlen=seq_length + 1, padding='pre')
    X = seq[:, :-1]
    y = seq[:, -1]
    return X, y


def build_lstm_model(total_words, seq_length):
    """Создание LSTM модели"""
    model = Sequential([
        Embedding(total_words, 256, input_length=seq_length),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        Dropout(0.3),
        Dense(total_words, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_gru_model(total_words, seq_length):
    """Создание GRU модели"""
    model = Sequential([
        Embedding(total_words, 256, input_length=seq_length),
        GRU(512, return_sequences=True),
        Dropout(0.3),
        GRU(512),
        Dropout(0.3),
        Dense(total_words, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_text(model, tokenizer, seed_text, seq_length, num_words=50, temp=0.7):
    """Генерация текста"""
    generated = seed_text.split()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([' '.join(generated[-seq_length:])])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')

        probs = model.predict(token_list, verbose=0)[0]
        probs = np.log(probs) / temp
        exp_probs = np.exp(probs)
        preds = exp_probs / np.sum(exp_probs)

        next_idx = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_idx, '<?>')
        generated.append(next_word)

        text = ' '.join(generated)
    return re.sub(r'\s([,.!?])', r'\1', text).capitalize()


def main():
    authors = {
        'chekhov': 'Чехов.txt',
        'mayakovsky': 'Маяковский.txt',
        'dostoevsky': 'Достоевский.txt'
    }

    seq_length = 20
    results = []

    for author, filename in authors.items():
        print(f"\n=== Обработка {author.capitalize()} ===")

        tokens = load_and_preprocess(filename)
        if not tokens:
            continue

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokens)
        total_words = len(tokenizer.word_index) + 1
        print(f"Уникальных слов: {total_words}")

        X, y = create_sequences(tokens, tokenizer, seq_length)
        if X is None:
            continue

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # LSTM модель
        print("\n[LSTM Model]")
        lstm_model = build_lstm_model(total_words, seq_length)
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=256,
            epochs=1,
            callbacks=[EarlyStopping(patience=3)],
            verbose=1
        )

        # GRU модель
        print("\n[GRU Model]")
        gru_model = build_gru_model(total_words, seq_length)
        gru_history = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=256,
            epochs=1,
            callbacks=[EarlyStopping(patience=3)],
            verbose=1
        )

        # Генерация примеров
        seed = ' '.join(tokens[:seq_length])
        lstm_text = generate_text(lstm_model, tokenizer, seed, seq_length)
        gru_text = generate_text(gru_model, tokenizer, seed, seq_length)

        results.extend([
            {
                'author': author,
                'model': 'LSTM',
                'vocab_size': total_words,
                'final_accuracy': lstm_history.history['accuracy'][-1],
                'val_accuracy': lstm_history.history['val_accuracy'][-1],
                'sample': lstm_text[:200] + "..."
            },
            {
                'author': author,
                'model': 'GRU',
                'vocab_size': total_words,
                'final_accuracy': gru_history.history['accuracy'][-1],
                'val_accuracy': gru_history.history['val_accuracy'][-1],
                'sample': gru_text[:200] + "..."
            }
        ])

    # Сохранение результатов
    if results:
        df = pd.DataFrame(results)
        df.to_csv('text_generation_results.csv', index=False)
        print("\nРезультаты сохранены в text_generation_results.csv")
        print(df[['author', 'model', 'val_accuracy', 'sample']].to_string())
    else:
        print("Не удалось получить результаты для сохранения")


if __name__ == '__main__':
    main()