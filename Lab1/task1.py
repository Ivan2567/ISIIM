import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import random


# Загрузка и подготовка данных
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def prepare_data(text, seq_length=100):
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    num_chars = len(text)
    num_vocab = len(chars)

    print(f"Total Characters: {num_chars}")
    print(f"Total Vocab: {num_vocab}")

    dataX = []
    dataY = []
    for i in range(0, num_chars - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    X = np.reshape(dataX, (len(dataX), seq_length))
    y = np.array(dataY)

    return X, y, num_vocab, char_to_int, int_to_char


# Создание модели LSTM
def create_lstm_model(seq_length, num_vocab, embedding_dim, lstm_units):
    model = Sequential([
        Embedding(num_vocab, embedding_dim, input_length=seq_length),
        LSTM(lstm_units, return_sequences=False),
        Dense(num_vocab, activation='softmax')
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


# Создание модели GRU
def create_gru_model(seq_length, num_vocab, embedding_dim, gru_units):
    model = Sequential([
        Embedding(num_vocab, embedding_dim, input_length=seq_length),
        GRU(gru_units, return_sequences=False),
        Dense(num_vocab, activation='softmax')
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


# Обучение модели
def train_model(model, X, y, epochs=20, batch_size=64, model_name='model'):
    checkpoint_path = f"Checkpoints/{model_name}_checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback]
    )

    return history


# Генерация текста
def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, num_chars):
    generated = seed_text
    for _ in range(num_chars):
        x = np.array([[char_to_int[char] for char in seed_text[-seq_length:]]])
        pred = model.predict(x, verbose=0)
        next_char = int_to_char[np.argmax(pred)]
        generated += next_char
        seed_text += next_char
    return generated


# Основная функция
def main():
    authors = {
        'chekhov': 'Чехов.txt',
        'mayakovsky': 'Маяковский.txt',
        'dostoevsky': 'Достоевский.txt'
    }

    seq_length = 50
    embedding_dim = 128
    lstm_units = 128
    gru_units = 128
    epochs = 1
    batch_size = 256

    for author, filename in authors.items():
        print(f"\n--- Обработка автора: {author} ---")
        text = load_text(filename)
        X, y, num_vocab, char_to_int, int_to_char = prepare_data(text, seq_length)

        # LSTM модель
        print("\nОбучение LSTM модели...")
        lstm_model = create_lstm_model(seq_length, num_vocab, embedding_dim, lstm_units)
        lstm_history = train_model(lstm_model, X, y, epochs, batch_size, f"{author}_lstm")

        # GRU модель
        print("\nОбучение GRU модели...")
        gru_model = create_gru_model(seq_length, num_vocab, embedding_dim, gru_units)
        gru_history = train_model(gru_model, X, y, epochs, batch_size, f"{author}_gru")

        # Генерация текста
        seed_text = text[:seq_length]
        print("\nГенерация текста LSTM:")
        lstm_generated = generate_text(lstm_model, seed_text, char_to_int, int_to_char)
        print(lstm_generated[:200] + "...")

        print("\nГенерация текста GRU:")
        gru_generated = generate_text(gru_model, seed_text, char_to_int, int_to_char)
        print(gru_generated[:200] + "...")

        # Заполнение таблиц (вывод в консоль)
        print("\nТаблица 1 – LSTM сеть")
        print(
            "№ п/п\tLSTM слой\tСлой эмбендинга слов\tВыходной слой\tФункция активации\tФункция потерь\tФункция оптимизации\tМетрика")
        print(f"1\t{lstm_units}\t{embedding_dim}\t{num_vocab}\tsoftmax\tSparseCategoricalCrossentropy\tAdam\taccuracy")

        print("\nТаблица 2 – работа LSTM сети")
        print("№ п/п\tКоличество эпох\tРазмер бэтча\tМетрика")
        print(f"1\t{epochs}\t{batch_size}\t{lstm_history.history['accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()