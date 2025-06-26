import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from datetime import datetime


# Загрузка и подготовка данных
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def prepare_data(text, seq_length):
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


# Создание модели LSTM с настраиваемой активацией
def create_lstm_model(seq_length, num_vocab, embedding_dim, lstm_units, lstm_activation, output_activation):
    model = Sequential([
        Embedding(num_vocab, embedding_dim, input_length=seq_length),
        LSTM(lstm_units, return_sequences=False, activation=lstm_activation, dropout=0.2, recurrent_dropout=0.2),
        # LSTM(lstm_units,
        #      activation=lstm_activation,
             # input_shape=(seq_length, 1)),
        # LSTM(lstm_units, return_sequences=False, activation=lstm_activation),
        Dense(num_vocab, activation=output_activation)
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    return model


# Создание модели GRU с настраиваемой активацией
def create_gru_model(seq_length, num_vocab, embedding_dim, gru_units, gru_activation, output_activation):
    model = Sequential([
        Embedding(num_vocab, embedding_dim, input_length=seq_length),
        GRU(gru_units, return_sequences=False, activation=gru_activation, dropout=0.2, recurrent_dropout=0.2),
        # GRU(gru_units,
            # activation=gru_activation,
            # input_shape=(seq_length, 1)),
        # GRU(gru_units, return_sequences=False, activation=gru_activation),
        Dense(num_vocab, activation=output_activation)
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    return model


# Обучение модели
def train_model(model, X, y, epochs, batch_size, model_name):
    checkpoint_path = f"Checkpoints/{model_name}_checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cp_callback],
    )

    return history


# Генерация текста
def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, num_chars=200):
    generated = seed_text
    for _ in range(num_chars):
        x = np.array([[char_to_int[char] for char in seed_text[-seq_length:]]])
        pred = model.predict(x, verbose=0)
        next_char = int_to_char[np.argmax(pred)]
        generated += next_char
        seed_text += next_char
    return generated


# Эксперимент с разными гиперпараметрами
def run_experiment(author, text, seq_length, hyperparams, model_type):
    X, y, num_vocab, char_to_int, int_to_char = prepare_data(text, seq_length)

    results = []
    for hp in hyperparams:
        embedding_dim = hp['embedding_dim']
        units = hp['units']
        epochs = hp['epochs']
        batch_size = hp['batch_size']
        rnn_activation = hp['rnn_activation']
        output_activation = hp['output_activation']

        if model_type == 'lstm':
            model = create_lstm_model(
                seq_length, num_vocab,
                embedding_dim, units,
                rnn_activation, output_activation
            )
        elif model_type == 'gru':
            model = create_gru_model(
                seq_length, num_vocab,
                embedding_dim, units,
                rnn_activation, output_activation
            )

        model_name = (f"{author}_{model_type}_emb{embedding_dim}_units{units}_"
                      f"rnnact{rnn_activation}_outact{output_activation}_"
                      f"ep{epochs}_bs{batch_size}")
        print(f"\nTraining {model_name}...")

        history = train_model(model, X, y, epochs, batch_size, model_name)
        final_accuracy = history.history['accuracy'][-1]

        seed_text = text[:seq_length]
        generated_text = generate_text(model, seed_text, char_to_int, int_to_char, seq_length)

        result = {
            'author': author,
            'model_type': model_type,
            'embedding_dim': embedding_dim,
            'units': units,
            'rnn_activation': rnn_activation,
            'output_activation': output_activation,
            'epochs': epochs,
            'batch_size': batch_size,
            'final_accuracy': final_accuracy,
            'generated_text_sample': generated_text[:200] + "...",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(result)

        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Sample: {generated_text[:200]}...")

    return results


# Основная функция
def main():
    authors = {
        'chekhov': 'Чехов.txt',
        # 'mayakovsky': 'Маяковский.txt',
        # 'dostoevsky': 'Достоевский.txt'
        'kor': 'Корь.txt',
        # 'chesotka': 'Чесотка.txt',
        # 'shizofrenia': 'Шизофрения.txt'
    }

    seq_length = 30

    # Доступные функции активации
    rnn_activations = [
        'tanh',
        'relu',
                       # 'sigmoid'
                       ]
    output_activations = [
        'softmax',
        # 'linear',
    ]

    # Гиперпараметры для перебора
    base_hyperparams = [
        {'embedding_dim': 128, 'units': 128, 'epochs': 25, 'batch_size': 64},
        {'embedding_dim': 256, 'units': 256, 'epochs': 50, 'batch_size': 64}
    ]

    # Создаем все комбинации гиперпараметров с разными функциями активации
    hyperparams = []
    for base in base_hyperparams:
        for rnn_act in rnn_activations:
            for out_act in output_activations:
                hp = base.copy()
                hp['rnn_activation'] = rnn_act
                hp['output_activation'] = out_act
                hyperparams.append(hp)

    all_results = []

    for author, filename in authors.items():
        print(f"\n--- Processing {author} ---")
        text = load_text(filename)

        # Тестируем LSTM
        lstm_results = run_experiment(author, text, seq_length, hyperparams, 'lstm')
        all_results.extend(lstm_results)

        # Тестируем GRU
        gru_results = run_experiment(author, text, seq_length, hyperparams, 'gru')
        all_results.extend(gru_results)

    # Сохраняем результаты в CSV
    df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    results_file = f"results/text_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to '{results_file}'")

    # Выводим сводную таблицу
    print("\nSummary Table:")
    print(df[['author', 'model_type', 'embedding_dim', 'units',
              'rnn_activation', 'output_activation',
              'epochs', 'batch_size', 'final_accuracy']].to_string())


if __name__ == "__main__":
    main()