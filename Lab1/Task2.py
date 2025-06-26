import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Загрузка данных (замените на ваш путь к файлу)
data = pd.read_csv('diabetes_012.csv')  # Убедитесь, что файл имеет правильные поля

# Разделение на признаки и целевую переменную
X = data.drop('Diabetes_012', axis=1).values
y = data['Diabetes_012'].values

# Преобразование целевой переменной в one-hot encoding
y = to_categorical(y, num_classes=3)

# Разделение на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование данных для LSTM/GRU (добавление временного измерения)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Создание модели LSTM
def create_lstm_model():
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Создание модели GRU
def create_gru_model():
    model = Sequential([
        GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Обучение моделей
lstm_model = create_lstm_model()
gru_model = create_gru_model()

history_lstm = lstm_model.fit(X_train, y_train,
                             epochs=10,
                             batch_size=32,
                             validation_data=(X_test, y_test),
                             verbose=1)

history_gru = gru_model.fit(X_train, y_train,
                           epochs=10,
                           batch_size=32,
                           validation_data=(X_test, y_test),
                           verbose=1)

# Визуализация результатов
plt.figure(figsize=(14, 6))

# График точности LSTM
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='LSTM Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Test Accuracy')
plt.title('LSTM Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# График точности GRU
plt.subplot(1, 2, 2)
plt.plot(history_gru.history['accuracy'], label='GRU Train Accuracy')
plt.plot(history_gru.history['val_accuracy'], label='GRU Test Accuracy')
plt.title('GRU Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Получение предсказаний
y_pred_lstm = lstm_model.predict(X_test)
y_pred_gru = gru_model.predict(X_test)

# Преобразование one-hot обратно в классы
y_test_classes = np.argmax(y_test, axis=1)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
y_pred_gru_classes = np.argmax(y_pred_gru, axis=1)

# 1. График реальных vs предсказанных значений (LSTM)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(y_test_classes)), y_test_classes, color='blue', label='Реальные значения', alpha=0.6)
plt.plot(range(len(y_pred_lstm_classes)), y_pred_lstm_classes, color='red', label='LSTM Прогноз', alpha=0.6)
plt.title('Сравнение реальных и предсказанных значений (LSTM)')
plt.xlabel('Номер примера')
plt.ylabel('Класс (0-2)')
plt.legend()

# 2. График реальных vs предсказанных значений (GRU)
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test_classes)), y_test_classes, color='blue', label='Реальные значения', alpha=0.6)
plt.plot(range(len(y_pred_gru_classes)), y_pred_gru_classes, color='green', label='GRU Прогноз', alpha=0.6)
plt.title('Сравнение реальных и предсказанных значений (GRU)')
plt.xlabel('Номер примера')
plt.ylabel('Класс (0-2)')
plt.legend()
plt.tight_layout()
plt.show()

# Оценка моделей на тестовых данных
lstm_score = lstm_model.evaluate(X_test, y_test, verbose=0)
gru_score = gru_model.evaluate(X_test, y_test, verbose=0)

print(f"LSTM Test Accuracy: {lstm_score[1]*100:.2f}%")
print(f"GRU Test Accuracy: {gru_score[1]*100:.2f}%")