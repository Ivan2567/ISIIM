import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def limit_images_per_class(dataset_path, num_images_per_class):

    if not os.path.exists(dataset_path):
        print(f"Путь к датасету не найден: {dataset_path}")
        return

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            if len(images) > num_images_per_class:
                # Сортируем изображения, чтобы удаление было детерминированным (опционально)
                images.sort()
                for i in range(num_images_per_class, len(images)):
                    image_to_remove = os.path.join(class_dir, images[i])
                    try:
                        os.remove(image_to_remove)
                        # print(f"Удалено: {image_to_remove}")
                    except OSError as e:
                        print(f"Ошибка при удалении файла {image_to_remove}: {e}")
            #print(f"Класс '{class_name}': осталось {len(os.listdir(class_dir))} изображений.")

def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):

    if not os.path.exists(source_dir):
        print(f"Исходный путь к датасету не найден: {source_dir}")
        return

    # Удаление тренировочных и тестовых папок, если они существуют
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Создание тренировочных и тестовых папок
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Проходим по каждой подпапке (классу) в исходной директории
    for class_name in os.listdir(source_dir):
        class_source_dir = os.path.join(source_dir, class_name)

        # Убедимся, что это директория
        if os.path.isdir(class_source_dir):
            # Создаем подпапки для этого класса в тренировочной и тестовой директориях
            class_train_dir = os.path.join(train_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # Получаем список всех файлов (изображений) в текущем классе
            images = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]

            # Разделяем список изображений на тренировочную и тестовую части
            if len(images) > 0:
                train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

                # Копируем тренировочные изображения в соответствующую папку
                for img in train_images:
                    src_path = os.path.join(class_source_dir, img)
                    dest_path = os.path.join(class_train_dir, img)
                    shutil.copy(src_path, dest_path)

                # Копируем тестовые изображения в соответствующую папку
                for img in test_images:
                    src_path = os.path.join(class_source_dir, img)
                    dest_path = os.path.join(class_test_dir, img)
                    shutil.copy(src_path, dest_path)

            #print(f"Разделен класс '{class_name}': {len(train_images)} для тренировки, {len(test_images)} для теста.")

def load_dataset(train_dataset_path, test_dataset_path, image_size=(224, 224), batch_size=32):

    try:
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory(
            train_dataset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb')

        test_generator = datagen.flow_from_directory(
            test_dataset_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb')
        return train_generator, test_generator
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return None, None

def create_model_vgg16(classes, input_shape=(224, 224, 3), activation_block1='relu', activation_block2='relu',
                       activation_block3='relu', activation_block4='relu', activation_block5='relu',
                       activation_dense='relu'):


    # Построение и компиляция модели VGG16
    model = Sequential()

    # Блок 1: два слоя свёртки + pooling
    model.add(Input(input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation_block1, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation=activation_block1, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 2: два слоя свёртки + pooling
    model.add(Conv2D(128, (3, 3), activation=activation_block2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation=activation_block2, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 3: три слоя свёртки + pooling
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 4: три слоя свёртки + pooling
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 5: три слоя свёртки + pooling
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Полносвязная сеть
    model.add(Flatten())
    model.add(Dense(4096, activation=activation_dense))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation=activation_dense))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

    return model

def create_model_vgg19(classes, input_shape=(224, 224, 3), activation_block1='relu', activation_block2='relu',
                       activation_block3='relu', activation_block4='relu', activation_block5='relu',
                       activation_dense='relu'):


    # Построение и компиляция модели VGG19
    model = Sequential()

    # Блок 1: два слоя свёртки + pooling
    model.add(Input(input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation_block1, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation=activation_block1, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 2: два слоя свёртки + pooling
    model.add(Conv2D(128, (3, 3), activation=activation_block2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation=activation_block2, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 3: четыре слоя свёртки + pooling
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation=activation_block3, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 4: четыре слоя свёртки + pooling
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block4, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Блок 5: четыре слоя свёртки + pooling
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation=activation_block5, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Полносвязная сеть
    model.add(Flatten())
    model.add(Dense(4096, activation=activation_dense))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation=activation_dense))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

    return model

def train_model(model, train_generator, test_generator, epochs, batch_size, verbose):

    history = model.fit(
        train_generator,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=test_generator,
    )

    return history, model

# Ограничение количества изображений на класс
limit_images_per_class('labs/lab2/task1/content', 2000)

# Деление выборки на тренировочную и тестовую
split_dataset('labs/lab2/task1/content', 'labs/lab2/task1/dataset/train', 'labs/lab2/task1/dataset/test', test_size=0.2, random_state=42)

# Создание генераторов для тренировки и тестирования
train_generator, test_generator = load_dataset('labs/lab2/task1/dataset/train', 'labs/lab2/task1/dataset/test', image_size=(64, 64))

# Создание папки для сохранения обученных моделей
os.makedirs('labs/lab2/task1/models', exist_ok=True)

# Тренировка и вывод точности моделей VGG16
hidden_layer_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish']
models_vgg16 = []
trained_models_vgg16 = []
historys_vgg16 = []
for i in range(len(hidden_layer_activations)):
    model = create_model_vgg16(6, (64, 64, 3), hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i])
    models_vgg16.append(model)
    history, trained_model = train_model(model, train_generator, test_generator, 5, 32, 0)
    trained_models_vgg16.append(trained_model)
    historys_vgg16.append(history)
    trained_model.save(f'labs/lab2/task1/models/model_vgg16({i+1}).keras')
    print('VGG16(' + str(i+1) + '): ' + 'Accuracy = ' + '{:.5f}'.format(historys_vgg16[i].history['accuracy'][-1]))

# Тренировка и вывод точности моделей VGG19
models_vgg19 = []
trained_models_vgg19 = []
historys_vgg19 = []
for i in range(len(hidden_layer_activations)):
    model = create_model_vgg19(6, (64, 64, 3), hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i], hidden_layer_activations[i])
    models_vgg19.append(model)
    history, trained_model = train_model(model, train_generator, test_generator, 5, 32, 0)
    trained_models_vgg19.append(trained_model)
    historys_vgg19.append(history)
    trained_model.save(f'labs/lab2/task1/models/model_vgg19({i+1}).keras')
    print('VGG19(' + str(i+1) + '): ' + 'Accuracy = ' + '{:.5f}'.format(historys_vgg19[i].history['accuracy'][-1]))
