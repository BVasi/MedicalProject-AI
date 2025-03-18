import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class PneumoniaModelTrainer:
    def __init__(self, dataset_dir, img_size=(224, 224), batch_size=32, epochs=10):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = self.create_model()
        self.train_generator, self.val_generator, self.test_generator = self.prepare_data()


    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def prepare_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        return train_generator, val_generator, test_generator


    def train(self):
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator
        )
        return history


    def save_model(self, model_path="pneumonia_model.h5"):
        self.model.save(model_path)
