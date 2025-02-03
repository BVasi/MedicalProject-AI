from model.ImageProcessing.image_processor import ImageProcessor
import tensorflow as tf
import os

class PneumoniaModelPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)


    def predict(self, image_array):
        if not self.model:
            return None
        prediction = self.model.predict(image_array)
        return round(float(prediction[0][0]), 3)


    def test_accuracy(self, dataset_dir):
        def load_images_from_directory(directory):
            images = []
            labels = []
            for label_dir in ['NORMAL', 'PNEUMONIA']:
                label_path = os.path.join(directory, label_dir)
                label = 0 if label_dir == 'NORMAL' else 1

                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    try:
                        with open(image_path, 'rb') as img_file:
                            image_bytes = img_file.read()
                            image_array = ImageProcessor.prepare_image(image_bytes)
                            images.append(image_array[0])
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_name}: {e}")

            return images, labels
        test_images, test_labels = load_images_from_directory(os.path.join(dataset_dir, 'test'))
        test_images = tf.convert_to_tensor(test_images)
        test_labels = tf.convert_to_tensor(test_labels)
        predictions = self.model.predict(test_images)
        predictions = [1 if p > 0.5 else 0 for p in predictions]
        correct_predictions = sum([1 for p, label in zip(predictions, test_labels) if p == label])
        accuracy = correct_predictions / len(test_labels)
        print(f"Test Accuracy: {accuracy:.2f}")
        return accuracy