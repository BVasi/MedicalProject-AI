import io
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

class ImageProcessor:
    @staticmethod
    def prepare_image(image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((150, 150))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
