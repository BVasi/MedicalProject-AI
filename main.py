import os
import json
import time
from constants import constants
from model.Azure.ServiceBus.service_bus_message_processor import ServiceBusMessageProcessor
from model.Azure.StorageAccount.storage_repository import StorageRepository
from model.ImageProcessing.image_processor import ImageProcessor
from model.AI.model_predictor import PneumoniaModelPredictor
from model.AI.model_trainer import PneumoniaModelTrainer


def get_configuration():
    with (open(constants.CONFIG_FILE_PATH, constants.READ_FILE)) as file:
        configuration = json.load(file)
    return configuration


def main(): #to do: refactor + more checking
    configuration = get_configuration()
    message_processor = ServiceBusMessageProcessor(
        configuration[constants.SERVICE_BUS][constants.CONNECTION_STRING],
        configuration[constants.SERVICE_BUS][constants.QUEUE_NAME]
    )
    storage_repository = StorageRepository(
        configuration[constants.BLOB_STORAGE][constants.CONNECTION_STRING],
        configuration[constants.BLOB_STORAGE][constants.CONTAINER_NAME],
        configuration[constants.TABLE_STORAGE][constants.TABLE_NAME]
    )
    if not os.path.exists(configuration[constants.MODEL_PATH]):
        pneumonia_trainer = PneumoniaModelTrainer(configuration[constants.DATASET_PATH])
        pneumonia_trainer.train()
        pneumonia_trainer.save_model(configuration[constants.MODEL_PATH])
    pneumonia_predictor = PneumoniaModelPredictor(configuration[constants.MODEL_PATH])
    while True:
        if not message_processor.has_message():
            time.sleep(constants.WAITING_TIME)
            continue
        message = message_processor.get_message()
        if not message:
            continue
        image = storage_repository.get_image(message)
        if not image:
            continue
        image_array = ImageProcessor.prepare_image(image)
        prediction = pneumonia_predictor.predict(image_array)
        storage_repository.save_to_table(message, prediction)
        message_processor.complete_message()


if __name__ == "__main__":
    main()
