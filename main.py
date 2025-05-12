import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
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


def process_message(message, storage_repository, pneumonia_predictor, message_processor):
    image = storage_repository.get_image(str(message))
    if not image:
        return
    image_array = ImageProcessor.prepare_image(image)
    prediction = pneumonia_predictor.predict(image_array)
    storage_repository.save_to_table(str(message), prediction)
    message_processor.complete_message(message)


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
    with ThreadPoolExecutor(max_workers=constants.MAX_WORKERS) as executor:
        while True:
            message_processor.queue_messages()
            message = message_processor.get_oldest_message()
            if not message:
                time.sleep(constants.WAITING_TIME)
                continue
            executor.submit(process_message, message, storage_repository, pneumonia_predictor, message_processor)


if __name__ == "__main__":
    main()
