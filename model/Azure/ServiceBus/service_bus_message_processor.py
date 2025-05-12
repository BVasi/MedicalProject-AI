from azure.servicebus import ServiceBusClient
from azure.servicebus import ServiceBusReceivedMessage
from queue import Queue
import threading


class ServiceBusMessageProcessor:
    def __init__(self, connection_string, queue_name):
        self.connection_string = connection_string
        self.queue_name = queue_name
        self.service_bus_client = ServiceBusClient.from_connection_string(self.connection_string)
        self.receiver = None
        self.message_queue = Queue()
        self.lock = threading.Lock()


    def queue_messages(self):
        with self.lock:
            if not self.receiver:
                self.receiver = self.service_bus_client.get_queue_receiver(queue_name=self.queue_name, max_wait_time=5)
            message = next(self.receiver, None)
            if not message:
                return
            self.message_queue.put(message)


    def get_oldest_message(self):
        with self.lock:
            if not self.message_queue.empty():
                return self.message_queue.get()
            return None


    def complete_message(self, message):
        with self.lock:
            if not message or not self.receiver:
                return
            self.receiver.complete_message(message)
