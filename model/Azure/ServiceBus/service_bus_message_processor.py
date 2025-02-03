from azure.servicebus import ServiceBusClient
from azure.servicebus import ServiceBusReceivedMessage


class ServiceBusMessageProcessor:
    def __init__(self, connection_string, queue_name):
        self.connection_string = connection_string
        self.queue_name = queue_name
        self.service_bus_client = ServiceBusClient.from_connection_string(self.connection_string)
        self.receiver = None
        self.current_message = None


    def has_message(self):
        if not self.receiver:
            self.receiver = self.service_bus_client.get_queue_receiver(queue_name=self.queue_name, max_wait_time=5)
        message = next(self.receiver, None)
        if message:
            self.current_message = message
            return True
        else:
            self.current_message = None
            return False


    def get_message(self):
        if self.current_message:
            return str(self.current_message)
        else:
            return None


    def complete_message(self):
        if not self.current_message or not self.receiver:
            return
        self.receiver.complete_message(self.current_message)
        self.current_message = None
