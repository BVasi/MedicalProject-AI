from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, TableClient


class StorageRepository:
    def __init__(self, connection_string, container_name, table_name):
        self.connection_string = connection_string
        self.container_name = container_name
        self.table_name = table_name

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.blob_client = None

        self.table_service_client = TableServiceClient.from_connection_string(self.connection_string)
        self.table_client = None


    def get_image(self, path):
        if not self.blob_client or self.blob_client.blob_name != path:
            self.blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=path)
        return self.blob_client.download_blob().readall() if self.blob_client.exists() else None


    def save_to_table(self, path, prediction): #to do: refactor
        if not self.table_client:
            self.table_client = self.table_service_client.get_table_client(table_name=self.table_name)
        entity = {
            'PartitionKey': path.split('/')[0],
            'RowKey': path.split('/')[1],
            'confidenceScore': prediction
        }
        self.table_client.upsert_entity(entity=entity)