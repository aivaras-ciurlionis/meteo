from azure.storage.blob import BlockBlobService
from os import path


def upload_file(file, filename, blob_service, container):
    blob_service.create_blob_from_path(container, filename, file)


class BlobUploader:

    def upload_actual(self, actualFiles, source_dir):
        self.upload_files(actualFiles, 'actual', source_dir)

    def upload_results(self, predictionResults, source_dir):
        for algorithm in predictionResults:
            self.upload_files(algorithm['files'], 'predicted', source_dir)

    def upload_files(self, files, container, source_dir):
        block_blob_service = BlockBlobService(
            account_name='meteorologydata',
            account_key='uDDmdk4JCIQGRdGEuaQxD5hu5kPBVqnR7IX0u4Yr7EBaUzL8zU54NsR0Y0PDzVwWv70BLD6Ut87kPrornXPjrg=='
        )
        for file in files:
            upload_file(path.join(source_dir, file), file, block_blob_service, container)
