import os
import arrow


class ImageSequencesLoader:
    srcFolder = ''
    DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'

    def select_folder(self, folder_name):
        self.srcFolder = folder_name
        return self

    def load_sequences(self):
        files = os.listdir(self.srcFolder)
        files.sort()
        first_image_name = os.path.splitext(files[0])[0]
        date = arrow.get(first_image_name, self.DATE_FORMAT)
        sequences = []
        new_sequence = [files[0]]
        for file_name in files[1:]:
            date = date.replace(minutes=15)
            image_date_name = os.path.splitext(file_name)[0]
            image_date = arrow.get(image_date_name, self.DATE_FORMAT)
            if image_date == date:
                new_sequence.append(file_name)
            else:
                sequences.append(new_sequence.copy())
                new_sequence = [file_name]
                date = arrow.get(file_name, self.DATE_FORMAT)
        sequences.append(new_sequence)
        return sequences
