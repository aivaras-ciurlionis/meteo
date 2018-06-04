import os
import arrow


class ImageSequencesLoader:
    srcFolder = ''
    startDate = None
    endDate = None
    DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'
    DATE_FORMAT2 = 'YYYY-M-DD--HH-mm-ss'

    def select_folder(self, folder_name):
        self.srcFolder = folder_name
        return self

    def set_date_range(self, start_date, end_date):
        if (start_date is not None):
            self.startDate = arrow.get(start_date)
        if (end_date is not None):
            self.endDate = arrow.get(end_date)
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
            try:
                image_date = arrow.get(image_date_name, self.DATE_FORMAT)
            except:
                image_date = arrow.get(image_date_name, self.DATE_FORMAT2)

            if image_date == date:
                new_sequence.append(file_name)
            else:
                filtered_files = self.filter_dates(new_sequence.copy())
                if len(filtered_files) > 0:
                    sequences.append(filtered_files)
                new_sequence = [file_name]
                try:
                    date = arrow.get(file_name, self.DATE_FORMAT)
                except:
                    date = arrow.get(file_name, self.DATE_FORMAT2)

        if self.startDate is not None and self.endDate is not None:
            sequences.append(self.filter_dates(new_sequence))
        else:
            sequences.append(new_sequence)
        print(sequences)
        return sequences

    def filter_dates(self, files):
        if self.startDate is None or self.endDate is None:
            return files
        filtered = []
        for file in files:
            image_date_name = os.path.splitext(file)[0]
            try:
                image_date = arrow.get(image_date_name, self.DATE_FORMAT)
            except:
                image_date = arrow.get(image_date_name, self.DATE_FORMAT2)
            if self.startDate <= image_date <= self.endDate:
                filtered.append(file)
        return filtered
