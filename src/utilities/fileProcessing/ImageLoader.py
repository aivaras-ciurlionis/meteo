from PIL import Image
from os import path

class ImageLoader:
    imageSequenceFileNames = []
    imagesFolder = ''

    def set_image_folder(self, folder):
        self.imagesFolder = folder
        return self

    def set_sequence(self, sequence):
        self.imageSequenceFileNames = sequence
        return self

    def load_sequence_images(self):
        images = []
        for image_name in self.imageSequenceFileNames:
            image_location = path.join(self.imagesFolder, image_name)
            image = Image.open(image_location)
            images.append(image)
        return images