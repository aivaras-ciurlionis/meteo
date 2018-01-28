from PIL import Image
from os import path


class ImageLoader:
    imageSequenceFileNames = []
    imagesFolder = ''
    maxImages = 50

    def set_image_folder(self, folder):
        self.imagesFolder = folder
        return self

    def set_sequence(self, sequence):
        self.imageSequenceFileNames = sequence
        return self

    def set_max_images(self, max_images):
        self.maxImages = max_images
        return self

    def load_sequence_images(self):
        images = []
        for index, image_name in enumerate(self.imageSequenceFileNames):
            if index > self.maxImages:
                break
            image_location = path.join(self.imagesFolder, image_name)
            image = Image.open(image_location)
            images.append(image)
            print(image.filename)
        return images
