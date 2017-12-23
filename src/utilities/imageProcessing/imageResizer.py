from PIL import Image


class ImageResizer:
    images=[]

    def set_images(self, images):
        self.images = images
        return self

    def resize_images(self, new_size):
        for image in self.images:
            self.resize_image(image, new_size)

    def resize_image(self, image, new_size):
        image.thumbnail(new_size, Image.NONE)
