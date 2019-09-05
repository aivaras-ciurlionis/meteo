from PIL import Image


class ImageResizer:
    images=[]
    cropAmount=0

    def set_images(self, images):
        self.images = images
        return self

    def set_crop_amount(self, cropAmount):
        self.cropAmount = cropAmount
        return self

    def resize_images(self, new_size):
        for image in self.images:
            self.resize_image(image, new_size)

    def resize_image(self, image, new_size):
        if self.cropAmount > 0:
            s = image.size[0]
            crop_amount = int(s * self.cropAmount)
            image.crop((crop_amount, crop_amount, s - crop_amount, s - crop_amount))
        image.thumbnail(new_size, Image.NONE)
