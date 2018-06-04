from PIL import Image
from os import path


class ImagesConcater:

    def concat_images(self, working_dir, result_file):
        img = Image.new('RGBA', (256*3, 256*3))
        i = 0
        for x in range(0,3):
            for y in range(0,3):
                image_location = path.join(working_dir, str(i)+".png")
                temp = Image.open(image_location)
                img.paste(temp, (256*x, 256*y))
                i += 1
        img.save(result_file)
        return self