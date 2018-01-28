class AlgorithmPredictionResult:
    images = []
    image_accuracy = []

    def __init__(self, images, img_accuracy):
        self.images = images
        self.image_accuracy = img_accuracy
