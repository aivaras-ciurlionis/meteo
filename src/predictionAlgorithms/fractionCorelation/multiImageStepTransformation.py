from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation


class MultiImageStepTransformation(BaseTransformation):
    transformations = []
    name = 'Multi image fraction'
    source_count = 4

    def __init__(self, transformation_algorithm, source_count=4):
        self.name += ' ' + transformation_algorithm[0]
        self.transformations = transformation_algorithm[1]
        self.source_count = source_count
        super().__init__(transformation_algorithm)

    def predict(self, source_images, count):
        vectors = []
        for i in range(2, self.source_count+2):
            best_vector = self.find_best_movement_vector\
            (
                source_images,
                source_images[-i],
                source_images[-i + 1]
            )
            vectors.append(best_vector)
        average_vector = self.get_average_vector(vectors)
        print(average_vector)
        return self.generate_images(source_images, average_vector, count)

    def get_average_vector(self, vectors):
        average_vector = []
        transformation_count = len(self.transformations)
        for i in range(transformation_count):
            s = 0
            for j in range(self.source_count-1):
                s += vectors[j][i]
            average_vector.append(s/(self.source_count))
        return average_vector

    def generate_images(self, images, best_movement_vector, count):
        generated_images = []
        original_size = images[-1].size
        new_size = tuple(self.source_count*x for x in images[-1].size)
        working_image = images[-1].resize(new_size)
        for index in range(count):
            for i, transformation in enumerate(self.transformations):
                algorithm = transformation[0]
                working_image = algorithm(working_image, int(best_movement_vector[i] * self.source_count))
            generated_images.append(working_image.resize(original_size))
        return generated_images
