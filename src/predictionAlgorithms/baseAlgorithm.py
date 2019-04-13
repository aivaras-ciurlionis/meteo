class BaseAlgorithm:
    name = 'Base'
    step = 1
    size = 128
    base = 4

    def predict(self, source_images, count):
        return source_images

    def with_step(self, step):
        self.step = step
        return self

    def with_size(self, size):
        self.size = size
        return self

    def with_base(self, base):
        self.base = base
        return self
