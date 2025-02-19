class TransformKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, sample):
        # Apply the transform only on the designated key.
        sample[self.key] = self.transform(sample[self.key])
        return sample