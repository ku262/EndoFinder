class RepeatedAugmentationTransform:

    def __init__(self, transform, copies=2, key="input", target_key="mask"):
        self.transform = transform
        self.copies = copies
        self.key = key
        self.target_key = target_key

    def __call__(self, record):
        record = record.copy()
        img = record.pop(self.key)
        mask = record.pop(self.target_key)
        for i in range(self.copies):
            record[f"{self.key}{i}"], record[f"{self.target_key}{i}"] = self.transform(img, mask)
        return record
