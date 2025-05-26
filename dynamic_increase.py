from ultralytics import YOLO
from ultralytics.trainer import DetectionTrainer

class DynamicFlipLoader(LoadImagesAndLabels):
    def __init__(self, ..., total_epochs=100, max_flipud=0.45):
        self.total_epochs = total_epochs
        self.cur_epoch = 0
        self.max_flipud = max_flipud
        super().__init__(...)

    def set_epoch(self, epoch):  # this is what you call from trainer
        self.cur_epoch = epoch

    def get_dynamic_flipud(self):
        # Adjust based on current epoch
        return min(self.max_flipud, self.cur_epoch / self.total_epochs * self.max_flipud)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        flipud_prob = self.get_dynamic_flipud()
        if random.random() < flipud_prob:
            # Apply vertical flip
            img = np.flipud(sample[0]).copy()
            if len(sample[1]):
                sample[1][:, 2] = 1 - sample[1][:, 2]
            return (img, sample[1], *sample[2:])
        return sample

class MyTrainer(DetectionTrainer):
    def before_train_epoch(self):
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(self.epoch)