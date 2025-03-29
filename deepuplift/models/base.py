
import numpy as np

class EarlyStopper:
    # patience（默认值为 15）代表可以容忍验证损失连续不下降的轮次数量，
    # min_delta（默认值为 0）表示验证损失变化的最小差值（只有损失上升幅度超过该差值才会计数）
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


