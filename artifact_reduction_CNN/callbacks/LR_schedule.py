import numpy as np


class Scheduler:
    # Learning rate scheduler function
    def __init__(self, scheduler_mode='power_decay', lr_power=0.9, epochs=250, lr_base=0.005):
        # Save parameters
        self.scheduler_mode = scheduler_mode
        self.lr_base = float(lr_base)
        self.lr_power = float(lr_power)
        self.epochs = float(epochs)

        # Get schedule function: a function that takes an epoch index (integer, indexed from 0) and current learning
        # rate (float) as inputs and returns a new learning rate as output (float).
        if self.scheduler_mode == 'power_decay':
            self.scheduler_function = self.power_decay_scheduler
        elif self.scheduler_mode == 'exp_decay':
            self.scheduler_function = self.exp_decay_scheduler
        elif self.scheduler_mode == 'no_decay':
            self.scheduler_function = self.constant_scheduler
        elif self.scheduler_mode == 'step_exp_decay':
            self.scheduler_function = self.step_exp_decay_scheduler
        elif self.scheduler_mode == 'boundary_exp_decay':
            self.scheduler_function = self.boundary_exp_decay_scheduler
        elif self.scheduler_mode == 'progressive_drops':
            self.scheduler_function = self.progressive_drops_scheduler
        elif self.scheduler_mode == 'piecewise_linear_decay':
            self.scheduler_function = self.piecewise_linear_scheduler
        else:
            raise ValueError('Unknown scheduler: ' + self.scheduler_mode)

    # equal to linear decay when lr_power=1
    def power_decay_scheduler(self, epoch, lr):
        return self.lr_base * ((1 - float(epoch)/self.epochs) ** self.lr_power)

    def exp_decay_scheduler(self, epoch, lr):
        return self.lr_base * (self.lr_power ** epoch)

    def constant_scheduler(self, epoch, lr):
        return self.lr_base

    def step_exp_decay_scheduler(self, epoch, lr):
        step_size = 5
        return self.lr_base * (self.lr_power ** np.floor(epoch/step_size))

    def boundary_exp_decay_scheduler(self, epoch, lr):
        lr_end = self.lr_base / self.lr_power
        decay_rate = np.exp((1 / (self.epochs-1)) * np.log(lr_end / self.lr_base))
        return self.lr_base * np.power(decay_rate, epoch)

    def progressive_drops_scheduler(self, epoch, lr):
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * self.epochs:
            new_lr = self.lr_base / 1000
        elif epoch > 0.75 * self.epochs:
            new_lr = self.lr_base / 100
        elif epoch > 0.5 * self.epochs:
            new_lr = self.lr_base / 10
        else:
            new_lr = self.lr_base
        return new_lr

    def piecewise_linear_scheduler(self, epoch, lr):
        start_decay_epoch = self.epochs / 2
        decay_rate = self.lr_base/(self.epochs - start_decay_epoch + 1)
        if epoch >= start_decay_epoch:
            new_lr = self.lr_base - decay_rate * (epoch - start_decay_epoch + 1)
        else:
            new_lr = self.lr_base
        return new_lr