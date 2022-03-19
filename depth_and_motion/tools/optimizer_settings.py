from depth_and_motion.static_names import (
    NO_SCHEDULER,
    RESTARTS_SCHEDULER,
    CYCLIC_TRIANGULAR_SCHEDULER,
    CYCLIC_COSINE_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    SGD,
    ADAM,
    ADAMW,
    RMSPROP,
)

class SchedulerSettings:
    """
    The class contains parameters for different schedulers.
    """

    def __init__(self):
        pass

    # Supported schedulers
    @staticmethod
    def no_scheduler(initial_learning_rate=1e-4):
        """
        Constant learning rate scheduler.
        """
        scheduler = {
            "name": NO_SCHEDULER,
            "initial_learning_rate": initial_learning_rate,
        }
        return scheduler


class OptimizerSettings:
    def __init__(self):
        pass

    @staticmethod
    def adam_optimizer(beta_1=0.99, beta_2=0.999, epsilon=1e-07, amsgrad=False):
        """
        Adam optimizer.
        See: https://arxiv.org/abs/1412.6980
        """
        optimizer = {
            "name": ADAM,
            "beta_1": beta_1,  # can be single value or list [min max] with cyclic scheduler
            "beta_2": beta_2,  # can be single value or list [min max] with cyclic scheduler
            "epsilon": epsilon,
            "amsgrad": amsgrad,
        }
        return optimizer