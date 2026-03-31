from internal.optimizers import *
import cpu_adam

@dataclass
class CPUAdam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        return cpu_adam.CPUAdam(
            params,
            lr,
            *args,
            **kwargs,
        )