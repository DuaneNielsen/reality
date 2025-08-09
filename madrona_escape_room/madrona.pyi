from typing import Any

class CudaSync:
    """
    None
    """

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def wait(self, arg: int, /) -> None: ...

class ExecMode:
    """
    None
    """

    CPU: ExecMode

    CUDA: ExecMode

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class Tensor:
    """
    None
    """

    def __init__(self, arg: "Any", /) -> None: ...
    def to_jax(*args, **kwargs):
        """
        to_jax(self) -> jaxlib.xla_extension.DeviceArray[]
        """
        ...

    def to_torch(*args, **kwargs):
        """
        to_torch(self) -> torch.Tensor[]
        """
        ...

class TrainInterface:
    """
    None
    """

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def step_inputs(self) -> dict: ...
    def step_outputs(self) -> dict: ...
