import torch 
import contextlib

@contextlib.contextmanager
def _no_grad_sync(model: torch.nn.Module):
    model.set_requires_all_reduce(False)
    try:
        yield
    finally:
        model.set_requires_all_reduce(True)


@contextlib.contextmanager
def no_grad_sync(models: list[torch.nn.Module], enable: bool = False):
    if not enable:
        yield
        return

    with contextlib.ExitStack() as stack:
        for m in models:
            ctx = _no_grad_sync(m) if hasattr(m, "set_requires_all_reduce") else contextlib.nullcontext()
            stack.enter_context(ctx)
        yield