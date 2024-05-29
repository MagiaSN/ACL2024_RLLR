import torch.utils.checkpoint
_original_checkpoint_func = torch.utils.checkpoint.checkpoint
def checkpoint_wrapper(*args, use_reentrant=False, **kwargs):
    return _original_checkpoint_func(*args, use_reentrant=use_reentrant, **kwargs)
torch.utils.checkpoint.checkpoint = checkpoint_wrapper

from llmtuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
