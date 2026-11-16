"""FIFO CPU preparation with main-thread randomness and bounded lookahead."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from torch.utils.data import default_collate


class _Plans(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.plan(index)


def _identity(plans):
    return plans


@contextmanager
def joint_batches(dataset, *, batch_size, generator, shuffle=False, prefetch=False, prefetch_batches=2):
    """Bounded batches ahead plus the consumer's batch; join on every exit.

    DataLoader still controls seeded index order and base-seed consumption.
    Plans consume CPU RNG on the main thread. The worker only reads/copies and
    transforms tensors. The current joint model consumes no CPU randomness;
    a future stochastic CPU model needs a separate, versioned RNG contract.
    No worker, future, or unconsumed queue survives an epoch/checkpoint boundary.
    """
    if type(prefetch_batches) is not int or prefetch_batches < 1:
        raise ValueError("prefetch_batches must be a positive integer")
    loader = torch.utils.data.DataLoader(_Plans(dataset) if prefetch else dataset,
        batch_size=batch_size,shuffle=shuffle,num_workers=0,generator=generator,
        collate_fn=_identity if prefetch else None)
    if not prefetch:
        yield iter(loader)
        return
    def prepare(plans):
        return default_collate([dataset.load_plan(plan) for plan in plans])
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="joint-loading") as executor:
        pending = deque()
        plans = iter(loader)
        def refill():
            while len(pending) < prefetch_batches:
                try:
                    plan = next(plans)
                except StopIteration:
                    break
                pending.append(executor.submit(prepare, plan))
        def batches():
            refill()
            while pending:
                batch = pending.popleft().result()
                refill()
                yield batch
        iterator = batches()
        try:
            yield iterator
        finally:
            iterator.close()
            for future in pending:
                future.cancel()
            # executor.__exit__ joins any active preparation, even on failure.
