"""Upload and retain immutable tc02 prefill metadata."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import lru_cache

import torch


class _CaptureMetadata:
    def __init__(self):
        self.chunks = {}
        self.plans = {}

    def prepare(self):
        for _, _, _, _, event, streams in tuple(self.chunks.values()):
            if streams is None:
                event.prepare_for_capture()
        for _, _, event, streams in tuple(self.plans.values()):
            if event is not None and streams is None:
                event.prepare_for_capture()


capture_metadata = ContextVar("flashqla_capture_metadata", default=None)


@contextmanager
def prefill_graph_metadata():
    """Retain the warmup layouts for a tc02 Graph capture.

    Warm up each forward within this scope, call the yielded preparation
    function, then capture before leaving the scope. The captured Graph owns
    the selected metadata through replay, including after the scope ends.
    """
    entries = _CaptureMetadata()
    token = capture_metadata.set(entries)
    try:
        yield entries.prepare
    finally:
        capture_metadata.reset(token)
        entries.chunks.clear()
        entries.plans.clear()


@lru_cache(maxsize=None)
def _upload_stream(device):
    return torch.cuda.Stream(device=device)


def metadata_upload(device, asynchronous):
    return torch.cuda.stream(_upload_stream(device)) if asynchronous else nullcontext()


def metadata_tensor(values, dtype, device, asynchronous):
    if asynchronous:
        return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(values, dtype=dtype, device=device)
