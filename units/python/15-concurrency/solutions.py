"""Reference answers for 15 - concurrency and async."""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def fetch_all(urls, fetch, max_workers=8):
    """Call `fetch` on every url concurrently, returning results IN ORDER.

    Threads are the right tool here because `fetch` is assumed to be
    IO-bound: it releases the GIL while waiting, so the waits overlap.

    Executor.map preserves input order even though completion order varies -
    which is why you do not need to sort the results yourself.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fetch, urls))


class Counter:
    """A counter that stays correct under concurrent increments.

    `self.value += 1` is read-modify-write: three bytecodes with a chance to
    be interrupted between them. The GIL does NOT make it atomic. A lock -
    or itertools.count, or a queue - is what makes it safe.
    """

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1


def producer_consumer(items, work, consumers=4):
    """Push items through `consumers` worker threads; return results unordered.

    A bounded queue applies backpressure: once it is full, producers block
    instead of buffering the whole input in memory. That is the difference
    between a pipeline that streams and one that OOMs on a big input.
    """
    task_queue = queue.Queue(maxsize=consumers * 2)
    results = []
    results_lock = threading.Lock()
    sentinel = object()

    def consume():
        while True:
            item = task_queue.get()
            try:
                if item is sentinel:
                    return
                value = work(item)
                with results_lock:
                    results.append(value)
            finally:
                task_queue.task_done()

    threads = [threading.Thread(target=consume, daemon=True)
               for _ in range(consumers)]
    for thread in threads:
        thread.start()
    for item in items:
        task_queue.put(item)
    for _ in threads:
        task_queue.put(sentinel)
    for thread in threads:
        thread.join()
    return results


async def gather_limited(coro_factories, limit):
    """Run coroutines concurrently, at most `limit` at a time, in order.

    asyncio.gather starts *everything* at once. A semaphore is how you cap
    concurrency - the standard shape for calling a rate-limited API or a
    GPU-backed service without swamping it.

    Takes factories, not coroutines: a coroutine object can only be awaited
    once, so the caller cannot safely build them all up front for a retry.
    """
    semaphore = asyncio.Semaphore(limit)

    async def run(factory):
        async with semaphore:
            return await factory()

    return await asyncio.gather(*(run(f) for f in coro_factories))


async def with_timeout(coro, seconds, default=None):
    """Await `coro`, returning `default` if it takes longer than `seconds`.

    asyncio.wait_for cancels the task on timeout - it does not leave it
    running in the background. Swallowing TimeoutError without cancelling is
    the usual leak.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return default


class Batcher:
    """Collects items and flushes them once the batch is full.

    The core of every batched inference server: amortise a fixed per-call
    cost over many inputs. Real ones also flush on a timer so a partial
    batch does not wait forever.
    """

    def __init__(self, size, flush):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.flush_fn = flush
        self.pending = []
        self.batches = 0

    def add(self, item):
        self.pending.append(item)
        if len(self.pending) >= self.size:
            return self.flush()
        return None

    def flush(self):
        if not self.pending:
            return None
        batch, self.pending = self.pending, []
        self.batches += 1
        return self.flush_fn(batch)


def cpu_bound_is_not_helped_by_threads():
    """Return the reason threads do not speed up CPU-bound Python.

    One sentence, because this is an interview answer, not an essay.
    """
    return ("The GIL lets only one thread execute Python bytecode at a time, "
            "so CPU-bound work is serialised; use processes instead.")
