"""Tests for 15 - concurrency and async.

Concurrency tests go flaky the moment they assert on wall-clock time, so
these assert on *observable structure* instead: ordering, counts, peak
concurrency recorded by the workers themselves, and whether a cancelled
task actually stopped. The only sleeps are tiny and are there to force
interleaving, never to be measured.
"""

import asyncio
import importlib
import os
import threading
import time
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))


class ConcurrencyProbe:
    """Records the peak number of workers running at the same time."""

    def __init__(self):
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def exit(self):
        with self._lock:
            self.current -= 1


class TestFetchAll(unittest.TestCase):
    def test_returns_results_in_input_order(self):
        """Completion order is not input order - the result must be."""
        def fetch(url):
            # Later urls finish first, so anything relying on completion
            # order comes back reversed.
            time.sleep((10 - url) * 0.005)
            return url * 2

        self.assertEqual(m.fetch_all(list(range(10)), fetch),
                         [n * 2 for n in range(10)])

    def test_empty_input(self):
        self.assertEqual(m.fetch_all([], lambda url: url), [])

    def test_calls_fetch_once_per_url(self):
        calls = []
        lock = threading.Lock()

        def fetch(url):
            with lock:
                calls.append(url)
            return url

        m.fetch_all(["a", "b", "c"], fetch)
        self.assertEqual(sorted(calls), ["a", "b", "c"])

    def test_actually_runs_concurrently(self):
        """Serial execution would leave the peak at 1."""
        probe = ConcurrencyProbe()

        def fetch(url):
            probe.enter()
            time.sleep(0.02)
            probe.exit()
            return url

        m.fetch_all(list(range(8)), fetch, max_workers=4)
        self.assertGreater(probe.peak, 1,
                           "no overlap - the fetches ran one after another")

    def test_respects_max_workers(self):
        probe = ConcurrencyProbe()

        def fetch(url):
            probe.enter()
            time.sleep(0.02)
            probe.exit()
            return url

        m.fetch_all(list(range(12)), fetch, max_workers=3)
        self.assertLessEqual(probe.peak, 3)

    def test_propagates_exceptions(self):
        """A worker failure must surface, not be swallowed by the pool."""
        class FetchFailed(Exception):
            pass

        def fetch(url):
            raise FetchFailed("boom")

        with self.assertRaises(FetchFailed):
            m.fetch_all([1], fetch)


class TestCounter(unittest.TestCase):
    def test_starts_at_zero(self):
        self.assertEqual(m.Counter().value, 0)

    def test_counts_single_threaded(self):
        counter = m.Counter()
        for _ in range(5):
            counter.increment()
        self.assertEqual(counter.value, 5)

    def test_survives_concurrent_increments(self):
        """`self.value += 1` is read-modify-write; the GIL does not save you.

        Enough iterations that an unlocked version loses updates reliably.
        """
        counter = m.Counter()
        threads = [threading.Thread(
            target=lambda: [counter.increment() for _ in range(20_000)])
            for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(counter.value, 80_000,
                         "lost updates - the increment is not atomic")

    def test_instances_are_independent(self):
        a, b = m.Counter(), m.Counter()
        a.increment()
        self.assertEqual((a.value, b.value), (1, 0))


class TestProducerConsumer(unittest.TestCase):
    def test_every_item_is_processed_exactly_once(self):
        out = m.producer_consumer(list(range(50)), lambda n: n * n)
        self.assertEqual(sorted(out), sorted(n * n for n in range(50)))

    def test_empty_input(self):
        self.assertEqual(m.producer_consumer([], lambda n: n), [])

    def test_more_consumers_than_items(self):
        out = m.producer_consumer([1, 2], lambda n: n, consumers=8)
        self.assertEqual(sorted(out), [1, 2])

    def test_all_workers_are_used(self):
        seen = set()
        lock = threading.Lock()

        def work(item):
            with lock:
                seen.add(threading.current_thread().name)
            time.sleep(0.01)
            return item

        m.producer_consumer(list(range(20)), work, consumers=4)
        self.assertGreater(len(seen), 1, "only one thread did any work")

    def test_terminates_and_does_not_hang(self):
        """A missing sentinel leaves consumers blocked in queue.get forever."""
        done = threading.Event()

        def run():
            m.producer_consumer(list(range(10)), lambda n: n)
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=10)
        self.assertTrue(done.is_set(), "producer_consumer did not return")


class TestGatherLimited(unittest.IsolatedAsyncioTestCase):
    async def test_results_are_in_input_order(self):
        def make(n):
            async def run():
                await asyncio.sleep((10 - n) * 0.005)
                return n
            return run

        out = await m.gather_limited([make(n) for n in range(10)], limit=4)
        self.assertEqual(out, list(range(10)))

    async def test_empty_input(self):
        self.assertEqual(await m.gather_limited([], limit=2), [])

    async def test_never_exceeds_the_limit(self):
        state = {"current": 0, "peak": 0}

        def make(n):
            async def run():
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                await asyncio.sleep(0.01)
                state["current"] -= 1
                return n
            return run

        await m.gather_limited([make(n) for n in range(20)], limit=3)
        self.assertEqual(state["peak"], 3)

    async def test_limit_larger_than_the_work_is_fine(self):
        def make(n):
            async def run():
                return n
            return run

        out = await m.gather_limited([make(n) for n in range(3)], limit=99)
        self.assertEqual(out, [0, 1, 2])

    async def test_actually_overlaps(self):
        """A sequential await loop would leave the peak at 1."""
        state = {"current": 0, "peak": 0}

        def make(n):
            async def run():
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                await asyncio.sleep(0.01)
                state["current"] -= 1
                return n
            return run

        await m.gather_limited([make(n) for n in range(6)], limit=3)
        self.assertGreater(state["peak"], 1, "the coroutines ran serially")


class TestWithTimeout(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_value_when_fast_enough(self):
        async def quick():
            return "ok"

        self.assertEqual(await m.with_timeout(quick(), 1.0), "ok")

    async def test_returns_the_default_when_too_slow(self):
        async def slow():
            await asyncio.sleep(5)
            return "ok"

        self.assertEqual(await m.with_timeout(slow(), 0.02, "fallback"),
                         "fallback")

    async def test_default_defaults_to_none(self):
        async def slow():
            await asyncio.sleep(5)

        self.assertIsNone(await m.with_timeout(slow(), 0.02))

    async def test_the_slow_coroutine_is_cancelled_not_left_running(self):
        """Swallowing TimeoutError without cancelling is the classic leak."""
        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        await m.with_timeout(slow(), 0.02)
        await asyncio.sleep(0.01)
        self.assertTrue(cancelled.is_set(), "the coroutine is still running")

    async def test_other_exceptions_are_not_swallowed(self):
        async def broken():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            await m.with_timeout(broken(), 1.0)


class TestBatcher(unittest.TestCase):
    def setUp(self):
        self.flushed = []

    def record(self, batch):
        self.flushed.append(list(batch))
        return len(batch)

    def test_does_not_flush_before_the_batch_is_full(self):
        batcher = m.Batcher(3, self.record)
        self.assertIsNone(batcher.add("a"))
        self.assertIsNone(batcher.add("b"))
        self.assertEqual(self.flushed, [])
        self.assertEqual(batcher.pending, ["a", "b"])

    def test_flushes_when_full_and_returns_the_callback_result(self):
        batcher = m.Batcher(2, self.record)
        batcher.add("a")
        self.assertEqual(batcher.add("b"), 2)
        self.assertEqual(self.flushed, [["a", "b"]])
        self.assertEqual(batcher.pending, [])

    def test_counts_batches(self):
        batcher = m.Batcher(2, self.record)
        for item in "abcd":
            batcher.add(item)
        self.assertEqual(batcher.batches, 2)
        self.assertEqual(self.flushed, [["a", "b"], ["c", "d"]])

    def test_manual_flush_of_a_partial_batch(self):
        batcher = m.Batcher(5, self.record)
        batcher.add("a")
        self.assertEqual(batcher.flush(), 1)
        self.assertEqual(self.flushed, [["a"]])
        self.assertEqual(batcher.pending, [])

    def test_flushing_nothing_is_a_no_op(self):
        batcher = m.Batcher(5, self.record)
        self.assertIsNone(batcher.flush())
        self.assertEqual(self.flushed, [])
        self.assertEqual(batcher.batches, 0)

    def test_size_of_one_flushes_every_add(self):
        batcher = m.Batcher(1, self.record)
        batcher.add("a")
        batcher.add("b")
        self.assertEqual(self.flushed, [["a"], ["b"]])

    def test_rejects_a_non_positive_size(self):
        with self.assertRaises(ValueError):
            m.Batcher(0, self.record)

    def test_the_flushed_batch_is_detached_from_pending(self):
        """Handing the callback the live list and clearing it loses the batch."""
        batcher = m.Batcher(2, self.record)
        batcher.add("a")
        batcher.add("b")
        batcher.add("c")
        self.assertEqual(self.flushed[0], ["a", "b"])


class TestGilExplanation(unittest.TestCase):
    def test_mentions_the_gil_and_processes(self):
        answer = m.cpu_bound_is_not_helped_by_threads().lower()
        self.assertIn("gil", answer)
        self.assertIn("process", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
