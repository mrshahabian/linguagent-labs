"""Reference solution for the Package Registry mock (Levels 1-5)."""


class Registry:
    """An in-memory package registry with TTLs, snapshots and download counts.

    State:
        _packages  name -> (size, start, expiry_or_None)
        _backups   timestamp -> {name: (size, remaining_or_None)}
        _downloads name -> count  (attached to the NAME, survives everything)
    """

    def __init__(self):
        self._packages = {}
        self._backups = {}
        self._downloads = {}

    # ---------------------------------------------------------- internals

    def _alive_size(self, timestamp, name):
        """Size if `name` is alive at `timestamp`, else None."""
        entry = self._packages.get(name)
        if entry is None:
            return None
        size, start, expiry = entry
        if timestamp < start:
            return None
        if expiry is not None and timestamp >= expiry:
            return None
        return size

    def _alive_items(self, timestamp):
        for name in self._packages:
            size = self._alive_size(timestamp, name)
            if size is not None:
                yield name, size

    @staticmethod
    def _format(pairs, n):
        ordered = sorted(pairs, key=lambda kv: (-kv[1], kv[0]))
        return [f"{name}({value})" for name, value in ordered[:n]]

    # ------------------------------------------------- Levels 1-2 (t = 0)

    def publish(self, name, size):
        return self.publish_at(0, name, size)

    def get(self, name):
        return self.get_at(0, name)

    def delete(self, name):
        return self.delete_at(0, name)

    def search(self, prefix):
        return self.search_at(0, prefix)

    def total_size(self):
        return sum(size for _, size in self._alive_items(0))

    # ------------------------------------------------- Level 3 (time)

    def publish_at(self, timestamp, name, size, ttl=None):
        was_alive = self._alive_size(timestamp, name) is not None
        expiry = None if ttl is None else timestamp + ttl
        self._packages[name] = (size, timestamp, expiry)
        return not was_alive

    def get_at(self, timestamp, name):
        return self._alive_size(timestamp, name)

    def delete_at(self, timestamp, name):
        if self._alive_size(timestamp, name) is None:
            return False
        del self._packages[name]
        return True

    def search_at(self, timestamp, prefix):
        matches = [
            (name, size)
            for name, size in self._alive_items(timestamp)
            if name.startswith(prefix)
        ]
        return self._format(matches, 5)

    # ------------------------------------------------- Level 4 (snapshots)

    def backup(self, timestamp):
        snapshot = {}
        for name, size in self._alive_items(timestamp):
            _, _, expiry = self._packages[name]
            remaining = None if expiry is None else expiry - timestamp
            snapshot[name] = (size, remaining)
        self._backups[timestamp] = snapshot
        return len(snapshot)

    def restore(self, timestamp, timestamp_to_restore):
        snapshot = self._backups[timestamp_to_restore]
        self._packages = {
            name: (
                size,
                timestamp,
                None if remaining is None else timestamp + remaining,
            )
            for name, (size, remaining) in snapshot.items()
        }

    # ------------------------------------------------- Level 5 (downloads)

    def download_at(self, timestamp, name):
        if self._alive_size(timestamp, name) is None:
            return None
        self._downloads[name] = self._downloads.get(name, 0) + 1
        return self._downloads[name]

    def top_downloaded(self, n):
        counted = [(name, c) for name, c in self._downloads.items() if c > 0]
        return self._format(counted, n)
