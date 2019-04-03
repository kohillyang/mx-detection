import Queue
import threading


def _worker(receive_queue, feed_queue, dataset):
    # type: (Queue.Queue, Queue.Queue, Any) -> Any
    while True:
        if feed_queue.empty():
            break
        idx = feed_queue.get()
        if idx is None:
            return
        d = dataset[idx]
        receive_queue.put(d)


class _MultiThreadIter(object):
    def __init__(self, dataset, batchsize, nthread=8, shuffle=True, last_batch="discard"):
        self.dataset = dataset
        self.batchsize = batchsize
        self.nthread = nthread
        assert last_batch == "discard"
        self._feed_queue = Queue.Queue(maxsize=0)
        self._receive_queue = Queue.Queue(maxsize=16)

        self._batch_idx = 0
        if shuffle:
            self._sample = list(range(len(dataset)))
            import random
            random.shuffle(self._sample)
            self._sample = self._sample[:(len(self)*self.batchsize)]
        else:
            self._sample = list(range(len(self) * batchsize))
        for sample_idx in self._sample:
            self._feed_queue.put(sample_idx)
        self._threads = []
        for i in range(nthread):
            t = threading.Thread(target=_worker, args=(self._receive_queue, self._feed_queue, self.dataset))
            t.setDaemon(True)
            t.start()

    def __len__(self):
        return len(self.dataset) // self.batchsize

    def __next__(self):
        return self.next()

    def next(self):
        if self._batch_idx >= len(self):
            for _ in self._threads:
                self._feed_queue.put(None)
            for t in self._threads:
                t.join()
            raise StopIteration
        self._batch_idx += 1
        r = [self._receive_queue.get() for _ in range(self.batchsize)]
        return r


class DataLoader(object):
    def __init__(self, dataset, batchsize, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.batchsize = batchsize

    def __iter__(self):
        return _MultiThreadIter(self.dataset, self.batchsize, **self.kwargs)

    def next(self):
        return self.q.get(block=True)

    def __len__(self):
        return len(self.dataset) // self.batchsize
