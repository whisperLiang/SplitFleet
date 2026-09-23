"""Group client RPCs for batched server-model execution."""

import asyncio


class ClientRequestGroup:
    """Container class that contains all the multiple requests from one or more clients.
    A ClientRequestGroup is a collection of client requests that:
    1. are meant to be processed by the same server model
    2. trigger the same method on the server model
    Thus, a ClientRequestGroup is a collection of requests that are meant to be processed together.
    """
    def __init__(self, sid):
        self.sid = sid
        self.batches = []
        self.events = []
        self.cids = []
        self._is_ready = asyncio.Event()
        self._initializing_state = True

    def add(self, batch_data, event, cid):
        self.batches.append(batch_data)
        self.events.append(event)
        self.cids.append(cid)

    def mark_as_ready(self):
        self._is_ready.set()

    def is_ready(self):
        return self._is_ready.is_set()

    def get_num_batches(self):
        return len(self.batches)

    def is_new(self):
        state = self._initializing_state
        self._initializing_state = False
        return state

    def get_data(self):
        # sort by cid to get consistent order
        batches = [x for _, x in sorted(zip(self.cids, self.batches), key=lambda pair: pair[0])]
        events = [x for _, x in sorted(zip(self.cids, self.events), key=lambda pair: pair[0])]
        return batches, events
