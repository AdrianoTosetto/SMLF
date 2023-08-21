from abc import abstractmethod
from typing import Callable


class Event():
    def __init__(self, payload: dict) -> None:
        self.payload = payload

ObserverHandler = Callable[[Event], None]

class Observable():
    def __init__(self):
        self.observers = []

    def add(self, observer: ObserverHandler):
        self.observers.append(observer)

    def remove(self, observer: ObserverHandler):
        self.observers.remove(observer)

    def notify(self, event: Event):
        for handler in self.observers:
            handler(event)
