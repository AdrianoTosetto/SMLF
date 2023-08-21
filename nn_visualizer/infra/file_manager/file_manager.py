from __future__ import annotations
from enum import Enum

from multipledispatch import dispatch


class File():

    class OpenPolicy(Enum):
        read = 0
        append = 1
        write = 2

    def __init__(self) -> None:
        pass

    def get_file_name(self) -> str:
        pass

    @classmethod
    def open_from_file_name(cls, path: str, open_policy: OpenPolicy = OpenPolicy.read) -> File:
        pass

    def open(self) -> int:
        pass

    def close(self) -> int:
        pass

    def read(self) -> str:
        pass

    @dispatch(str)
    def write(self, content: str) -> None:
        pass

    @dispatch(bytearray)
    def write(self, content: bytearray) -> None:
        pass

class FileManager():
    def __init__(self):
        pass

    def open_file(self) -> File:
        pass

    def open_string_file(self):
        pass
