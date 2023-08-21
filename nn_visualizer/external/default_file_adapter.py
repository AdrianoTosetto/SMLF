from __future__ import annotations

from multipledispatch import dispatch

from nn_visualizer.infra.file_manager.file_manager import File


class DefaultFileAdapter(File):
    def __init__(self, path: str = None,  open_policy: File.OpenPolicy = File.OpenPolicy.write) -> None:
        File.__init__(self)
        self.path = path
        self.policy = open_policy
        self.content_str = None
        self.adaptee = None

    def open(self) -> int:
        self.adaptee = open(self.path, 'r+')
        self.content_str = self.adaptee.read()

        return 0

    def close(self) -> int:
        self.adaptee.close()
        self.content_str = None

        return 0

    def read(self) -> str:
        return self.content_str

    @dispatch(str)
    def write(self, content: str) -> int:
        if self.policy == File.OpenPolicy.read:
            raise Exception('Can\'t write in read mode')

        new_content = content

        if self.policy == File.OpenPolicy.write:
            return self.adaptee.write(new_content)
        else:
            return self.adaptee.write(self.content_str + new_content)

    @dispatch(bytearray)
    def write(self, content: bytearray) -> None:
        pass

    def get_file_name(self) -> str:
        return self.path.split("/")[-1]

    @classmethod
    def open_from_file_name(cls, path: str, open_policy: File.OpenPolicy = File.OpenPolicy.write) -> DefaultFileAdapter:
        file = cls(path, open_policy)

        return file
