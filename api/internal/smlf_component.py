from abc import ABC, abstractmethod


'''
    every smlf component inherits from this class for implementing methods:
    save state:
        for layers, saves weights and biases
        for other components saves internal configurations
'''

class SMLFComponent:
    def __init__(self):
        pass

    @abstractmethod
    def name(self): str

    @abstractmethod
    def component_name(self) -> str:
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def to_string_json(self):
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_obj: dict):
        pass

    @classmethod
    @abstractmethod
    def from_string_json(cls, json_str: str):
        pass
