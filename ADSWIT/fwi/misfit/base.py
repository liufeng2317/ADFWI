from abc import abstractmethod

class Misfit():
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def forward(self):
        pass