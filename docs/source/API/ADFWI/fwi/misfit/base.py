from abc import abstractmethod

class Misfit():
    """
    Abstract base class for Misfit functions.
    
    This class serves as a template for creating specific misfit functions used in inversion problems.
    """

    def __init__(self) -> None:
        """
        Initializes the Misfit class.
        
        This constructor does not require any parameters and is designed to be inherited by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self):
        """
        Abstract method for computing the forward misfit.

        This method must be implemented by subclasses to compute the forward misfit based on specific data and models.

        Returns
        -------
        result : float
            The computed misfit value.
        """
        pass
