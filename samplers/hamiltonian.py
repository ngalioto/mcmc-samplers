import torch
import delayed_rejection
from proposals import Hamiltonian_Dynamics

class Hamiltonian_Monte_Carlo(Delayed_Rejection):

    def __init__(
            self,
            target
    ):
        super().__init__(
            target = target,
            proposals = [Hamiltonian_Dynamics()]
        )
