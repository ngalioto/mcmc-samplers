import torch
from proposals import Proposal

class Hamiltonian_Dynamics(Proposal):
    '''
    Put this on hold
    '''
    
    def __init__(
            self
    ):
        # number of steps
        # step size
        super().__init__()

    @property
    def is_symmetric(
            self
    ):
        return False

    def _leapfrog(
            self
    ):
        pass

    def sample(
            self
    ):
        # sample momentum
        # run leapfrog integrator
            # position.log_prob = self.target(position)
            # grad = - position.log_prob.backward() ? 
        # save the momentum
        # return the position
        pass

    def log_prob(
            self
    ):
        # return kinetic energy
        pass
