import torch


class Fill:
    def fill(self, path_coords, insert_count, weights, transition_data):
        """
        Interface method for fill methods.

        Parameters
        ----------
        path_coords : torch.FloatTensor
            The coordinates of the pivots.
        insert_count : int
            How many pivots to insert.
        weights : torch.FloatTensor
            The relative distances between existing pivots.
        transition_data : dict
            Further metrics of the existing paths, such as the sub-sampled loss.

        Returns
        -------
        torch.FloatTensor, torch.FloatTensor
            The new coordinates and new relative target distances.
        """
        raise NotImplementedError
