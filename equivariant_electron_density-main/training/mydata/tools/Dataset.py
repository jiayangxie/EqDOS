from torch.utils.data import Dataset
import torch
import numpy as np

class Dataset(Dataset):
    def __init__(self, props):
        self.props = props
        for key, value in self.props.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        result = {}
        for key in self.__dict__:
            if isinstance(getattr(self, key), list) and len(getattr(self, key)) > index:
                result[key] = getattr(self, key)[index]
        return result

    def __len__(self):
        return len(self.pos)

    def save(self, path):
        torch.save(self.props, path)

    def generate_neighbor_list(self,
                               cutoff,
                               undirected=True,
                               key='nbr_list',
                               offset_key='offsets'):
        """Generates a neighbor list for each one of the atoms in the dataset.
            By default, does not consider periodic boundary conditions.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
            undirected (bool, optional): Description

        Returns:
            TYPE: Description
        """
        if 'lattice' not in self.props:
            self.props[key] = [self.get_neighbor_list(nxyz[:, 1:4], cutoff, undirected) for nxyz in self.props['nxyz']]
            self.props[offset_key] = [
                torch.sparse.FloatTensor(nbrlist.shape[0], 3)
                for nbrlist in self.props[key]
            ]
        else:
            self._get_periodic_neighbor_list(cutoff=cutoff,
                                             undirected=undirected,
                                             offset_key=offset_key,
                                             nbr_key=key)
            return self.props[key], self.props[offset_key]

        return self.props[key]

