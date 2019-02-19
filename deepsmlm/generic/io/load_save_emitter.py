from abc import ABC, abstractmethod  # abstract class
import numpy as np
import torch
import scipy.io as sio

from deepsmlm.generic.emitter import EmitterSet


"""
The general interface is to provide the following variables:
    xyz             matrix of size N x 3 or N x 2 in 2D
    phot            matrix of size N 
    frame_ix        matrix of size N (integer, starting with 0 for pythonic world, matlabian world will be -= 1)
    extent          tuple of tuples. extent = ((xmin, xmax), (ymin, ymax), (zmin zmax)). 
                    Pixel / Voxelsize will be determined by this specification
    
    frames          matrix / tensor of size N x 1 x H x W
"""


class BinaryInterface(ABC):
    """
    Abstract class to specify binary interfaces
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_binary(self):
        raise NotImplementedError

    @abstractmethod
    def save_binary(self):
        raise NotImplementedError


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


"""---------------------------------------------- Interface Definitions ---------------------------------------------"""


class MatlabInterface(BinaryInterface):

    def __init__(self,
                 xyz_key='xyz', phot_key='phot', fix_key='frame_ix',
                 extent_key='extent', frame_key='frames', id_key='id'):
        super().__init__()

        self.xyz_key = xyz_key
        self.phot_key = phot_key
        self.fix_key = fix_key
        self.id_key = id_key

        self.extent_key = extent_key
        self.frame_key = frame_key

    def load_binary(self, mat_file):
        bin = sio.loadmat(mat_file)

        emitter_set = EmitterSet(xyz=torch.from_numpy(bin[self.xyz_key]),
                                 phot=torch.from_numpy(bin[self.phot_key]).squeeze(),
                                 frame_ix=torch.from_numpy(bin[self.fix_key] - 1).squeeze(),
                                 id=torch.from_numpy(bin[self.id_key]).squeeze())

        extent = totuple(bin[self.extent_key])
        frames = torch.from_numpy(bin[self.frame_key].astype(np.int64)).type(torch.FloatTensor)
        """
        Note that we need to transpose the input coming from matlab because there we don't have the issue with regard 
        to coordinate axis order and image axis. 
        """
        frames = frames.transpose(-1, -2)

        return emitter_set, extent, frames

    def save_binary(self, emitter_set, mat_file):
        raise NotImplementedError


class NumpyInterface(BinaryInterface):

    def __init__(self,
                 xyz_key='xyz', phot_key='phot', fix_key='frame_ix',
                 extent_key='extent', frame_key='frames', id_key='id'):
        super().__init__()

        self.xyz_key = xyz_key
        self.phot_key = phot_key
        self.fix_key = fix_key
        self.id_key = id_key

        self.extent_key = extent_key
        self.frame_key = frame_key

    def load_binary(self, mat_file):
        bin = np.load(mat_file)

        emitter_set = EmitterSet(xyz=torch.from_numpy(bin[self.xyz_key]),
                                 phot=torch.from_numpy(bin[self.phot_key]),
                                 frame_ix=torch.from_numpy(bin[self.fix_key]),
                                 id=torch.from_numpy(bin[self.id_key]))

        extent = totuple(bin[self.extent_key])
        frames = torch.from_numpy(bin[self.frame_key])

        return emitter_set, extent, frames

    def save_binary(self, emitter_set, mat_file):
        raise NotImplementedError