import torch
import numpy as np
import math
from pathlib import Path
from typing import Tuple, Union

from ..generic import EmitterSet
from . import psf_kernel


class Simulation:
    """
    A simulation class that holds the necessary modules, i.e. an emitter source (either a static EmitterSet or
    a function from which we can sample emitters), a psf background and noise. You may also specify the desired frame
    range, i.e. the indices of the frames you want to have as output. If they are not specified, they are automatically
    determined but may vary with new sampled emittersets.

    Attributes:
        em (EmitterSet): Static EmitterSet
        em_sampler: instance with 'sample()' method to sample EmitterSets from
        frame_range: frame indices between which to compute the frames. If None they will be
        auto-determined by the psf implementation.
        psf: psf model with forward method
        background (Background): background implementation
        noise (Noise): noise implementation
    """

    def __init__(self, psf: psf_kernel.PSF, em_sampler=None, background=None, noise=None,
                 frame_range: Tuple[int, int] = None):
        """
        Init Simulation.

        Args:
            psf: point spread function instance
            em_sampler: callable that returns an EmitterSet upon call
            background: background instance
            noise: noise instance
            frame_range: limit frames to static range
        """

        self.em_sampler = em_sampler
        self.frame_range = frame_range if frame_range is not None else (None, None)

        self.psf = psf
        self.background = background
        self.noise = noise

    def sample(self):
        """
        Sample a new set of emitters and forward them through the simulation pipeline.

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        emitter = self.em_sampler()
        frames, bg = self.forward(emitter)
        return emitter, frames, bg

    def forward(self, em: EmitterSet, ix_low: Union[None, int] = None, ix_high: Union[None, int] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Forward an EmitterSet through the simulation pipeline. 
        Setting ix_low or ix_high overwrites the frame range specified in the init.

        Args:
            em (EmitterSet): Emitter Set
            ix_low: lower frame index
            ix_high: upper frame index (inclusive)

        Returns:
            torch.Tensor: simulated frames
            torch.Tensor: background frames (e.g. to predict the bg seperately)
        """

        if ix_low is None:
            ix_low = self.frame_range[0]

        if ix_high is None:
            ix_high = self.frame_range[1]

        frames = self.psf.forward(em.xyz_px, em.phot, em.frame_ix,
                                  ix_low=ix_low, ix_high=ix_high)

        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, that background is assumed to be independent of the 
        emitter position / signal.
        """
        if self.background is not None:
            frames, bg_frames = self.background.forward(frames)
        else:
            bg_frames = None

        if self.noise is not None:
            frames = self.noise.forward(frames)

        return frames, bg_frames

class CellSimulation:
    """
    A simulation class, that will spit out emitters, frames, and background frames
    where psf of the emitters is overlaid on the frame, and background frame is the 
    cell background. Use the index slicing operations on these 3 items to check
    individual datapoints

    Attributes:

    """

    def __init__(self, psf: psf_kernel.PSF, param=None, noise=None,
                frame_range: Tuple[int, int] = None, 
            ):
        """
        Init Simulation.
        
        Args:
            psf: point spread function instance
            param: parameters RecursiveNamespace object
            noise: noise instance
            frame_range: limit frame to static range
        """
        self.frame_range = frame_range if frame_range is not None else (None, None)
        self.psf = psf
        self.noise = noise

    def sample(self):
        """

        Returns:
            EmitterSet: sampled emitters
            torch.Tensor: simulated frames
            torch.Tensor: background frames
        """

        # these are used to make approriate targets in the dataset 
        # used by the network. This function is called once per epoch
        # see how many frames are needed.
        num_frames = self.frame_range[1] - self.frame_range[0] + 1
        photon_range = param.Simulation.photon_range

        # emitter_av is a number between 1 and 2
        em_avg = param.Simulation.emitter_av
        xy_unit = param.Simulation.xy_unit
        px_size = param.Camera.px_size


        # Load the number of files required to do num_frames images
        cellbg_dir = Path(param.Simulation.cellbg_dir) 
        if (cellbg_dir.exists() == False):
            raise ValueError("Cell background images directory not present ..")
        n_images_per_file = int(param.Simulation.n_bg_images_per_file)
        filenames = sorted(list(cellbg_dir.glob('*' + param.Simulation.bg_images_fileformat)))
        n_available = len(filenames)
        n_req_files = math.ceil(num_frames / n_images_per_file)

        # sample required files from
        file_indices2read = list(np.random.choice(np.arange(n_available), size=n_req_files))

        cell_bg = []
        mask_bg = []
        for file_i in file_indices2read:
            filename2read = filenames[file_i]
            data = np.load(filename2read)
            cell_bg.append(data[param.Simulation.bg_type])
            mask_bg.append(data['mask'])
        cell_bg = np.concatenate(cell_bg, axis=0)[:num_frames]
        cell_bg *= param.Simulation.bg_scale_factor
        mask_bg = np.concatenate(mask_bg, axis=0)[:num_frames]

        # sample dots from the masks to generate emitter sets
        # xyz, phot, frame_ix, id, xy_unit, px_size
        if (em_avg > 2.0  or em_avg < 1.0):
            raise ValueError("Average number of emitters should be between 1.0 and 2.0")
        n_emitters_per_frame = np.random.choice([1, 2], num_frames, p=[(2.0 - em_avg), 1.0 - (2.0 - em_avg)])
        frame_ix = np.repeat(np.arange(num_frames), n_emitters_per_frame)

        n_emitters = np.sum(n_emitters_per_frame) 

        # x, y are pixel coordinates of the top left corner 
        x = np.zeros(n_emitters,)
        y = np.zeros(n_emitters,)
        counter = 0
        for i in range(num_frames):
            p_x, p_y = np.nonzero(mask_bg[i])
            p_i = np.random.randint(low=0, high=len(p_x), size=n_emitters_per_frame[i])
            x[counter: counter+n_emitters_per_frame[i]] = p_x[p_i]
            y[counter: counter+n_emitters_per_frame[i]] = p_y[p_i]
            counter += n_emitters_per_frame[i]
        
        # 
        z_extent = param.Simulation.emitter_extent[2]
        xyz = torch.zeros((n_emitters, 3))
        xyz[:, 0] = torch.from_numpy(x.astype('float32')) 
        xyz[:, 0] += torch.rand((n_emitters,)) - 0.5
        xyz[:, 1] = torch.from_numpy(y.astype('float32'))
        xyz[:, 1] += torch.rand((n_emitters,)) - 0.5
        xyz[:, 2] = (z_extent[1] - z_extent[0]) * torch.rand((n_emitters,))  + z_extent[0]

        # photon_counts
        photon_counts = torch.randint(*param.Simulation.photon_range, (n_emitters,))
        # frame_iex
        frame_ix = torch.from_numpy(frame_ix).long()

        emitter = EmitterSet(xyz=xyz, phot=photon_counts,
                        frame_ix=frame_ix,
                        id=torch.arange(n_emitters,).long(),
                        xy_unit=xy_unit,
                        px_size=px_size
        )
        frames, bg = self.forward(emitter, cell_bg)

        return emitter, frames, bg
    
    def forward(self, em: EmitterSet, cell_bg: torch.Tensor, ix_low:Union[None, int] = None,
            ix_high: Union[None, int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward an emitter set through the simulation pipeline.
        Setting ix_low or ix_high overwrites the frame range specified in the init.

        Args:
            ix_low: lower frame index
            ix_high: upper frame index (inclusive)
        
        Returns:
            torch.Tensor: simulated frames
            torch.Tensor: background frames (to predict background seperately)
        """

        if ix_low is None:
            ix_low = self.frame_range[0]
        if ix_high is None:
            ix_high = self.frame_range[1]

        # forward the emitters and get the PSF simulation stack

        psf_frames = self.psf.forward(em.xyz_px, em.phot, em.frame_ix, 
                                  ix_low=ix_low, ix_high=ix_high)

        # pass the cell_bg back to the camera to go into photons as you are in ADU
        #cell_bg =  self.noise.backward(cell_bg)

        # place the frames of simulated psf on top of the background

        
        return frames, bg_frames

    