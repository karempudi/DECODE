import configparser
import itertools as iter
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 4 * plt.rcParams["figure.dpi"]
import functools
import pandas as pd
import sys
import torch
import warnings
from multiprocessing.dummy import Pool as ThreadPool
from missingFromNumpy import repeat_np, splitbatchandrunfunc


class Simulation:
    """
    A class representing a smlm simulation

    an image is represented according to pytorch convention, i.e.
    (N, C, H, W) in 2D - batched
    (C, H, W) in 2D - single image
    (N, C, D, H, W) in 3D - batched
    (C, D, H, W) in 2D - single image
    (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d)
    """
    def __init__(self, em_mat=None, cont_mat=None, img_size=(64, 64), sigma=1.5, upscale=1,
                 bg_value=10, background=None, psf=None, psf_hr=None, performance=None, poolsize=4,
                 use_cuda=True):
        """
        Initialise a simulation
        :param emitter_mat: matrix of emitters. N x 6. 0-2 col: xyz, 3 photon, 4: frameix, 5 emit id
        :param img_size: tuple of image dimension
        :param img_size_hr: tuple of high_resolution image dimension (i.e. image with delta function psf)
        :param background: function to generate background, comprises a fct for pre capture bg and post bg
                            input: image
        :param psf: psf function
        :param psf_hr: psf function for high resolution image (e.g. delta function psf)
        :param use_cuda: perform calculations as much as possible in cuda. support is rudimentary atm.

        :param performance: function to evaluate performance of prediction

        class own
        :param predictions: array of instances of predicted emitters
        :param image: camera image
        """

        self.emitter_mat = em_mat
        self.contaminator_mat = cont_mat

        self.img_size = img_size
        self.upscale = upscale
        self.img_size_hr = (self.img_size[0] * self.upscale, self.img_size[1] * self.upscale)
        self.image = None
        self.image_hr = None

        self.predictions = None

        self.sigma = sigma
        self.bg_value = bg_value
        self.psf = psf
        self.psf_hr = psf_hr
        self.background = background
        self.performance = performance
        self.poolsize = poolsize
        # use cuda only if it's really possible, i.e. a cuda device is detected
        self.use_cuda = use_cuda if torch.cuda.is_available() else False

        # use standard psf's if all psf functions are none
        if not any([self.psf, self.psf_hr, self.background]):
            self.init_standard_psf()

    @property
    def num_emitters(self):
        return self.emitter_mat.shape[0]

    @property
    def num_frames(self):
        em_matrix = self.get_emitter_matrix('all')
        return int(torch.max(em_matrix[:, 4]) + 1)

    @property
    def simulation_extent(self):
        return self.get_extent(self.img_size)

    @property
    def simulation_extent_hr(self):
        return self.get_extent(self.img_size_hr)

    def init_standard_psf(self):
        self.background = lambda img: noise_psf(img, bg_poisson=self.bg_value)
        self.psf = lambda pos, phot: gaussian_expect(pos, self.sigma, phot, img_shape=self.img_size)
        self.psf_hr = lambda pos, phot: delta_psf(pos, phot, img_shape=self.img_size_hr,
                                                  xextent=torch.tensor([0, self.img_size[0]], dtype=torch.float),
                                                  yextent=torch.tensor([0, self.img_size[1]], dtype=torch.float))

    def camera_image(self, psf=None, bg=True, postupscaling=False, emitter_kind='all'):
        # get emitter matrix
        em_mat = self.get_emitter_matrix(kind=emitter_kind)

        pool = ThreadPool(self.poolsize)
        frame_list = pool.starmap(self.get_frame_wrapper, zip(list(range(self.num_frames)),
                                                              iter.repeat(em_mat),
                                                              iter.repeat(psf),
                                                              iter.repeat(bg),
                                                              iter.repeat(postupscaling)))
        img = torch.stack(frame_list, dim=0).type(torch.int16)
        if postupscaling:
            return splitbatchandrunfunc(img, upscale, (self.upscale, (2, 3)),
                                        batch_size_target=10000, to_cuda=self.use_cuda)
        else:
            return img

    def directed_distance(em_mat, px_positions, image_shape):
        em_mat = torch.from_numpy(em_mat)
        image_shape = torch.from_numpy(image_shape)
        # get index of emitter which is closest, remember the order
        ix_closest_emitter = pairwise_distances(px_positions, em_mat).min(1)[1]

        # place cordinates of closest emitter
        directed_distance = (em_mat[ix_closest_emitter, :] - px_positions).view(em_mat.shape[1], *image_shape)

        return directed_distance

    def get_frame_wrapper(self, ix, em_mat, psf, bg=True, postupscaling=False):
        em_in_frame = em_mat[em_mat[:, 4] == ix, :]
        return self.get_frame(em_in_frame[:, :3], em_in_frame[:, 3], psf=psf, bg=bg)

    def get_frame(self, pos, phot, psf, bg=True):
        if psf is None:
            psf = self.psf

        frame = psf(pos, phot)
        if bg:
            frame = self.background(frame)
        return frame

    def em_mat_frame(self, ix, em_mat=None):
        if em_mat is None:
            em_mat = self.emitter_mat
        return em_mat[em_mat[:, 4] == ix, :]

    @functools.lru_cache()
    def get_emitter_matrix(self, kind='emitter'):
        """
        :param emitters:
        :return: Matrix number of emitters x 5. (x,y,z, photon count, frameix)
        """
        if kind == 'all':
            return torch.cat([self.emitter_mat, self.contaminator_mat], dim=0)
        elif kind == 'emitter':
            return self.emitter_mat
        elif kind == 'contaminator':
            return self.contaminator_mat
        else:
            raise ValueError('Not supported kind.')

    def get_emitter_matrix_frame(self, ix, kind=None):
        em_mat = self.get_emitter_matrix(kind=kind)
        return em_mat[em_mat[:, 4] == ix, :4]

    @functools.lru_cache()
    def get_extent(self, img_size):
        return [-0.5, img_size[0]/self.upscale - 0.5, img_size[1]/self.upscale - 0.5, -0.5]

    def plot_frame(self, ix, image=None, crosses=True):
        if image is None:
            image = self.image

        plt.imshow(image[ix, 0, :, :].numpy(), cmap='gray', extent=self.get_extent(image.shape[2:]))

        if crosses is True:
            ground_truth = self.get_emitter_matrix_frame(ix, kind='emitter')
            plt.plot(ground_truth[:, 0].numpy(), ground_truth[:, 1].numpy(), 'rx')

        if self.predictions is not None:
            raise NotImplementedError()
            plt.plot(localisation[:, 0].numpy(), localisation[:, 1].numpy(), 'bo')

    def run_simulation(self, plot_sample=False):
        """
        Frame will be in image_size, upscale afterwards so that it looks like image_size
        where it's actually in image_size_hr. For the case of finer target, upscaling is part of the psf, since the original
        output of the 'delta shaped psf' must be in high resolution already.
        """
        self.image = self.camera_image(postupscaling=True)
        self.image_hr = self.camera_image(psf=self.psf_hr, bg=False, postupscaling=False, emitter_kind='emitter')
        print("Generating {} samples done.".format(self.num_frames))

        if plot_sample:
            ix = np.random.randint(1, self.num_frames-2)
            print('You see frame {} {} {}'.format(ix - 1, ix, ix + 1))

            plt.subplot(321)
            self.plot_frame(ix-1, self.image, True)

            plt.subplot(322)
            self.plot_frame(ix-1, self.image_hr, False)

            plt.subplot(323)
            self.plot_frame(ix, self.image, True)

            plt.subplot(324)
            self.plot_frame(ix, self.image_hr, False)

            plt.subplot(325)
            self.plot_frame(ix+1, self.image, True)

            plt.subplot(326)
            self.plot_frame(ix+1, self.image_hr, False)

            plt.show()

    def write_to_binary(self, outfile):
        np.savez_compressed(outfile, frames=self.image.numpy(), frames_hr=self.image_hr.numpy(),
                            emitters=self.get_emitter_matrix().numpy(), len=self.num_frames,
                            extent=self.simulation_extent)
        print("Saving simulation to {}.".format(outfile))


class Args:
    """
    Simple class to init from configuration file
    """
    def __init__(self, ini_file=None):
        self.ini_file = ini_file
        self.config = configparser.ConfigParser()

        self.binary_path = None
        self.positions_csv = None
        self.image_size = None
        self.upscale_factor = None
        self.emitter_p_frame = None
        self.total_frames = None
        self.bg_value = None
        self.sigma = None
        self.use_cuda = None
        self.dimension = None

    def parse(self, section='DEFAULT', from_variable=False):
        if not from_variable:
            self.config.read(self.ini_file)

        self.binary_path = self.config[section]['binary_path']
        self.positions_csv = None if self.config[section]['positions_csv'] is 'None' else self.config[section]['positions_csv']
        self.image_size = eval(self.config[section]['image_size'])
        self.upscale_factor = int(self.config[section]['upscale_factor'])
        self.emitter_p_frame = int(self.config[section]['emitter_p_frame'])
        self.total_frames = int(self.config[section]['total_frames'])
        self.bg_value = int(self.config[section]['bg_value'])
        self.sigma = torch.tensor([float(self.config[section]['sigma']),
                              float(self.config[section]['sigma'])])
        self.poolsize = int(self.config[section]['poolsize'])
        self.use_cuda = eval(self.config[section]['use_cuda'])
        self.dimension = int(self.config[section]['dimension'])


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):  # not numerically stable but fast
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def expanded_pairwise_distances(x, y=None):  # numerically stable but slow
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances


def upscale(img, scale=1, img_dims=(2, 3)):  # either 2D or 3D batch of 2D
    return repeat_np(repeat_np(img, scale, img_dims[0]), scale, img_dims[1])


def dist_phot_lifetime(start_frame, lifetime, photon_count):
    # return photon count per frame index
    raise RuntimeError("Not yet supported.")


def emitters_from_csv(csv_file, img_size, cont_radius=3):
    emitters_matlab = pd.read_csv(csv_file)

    emitters_matlab = torch.from_numpy(emitters_matlab.iloc[:, :].as_matrix()).type(torch.float32)
    em_mat = torch.cat((emitters_matlab[:, 2:5] * (img_size[0] + 2 * cont_radius) - cont_radius,  # transform from 0, 1
                        emitters_matlab[:, 5:6],
                        emitters_matlab[:, 1:2] - 1,  # index shift from matlab to python
                        torch.zeros_like(emitters_matlab[:, 0:1])), dim=1)

    warnings.warn('Emitter ID not implemented yet.')

    return split_emitter_cont(em_mat, img_size)


def split_emitter_cont(em_mat, img_size):
    """

    :param em_mat: matrix of all emitters
    :param img_size: img_size in px (not upscaled)
    :return: emitter_matrix and contaminator matrix. contaminators are emitters which are outside the image
    """

    if img_size[0] != img_size[1]:
        raise NotImplementedError("Image must be square at the moment because otherwise the following doesn't work.")

    is_emit = torch.mul((em_mat[:, :2] >= 0).all(1), (em_mat[:, :2] <= img_size[0] - 1).all(1))
    is_cont = ~is_emit

    emit_mat, cont_mat = em_mat[is_emit, :], em_mat[is_cont, :]
    return emit_mat, cont_mat


def random_emitters(emitter_per_frame, frames, lifetime, img_size, cont_radius=3, z_sigma=3):
    """

    :param emitter_per_frame:
    :param frames:
    :param lifetime:
    :param img_size:
    :param cont_radius:
    :return: emitter-matrix (em) (N x 6), N number of emitters in frames.
             em[:, 0:3] = x,y,z
             em[:, 3] = photon count
             em[:, 4] = start frame
             em[:, 5] = emit_id
    """

    if lifetime is None:  # assume 1 frame emitters
        num_emitters = emitter_per_frame * frames
        positions = torch.rand(num_emitters, 3) * (img_size[0] + 2 * cont_radius) - cont_radius
        if z_sigma != 0:
            positions[:, 2] = torch.randn_like(positions[:, 2]) * z_sigma
        else:
            positions[:, 2] = 0.

        start_frame = torch.randint(0, frames, (num_emitters, 1))  # start on state is distributed uniformly
        lifetime_per_emitter = 1

        emit_id =  torch.arange(0, num_emitters).unsqueeze(0).transpose(0, 1)
        photon_count = torch.randint(800, 4000, (num_emitters, 1))
    else:  # prototype
        raise NotImplementedError
        num_emitters = torch.round(emitter_per_frame * frames / lifetime)  # roughly
        positions = torch.uniform(num_emitters, 3) * (img_size[0] + 2 * cont_radius) - cont_radius  # place emitters entirely randomly
        # start_frame = np.random.uniform(0, frames, (num_emitters, 1))  # start on state is distributed uniformly
        lifetime_per_emitter = torch.zeros(num_emitters).exponential(lifetime)

    emit_all = torch.cat((positions, photon_count.float(), start_frame.float(), emit_id.float()), 1)

    return split_emitter_cont(emit_all, img_size)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # no .ini file specified
        ini_file = 'SimulationDefault.ini'
        section = 'DEFAULT'
    else:
        ini_file = sys.argv[1]
        section = sys.argv[2]

    args = Args(ini_file)
    """
    Alternatively you can specify here directly and call .parse(from_variable=True)

    args.config['DirectConfiguration'] = {'binary_path': 'data/temp.npz',
                                          'positions_csv' : 'None',
                                          'image_size': '(32, 32)',
                                          'upscale_factor': '8',
                                          'emitter_p_frame': '15',
                                          'total_frames': '5',
                                          'bg_value': '10',
                                          'sigma': '1.5',
                                          'poolsize': '10',
                                          'use_cuda': 'True',
                                          'dimension': '2'}
    """

    args.parse(section=section)
    args.config['DirectConfiguration'] = {'binary_path': 'data/temp.npz',
                                          'positions_csv': 'None',
                                          'image_size': '(16, 16)',
                                          'upscale_factor': '1',
                                          'emitter_p_frame': '10',
                                          'total_frames': '10',
                                          'bg_value': '10',
                                          'sigma': '1.5',
                                          'poolsize': '10',
                                          'use_cuda': 'True',
                                          'dimension': '2'}
    args.parse(section='DirectConfiguration', from_variable=True)

    if args.positions_csv is None:
        if args.dimension == 2:
            z_sigma = 0
        else:
            z_sigma = 3

        emit, cont = random_emitters(args.emitter_p_frame, args.total_frames, None, args.image_size, z_sigma=z_sigma)
    else:
        emit, cont = emitters_from_csv(args.positions_csv, args.image_size)

    sim = Simulation(emit, cont,
                     img_size=args.image_size,
                     sigma=args.sigma,
                     upscale=args.upscale_factor,
                     bg_value=args.bg_value,
                     background=None,
                     psf=None,
                     psf_hr=None,
                     poolsize=args.poolsize,
                     use_cuda=args.use_cuda)

    sim.run_simulation(plot_sample=True)
    sim.write_to_binary(args.binary_path)