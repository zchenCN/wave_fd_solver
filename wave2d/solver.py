"""
Finite difference solver for 2d wave equations
with perfectly matched layer

@author: zchen
@date: 20220-12-20
"""

import numpy as np

def ricker(dt, nt, peak_time, dominant_freq):
    """Ricker wavelet with specific dominant frequency"""
    t = np.arange(-peak_time, dt * nt - peak_time, dt, dtype=np.float32)
    w = ((1.0 - 2.0 * (np.pi**2) * (dominant_freq**2) * (t**2))
        * np.exp(-(np.pi**2) * (dominant_freq**2) * (t**2)))
    return w 

class Solver:
    def __init__(self, model, h, dt, nt, peak_time, dominant_freq,
            sources_xz, receivers_xz, pml_width=10, pad_width=10):
        
        # Mesh
        self.nptz, self.nptx = model.shape # number of grid points
        self.nz, self.nx = self.nptz - 1, self.nptx - 1
        self.nt = int(nt)
        self.h = np.float64(h)
        self.dt = np.float64(dt)

        # Sources and receivers
        self.sources_xz = sources_xz
        self.receivers_xz = receivers_xz
        self.num_shots = len(sources_xz)

        # CFL
        max_vel = model.max()
        min_vel = model.min()
        cfl = max_vel * dt / h
        assert cfl < 1 
        print(f'CFL number is {cfl}')

        # source time function
        self.peak_time = peak_time 
        self.dominant_freq = dominant_freq
        self.source_time = ricker(self.dt, self.nt, self.peak_time, self.dominant_freq)

        # PML
        self.pml_width = pml_width
        self.pad_width = pad_width

        # pad
        self.total_pad = pml_width + pad_width
        self.nptx_padded = self.nptx + 2 * self.total_pad 
        self.nptz_padded = self.nptz + 2 * self.total_pad
        self.model_padded = np.pad(model, ((self.total_pad, self.total_pad), (self.total_pad, self.total_pad)), 'edge')

        # dampling factor
        profile = 40.0 + 60.0 * np.arange(pml_width, dtype=np.float64)

        self.sigma_x = np.zeros(self.nptx_padded, np.float64)
        self.sigma_x[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.sigma_x[-self.total_pad:-self.pad_width] = profile
        self.sigma_x[:self.pad_width] = self.sigma_x[self.pad_width]
        self.sigma_x[-self.pad_width:] = self.sigma_x[-self.pad_width-1]
        self.sigma_x = np.tile(self.sigma_x, (self.nptz_padded, 1))

        self.sigma_z = np.zeros(self.nptz_padded, np.float64)
        self.sigma_z[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.sigma_z[-self.total_pad:-self.pad_width] = profile
        self.sigma_z[:pad_width] = self.sigma_z[pad_width]
        self.sigma_z[-self.pad_width:] = self.sigma_z[-self.pad_width-1]
        self.sigma_z = np.tile(self.sigma_z.reshape(-1, 1), (1, self.nptx_padded))


        # wavefield
        self.prev_wavefield = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64) # previous wavefield
        self.cur_wavefield = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64) # current wavefield

        # auxiliary function
        self.cur_psi = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        self.cur_phi = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64) 

    def first_x_deriv(self, f):
        """First derivative of f with respect to x"""
        fx = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        fx[:, :, self.pad_width:-self.pad_width] = (
                                            5 * f[:, :, self.pad_width-6:-self.pad_width-6]
                                          - 72 * f[:, :, self.pad_width-5:-self.pad_width-5]
                                          + 495 * f[:, :, self.pad_width-4:-self.pad_width-4]
                                          - 2200 * f[:, :, self.pad_width-3:-self.pad_width-3]
                                          + 7425 * f[:, :, self.pad_width-2:-self.pad_width-2]
                                          - 23760 * f[:, :, self.pad_width-1:-self.pad_width-1]
                                          + 23760 * f[:, :, self.pad_width+1:-self.pad_width+1]
                                          - 7425 * f[:, :, self.pad_width+2:-self.pad_width+2]
                                          + 2200 * f[:, :, self.pad_width+3:-self.pad_width+3]
                                          - 495 * f[:, :, self.pad_width+4:-self.pad_width+4]
                                          + 72 * f[:, :, self.pad_width+5:-self.pad_width+5]
                                          - 5 * f[:, :, self.pad_width+6:-self.pad_width+6]) / (27720*self.h)
        return fx

    def first_z_deriv(self, f):
        """First derivative of f with respect to z"""
        fz = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        fz[:, self.pad_width:-self.pad_width, :] = (
                                            5 * f[:, self.pad_width-6:-self.pad_width-6, :]
                                          - 72 * f[:, self.pad_width-5:-self.pad_width-5, :]
                                          + 495 * f[:, self.pad_width-4:-self.pad_width-4, :]
                                          - 2200 * f[:, self.pad_width-3:-self.pad_width-3, :]
                                          + 7425 * f[:, self.pad_width-2:-self.pad_width-2, :]
                                          - 23760 * f[:, self.pad_width-1:-self.pad_width-1, :]
                                          + 23760 * f[:, self.pad_width+1:-self.pad_width+1, :]
                                          - 7425 * f[:, self.pad_width+2:-self.pad_width+2, :]
                                          + 2200 * f[:, self.pad_width+3:-self.pad_width+3, :]
                                          - 495 * f[:, self.pad_width+4:-self.pad_width+4, :]
                                          + 72 * f[:, self.pad_width+5:-self.pad_width+5, :]
                                          - 5 * f[:, self.pad_width+6:-self.pad_width+6, :]) / (27720*self.h)
        return fz



    def second_x_deriv(self, f):
        """Second derivative of f with respect to x"""
        fxx = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        fxx[:, :, self.pad_width:-self.pad_width] = (
                                          - 735 * f[:, :, self.pad_width-8:-self.pad_width-8]
                                          + 15360 * f[:, :, self.pad_width-7:-self.pad_width-7]
                                          -  156800 * f[:, :, self.pad_width-6:-self.pad_width-6]
                                          + 1053696 * f[:, :, self.pad_width-5:-self.pad_width-5]
                                          - 5350800 * f[:, :, self.pad_width-4:-self.pad_width-4]
                                          + 22830080 * f[:, :, self.pad_width-3:-self.pad_width-3]
                                          - 94174080 * f[:, :, self.pad_width-2:-self.pad_width-2]
                                          + 538137600 * f[:, :, self.pad_width-1:-self.pad_width-1]
                                          - 924708642 * f[:, :, self.pad_width:-self.pad_width]
                                          + 538137600 * f[:, :, self.pad_width+1:-self.pad_width+1]
                                          - 94174080 * f[:, :, self.pad_width+2:-self.pad_width+2]
                                          + 22830080 * f[:, :, self.pad_width+3:-self.pad_width+3]
                                          - 5350800 * f[:, :, self.pad_width+4:-self.pad_width+4]
                                          + 1053696 * f[:, :, self.pad_width+5:-self.pad_width+5]
                                          - 156800 * f[:, :, self.pad_width+6:-self.pad_width+6]
                                          + 15360 * f[:, :, self.pad_width+7:-self.pad_width+7]
                                          - 735 * f[:, :, self.pad_width+8:-self.pad_width+8]
                                          ) / (302702400*self.h**2)
        return fxx

    def second_z_deriv(self, f):
        """Second derivative of f with respect to z"""
        fzz = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        fzz[:, self.pad_width:-self.pad_width, :] = (
                                          - 735 * f[:, self.pad_width-8:-self.pad_width-8, :]
                                          + 15360 * f[:, self.pad_width-7:-self.pad_width-7, :]
                                          -  156800 * f[:, self.pad_width-6:-self.pad_width-6, :]
                                          + 1053696 * f[:, self.pad_width-5:-self.pad_width-5, :]
                                          - 5350800 * f[:, self.pad_width-4:-self.pad_width-4, :]
                                          + 22830080 * f[:, self.pad_width-3:-self.pad_width-3, :]
                                          - 94174080 * f[:, self.pad_width-2:-self.pad_width-2, :]
                                          + 538137600 * f[:, self.pad_width-1:-self.pad_width-1, :]
                                          - 924708642 * f[:, self.pad_width:-self.pad_width, :]
                                          + 538137600 * f[:, self.pad_width+1:-self.pad_width+1, :]
                                          - 94174080 * f[:, self.pad_width+2:-self.pad_width+2, :]
                                          + 22830080 * f[:, self.pad_width+3:-self.pad_width+3, :]
                                          - 5350800 * f[:, self.pad_width+4:-self.pad_width+4, :]
                                          + 1053696 * f[:, self.pad_width+5:-self.pad_width+5, :]
                                          - 156800 * f[:, self.pad_width+6:-self.pad_width+6, :]
                                          + 15360 * f[:, self.pad_width+7:-self.pad_width+7, :]
                                          - 735 * f[:, self.pad_width+8:-self.pad_width+8, :]
                                          ) / (302702400*self.h**2)
        return fzz

    def laplacian(self, f):
        return self.second_x_deriv(f) + self.second_z_deriv(f)

    def one_step(self, nt):
        nabla_u = self.laplacian(self.cur_wavefield)
        phi_x = self.first_x_deriv(self.cur_phi)
        psi_z = self.first_z_deriv(self.cur_psi)
        ux = self.first_x_deriv(self.cur_wavefield)
        uz = self.first_z_deriv(self.cur_wavefield)

        next_wavefield = self.dt**2 * self.model_padded**2 * (nabla_u + phi_x + psi_z) \
                    - self.dt**2 * self.sigma_x * self.sigma_z * self.cur_wavefield\
                    + (self.dt/2) * (self.sigma_x + self.sigma_z) * self.prev_wavefield\
                    + 2 * self.cur_wavefield - self.prev_wavefield
        
        next_wavefield = next_wavefield / (1 + (self.dt/2) * (self.sigma_x + self.sigma_z))
    
        # add source 
        sx = self.sources_xz[:, 1] + self.total_pad
        sz = self.sources_xz[:, 0] + self.total_pad 
        next_wavefield[range(self.num_shots), sz, sx] += self.dt**2 * self.source_time[nt]

        next_phi = -self.dt * self.sigma_x * self.cur_phi \
            + self.dt * (self.sigma_z - self.sigma_x) * ux + self.cur_phi
        
        next_psi = -self.dt * self.sigma_z * self.cur_psi \
            + self.dt * (self.sigma_x - self.sigma_z) * uz + self.cur_psi

        return next_wavefield, next_phi, next_psi

    def step(self, save_nt=None):
        if save_nt is not None:
            self.wavefield = []
        if self.receivers_xz is not None:
            num_receivers = len(self.receivers_xz)
            self.seismogram = np.zeros((self.num_shots, num_receivers, len(self.source_time)))
        for nt in range(len(self.source_time)):
            next_wavefield, next_phi, next_psi = self.one_step(nt)
            self.cur_wavefield, self.prev_wavefield = next_wavefield, self.cur_wavefield 
            self.cur_phi, self.cur_psi = next_phi, next_psi 
            actual_wavefield = next_wavefield[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]

            if self.receivers_xz is not None:
                self.seismogram[:, :, nt] = actual_wavefield[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
            if save_nt is not None and nt in save_nt:
                self.wavefield.append(actual_wavefield)

