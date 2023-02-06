"""
Staggered grid Finite difference solver for 2d wave equations
with perfectly matched layer 

@author: zchen
@date: 2022-12-22
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

        # Source time function
        self.peak_time = peak_time 
        self.dominant_freq = dominant_freq
        self.source_time = ricker(self.dt, self.nt, self.peak_time, self.dominant_freq)

        # PML 
        self.pml_width = pml_width 
        self.pad_width = pad_width 

        # Pad 
        self.total_pad = pml_width + pad_width
        self.nptx_padded = self.nptx + 2 * self.total_pad 
        self.nptz_padded = self.nptz + 2 * self.total_pad
        self.model_padded = np.pad(model, ((self.total_pad, self.total_pad), (self.total_pad, self.total_pad)), 'edge')

        # Dampling factor 
        profile = 40.0 + 60.0 * np.arange(pml_width, dtype=np.float64) 
        profile_h = 40.0 + 60.0 * (np.arange(pml_width, dtype=np.float64) - 0.5) # profile at half grid 

        self.sigma_x = np.zeros(self.nptx_padded, np.float64)
        self.sigma_x[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.sigma_x[-self.total_pad:-self.pad_width] = profile
        self.sigma_x[:self.pad_width] = self.sigma_x[self.pad_width]
        self.sigma_x[-self.pad_width:] = self.sigma_x[-self.pad_width-1]

        self.sigma_xh = np.zeros(self.nptx_padded - 1, np.float64)
        self.sigma_xh[self.total_pad-1:self.pad_width-1:-1] = profile_h 
        self.sigma_xh[-self.total_pad:-self.pad_width] = profile_h 
        self.sigma_xh[:self.pad_width] = self.sigma_x[self.pad_width]
        self.sigma_xh[-self.pad_width:] = self.sigma_x[-self.pad_width-1]

        self.sigma_z = np.zeros(self.nptz_padded, np.float64)
        self.sigma_z[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.sigma_z[-self.total_pad:-self.pad_width] = profile
        self.sigma_z[:pad_width] = self.sigma_z[pad_width]
        self.sigma_z[-self.pad_width:] = self.sigma_z[-self.pad_width-1]

        self.sigma_zh = np.zeros(self.nptz_padded - 1, np.float64)
        self.sigma_zh[self.total_pad-1:self.pad_width-1:-1] = profile_h
        self.sigma_zh[-self.total_pad:-self.pad_width] = profile_h 
        self.sigma_zh[:self.pad_width] = self.sigma_z[self.pad_width]
        self.sigma_zh[-self.pad_width:] = self.sigma_z[-self.pad_width-1]

        self.sigma_x = np.tile(self.sigma_x, (self.nptz_padded, 1))
        self.sigma_xh = np.tile(self.sigma_xh, (self.nptz_padded, 1))
        self.sigma_z = np.tile(self.sigma_z.reshape(-1, 1), (1, self.nptx_padded))
        self.sigma_zh = np.tile(self.sigma_zh.reshape(-1, 1), (1, self.nptx_padded))

        # Wavefield and auxiliary function
        self.cur_px = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        self.cur_pz = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)        
        self.cur_qx = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded-1), np.float64)
        self.cur_qz = np.zeros((self.num_shots, self.nptz_padded-1, self.nptx_padded), np.float64)

    def _qx_first_x_deriv(self, qx): 
        qx_x = (qx[:, :, 1:] - qx[:, :, :-1]) / self.h
        qx_x = np.pad(qx_x, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=0.0)
        assert qx_x.shape == (self.num_shots, self.nptz_padded, self.nptx_padded)
        return qx_x

    def _qz_first_z_deriv(self, qz):
        qz_z = (qz[:, 1:, :] - qz[:, :-1, :]) / self.h 
        qz_z = np.pad(qz_z, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=0.0)
        assert qz_z.shape == (self.num_shots, self.nptz_padded, self.nptx_padded)
        return qz_z 

    def _p_first_x_deriv(self, p):
        p_x = (p[:, :, 1:] - p[:, :, :-1]) / self.h 
        assert p_x.shape == (self.num_shots, self.nptz_padded, self.nptx_padded-1)
        return p_x 

    def _p_first_z_derive(self, p):
        p_z = (p[:, 1:, :] - p[:, :-1, :]) / self.h 
        assert p_z.shape == (self.num_shots, self.nptz_padded-1, self.nptx_padded)
        return p_z 

    def _one_step(self, nt):
        qx_x = self._qx_first_x_deriv(self.cur_qx)
        qz_z = self._qz_first_z_deriv(self.cur_qz)

        next_px = (self.dt * self.model_padded ** 2 * qx_x 
                - self.dt * self.sigma_x * self.cur_px
                + self.cur_px)
        next_pz = (self.dt * self.model_padded**2 * qz_z
                - self.dt * self.sigma_z * self.cur_pz
                + self.cur_pz)
        next_p = next_px + next_pz 

        # Add source 
        sx = self.sources_xz[:, 1] + self.total_pad
        sz = self.sources_xz[:, 0] + self.total_pad 
        next_p[range(self.num_shots), sz, sx] += self.dt * self.source_time[nt]

        p_x = self._p_first_x_deriv(next_p)
        p_z = self._p_first_z_derive(next_p)
        next_qx = self.dt * p_x - self.dt * self.sigma_xh * self.cur_qx + self.cur_qx 
        next_qz = self.dt * p_z - self.dt * self.sigma_zh * self.cur_qz + self.cur_qz 

        return next_p, next_px, next_pz, next_qx, next_qz
        
    def step(self, save_nt=None):
        if save_nt is not None:
            self.wavefield = []

        if self.receivers_xz is not None:
            num_receivers = len(self.receivers_xz)
            self.seismogram = np.zeros((self.num_shots, num_receivers, len(self.source_time)))

        for nt in range(self.nt):
            next_p, next_px, next_pz, next_qx, next_qz = self._one_step(nt)
            self.cur_px, self.cur_pz, self.cur_qx, self.cur_qz = next_px, next_pz, next_qx, next_qz
            actual_wavefield = next_p[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]

            if self.receivers_xz is not None:
                self.seismogram[:, :, nt] = actual_wavefield[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
            if save_nt is not None and nt in save_nt:
                self.wavefield.append(actual_wavefield)