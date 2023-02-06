"""
Staggered grid Finite difference solver for 2d elastic wave equations
with perfectly matched layer 

For more details, please refernce this paper: https://hal.inria.fr/inria-00073219/file/RR-3471.pdf

@author: zchen
@date: 2023-01-05
"""

import numpy as np
from scipy.interpolate import griddata

def ricker(dt, nt, peak_time, dominant_freq):
    """Ricker wavelet with specific dominant frequency"""
    t = np.arange(-peak_time, dt * nt - peak_time, dt, dtype=np.float32)
    w = ((1.0 - 2.0 * (np.pi**2) * (dominant_freq**2) * (t**2))
        * np.exp(-(np.pi**2) * (dominant_freq**2) * (t**2)))
    return w 


class Solver:
    def __init__(self, rho, vp, vs, 
                nx, nz, h, dt, nt, peak_time, dominant_freq,
                sources_xz, receivers_xz=None, 
                pml_width=10, pad_width=10):

        # Mesh
        self.nz, self.nx = int(nz), int(nx)
        self.nptz, self.nptx = nz + 1, nx + 1 # number of grid points
        self.nt = int(nt)
        self.h = np.float64(h)
        self.dt = np.float64(dt)
 
        # Sources and receivers
        self.sources_xz = sources_xz
        self.receivers_xz = receivers_xz
        self.num_shots = len(sources_xz)

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

        # Model
        # homogenenous
        if isinstance(rho, float) and isinstance(vp, float) and isinstance(vs, float):
            self.rho = np.ones((self.nptz, self.nptx)) * rho 
            self.vp = np.ones((self.nptz, self.nptx)) * vp 
            self.vs = np.ones((self.nptz, self.nptx)) * vs
        else:
            self.rho = rho 
            self.vp = vp 
            self.vs = vs 
        self.rho_padded = np.pad(self.rho, ((self.total_pad, self.total_pad), (self.total_pad, self.total_pad)), 'edge')
        self.vp_padded = np.pad(self.vp, ((self.total_pad, self.total_pad), (self.total_pad, self.total_pad)), 'edge')
        self.vs_padded = np.pad(self.vs, ((self.total_pad, self.total_pad), (self.total_pad, self.total_pad)), 'edge')
        self.mu_padded = self.rho_padded * self.vs_padded**2
        self.ld_padded = self.rho_padded * self.vp_padded**2 - 2 * self.mu_padded
        x = np.arange(self.nptx_padded) * h
        z = np.arange(self.nptz_padded) * h
        X, Z = np.meshgrid(x, z)
        points = np.hstack((X.flatten()[:, np.newaxis], Z.flatten()[:, np.newaxis]))
        xh = x[:-1] + h/2
        zh = z[:-1] + h/2
        # ld, mu on (i+1/2, j)
        grid_x, grid_z = np.meshgrid(xh, z)
        points_ihj = np.hstack((grid_x.flatten()[:, np.newaxis], grid_z.flatten()[:, np.newaxis]))
        self.ld_padded_ihj = griddata(points, self.ld_padded.flatten()[:, np.newaxis], points_ihj, method='cubic')
        self.ld_padded_ihj = self.ld_padded_ihj.reshape(self.nptz_padded, self.nptx_padded-1)
        self.mu_padded_ihj = griddata(points, self.mu_padded.flatten()[:, np.newaxis], points_ihj, method='cubic')
        self.mu_padded_ihj = self.mu_padded_ihj.reshape(self.nptz_padded, self.nptx_padded-1)
        # mu on (i, j+1/2)
        grid_x, grid_z = np.meshgrid(x, zh)
        points_ijh = np.hstack((grid_x.flatten()[:, np.newaxis], grid_z.flatten()[:, np.newaxis]))
        self.mu_padded_ijh = griddata(points, self.mu_padded.flatten()[:, np.newaxis], points_ijh, method='cubic')
        self.mu_padded_ijh = self.mu_padded_ijh.reshape(self.nptz_padded-1, self.nptx_padded)
        # rho on (i+1/2, j+1/2)
        grid_x, grid_z = np.meshgrid(xh, zh)
        points_ihjh = np.hstack((grid_x.flatten()[:, np.newaxis], grid_z.flatten()[:, np.newaxis]))
        self.rho_padded_ihjh = griddata(points, self.rho_padded.flatten()[:, np.newaxis], points_ihjh, method='cubic')
        self.rho_padded_ihjh = self.rho_padded_ihjh.reshape(self.nptz_padded-1, self.nptx_padded-1)


        # Dampling factor
        profile = 40.0 + 60.0 * np.arange(pml_width, dtype=np.float64) 
        profile_h = 40.0 + 60.0 * (np.arange(pml_width, dtype=np.float64) - 0.5) # profile at half grid 

        self.d_x = np.zeros(self.nptx_padded, np.float64)
        self.d_x[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.d_x[-self.total_pad:-self.pad_width] = profile
        self.d_x[:self.pad_width] = self.d_x[self.pad_width]
        self.d_x[-self.pad_width:] = self.d_x[-self.pad_width-1]

        self.d_xh = np.zeros(self.nptx_padded - 1, np.float64)
        self.d_xh[self.total_pad-1:self.pad_width-1:-1] = profile_h 
        self.d_xh[-self.total_pad:-self.pad_width] = profile_h 
        self.d_xh[:self.pad_width] = self.d_x[self.pad_width]
        self.d_xh[-self.pad_width:] = self.d_x[-self.pad_width-1]

        self.d_y = np.zeros(self.nptz_padded, np.float64)
        self.d_y[self.total_pad-1:self.pad_width-1:-1] = profile 
        self.d_y[-self.total_pad:-self.pad_width] = profile
        self.d_y[:pad_width] = self.d_y[pad_width]
        self.d_y[-self.pad_width:] = self.d_y[-self.pad_width-1]

        self.d_yh = np.zeros(self.nptz_padded - 1, np.float64)
        self.d_yh[self.total_pad-1:self.pad_width-1:-1] = profile_h
        self.d_yh[-self.total_pad:-self.pad_width] = profile_h 
        self.d_yh[:self.pad_width] = self.d_y[self.pad_width]
        self.d_yh[-self.pad_width:] = self.d_y[-self.pad_width-1]

        self.dx_ij = np.tile(self.d_x, (self.nptz_padded, 1))
        self.dx_ijh = np.tile(self.d_x, (self.nptz_padded-1, 1))
        self.dx_ihj = np.tile(self.d_xh, (self.nptz_padded, 1))
        self.dx_ihjh = np.tile(self.d_xh, (self.nptz_padded-1, 1))
        self.dy_ij = np.tile(self.d_y.reshape(-1, 1), (1, self.nptx_padded))
        self.dy_ihj = np.tile(self.d_y.reshape(-1, 1), (1, self.nptx_padded-1))
        self.dy_ijh = np.tile(self.d_yh.reshape(-1, 1), (1, self.nptx_padded))
        self.dy_ihjh = np.tile(self.d_yh.reshape(-1, 1), (1, self.nptx_padded-1))

        # Stress and velocity
        self.cur_vx_x = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        self.cur_vx_y =  np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded), np.float64)
        self.cur_vy_x = np.zeros((self.num_shots, self.nptz_padded-1, self.nptx_padded-1), np.float64)
        self.cur_vy_y = np.zeros((self.num_shots, self.nptz_padded-1, self.nptx_padded-1), np.float64)
        self.cur_sxx_x = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded-1), np.float64)
        self.cur_sxx_y = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded-1), np.float64)
        self.cur_syy_x = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded-1), np.float64)
        self.cur_syy_y = np.zeros((self.num_shots, self.nptz_padded, self.nptx_padded-1), np.float64)
        self.cur_sxy_x = np.zeros((self.num_shots, self.nptz_padded-1, self.nptx_padded), np.float64)
        self.cur_sxy_y = np.zeros((self.num_shots, self.nptz_padded-1, self.nptx_padded), np.float64)

    def _sxx_first_x_deriv(self, sxx):
        sxx_x = (sxx[:, :, 1:] - sxx[:, :, :-1]) / self.h
        sxx_x = np.pad(sxx_x, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=0.0)
        assert sxx_x.shape == (self.num_shots, self.nptz_padded, self.nptx_padded)
        return sxx_x 

    def _sxy_first_x_deriv(self, sxy):
        sxy_x = (sxy[:, :, 1:] - sxy[:, :, :-1]) / self.h 
        assert sxy_x.shape == (self.num_shots, self.nptz_padded-1, self.nptx_padded-1)
        return sxy_x

    def _sxy_first_y_deriv(self, sxy):
        sxy_y = (sxy[:, 1:, :] - sxy[:, :-1, :]) / self.h 
        sxy_y = np.pad(sxy_y, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=0.0)
        assert sxy_y.shape == (self.num_shots, self.nptz_padded, self.nptx_padded)
        return sxy_y 

    def _syy_first_y_deriv(self, syy):
        syy_y = (syy[:, 1:, :] - syy[:, :-1, :]) / self.h 
        assert syy_y.shape == (self.num_shots, self.nptz_padded-1, self.nptx_padded-1)
        return syy_y 

    def _vx_first_x_deriv(self, vx):
        vx_x = (vx[:, :, 1:] - vx[:, :, :-1]) / self.h 
        assert vx_x.shape == (self.num_shots, self.nptz_padded, self.nptx_padded-1)
        return vx_x 

    def _vx_first_y_deriv(self, vx):
        vx_y = (vx[:, 1:, :] - vx[:, :-1, :]) / self.h 
        assert vx_y.shape == (self.num_shots, self.nptz_padded-1, self.nptx_padded)
        return vx_y

    def _vy_first_x_deriv(self, vy):
        vy_x = (vy[:, :, 1:] - vy[:, :, :-1]) / self.h
        vy_x = np.pad(vy_x, ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=0.0)
        assert vy_x.shape == (self.num_shots, self.nptz_padded-1, self.nptx_padded)
        return vy_x 

    def _vy_first_y_deriv(self, vy):
        vy_y = (vy[:, 1:, :] - vy[:, :-1, :]) / self.h
        vy_y = np.pad(vy_y, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=0.0)
        assert vy_y.shape == (self.num_shots, self.nptz_padded, self.nptx_padded-1)
        return vy_y 

    def _one_step(self, nt):
        cur_vx = self.cur_vx_x + self.cur_vx_y 
        cur_vy = self.cur_vy_x + self.cur_vy_y 


        vx_x = self._vx_first_x_deriv(cur_vx)
        vx_y = self._vx_first_y_deriv(cur_vx)
        vy_x = self._vy_first_x_deriv(cur_vy)
        vy_y = self._vy_first_y_deriv(cur_vy)

        next_sxx_x = (
            (self.ld_padded_ihj + 2 * self.mu_padded_ihj) * self.dt * vx_x 
            + self.cur_sxx_x 
            - self.dt * self.d_xh * self.cur_sxx_x / 2
        )
        next_sxx_x /= (1 + self.dt * self.d_xh / 2)

        next_sxx_y = (
            self.ld_padded_ihj * self.dt * vy_y 
            + self.cur_sxx_y
            - self.dt * self.dy_ihj * self.cur_sxx_y / 2
        )
        next_sxx_y /= (1 + self.dt * self.dy_ihj / 2)

        next_syy_x = (
            self.ld_padded_ihj * self.dt * vx_x 
            + self.cur_syy_x
            - self.dt * self.dx_ihj * self.cur_syy_x / 2
        )
        next_syy_x /= (1 + self.dx_ihj * self.dt / 2)

        next_syy_y = (
            (self.ld_padded_ihj + 2 * self.mu_padded_ihj) * self.dt * vy_y 
            + self.cur_syy_y 
            - self.dt * self.dy_ihj * self.cur_syy_y / 2
        )
        next_syy_y /= (1 + self.dy_ihj * self.dt / 2)

        next_sxy_x = (
            self.mu_padded_ijh * self.dt * vy_x 
            + self.cur_sxy_x 
            - self.dt * self.dx_ijh * self.cur_sxy_x / 2
        )
        next_sxy_x /= (1 + self.dt * self.dx_ijh / 2)

        next_sxy_y = (
            self.mu_padded_ijh * self.dt * vx_y 
            + self.cur_sxy_y
            - self.dt * self.dy_ijh * self.cur_sxy_y / 2
        )
        next_sxy_y /= (1 + self.dy_ijh * self.dt / 2)

        next_sxx = next_sxx_x + next_sxx_y 
        next_sxy = next_sxy_x + next_sxy_y
        next_syy = next_syy_x + next_syy_y

        sxx_x = self._sxx_first_x_deriv(next_sxx)
        sxy_y = self._sxy_first_y_deriv(next_sxy)
        sxy_x = self._sxy_first_x_deriv(next_sxy)
        syy_y = self._syy_first_y_deriv(next_syy)

        next_vx_x = (
            self.dt * sxx_x / self.rho_padded
            + self.cur_vx_x 
            - self.dx_ij * self.dt * self.cur_vx_x / 2
        ) 
        next_vx_x /= (1 + self.dt * self.dx_ij / 2)

        next_vx_y = ( 
            self.dt * sxy_y / self.rho_padded
            + self.cur_vx_y 
            - self.dy_ij * self.dt * self.cur_vx_y / 2
        )
        next_vx_y /= (1 + self.dt * self.dy_ij / 2)

        next_vy_x = ( 
            self.dt *  sxy_x / self.rho_padded_ihjh
            + self.cur_vy_x 
            - self.dt * self.dx_ihjh * self.cur_vy_x / 2
        )
        next_vy_x /= (1 + self.dt * self.dx_ihjh / 2)

        next_vy_y = (
            self.dt * syy_y / self.rho_padded_ihjh
            + self.cur_vy_y 
            - self.dt * self.dy_ihjh * self.cur_vy_y / 2
        )
        next_vy_y /= (1 + self.dt * self.dy_ihjh / 2)
                

        # Add source 
        sx = self.sources_xz[:, 1] + self.total_pad
        sz = self.sources_xz[:, 0] + self.total_pad 
        next_vx_x[range(self.num_shots), sz, sx] += self.dt * self.source_time[nt]

        next_vx = next_vx_x + next_vx_y 
        next_vy = next_vy_x + next_vy_y 

        return (
            next_vx, next_vy, 
            next_vx_x, next_vx_y, 
            next_vy_x, next_vy_y,
            next_sxx_x, next_sxx_y,
            next_sxy_x, next_sxy_y,
            next_syy_x, next_syy_y       
            )

    def step(self, save_nt=None):
        if save_nt is not None:
            self.vx = []
            self.vy = []

        if self.receivers_xz is not None:
            num_receivers = len(self.receivers_xz)
            self.seismogram_vx = np.zeros((self.num_shots, num_receivers, len(self.source_time)))
            self.seismogram_vy = np.zeros((self.num_shots, num_receivers, len(self.source_time)))

        for nt in range(self.nt):
            state = self._one_step(nt)
            next_vx, next_vy = state[0], state[1]
            actual_vx = next_vx[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
            actual_vy = next_vy[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
             
            if self.receivers_xz is not None:
                self.seismogram_vx[:, :, nt] = actual_vx[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
                self.seismogram_vy[:, :, nt] = actual_vy[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
            if save_nt is not None and nt in save_nt:
                self.vx.append(actual_vx)
                self.vy.append(actual_vy)

            self.cur_vx_x, self.cur_vx_y = state[2], state[3]
            self.cur_vy_x, self.cur_vy_y = state[4], state[5]
            self.cur_sxx_x, self.sxx_y = state[6], state[7]
            self.cur_sxy_x, self.cur_sxy_y = state[8], state[9]
            self.cur_syy_x, self.cur_syy_y = state[10], state[11]










    
