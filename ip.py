import numpy as np
import scipy.constants as const
from types import SimpleNamespace

class InputParameters:

    def __init__(self, nt, dri = 10e3):
        # values are based on the observations of 67P around Jan 2015 (Heritier, 2018)
        self.Rn = 1e3 # radius of nucleus, m
        self.dRi = dri # ionization range, m
        self.Q = 1e26 # gas production rate
        self.nu = 1e-7 # ionization frequency, s^-1
        self.u = 1e3 # radial velocity of neutrals, m
        self.E0_e = 10 * const.e # initial electron energy, J
        self.E0_i = 0.1 * const.e # initial ion energy, J
        self.mi_me_ratio = 1e2 # mass ratio of i to e
        self.n_max = 250 * 1e6 # maximum expected plasma number density, m^-3
        self.dN = 10 # number of shells produced for a plasma species, s^-1
        self.kT = 10 # reference energy, eV

        # scaling constants
        self.st = np.sqrt(const.m_e * const.epsilon_0 / self.n_max) / const.e
        self.dt = 0.1 * self.st

        self.sN = int(np.ceil(self.nu * self.Q / self.u * self.dt * self.dRi / self.dN)) # dP / dN
        self.sr = np.sqrt(self.kT * const.epsilon_0 / (self.n_max * const.e))
        self.sm = const.m_e
        self.sq = const.e
        self.sF = self.kT # potential, V
        self.sE = self.kT * const.e # energy, J
        self.sv = np.sqrt(self.kT * const.e / const.m_e)

        self.nt = nt
        self.t = self.nt * self.dt

    def normalize(self):
        return SimpleNamespace(
            nt = self.nt,
            t = self.t / self.st,
            Rn = self.Rn / self.sr,
            dRi = self.dRi / self.sr,
            u = self.u / self.sv,
            mi_me_ratio = self.mi_me_ratio,
            E0_e = self.E0_e / self.sE, # initial e energy
            E0_i = self.E0_i / self.sE, # initial i energy
            dN = self.dN,
            st = self.st,
            dt = self.dt / self.st,
            sN = self.sN,

            sr = self.sr,
            sm = self.sm,
            sq = self.sq,
            sF = self.sF,
            sE = self.sE,
            sv = self.sv,
            k_e = self.sN * self.sq / (self.sF * self.sr * 4 * const.pi * const.epsilon_0)
        )

    def print(self, file = 'ip.txt'):
        with open(file, 'wt') as f:
            p = self.normalize()
            f.write('Scaling params:\n')
            ns = ['sN', 'sr', 'sv', 'sE', 'sF', 'st']
            for n in ns:
                f.write('    {} = {:.5g}\n'.format(n, getattr(p, n)))
            f.write('Total time {:.5g} ({:.5g} s)\n'.format(p.t, self.t))
            f.write('Total iteration steps {:.5g}\n'.format(p.nt))
            f.write('Time step {:.5g} ({:.5g} s)\n'.format(p.dt, self.dt))
            f.write('Radius of the nucleus {:.5g} ({:.5g} m)\n'.format(p.Rn, self.Rn))
            f.write('Ionization distance {:.5g} ({:.5g} m)\n'.format(p.dRi, self.dRi))
            f.write('Initial energy of e {:.5g} ({:.5g} eV), i {:.5g} ({:.5g} eV)\n'.format(p.E0_e, self.E0_e / const.e, p.E0_i, self.E0_i / const.e))
            v_i = np.sqrt(2 * p.E0_i / p.mi_me_ratio)
            v_e = np.sqrt(2 * p.E0_e)
            f.write('Initial velocity of e {:.5g} ({:.5g} m/s), i {:.5g} ({:.5g} m/s)\n'.format(v_e, v_e * p.sv, v_i, v_i * p.sv))
            t_i = p.dRi / v_i
            t_e = p.dRi / v_e
            f.write('Transport time of e {:.5g} ({:.5g} s), i {:.5g} ({:.5g} s)\n'.format(t_e, t_e * p.st, t_i, t_i * p.st))
            v_s = np.sqrt(self.E0_e / self.mi_me_ratio / const.m_e)
            f.write('Ion acoustic velocity {:.5g} ({:.5g} m/s)\n'.format(v_s / p.sv, v_s))
            # max_phi
            phi_factor = 1 / (4 * const.pi * const.epsilon_0) * const.e * self.nu * self.Q / self.u * self.dt * self.dRi
            def max_phi_func(dn):
                return phi_factor / dn * (1 / self.Rn - 1 / (self.Rn + self.dRi / (1000 * dn)))
            max_phi = max_phi_func(self.dN)
            f.write('Maximum expected potantial drop {:.5g} ({:.5g} V)\n'.format(max_phi / p.sF, max_phi))
            max_n = self.nu * self.Q / (16 * const.pi * self.Rn * self.u * v_i * p.sv)
            f.write('Maximum expected ion number density {:.5g} cm^-3\n'.format(max_n / 1e6))

