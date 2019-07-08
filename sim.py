#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ip import InputParameters
import numpy as np
import numpy.random as rnd
import pickle
import scipy.constants as const
import scipy.stats as stats
import sys
import time
from types import SimpleNamespace

class Simulation:
    testParticles = None

    # nt - number of time steps
    def __init__(self, params, callbacks = []):
        self.ip = params
        self.p = params.normalize()
        self.nt = self.p.nt # total number of sim steps to run

        bufSize = 2 * self.p.dN * self.nt
        self.__addArray('r', bufSize, np.float64)
        self.__addArray('vr', bufSize, np.float64)
        self.__addArray('pt', bufSize, np.float64)
        self.__addArray('q', bufSize, np.int8)

        self.Qn = 0 # charge of the nucleus, e
        self.num = 0 # current number of ions
        self.eConst = np.array([self.p.k_e, np.inf, self.p.k_e / self.p.mi_me_ratio])

        self.callbacks = callbacks

        self.rbins = np.arange(self.p.Rn, self.p.Rn + 1.3*self.p.dRi, 10)
        self.rbins2 = np.arange(self.p.Rn, self.p.Rn + 1.3*self.p.dRi, 100)

    def __addArray(self, a, size, type):
        newArray = np.empty(size, dtype = type)
        setattr(self, a + '_buf', newArray)
        setattr(self, a, newArray[:0])
        setattr(self, a + '_buf2', np.empty(size, dtype = type))

    # for all arrays: moves [start:end] num postions to the right
    def __moveArrays(self, start, end, num):
        for a in [self.r_buf, self.vr_buf, self.pt_buf, self.q_buf]:
            np.copyto(a[start + num : end + num], a[start : end])

    def __updateArrayView(self):
        for name in ['r', 'vr', 'pt', 'q']:
            setattr(self, name, getattr(self, name + '_buf')[:self.num])

    def createTestParticles(self):
        v_i = np.sqrt(2 * self.p.E0_i / self.p.mi_me_ratio)
        v_e = np.sqrt(2 * self.p.E0_e)
        num = 3 # number of locations to inject particles

        # injecting 1 ion and 5 electrons per location
        r = np.linspace(self.p.Rn, self.p.Rn + self.p.dRi, num + 2)[1 : -1]
        r_e = np.repeat(r, 5)
        phi_e = np.tile(np.linspace(0, np.pi, 5), num)

        self.testParticles = [
            # e
            SimpleNamespace(
                r = r_e.copy(),
                vr = np.cos(phi_e) * v_e,
                pt = np.square(v_e * np.sin(phi_e) * r_e)),
            # i
            SimpleNamespace(
                r = r.copy(),
                vr = np.full((num, ), v_i, np.float64))
        ]

        # cumulative charge at the particle locations
        indices = np.searchsorted(self.r, r)
        lastIdx = indices[num - 1] + 1
        q = self.r_buf2[:lastIdx]
        np.cumsum(self.q[:lastIdx], dtype = np.float64, out = q)
        curQ = q[indices]

        # update velocity by half-step (required by leapfrog integration)
        # ions
        ions = self.testParticles[1]
        ions.vr += self.p.dt / 2 * self.eConst[2] * curQ / (ions.r * ions.r)
        # electrons
        els = self.testParticles[0]
        els.vr += self.p.dt / (2 * els.r * els.r) * (self.eConst[0] * (-np.repeat(curQ, 5)) + els.pt / els.r)

    def ionize(self):
        nr = self.p.Rn + self.p.dRi * rnd.random(self.p.dN) # positions of new ions/electrons
        nr[::-1].sort() # sort in descending order
        indices = np.searchsorted(self.r, nr)

        v_i = np.sqrt(2 * self.p.E0_i / self.p.mi_me_ratio)

        offset = 2 * indices.size
        maxIdx = self.num
        for i in range(0, indices.size):
            idx = indices[i]
            self.__moveArrays(idx, maxIdx, offset)
            offset -= 2
            idx += offset
            indices[i] = idx
            maxIdx = np.maximum(idx - offset, 0)

            r = nr[i]
            # ion
            self.r_buf[idx] = r
            self.q_buf[idx] = 1
            self.vr_buf[idx] = v_i
            self.pt_buf[idx] = 0
            # electron
            self.r_buf[idx + 1] = r
            self.q_buf[idx + 1] = -1
            cosPhi = 2 * rnd.random() - 1.0
            v_e = np.sqrt(2 * self.p.E0_e)
            vr = v_e * cosPhi
            pt = v_e*v_e * (1 - cosPhi*cosPhi) * r*r
            self.vr_buf[idx + 1] = vr
            self.pt_buf[idx + 1] = pt

        self.num += 2 * self.p.dN
        self.__updateArrayView()


        # advance v by dt/2 as required by leapfrog integration method
        # now we need to iterate over new ions in ascending order of r
        curQ = self.Qn
        startIdx = 0
        for idx in indices[::-1]:
            curQ += self.q[startIdx : idx].sum()
            startIdx = idx
            r = self.r[idx]
            # ion
            self.vr[idx] += self.p.dt / 2 * self.eConst[2] * curQ / (r * r)
            # electron
            self.vr[idx + 1] += self.p.dt / (2 * r * r) * (self.eConst[0] * (-1 * (curQ + 1)) + self.pt[idx + 1] / r)


    def move(self):
        # update r
        self.r += self.vr * self.p.dt
        # re-sort
        idx = np.argsort(self.r, kind = 'mergesort')
        for aname in ['r', 'vr', 'pt', 'q']:
            tmp = getattr(self, aname + '_buf2')
            a = getattr(self, aname + '_buf')
            np.take(a, idx, out = tmp[:self.num])
            setattr(self, aname + '_buf2', a)
            setattr(self, aname + '_buf', tmp)
        self.__updateArrayView()

        r, vr, pt, q, dt, num = self.r, self.vr, self.pt, self.q, self.p.dt, self.num
        # calc E
        tmp1 = self.r_buf2[:num]
        tmp1[0] = 0
        np.cumsum(q[:num - 1], out = tmp1[1 : num])
        tmp1 += self.Qn

        if self.testParticles is not None:
            ions = self.testParticles[1]
            indices = np.searchsorted(self.r, ions.r)
            ions.r += ions.vr * self.p.dt
            ions.vr += self.p.dt * self.eConst[2] * tmp1[indices] / (ions.r * ions.r)

            els = self.testParticles[0]
            indices = np.searchsorted(self.r, els.r)
            els.r += els.vr * self.p.dt
            els.vr += self.p.dt / (els.r * els.r) * (self.eConst[0] * (-1 * tmp1[indices]) + els.pt / els.r)

        # update vr
        tmp2 = self.vr_buf2[:num]
        tmpq = self.q_buf2[:num]
        # vr += dt / r**2 * ( eConst[q + 1] * (Qe * q) + pt / r)
        tmp1 *= q # tmp1 = Qe * q
        np.add(q, 1, out = tmpq) # tmpq = q + 1
        np.take(self.eConst, tmpq, out = tmp2) # tmp2 = eConst[q + 1]
        tmp1 *= tmp2 # tmp1 = eConst[q + 1] * Qe * q
        np.divide(pt, r, out = tmp2) # tmp2 = pt / r
        tmp1 += tmp2 # tmp1 = eConst[q + 1] * (Qe * q) + pt / r
        np.square(r, out = tmp2) # tmp2 = r * r
        tmp1 /= tmp2 # tmp1 = (eConst[q + 1] * (Qe * q) + pt / r) / (r * r)
        tmp1 *= dt
        vr += tmp1

        # absorb ions into the nucleus
        absorbIndex = np.searchsorted(r, self.p.Rn)
        if absorbIndex != 0:
            dQ = q[:absorbIndex].sum()
            self.Qn += dQ
            self.__moveArrays(absorbIndex, num, -absorbIndex)
            self.num -= absorbIndex
            self.__updateArrayView()

    def run(self):
        dataSaver = DataSaver()

        self.ct = 0

        dataSaver.preRun(self)

        t = time.time()

        for i in range(1, self.nt + 1):
            tt = time.time()

            self.ionize()
            self.move()
            self.ct += 1

            if i  % 100 == 0:
                curt = time.time()
                print("{} iteration step took {:.2e} s; total time elapsed {:g} s; N = {:g}".format(i, curt - tt, curt - t, self.num))
                sys.stdout.flush()


            dataSaver.update(self)

        t = time.time() - t
        print('{} simulation steps with dt = {:.2e} s (total sim time = {:.2e} s) and dN = {:.2g} took {:g} s'.format(self.nt, self.p.dt * self.p.st, self.nt * self.p.dt * self.p.st, self.p.dN, t))

        dataSaver.postRun(self)


class DataSaver:
    def preRun(self, s):
        self.Qn = np.empty(s.nt, np.int32)
        self.f = open('data10.bin', 'wb')
        self.f_tp  = open('test-particles.bin', 'wb')

    def postRun(self, s):
        np.save(self.f, self.Qn)

        self.f.close()
        self.f_tp.close()

        pickle.dump(s.ip, open( "sim-params.bin", "wb"))

    def update(self, s):
        self.Qn[s.ct - 1] = s.Qn

        if s.ct % 10 == 0:
            # calc cumulative charge at each shell
            Q = np.empty(s.num + 1, dtype = np.float64)
            Q[0] = 0
            np.cumsum(s.q, dtype = np.float64, out = Q[1 : s.num + 1])
            Q += s.Qn

            potR = np.insert(s.r, 0, s.p.Rn)

            potN = Q[s.num] * s.p.k_e / potR[s.num]  # potential at the last particle

            r_rev = np.diff(np.reciprocal(np.flip(potR)))
            pot_rev = np.flip(Q[0 : s.num]) * s.p.k_e * r_rev
            pot_rev = np.insert(pot_rev, 0, potN)
            pot = np.flip(np.cumsum(pot_rev))

            binsPot, _, binIdx = stats.binned_statistic(potR, pot, 'mean', s.rbins)
            np.save(self.f, binsPot)

            # particle number
            eIdx = np.where(s.q == -1)[0]
            r_e = s.r[eIdx]
            eCount, _, _ = stats.binned_statistic(r_e, None, 'count', s.rbins2)
            iIdx = np.where(s.q == 1)[0]
            iCount, _, _ = stats.binned_statistic(s.r[iIdx], None, 'count', s.rbins2)
            np.save(self.f, eCount)
            np.save(self.f, iCount)

        n = 200000
        if n > s.nt//2:
            n = s.nt//2
        if s.testParticles is not None:
            np.save(self.f_tp, s.testParticles[0].r)
            np.save(self.f_tp, s.testParticles[0].vr)
            np.save(self.f_tp, s.testParticles[1].r)
            np.save(self.f_tp, s.testParticles[1].vr)
        elif s.ct >= s.nt - n:
            s.createTestParticles()
            np.save(self.f_tp, s.testParticles[0].pt)

        if s.ct == s.nt:
            with open('final-state.bin', 'wb') as f:
                np.save(f, s.r)
                np.save(f, s.vr)
                np.save(f, s.pt)
                np.save(f, s.q)

