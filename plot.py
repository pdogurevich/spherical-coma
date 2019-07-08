# This scipt is written for ipython

import ip
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.constants as const
from types import SimpleNamespace

class SimData:
    def __init__(self):
        self.ip = pickle.load(open('sim-params.bin', 'rb'))

        self.p = self.ip.normalize()
        self.rbins = np.arange(self.p.Rn, self.p.Rn + 1.3*self.p.dRi, 10)
        self.rbins2 = np.arange(self.p.Rn, self.p.Rn + 1.3*self.p.dRi, 100)

        self.datadt = 10

        dataPoints = self.p.nt // self.datadt
        self.pot = np.empty((dataPoints, self.rbins.size - 1), np.float64)
        self.n = np.empty((2, dataPoints, self.rbins2.size - 1), np.float64)
        with open('data10.bin', 'rb') as f:
            for i in range(0, dataPoints):
                self.pot[i] = np.load(f)
                self.n[0, i] = np.load(f) # e
                self.n[1, i] = np.load(f) # i
            self.qn = np.load(f)

#%% load data
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

sd = SimData()

with open('final-state.bin', 'rb') as f:
    r = np.load(f) #r
    vr = np.load(f) #vr
    pt = np.load(f) #pt
    q = np.load(f) #q

num = r.size

#%%

def model_n0(r):
    v_i = np.sqrt(2 * sd.p.E0_i/sd.p.mi_me_ratio)
    return sd.ip.nu * sd.ip.Q / (4*np.pi * r * sd.p.u * v_i)*(1 - sd.p.Rn / r) \
        * sd.p.st * sd.p.st

#%% load test particles
dataPoints = 200000
num_i = 3
num_e = 15
els = SimpleNamespace(
    pt = np.empty(num_e, np.float64),
    r = np.empty((dataPoints, num_e), np.float64),
    vr = np.empty((dataPoints, num_e), np.float64))
ions = SimpleNamespace(
    r = np.empty((dataPoints, num_i), np.float64),
    vr = np.empty((dataPoints, num_i), np.float64))
with open('test-particles.bin', 'rb') as f:
    els.pt = np.load(f)
    for i in range(0, dataPoints):
        els.r[i] = np.load(f)
        els.vr[i] = np.load(f)
        ions.r[i] = np.load(f)
        ions.vr[i] = np.load(f)

#%% test particle trajectories
# electron phase plot
partnum = 2
pr = els.r[:, partnum]
pvr = els.vr[:, partnum]
ppt = els.pt[partnum]
pq = -1

# projected
Q = np.empty(num, dtype = np.int32)
Q[0] = 0
np.cumsum(q[:-1], out = Q[1:])
Q += sd.qn[-1]

size = 260000*5
nr = np.empty(size)
nvr = np.empty(size)
curr = pr[-1]
curvr = pvr[-1]
for t in range(size):
    idx = np.searchsorted(r, curr)
    curr += curvr * sd.p.dt
    curvr += sd.p.dt / (curr * curr) * (sd.p.k_e * (-1 * Q[idx]) + ppt / curr)
    nr[t] = curr
    nvr[t] = curvr

#%%
    
# phase space
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Radial velocity in $v_s$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')


ax.plot(pr[:-100000], pvr[:-100000], 'm', lw = 1)
ax.plot(pr[100000:], pvr[100000:], 'g', lw = 1)
#orbit = 119000 # high 2
orbit = 126000 # low 2
#orbit = 160000 # low 3
ax.plot(nr[:orbit], nvr[:orbit], '--b', lw = 1.5)

ax.plot(pr[0], pvr[0], 'xm', ms = 10)
ax.plot(pr[100000], pvr[100000], 'xg', ms = 10)
ax.plot(nr[0], nvr[0], 'xb', ms = 10)

fig.savefig('particle-{}-phase'.format(partnum))

# trajectory
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "X in $r_s$", ylabel = "Y in $r_s$")
ax.add_artist(plt.Circle((0, 0), sd.p.Rn, ls = '-', color = 'r', fill = True))
ax.grid(True, lw = 1, ls = ':', c = '0.8')
ax.add_artist(plt.Circle((0, 0), sd.p.Rn + sd.p.dRi, ls = '-', color='red', fill = False))

curPhi = 0
for tr, tvr, s in [(pr[:100000], pvr[:100000], 'm'), (pr[100000:], pvr[100000:], 'g'), (nr[:orbit], nvr[:orbit], '--b')]:
    phi = np.sqrt(ppt) * sd.p.dt / np.square(tr[:-1])
    phi = np.insert(phi, 0, curPhi)
    phi = np.cumsum(phi)
    x = tr * np.cos(phi)
    y = tr * np.sin(phi)
    ax.plot(x, y, s, lw = 1.2)
    curPhi = phi[-1]

dt = 2*orbit
orbits = nr[orbit:].size // dt
for i in range(0, orbits):
    tmin, tmax = orbit + i*dt, orbit + (i + 1)*dt
    tr, tvr, s = nr[tmin : tmax], nvr[tmin : tmax], '--'
    phi = np.sqrt(ppt) * sd.p.dt / np.square(tr[:-1])
    phi = np.insert(phi, 0, curPhi)
    phi = np.cumsum(phi)
    x = tr * np.cos(phi)
    y = tr * np.sin(phi)
    ax.plot(x, y, s, lw = 1.2)
    curPhi = phi[-1]

maxR = max(np.max(pr), np.max(nr[orbit:]))
ax.set_xlim(-maxR, maxR)
ax.set_ylim(-maxR, maxR)
ax.set_aspect('equal')
fig.savefig('particle-{}-config'.format(partnum))

#%% final pot
rbins = sd.rbins
pot = sd.pot * sd.ip.sF
ribin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
rmin, rmax = 0, ribin
x = rbins[rmin : rmax]

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Potential in V")
ax.plot(x, pot[-1, rmin:rmax], 'r', lw = 1, label = "\\phi$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
fig.savefig('potential')

#%% qn
qn = sd.qn

#tperiod = 500 # low
tperiod = 100 # high

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Time in $t_s$", ylabel = "Charge in $e$")
ax.plot(np.linspace(1, sd.p.nt//10, sd.p.nt), sd.p.sN * qn, '-b', lw = 1, label = "$q_n$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
fig.savefig('qn')

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Time in $t_s$", ylabel = "Charge in $N_s \\times  e$")
tsteps = 10 * tperiod
ax.plot(np.linspace(1, tperiod, tsteps), qn[-tsteps:], '-b', lw = 1, label = "$q_n$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
fig.savefig('qn-zoom')


#%% mean potential in time + stddev
rbins = sd.rbins
pot = sd.pot * sd.ip.sF
ribin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
rmin, rmax = 0, ribin
x = rbins[rmin : rmax]

tp = 100 # high
#tp = 500 # low
rs = np.array([640, 1500, 3000, 5000]) # high
#rs = np.array([850, 1500, 3000, 5000]) # low

tmin, tmax = 50000 - tp - 1, 50000
cs = np.array(['c', 'm', 'g', 'b'])

# MEAN in time
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Potential in V")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
for tau in [0.25, 0.5, 0.75, 1]:
    t = int(pot.shape[0] * tau)
    y = np.mean(pot[t - tp:t, rmin:rmax], axis = 0)
    ax.plot(x, y, lw = 1, label = "$\\tau = {:.2f}\, \\tau_{{\mathrm{{max}} }}$".format(tau))
ax.legend()
fig.savefig('potential-mean-t')

#%%
rmin, rmax = 0, ribin
x = rbins[rmin : rmax]
meanpot = np.mean(pot[tmin:tmax, rmin:rmax], axis = 0)
stdpot = np.std(pot[tmin:tmax, rmin:rmax], axis = 0)

# MEAN, STD, CUTS
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Potential in V")
ax.plot(x, meanpot, 'r', lw = 1, label = "$\\overline{{\\phi}}$")
ax.plot(x, meanpot + stdpot, '--k', lw = 1, label = "$\\sigma(\\phi)$")
ax.plot(x, meanpot - stdpot, '--k', lw = 1)
ax.grid(True, lw = 1, ls = ':', c = '0.8')
for i in range(rs.size):
    ax.axvline(rs[i], lw = 1.2, ls = ':', c = cs[i])
ax.legend()
fig.savefig('potential-mean')

# OSCILLATIONS AT CUTS
fig = plt.figure()
axes = fig.subplots(rs.size//2, 2, sharex = True, sharey = True).flatten()
x = np.arange(tmin, tmax) - tmin
for i in range(rs.size):
    ax = axes[i]
    curr = rs[i]
    rbin = np.searchsorted(rbins, curr)
    ax.plot(x, pot[tmin:tmax, rbin], lw = 1, c = cs[i], label = '$\\phi$ at $r = {:.0f} r_s$'.format(curr))    
    ax.grid(True, lw = 1, ls = ':', c = '0.8')
fig.savefig('potential-mean-intime-r')


# FIT FOR THE MEAN
# fit for n_i
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Potential in V")
from scipy.optimize import curve_fit
def func1(x, a, b):
    return a/x + b
def funcln(x, a, b):
    return a*np.log(x) + b

def model_phi(r):
    return 2/3 * 10 * np.log(model_n0(r)/model_n0(10*67210))

x = rbins[rmin : rmax]
ax.plot(x, meanpot, 'r', lw = 1.5, label = "$\\overline{{\\phi}}$")

rmin = np.searchsorted(rbins, 670) # high
#rmin = np.searchsorted(rbins, 1200) # low
x = rbins[rmin : rmax]
meanpot = meanpot[rmin:]
a, b = curve_fit(funcln, x, meanpot)[0]
ax.plot(x, funcln(x, a, b), '--k', lw = 1.3, label = "$\\ln{r}$ fit")
a, b = curve_fit(func1, x, meanpot)[0]
ax.plot(x, func1(x, a, b), ':k', lw = 1.3, label = "$1/r$ fit")
ax.legend()
fig.savefig('potential-mean-fit')

plt.show()

#%% 2d potential
rbins = sd.rbins
pot = sd.pot * sd.ip.sF
ribin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
rmin, rmax = 0, ribin
tmin, tmax = 20000 - 1, 50000

extent = [tmin + 1, tmax, sd.p.Rn, sd.p.Rn + sd.p.dRi]

fig = plt.figure()
ax = fig.add_subplot(111, ylabel = "Position in $r_s$", xlabel = "Time in $t_s$")
im = ax.imshow(pot[tmin:tmax].transpose(), origin = 'lower', aspect = 'auto', interpolation = 'None', extent = extent)
# Create colorbar
cbar = fig.colorbar(im, ax = ax)
cbar.ax.set_ylabel('Potential in V', rotation=-90, va="bottom")

fig.savefig('pot-2d')

#%% 2d E_k
rbins = sd.rbins
cr = 2000
rbin = np.searchsorted(rbins, cr)
ek = np.empty((50000, 50))

with open('data10.bin', 'rb') as f:
    for i in range(0, 50000):
        np.load(f) # pot
        np.load(f) # ne
        np.load(f) # i
        cek = np.load(f) # ke
        ek[i] = cek[rbin]
        ek[i] /= np.sum(ek[i])
        
#%%
tmin, tmax = 20000 - 1, 50000
emin, emax = 0, 25
extent = [tmin + 1, tmax, emin, emax]

fig = plt.figure()
fig.suptitle('Kinetic energy time evolution at r = {:.0f}'.format(cr))
ax = fig.add_subplot(111, ylabel = "Kinetic energy in eV", xlabel = "Time in $t_s$")
im = ax.imshow(ek[tmin:tmax, emin:emax].transpose(), origin = 'lower', aspect = 'auto', interpolation = 'None', extent = extent)
# Create colorbar
cbar = fig.colorbar(im, ax = ax)
cbar.ax.set_ylabel('Fraction of particles', rotation=-90, va="bottom")

fig.savefig('ek-{:.0f}-2d'.format(cr))

#%%
tmin, tmax = 20000 - 1, 50000
ces = [2, 5, 10, 15]
emin, emax = 0, 5
extent = [tmin + 1, tmax, emin, emax]

fig = plt.figure()
fig.suptitle('Fraction of particles with specific kinetic energy at $r = {:.0f}$'.format(cr))
ax = fig.add_subplot(111, ylabel = "Fraction of particles", xlabel = "Time in $t_s$")
for ce in ces:
    ax.plot(np.arange(tmin, tmax) + 1, ek[tmin:tmax, ce], label = '${}$ eV'.format(ce))
ax.legend()
fig.savefig('ek-{:.0f}-1d.png'.format(cr))

#%% final ek

rbins = sd.rbins
rbin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
ek = np.empty((rbins.size - 1, 50))

with open('data10.bin', 'rb') as f:
    for i in range(0, 50000):
        np.load(f) # pot
        np.load(f) # ne
        np.load(f) # i
        cek = np.load(f) # ke
        if i == 49999:
            ek = cek
#%%
emin, emax = 0, 30

extent = [sd.p.Rn, sd.p.Rn + sd.p.dRi, emin, emax]

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Kinetic energy in eV")
im = ax.imshow(ek[rmin:rmax, emin:emax].transpose() * sd.p.sN, origin = 'lower', aspect = 'auto', interpolation = 'None', extent = extent)
# Create colorbar
cbar = fig.colorbar(im, ax = ax)
cbar.ax.set_ylabel('Number of particles', rotation=-90, va="bottom")

fig.savefig('ek-final')


#%% final number density plots
import scipy.stats as stats
rbins = np.arange(sd.p.Rn, sd.p.Rn + sd.p.dRi + 100, 20)
# volume of each rbin in cm^3
shellv = 4/3 * const.pi * (np.power(rbins[1:], 3) - np.power(rbins[:-1], 3)) \
    * np.power(sd.p.sr, 3) * 1e6

eIdx = np.where(q == -1)[0]
n_e, _, _ = stats.binned_statistic(r[eIdx], None, 'count', rbins)
n_e /= shellv
n_e *= sd.p.sN
iIdx = np.where(q == 1)[0]
n_i, _, _ = stats.binned_statistic(r[iIdx], None, 'count', rbins)
n_i /= shellv
n_i *= sd.p.sN

rnbin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
x = rbins[ : rnbin]

# n_i, n_e, and n_i - n_e
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Number density in particles per cm$^{3}$")
ax2 = ax.twinx()

ax.set_xlim(sd.p.Rn - 100, sd.p.Rn + sd.p.dRi + 100)
ax.grid(True, lw = 1, ls = ':', c = '0.8')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
p3, = ax2.plot(x, (n_i[:rnbin] - n_e[:rnbin]) / (n_i[:rnbin] + n_e[:rnbin]), 'g', label = "$\\frac{n_i - n_e}{(n_i + n_e)}$", lw = 1)
p1, = ax.plot(x, n_e[:rnbin], 'b', label = "$n_e$")
p2, = ax.plot(x, n_i[:rnbin], '--r', label = "$n_i$")
ax2.hlines(0, sd.p.Rn, ax.get_xlim()[1], linestyles = '--', colors = '0.5', lw = 1)
ax.legend(handles = [p1, p2, p3])
fig.savefig('final-numdensity-1d')


# fit for n_i
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Number density in particles per cm$^{3}$")
from scipy.optimize import curve_fit
def func1(x, a, b):
    return a/x + b
def func2(x, a, b):
    return a/(x**2) + b

ax.set_xlim(sd.p.Rn - 100, sd.p.Rn + sd.p.dRi + 100)
x = rbins[:rnbin]
ax.plot(x, n_i[:rnbin], 'r', label = '$n_i$')
ax.plot(x, model_n0(x) / 1e6 / (sd.p.sr ** 3), '-.k', lw = 1, label = '$n^{0}$')

#peakbin = np.searchsorted(rbins, 1200) # low
peakbin = np.searchsorted(rbins, 660) # high
rmin, rmax = peakbin, rnbin
x = rbins[rmin : rmax]
n = n_i[rmin : rmax]
params = curve_fit(func1, x, n)
ax.plot(x, func1(x, params[0][0], params[0][1]), '--k', lw = 1.5, label = "$1/r$ fit")
params = curve_fit(func2, x, n)
ax.plot(x, func2(x, params[0][0], params[0][1]), ':k', lw = 1.5, label = "$1/r^2$ fit")
ax.legend()
fig.savefig('final-numdensity-fit')



fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$")

n = sd.ip.Q / sd.ip.u * (rbins[1:] - rbins[:-1])
n /= shellv

ax.set_xlim(sd.p.Rn - 100, sd.p.Rn + sd.p.dRi + 100)
x = rbins[ : rnbin]
ax.plot(x, n_e[:rnbin]/n[:rnbin], 'b', label = "$n_e / n$")

v_i = np.sqrt(2 * sd.ip.E0_i / (sd.ip.mi_me_ratio * const.m_e))
etta = sd.ip.mi_me_ratio * const.m_e * v_i**2 / (4/3 * sd.ip.E0_e)
koeff = sd.ip.nu / v_i * (np.sqrt(etta * const.pi) - 2*etta)
def f(r):
    return koeff * r
ax.plot(x, f(x), '--k', label = '$n_e^*/n$')
ax.grid(True, lw = 1, ls = ':', c = '0.8')
ax.legend()
fig.savefig('final-numdensity-ratio', bbox_inches = 'tight')


plt.show()

#%% number density evolution in time
import scipy.stats as stats
rbins = sd.rbins2
# volume of each rbin in cm^3
shellv = 4/3 * const.pi * (np.power(rbins[1:], 3) - np.power(rbins[:-1], 3)) \
    * np.power(sd.p.sr, 3) * 1e6

n_i = sd.n[1] / shellv * sd.p.sN

rnbin = np.searchsorted(rbins, sd.p.Rn + sd.p.dRi)
x = rbins[ : rnbin]

# n_i in time
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Number density in particles per cm$^{3}$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
for tau in [0.25, 0.5, 0.75, 1]:
    t = int(n_i.shape[0] * tau) - 1
    y = n_i[t, :rnbin]
    ax.plot(x, y, lw = 1, label = "$\\tau = {:.2f}\, \\tau_{{\mathrm{{max}} }}$".format(tau))
ax.legend()
fig.savefig('numdens-t')

#%%
size = 100000
nr = np.empty((3, size))
nvr = np.empty((3, size))
ns = ['r', 'g', 'b']
ks = [2.5, 10, 30]
ppt = np.sqrt(2 * sd.p.E0_e)**2 * 2000**2

for i in range(3):
    cr, cvr, ck = nr[i], nvr[i], ks[i]

    curr = 2000
    curvr = 0
    for t in range(size):
        cr[t] = curr
        cvr[t] = curvr
        curr += curvr * sd.p.dt
        curvr += sd.p.dt / (curr * curr) * (ck * sd.p.k_e * (-1) + ppt / curr)
        
# trajectory
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "X in $r_s$", ylabel = "Y in $r_s$")
ax.add_artist(plt.Circle((0, 0), 10, ls = '-', color = 'r', fill = True))
ax.grid(True, lw = 1, ls = ':', c = '0.8')

for i in range(3):
    tr, tvr, s = nr[i], nvr[i], ns[i]

    phi = np.sqrt(ppt) * sd.p.dt / np.square(tr[:-1])
    phi = np.insert(phi, 0, 0)
    phi = np.cumsum(phi)
    x = tr * np.cos(phi)
    y = tr * np.sin(phi)
    ax.plot(x, y, s, lw = 1.2)
    curPhi = phi[-1]
ax.set_aspect('equal')
fig.savefig('kepler-config')


fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "Position in $r_s$", ylabel = "Radial velocity in $v_s$")
ax.grid(True, lw = 1, ls = ':', c = '0.8')
for i in range(3):
    tr, tvr, s = nr[i], nvr[i], ns[i]
    ax.plot(tr, tvr, s, lw = 1)
fig.savefig('kepler-phase')

plt.show()
