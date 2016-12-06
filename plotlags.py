#!/usr/bin/env python
"""
Created on Wed May  6 13:55:26 2015

@author: John Swoboda
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import numpy as np
import seaborn as sns

from SimISR.IonoContainer import IonoContainer


if __name__== '__main__':
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    rc('text', usetex=True)
    fname = 'ACF/00lags.h5'
    ffit = 'Fitted/fitteddata.h5'
    Ionodata = IonoContainer.readh5(fname)
    Ionofit = IonoContainer.readh5(ffit)
    dataloc = Ionodata.Sphere_Coords
    angles = dataloc[:,1:]
    b = np.ascontiguousarray(angles).view(np.dtype((np.void, angles.dtype.itemsize * angles.shape[1])))
    _, idx, invidx = np.unique(b, return_index=True,return_inverse=True)

    Neind = sp.argwhere('Ne'==Ionofit.Param_Names)[0,0]
    beamnums = [0]
    beamlist = angles[idx]
    for ibeam in beamnums:
        curbeam = beamlist[ibeam]
        indxkep = np.argwhere(invidx==ibeam)[:,0]
        Ne_data = np.abs(Ionodata.Param_List[indxkep,0,0])*2.0
        Ne_fit = Ionofit.Param_List[indxkep,0,Neind]
        rng= dataloc[indxkep,0]
        curlocs = dataloc[indxkep]
        origNe = np.ones_like(Ne_data)*1e11
        rngin = rng

        print sp.nanmean(Ne_data/origNe)
        fig = plt.figure()
        plt.plot(Ne_data,rng,'bo',label='Data')
        plt.gca().set_xscale('log')
        plt.hold(True)
        plt.plot(origNe,rngin,'g.',label='Input')
        plt.plot(Ne_fit,rngin,'r*',label='Fit')
        plt.xlabel('$N_e$')
        plt.ylabel('Range km')
        plt.title('Ne vs Range for beam {0} {1}'.format(*curbeam))
        plt.legend(loc=1)

        plt.savefig('comp{0}'.format(ibeam))
        plt.close(fig)




