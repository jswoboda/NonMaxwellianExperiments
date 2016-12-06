#!/usr/bin/env python
"""
Created on Wed Jun 17 16:27:13 2015

@author: John Swoboda
"""

import os, inspect, glob,pdb
import scipy as sp
import scipy.fftpack as scfft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from ISRSpectrum.ISRSpectrum import ISRSpectrum
#from SimISR.specfunctions import makefitsurf
from SimISR.utilFunctions import readconfigfile
from SimISR.IonoContainer import IonoContainer
from SimISR.analysisplots import analysisdump
import SimISR.runsim as runsim
from turn2geodata import fit2geodata
from GeoData.GeoData import GeoData


def makestartfile(datapath):
    outdata = sp.array([[[[1e11,3e3],[1e11,3e3]]]])
    datalocs = sp.array([[225.0,15.0,89.0]])
    pnames = sp.array([['Ni','Ti'],['Ne','Te']])
    species=['O2+','e-']
    vel = sp.array([[[0,0,0]]])
    Ionoout = IonoContainer(datalocs,outdata,sp.array([0]),ver=1,
                            paramnames=pnames, species=species,velocity=vel)
    Ionoout.saveh5(os.path.join(datapath,'startdata.h5'))
#%% For stuff
def ke(item):
    if item[0].isdigit():
        return int(item.partition(' ')[0])
    else:
        return float('inf')

def runstuff(datapath,picklefilename):
    makestartfile(datapath)

    funcnamelist=['radardata','fitting']
    runsim.main(funcnamelist,datapath,os.path.join(datapath,picklefilename),True)
    fit2geodata(os.path.join(datapath,'Fitted','fitteddata.h5'))


    analysisdump(datapath,os.path.join(datapath,picklefilename),'Non Maxwellian')

def plotparams(datapath,indch):
    # read in the files
    geofile = os.path.join(datapath,'Fitted','fitteddataGEOD.h5')
    acffile = os.path.join(datapath,'ACF','00lags.h5')
    geod = GeoData.read_h5(geofile)
    acfIono = IonoContainer.readh5(acffile)
    picklefile = os.path.join(datapath,'config.ini')
    (sensdict,simparams) = readconfigfile(picklefile)
    #determine the ind
    dataloc = geod.dataloc
    rngloc = 210
    ind = sp.argmin(sp.absolute(dataloc[:,0]-rngloc))
    #
    Ne = geod.data['Ne'][ind]
    Ti = geod.data['Ti'][ind]
    Te = geod.data['Te'][ind]


    #make a spectrum
    npts = 128
    datablock = sp.array([[Ne[indch],Ti[indch]],[Ne[indch],Te[indch]]])
    specobj = ISRSpectrum(centerFrequency =sensdict['fc'],nspec = npts,sampfreq=sensdict['fs'])
    (omeg,specexample,rcs) = specobj.getspecsep(datablock,simparams['species'],rcsflag=True)
    specexample = rcs*npts*specexample/sp.absolute(specexample).sum()
    acf = acfIono.Param_List[ind,indch]
    origspec = scfft.fftshift(scfft.fft(acf,n=npts)).real
    (figmplf, [[ax1,ax2],[ax3,ax4]]) = plt.subplots(2, 2,figsize=(16, 12), facecolor='w')


    ax1.plot(sp.arange(len(Ne)),Ne)
    ax1.set_title('$N_e$')
    ax2.plot(sp.arange(len(Ti)),Ti)
    ax2.set_title('$T_i$')
    ax3.plot(sp.arange(len(Te)),Te)
    ax3.set_title('$T_e$');
    spec1 = ax4.plot(omeg*1e-3,origspec,label='Measured')
    spec2 = ax4.plot(omeg*1e-3,specexample,label='Fitted')
    ax4.set_title('Measured and Fitted Spectrums')
    ax4.set_xlabel('Frequency KHz')
    ax4.legend(loc = 1)

#    Nevals = sp.linspace(5e10,5e11,10)
#    Tivals = sp.linspace(1e3,2e5,10)
#    Tevals = sp.linspace(1e3,2e5,10)
#    xlist = [[1],Tivals,Nevals,Tevals,[0]]
#    outsurf = makefitsurf(xlist,acf,sensdict,simparams)
    return(figmplf)

if __name__ == "__main__":
    datapaths = ['Type1','Type2','Type3']
    Ti = [5000,7000,9000]
    specchoice=[10,10,10]
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    rc('text', usetex=True)

    for ip, cpath in enumerate(datapaths):
        runstuff(cpath,'config.ini')
        curfig = plotparams(cpath,specchoice[ip])
        figname = 'pltforTi-{0}.png'.format(Ti[ip])
        curfig.suptitle('For Case where $T_i$={0}'.format(Ti[ip]) )
        plt.savefig(figname,format='png',dpi = 600)
        plt.close(curfig)
