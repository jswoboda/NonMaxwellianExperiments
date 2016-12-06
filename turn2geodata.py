#!/usr/bin/env python
"""
Created on Tue Apr 14 18:06:56 2015

@author: John Swoboda
"""
import os, glob
from SimISR.IonoContainer import IonoContainer
from GeoData.GeoData import GeoData
from GeoData.utilityfuncs import readIono
if __name__ == "__main__":
    inputdir = '/Users/Bodangles/Documents/Python/PhantomProc/Origparams'
    outdir = '/Users/Bodangles/Documents/MATLAB/phantoms/Origparams'
    dirlist = glob.glob(os.path.join(inputdir,'*.mat'))
    keep = ['Ne','Te','Ti']
    for ifile in dirlist:
        Ionoin=IonoContainer.readmat(ifile)
        GD = GeoData(readIono,[Ionoin])
        for ikey in GD.data.keys():
            if ikey not in keep:
                del GD.data[ikey]
        fname = os.path.splitext(os.path.split(ifile)[1])[0]
        GD.write_h5(os.path.join(outdir,fname+'.h5'))
        print('Saved '+os.path.join(outdir,fname+'.h5'))
def fit2geodata(inputfile):
    Ionoin=IonoContainer.readh5(inputfile)
    keep = ['Ne','Te','Ti']
    GD=GeoData(readIono,[Ionoin])
    for ikey in GD.data.keys():
        if ikey not in keep:
            del GD.data[ikey]
    indir,fnameall=os.path.split(inputfile)
    fname = os.path.splitext(fnameall)[0]
    GD.write_h5(os.path.join(indir,fname+'GEOD.h5'))
    print('Saved '+os.path.join(indir,fname+'.h5'))