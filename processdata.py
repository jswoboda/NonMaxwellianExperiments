#!/usr/bin/env python
"""
Created on Tue Dec  1 14:26:38 2015

@author: John Swoboda
"""

import os,glob,inspect
import scipy as sp
from sricomptools import SRIparams2iono,SRIACF2iono
from RadarDataSim.runsim import main as runmain

def runsims(fitdir,simdir,configfile):
    fitconfig = os.path.join(fitdir,configfile)
    runmain(['fitting'],fitdir,fitconfig,False)
    
#    simconfig = os.path.join(simdir,configfile)
#    simlist = [ 'radardata', 'fitting']
#    runmain(simlist,simdir,simconfig,True)
    
    