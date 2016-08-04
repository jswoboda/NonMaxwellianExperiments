#!/usr/bin/env python
"""
Created on Fri Apr 15 15:00:12 2016
This script will be used to see if any changes will prevent the simulator from running.
@author: John Swoboda
"""

import os,inspect
import scipy as sp
import matplotlib.pyplot as plt
import pdb
from RadarDataSim.utilFunctions import readconfigfile,makeconfigfile,TempProfile,Chapmanfunc
from RadarDataSim.IonoContainer import IonoContainer
from  RadarDataSim.runsim import main as runsim 
from RadarDataSim.analysisplots import analysisdump
from GeoData.utilityfuncs import readMad_hdf5,readIono
from GeoData.GeoData import GeoData
from GeoData.plotting import rangevsparam

def plotdata(ionofile_in,ionofile_fit,madfile,time1):


    fig1,axmat =plt.subplots(2,2)
    axvec = axmat.flatten()
    paramlist = ['ne','te','ti','vo']
    paramlisti = ['Ne','Te','Ti','Vi']
    boundlist = [[0.,7e11],[500.,3200.],[500.,2500.],[-500.,500.]]
    IonoF = IonoContainer.readh5(ionofile_fit)
    IonoI = IonoContainer.readh5(ionofile_in)
    gfit = GeoData(readIono,[IonoF,'spherical'])
    ginp = GeoData(readIono,[IonoI,'spherical'])
    data1 = GeoData(readMad_hdf5,[madfile,['nel','te','ti','vo','dnel','dte','dti','dvo']])
    data1.data['ne']=sp.power(10.,data1.data['nel'])
    data1.data['dne']=sp.power(10.,data1.data['dnel'])
    handlist = []
    for inum,iax in enumerate(axvec):
        ploth = rangevsparam(data1,data1.dataloc[0,1:],time1,gkey=paramlist[inum],fig=fig1,ax=iax,it=False)
        handlist.append(ploth)
        ploth = rangevsparam(ginp,ginp.dataloc[0,1:],0,gkey=paramlisti[inum],fig=fig1,ax=iax,it=False)
        handlist.append(ploth)
        ploth = rangevsparam(gfit,gfit.dataloc[0,1:],0,gkey=paramlisti[inum],fig=fig1,ax=iax,it=False)
        handlist.append(ploth)
        iax.set_xlim(boundlist[inum])
    # with error bars
    fig1.suptitle('Comparison Without Error Bars')
    fig2,axmat2 =plt.subplots(2,2)
    axvec2 = axmat2.flatten()
    handlist2 = []
    for inum,iax in enumerate(axvec2):
        ploth = rangevsparam(data1,data1.dataloc[0,1:],time1,gkey=paramlist[inum],gkeyerr='d'+paramlist[inum],fig=fig2,ax=iax,it=False)
        handlist2.append(ploth)
        ploth = rangevsparam(ginp,ginp.dataloc[0,1:],0,gkey=paramlisti[inum],fig=fig2,ax=iax,it=False)
        handlist2.append(ploth)
        ploth = rangevsparam(gfit,gfit.dataloc[0,1:],0,gkey=paramlisti[inum],gkeyerr='n'+paramlisti[inum],fig=fig2,ax=iax,it=False)
        handlist2.append(ploth)
        iax.set_xlim(boundlist[inum])
    fig2.suptitle('Comparison With Error Bars')
    return (fig1,axvec,handlist,fig2,axvec2,handlist2)
def configfilesetup(testpath,npulses = 1400):
    """ This will create the configureation file given the number of pulses for 
        the test. This will make it so that there will be 12 integration periods 
        for a given number of pulses.
        Input
            testpath - The location of the data.
            npulses - The number of pulses. 
    """
    
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    defcon = os.path.join(curloc,'statsbase.ini')
    
    (sensdict,simparams) = readconfigfile(defcon)
    tint = simparams['IPP']*npulses
    ratio1 = tint/simparams['Tint']
    simparams['Tint']=ratio1 * simparams['Tint']
    simparams['Fitinter'] = ratio1 * simparams['Fitinter']
    simparams['TimeLim'] = 3*tint
    
    simparams['startfile']='startfile.h5'
    makeconfigfile(os.path.join(testpath,'stats.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
def makedata(testpath,tint):
    """ This will make the input data for the test case. The data will have cases
        where there will be enhancements in Ne, Ti and Te in one location. Each 
        case will have 3 integration periods. The first 3 integration periods will
        be the default set of parameters Ne=Ne=1e11 and Te=Ti=2000.
        Inputs
            testpath - Directory that will hold the data.
            tint - The integration time in seconds.
    """
    finalpath = os.path.join(testpath,'Origparams')
    if not os.path.isdir(finalpath):
        os.mkdir(finalpath)
    z = sp.linspace(50.,750,150)
    nz = len(z)
    Z_0 = 250.
    H_0=30.
    N_0=6.5e11
    c1 = Chapmanfunc(z,H_0,Z_0,N_0)+5e10
    z0=100.
    T0=600.
    Te,Ti = TempProfile(z,T0,z0)
    
    params = sp.zeros((nz,1,2,2))
    params[:,0,0,0] = c1
    params[:,0,1,0] = c1
    params[:,0,0,1] = Ti
    params[:,0,1,1] = Te
    coords = sp.column_stack((sp.zeros(nz),sp.zeros(nz),z))
    species=['O+','e-']
    times = sp.array([[0,1e3]])
    times2 = sp.column_stack((sp.arange(0,1),sp.arange(1,2)))*3*tint
    vel = sp.zeros((nz,1,3))
    vel2 = sp.zeros((nz,4,3))
    Icontstart = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel)
    Icont1 = IonoContainer(coordlist=coords,paramlist=params,times = times,sensor_loc = sp.zeros(3),ver =0,coordvecs =
        ['x','y','z'],paramnames=None,species=species,velocity=vel2)
        
    finalfile = os.path.join(finalpath,'0 stats.h5')
    Icont1.saveh5(finalfile)
    Icontstart.saveh5(os.path.join(testpath,'startfile.h5'))
    

def main(testpath,npulse = 1400 ,functlist = ['spectrums','radardata','fitting','analysis']):
    """ This function will call other functions to create the input data, config
        file and run the radar data sim. The path for the simulation will be 
        created in the Testdata directory in the RadarDataSim module. The new
        folder will be called BasicTest. The simulation is a long pulse simulation
        will the desired number of pulses from the user.
        Inputs
            npulse - Number of pulses for the integration period, default==100.
            functlist - The list of functions for the RadarDataSim to do.
    """
    
        
    curloc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    
    
    if not os.path.isdir(testpath):
        os.mkdir(testpath)
        
    functlist_default = ['spectrums','radardata','fitting']
    check_list = sp.array([i in functlist for i in functlist_default])
    check_run =sp.any( check_list) 
    functlist_red = sp.array(functlist_default)[check_list].tolist()

    
    configfilesetup(testpath,npulse)
    config = os.path.join(testpath,'stats.ini')
    (sensdict,simparams) = readconfigfile(config)
    makedata(testpath,simparams['Tint'])
    if check_run :
        runsim(functlist_red,testpath,config,True)
    if 'analysis' in functlist:
        analysisdump(testpath,config)

if __name__== '__main__':
    from argparse import ArgumentParser
    descr = '''
             This script will perform the basic run est for ISR sim.
            '''
    p = ArgumentParser(description=descr)
    p.add_argument('-b',"--basedir",help='The base directory for the stuff.')
    p.add_argument("-p", "--npulses",help='Number of pulses.',type=int,default=1400)
    p.add_argument('-f','--funclist',help='Functions to be uses',nargs='+',default=['spectrums','radardata','fitting','analysis'])#action='append',dest='collection',default=['spectrums','radardata','fitting','analysis'])
    
    p = p.parse_args()
    main(p.basedir,p.npulses,p.funclist)
   
