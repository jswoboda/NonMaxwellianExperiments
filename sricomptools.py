#!/usr/bin/env python
"""
Created on Tue Nov  3 15:38:50 2015

@author: John Swoboda
"""
import glob,os
from ioclass import h5file
import scipy as sp
from RadarDataSim.radarData import lagdict2ionocont
from RadarDataSim.utilFunctions import makeparamdicts, makesumrule, makeconfigfile
from RadarDataSim.IonoContainer import IonoContainer
from RadarDataSim.const.physConstants import v_Boltz

def SRIparams2iono(filename):

    fullfile = h5file(filename)
    fullfiledict = fullfile.readWholeh5file()

    #Size = Nrecords x Nbeams x Nranges x Nions+1 x 4 (fraction, temperature, collision frequency, LOS speed)
    fits = fullfiledict['/FittedParams']['Fits']
    (nt,nbeams,nrng,nspecs,nstuff) = fits.shape
    nlocs = nbeams*nrng
    fits = fits.transpose((1,2,0,3,4))
    fits = fits.reshape((nlocs,nt,nspecs,nstuff))
    #  Nrecords x Nbeams x Nranges
    Ne = fullfiledict['/FittedParams']['Ne']
    Ne = Ne.transpose((1,2,0))
    Ne = Ne.reshape((nlocs,nt))
    param_lists =sp.zeros((nlocs,nt,nspecs,2))
    param_lists[:,:,:,0] = fits[:,:,:,0]
    param_lists[:,:,:,1] = fits[:,:,:,1]
    param_lists[:,:,-1,0]=Ne
    Velocity = fits[:,:,0,3]


    if fullfiledict['/FittedParams']['IonMass']==16:
        species = ['O+','e-']
        pnames = sp.array([['Ni','Ti'],['Ne','Te']])

    time= fullfiledict['/Time']['UnixTime']
    time = time
    rng = fullfiledict['/FittedParams']['Range']
    bco = fullfiledict['/']['BeamCodes']
    angles = bco[:,1:3]
    (nang,nrg) = rng.shape

    allang = sp.tile(angles[:,sp.newaxis],(1,nrg,1))
    all_loc = sp.column_stack((rng.flatten(),allang.reshape(nang*nrg,2)))
    lkeep = ~ sp.any(sp.isnan(all_loc),1)
    all_loc = all_loc[lkeep]
    Velocity = Velocity[lkeep]
    param_lists = param_lists[lkeep]
    iono1 = IonoContainer(all_loc,param_lists,times=time,ver = 1,coordvecs = ['r','theta','phi'],
                          paramnames = pnames,species=species,velocity=Velocity)
                          
                          
                          
    # MSIS
    tn = fullfiledict['/MSIS']['Tn']
    tn = tn.transpose((1,2,0))
    tn = tn.reshape((nlocs,nt))
    
    
    startparams = sp.ones((nlocs,nt,2,2))
    startparams[:,:,0,1] = tn
    startparams[:,:,1,1] = tn
    startparams = startparams[lkeep]
    ionoS = IonoContainer(all_loc,startparams,times=time,ver = 1,coordvecs = ['r','theta','phi'],
                          paramnames = pnames,species=species)
    return iono1,ionoS


def SRIACF2iono(flist):
    for iflistn, iflist in enumerate(flist):

        for ifile,filename in enumerate(iflist):
            fullfile = h5file(filename)
            fullfiledict = fullfile.readWholeh5file()

            if fullfiledict['/Site']['Name'] =='Resolute North':
                radarname='risr'
            else:
                radarname='pfisr'
            Pcal=fullfiledict['/Rx']['Bandwidth']*fullfiledict['/Rx']['CalTemp']*v_Boltz

            bco = fullfiledict['/S/Data']['Beamcodes']
            beamlist = bco[0]
            time= fullfiledict['/Time']['UnixTime']
#            if ifile==0 and iflistn==0:
#                time0=time[0,0]
#            time=time-time0
            #nt x nbeams x lags x range x 2
            acfs = fullfiledict['/S/Data/Acf']['Data']
            acfs = acfs[:,:,:,:,0]+1j*acfs[:,:,:,:,1]
            (nt,nbeams,nlags,nrng) =acfs.shape
            acfrng = fullfiledict['/S/Data/Acf']['Range'].flatten()
            acf_pint = fullfiledict['/S/Data']['PulsesIntegrated']


            pwidth = fullfiledict['/S/Data']['Pulsewidth']

            # Pull in call and noise material because these will needed for fitting

            cal_pint = fullfiledict['/S/Cal']['PulsesIntegrated']
            #nt x nbeams x range
            caldata = fullfiledict['/S/Cal/Power']['Data']
            caldata = sp.nanmedian(caldata,axis=-1)
            caldata=caldata/cal_pint

            noise_pint = fullfiledict['/S/Noise']['PulsesIntegrated']
            noise_pwer = fullfiledict['/S/Noise/Power']['Data']
            noise_pwer = sp.nanmedian(noise_pwer,axis=-1)
            noise_pwer = noise_pwer/noise_pint


            caldata = caldata-noise_pwer
            caldataD = sp.tile(caldata[:,:,sp.newaxis,sp.newaxis],(1,1,nlags,nrng))
            #nt x nbeams x lags x range x 2
            noise_data = fullfiledict['/S/Noise/Acf']['Data']
            noise_data = noise_data[:,:,:,:,0]+1j*noise_data[:,:,:,:,1]
            nnrg = noise_data.shape[-1]
            caldataN = sp.tile(caldata[:,:,sp.newaxis,sp.newaxis],(1,1,nlags,nnrg))
            # subtract noise and
            acfs = Pcal*acfs/caldataD
            acfs = acfs.transpose((0,1,3,2))
            noise_data = Pcal*noise_data/caldataN
            noise_data = noise_data.transpose((0,1,3,2))
            # Create output dictionaries and output data
            if ifile==0:
                acfs_sum=acfs.copy()
                acf_pint_sum = acf_pint.copy()
                noise_data_sum = noise_data.copy()
                noise_pint_sum = noise_pint.copy()

            else:
                acfs_sum=acfs_sum+acfs
                acf_pint_sum = acf_pint_sum+acf_pint
                noise_data_sum = noise_data_sum+ noise_data
                noise_pint_sum = noise_pint_sum + noise_pint

        if iflistn==0:

            acf_acum=acfs_sum.copy()
            acf_pint_acum = acf_pint_sum.copy()
            noise_data_acum = noise_data_sum.copy()
            noise_pint_acum = noise_pint_sum.copy()
            Time_acum =time.copy()
        else:
            acf_acum=sp.vstack((acf_acum,acfs_sum))
            acf_pint_acum = sp.vstack((acf_pint_acum,acf_pint_sum))
            noise_data_acum = sp.vstack((noise_data_acum,noise_data_sum))
            noise_pint_acum= sp.vstack((noise_pint_acum,noise_pint_sum))
            Time_acum = sp.vstack((Time_acum,time))
    
    DataLags = {'ACF':acf_acum,'Pow':acf_acum[:,:,:,0].real,'Pulses':acf_pint_acum,'Time':Time_acum}
    NoiseLags = {'ACF':noise_data_acum,'Pow':noise_data_acum[:,:,:,0].real,'Pulses':noise_pint_acum,'Time':Time_acum}

    rng_vec = acfrng
    ts = fullfiledict['/Rx']['SampleTime']
    sumrule = makesumrule('long',fullfiledict['/S/Data']['Pulsewidth'],ts)
    minrg = -sumrule[0].min()
    maxrg = len(rng_vec)-sumrule[1].max()
    rng_lims = [rng_vec[minrg],rng_vec[maxrg]]# limits of the range gates
    IPP = .0087 #interpulse period in seconds


    simparams =   {'IPP':IPP, #interpulse period
                   'TimeLim':time[-1,1], # length of simulation
                   'RangeLims':rng_lims, # range swath limit
#                   'Pulse':pulse, # pulse shape
                   'Pulselength':pwidth,
                   'FitType' :'acf',
                   't_s': ts,
                   'Pulsetype':'long', # type of pulse can be long or barker,
                   'Tint':time[0,1]-time[0,0], #Integration time for each fitting
                   'Fitinter':time[1,0]-time[0,0], # time interval between fitted params
                   'NNs': 100,# number of noise samples per pulse
                   'NNp':100, # number of noise pulses
                   'dtype':sp.complex128, #type of numbers used for simulation
                   'ambupsamp':1, # up sampling factor for ambiguity function
                   'species':['O+','e-'], # type of ion species used in simulation
                   'numpoints':128} # number of points for each spectrum

    (sensdict,simparams) = makeparamdicts(beamlist,radarname,simparams)
    simparams['Rangegates'] =rng_vec
    simparams['Rangegatesfinal']=rng_vec[minrg:maxrg]

    ionolag, ionosigs = lagdict2ionocont(DataLags,NoiseLags,sensdict,simparams,time)

    return (ionolag,ionosigs,simparams,sensdict)

def makefitdirectory(sridir,srifile,fitdir,simdir):

    srifiles = [glob.glob(os.path.join(sridir,'*.dt{0}.h5'.format(i))) for i in range(3)]

    if [] in srifiles:
        srifiles.remove([])
     
    srifilesnew = [[None]*len(srifiles)]*len(srifiles[0])
    for ieln in range(len(srifiles[0])):
        srifilesnew[ieln] =[i[ieln] for i in srifiles ]

         
    
    # make iono conatainers out of the SRI data    
    (ionolag,ionosigs,simparams,sensdict) = SRIACF2iono(srifilesnew)
    
    paramsiono,startiono = SRIparams2iono(srifile)
    # Add diretories that are needed
    mkdirlist = [fitdir,simdir]

    mkdirlist.append( os.path.join(fitdir,'ACF'))
    mkdirlist.append(os.path.join(fitdir,'Origparams'))
    mkdirlist.append(os.path.join(simdir,'Origparams'))
    for idir in mkdirlist:    
        if not os.path.isdir(idir):
            os.mkdir(idir)
    
    # Save ACFs
    ionolag.saveh5(os.path.join(mkdirlist[-3],'00lags.h5'))
    ionosigs.saveh5(os.path.join(mkdirlist[-3],'00sigs.h5'))
    # Save fitted data
    paramsiono.saveh5(os.path.join(mkdirlist[-2],'0 srifits.h5'))
    paramsiono.saveh5(os.path.join(mkdirlist[-1],'0 srifits.h5'))
    # Save start data
    startiono.saveh5(os.path.join(mkdirlist[0],'startdata.h5'))
    startiono.saveh5(os.path.join(mkdirlist[1],'startdata.h5'))
    # Save ini files
    makeconfigfile(os.path.join(mkdirlist[0],'sridata.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    makeconfigfile(os.path.join(mkdirlist[1],'sridata.ini'),simparams['Beamlist'],sensdict['Name'],simparams)
    
    