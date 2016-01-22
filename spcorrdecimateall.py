#!/usr/bin/env python
"""
spcorrdecimateall.py

John Swoboda 
"""


import os
import numpy as np
import tables
import glob
import matplotlib.pyplot as plt
import pdb
import ioclass
from IQTools import CenteredLagProduct, FormatIQ

if __name__ == "__main__":
## Define input params Input
    spcor_dir = '/Volumes/ISRDrive/ISRData/20121207.002/'
    h5_files = glob.glob(os.path.join(spcor_dir,'*.dt2.h5'))
    #h5_files = ['/Volumes/ISRDrive/ISRData/20121207.002/d0330405.dt0.h5']
    h5_files.sort()
    print('File being processed '+ h5_files[0])
     ## Set up original beam patterns
    ptlen = 121
    rclen = 20*ptlen
    pattern1 = np.array([65228, 65288, 65225, 65291])
    npats = 10
    fullpat = np.array([pattern1[x%4] for x in np.arange(ptlen)])
    
    ## Output
    nLags = 12  
    Nranges = 228
    lags = np.arange(0,nLags)*20e-6  
    nrec_orig = 30
    mrecs_end = 27# this is the maximum number of recs at the end to deal with
    h5Paths = {'S'    :   ('/S',''),\
               'Data'           :   ('/S/Data',''),\
               'Data_Acf'       :   ('/S/Data/Acf',''),\
               'Data_Power'     :   ('/S/Data/Power',''),\
               }    

    # set up the output directory structure
    outpath = '/Volumes/ISRDrive/ISRDATA/OutData2/'
    outpaths = {x:os.path.join(outpath,'Pattern{:02}'.format(x)) for x in np.arange(npats)}
    for x in np.arange(npats):
        if not os.path.exists(outpaths[x]):
            os.makedirs(outpaths[x])
    
    file_recs = np.zeros(len(h5_files))
    file_count = 0
    ## Go through all files and get records
    for fname in h5_files: 
        #print('File being Looked at '+ fname)
        f_cur = tables.open_file(fname)
        file_recs[file_count] = f_cur.root.Raw11.Raw.RadacHeader.BeamCode.shape[0]
        f_cur.close()
        file_count+=1
        # get the sample location
    sample_ends = np.cumsum(file_recs)*rclen
    
    ## Get start point
    
    fname=h5_files[0]
    f_cur = tables.open_file(fname)
    all_beams_orig_mat =  f_cur.root.Raw11.Raw.RadacHeader.BeamCode.read()
    all_beams_orig = all_beams_orig_mat.flatten()
    f_cur.close()
    # Determine start point
    keepgoing = True
    stpnt = 0
    while keepgoing:
        subset_dat =all_beams_orig[stpnt:stpnt+ptlen]
        test_arr = subset_dat==fullpat
        if test_arr.all():
            break
        stpnt+=1
    
    ## Pull out the beam patterns
    # In this script x is used as the pattern iterator and y is used as the record indicator
    f_patternsdict = {(x):all_beams_orig[stpnt+x*2*ptlen:stpnt+(x+1)*2*ptlen] for x in np.arange(npats) }
    # on repeation of the beams
    patternsdict = {x:f_patternsdict[x][:(x+2)**2] for x in f_patternsdict.keys()}
    
    
    des_recs = 30    
    # determine the pattern
    patternlocdic_template = {(x):[(np.arange(x*2*ptlen+stpnt+y*rclen,(x+1)*2*ptlen+stpnt+y*rclen)) for y in np.arange(des_recs)] for x in np.arange(10)}
    
    
    ## Start loop for all files
    for file_num in np.arange(len(h5_files)):
        fname = h5_files[file_num]
        # bring in the contents of the full file because this structure will be needed when the new file is     
        # made
        fullfile = ioclass.h5file(fname)
        fullfiledict = fullfile.readWholeh5file()
        print('Main file being operated on: '+os.path.split(fname)[-1])
        # pull content that will be deleted
        all_data = fullfiledict['/Raw11/Raw/Samples']['Data']
        rng = fullfiledict['/S/Data/Acf']['Range']
        all_beams_mat = fullfiledict['/Raw11/Raw/RadacHeader']['BeamCode']
        
        txbaud = fullfiledict['/S/Data']['TxBaud']
        ambfunc = fullfiledict['/S/Data']['Ambiguity']
        pwidth = fullfiledict['/S/Data']['Pulsewidth']
        
        # Pull in call and noise material because these will needed for fitting
        beamcodes_cal = fullfiledict['/S/Cal']['Beamcodes']
        beamcodes_noise = fullfiledict['/S/Noise']['Beamcodes']
        cal_pint = fullfiledict['/S/Cal']['PulsesIntegrated']
        caldata = fullfiledict['/S/Cal/Power']['Data']
        noise_pint = fullfiledict['/S/Noise']['PulsesIntegrated']
        noise_pwer = fullfiledict['/S/Noise/Power']['Data']
        noise_data =fullfiledict['/S/Noise/Acf']['Data']
     
        
        # These keys lead to material will either conflict in the new file or will be unneccesary 
        dump = ['/S/Data','/S/Data/Acf','/S/Data/Power','/','/S','/Raw11/Raw/Samples']
        for key in dump:
            del fullfiledict[key]
        for key in fullfiledict:
            h5Paths[key] = (key,'')
    
        # Case for end file
        lastfile = file_num==len(h5_files)-1
        # add an extra record to the end of the arrays to deal with possible stralling data
        if not lastfile:
            fname2 = h5_files[file_num]
            # bring in the contents of the full file because this structure will be needed when the new file is     
            # made
            fullfile2 = ioclass.h5file(fname2)
            fullfiledict2 = fullfile2.readWholeh5file()
            all_data = np.concatenate((all_data,np.array([fullfiledict2['/Raw11/Raw/Samples']['Data'][0]])),0)
            all_beams_mat = np.concatenate((all_beams_mat,np.array([fullfiledict2['/Raw11/Raw/RadacHeader']['BeamCode'][0]])),0)
            #beamcodes_cal = np.concatenate((beamcodes_cal,np.array([fullfiledict2['/S/Cal']['Beamcodes'][0]])),0)
            #beamcodes_noise =np.concatenate(( beamcodes_noise,np.array([fullfiledict2['/S/Noise']['Beamcodes'][0]])),0)
            #cal_pint = np.concatenate((cal_pint, np.array([fullfiledict2['/S/Cal']['PulsesIntegrated'][0]])),0)
            #caldata = np.concatenate((caldata, np.array([fullfiledict2['/S/Cal/Power']['Data'][0]])),0)
            #noise_pint = np.concatenate((noise_pint, np.array([fullfiledict2['/S/Noise']['PulsesIntegrated'][0]])),0)
            #noise_pwer = np.concatenate((noise_pwer, np.array([fullfiledict2['/S/Noise/Power']['Data'][0]])),0)
            #noise_data =np.concatenate((noise_data, np.array([fullfiledict2['/S/Noise/Acf']['Data'][0]])),0)
            patternlocdic = patternlocdic_template.copy()
        else:
            
            des_recs = mrecs_end    
            # determine the pattern
            patternlocdic = {(x):[(np.arange(x*2*ptlen+stpnt+y*rclen,(x+1)*2*ptlen+stpnt+y*rclen)) for y in np.arange(des_recs)] for x in np.arange(10)}
            all_data = all_data[:des_recs+1]
            all_beams_mat = all_beams_mat[:des_recs+1]
            beamcodes_cal = beamcodes_cal[:des_recs]
            beamcodes_noise = beamcodes_noise[:des_recs]
            cal_pint = cal_pint[:des_recs]
            caldata = caldata[:des_recs]
            noise_pint = noise_pint[:des_recs]
            noise_pwer = noise_pwer[:des_recs]
            noise_data = noise_data[:des_recs]
        
        # first loop goes through patterns
        for x in np.arange(npats):
            
            # set up the outputfiles
            curoutpath =outpaths[x]
            bname = os.path.basename(fname)
            spl = bname.split('.')
            oname = os.path.join(curoutpath, spl[0]+'.' + spl[1] + '.proc.' + spl[2])
            
            # check the output files
            if os.path.exists(oname):
                os.remove(oname)
            # output file
            ofile = ioclass.outputFileClass()
            ofile.fname = oname
            ofile.openFile()
            ofile.h5Paths=h5Paths
            ofile.createh5groups()
            ofile.closeFile()
            
            # set up receivers and beams
            nrecs = len(patternlocdic[x])
            nbeams = len(patternsdict[x])        
                
            # Checks to make sure he arrays are set 
            
            #set up location arrays
            curbeams = patternsdict[x]
            cal_beam_loc = np.zeros(curbeams.shape)
            noise_beam_loc = np.zeros(curbeams.shape)
            cal_beam_loc = np.array([np.where(beamcodes_cal[0,:]==ib)[0][0] for ib in curbeams])
            noise_beam_loc = np.array([np.where(beamcodes_noise[0,:]==ib)[0][0] for ib in curbeams])
            #pdb.set_trace()
            
            # do all the call params
            fullfiledict['/S/Cal']['Beamcodes'] = beamcodes_cal[:,cal_beam_loc]
            fullfiledict['/S/Cal']['PulsesIntegrated'] = cal_pint[:,cal_beam_loc]
            fullfiledict['/S/Cal/Power']['Data'] = caldata[:,cal_beam_loc]
            # do all the noise params
            fullfiledict['/S/Noise']['Beamcodes'] = beamcodes_noise[:,noise_beam_loc]
            fullfiledict['/S/Noise']['PulsesIntegrated'] = noise_pint[:,noise_beam_loc]
            fullfiledict['/S/Noise/Power']['Data'] = noise_pwer[:,noise_beam_loc]
            fullfiledict['/S/Noise/Acf']['Data'] = noise_data[:,noise_beam_loc]
            irec = 0
            # second loop goes though all of the records
            for y in patternlocdic[x]:
                # determine the samples
                [arecs,asamps] = np.unravel_index(y,all_beams_mat.shape);
                # check if you've gone beyond the recordings
                arec_bey = np.any(arecs>=nrec_orig)
                # get the IQ data for all of the pulses in a pattern
                # this should keep the ordering
                fullIQ = FormatIQ(all_data,(arecs,asamps))
                
                # Beam by beam goes through the IQ data
                beamnum = 0
                # make holding arrays for acfs
                acf_rec = np.zeros((nbeams,nLags,Nranges,2))
                beams_rec =np.zeros((nbeams))
                pulsesintegrated = np.zeros((nbeams))
                pwr  = np.zeros((nbeams,Nranges))
                # fill in temp arrays
                for ibeam in patternsdict[x]:
                    cur_beam_loc = np.where(f_patternsdict[x]==ibeam)[0]
                    temp_lags = CenteredLagProduct(fullIQ[:,cur_beam_loc],nLags)
                    acf_rec[beamnum,:,:,0] = temp_lags.real.transpose()
                    acf_rec[beamnum,:,:,1] = temp_lags.imag.transpose()
                    beams_rec[beamnum] = ibeam
                    pulsesintegrated[beamnum] = len(cur_beam_loc)
                    pwr[beamnum,] = acf_rec[beamnum,0,:,0]
                    beamnum+=1
                    
                # pack the files with data from each record
                ofile.openFile()   
                ofile.createDynamicArray(ofile.h5Paths['Data_Power'][0]+'/Data',pwr)
                ofile.createDynamicArray(ofile.h5Paths['Data_Acf'][0]+'/Data',acf_rec)
                ofile.createDynamicArray(ofile.h5Paths['Data'][0]+'/PulsesIntegrated', pulsesintegrated)
                ofile.createDynamicArray(ofile.h5Paths['Data'][0]+'/Beamcodes',beams_rec)
                # pack the stuff that only is needed once
                if irec ==0:
                    ofile.createDynamicArray(ofile.h5Paths['Data_Acf'][0]+'/Range',rng[0,:])
                    ofile.createStaticArray(ofile.h5Paths['Data_Acf'][0]+'/Lags', lags[np.newaxis])
                    ofile.createDynamicArray(ofile.h5Paths['Data_Power'][0]+'/Range',rng[0,:][np.newaxis])
                    ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/TxBaud',txbaud)
                    ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/Ambiguity',ambfunc)
                    ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/Pulsewidth',pwidth)
                    # go through original file and get everything
                    for g_key in fullfiledict:
                        cur_group = fullfiledict[g_key]
                        for n_key in cur_group:
                            # 
                            if arec_bey and (type(cur_group[n_key])==np.ndarray):
                                #kluge
                                try:
                                    if cur_group[n_key].shape[0]==nrec_orig:
                                        cur_group[n_key][:-1] = cur_group[n_key][1:]
                                        cur_group[n_key][-1] = fullfiledict2[g_key][n_key][1]
                                except:
                                    pass
                                #check if last file
                            elif lastfile:
                                try:
                                    if cur_group[n_key].shape[0]==file_recs[-1]:
                                        cur_group[n_key] = cur_group[n_key][:mrecs_end]
                                except:
                                    pass
                            ofile.createStaticArray(ofile.h5Paths[g_key][0]+'/'+n_key,cur_group[n_key])
                # close the file
                ofile.closeFile()
                irec+=1
                
            print('\tData for Pattern '+str(x)+' has Finished')         
        
        
        
