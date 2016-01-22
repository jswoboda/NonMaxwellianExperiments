#!/usr/bin/env python

"""

"""

import tables, os, distutils.dir_util, scipy

class h5file():

    def __init__(self,fname):
        """ initialization function """
        self.fname = fname
        self.fhandle = None        
        return
    
    def openFile(self):
        """ open file self.fname """
        self.fhandle = tables.openFile(self.fname, mode = "a")
        return
    
    def readWholeh5file(self):
        
        h5file=tables.openFile(self.fname)
        output={}
        for group in h5file.walkGroups("/"):
            output[group._v_pathname]={}
            for array in h5file.listNodes(group, classname = 'Array'):
                output[group._v_pathname][array.name]=array.read()
        h5file.close()
         
        return output

class outputFileClass:
    
    def __init__(self):
        """ initialization function """
        self.h5Paths={}
        self.h5Attribs = {}
        self.fname = ''
        self.title = ''
        self.fhandle = None
        return
    
    def createFile(self, fname):
        """ create file fname, set self.fname and self.fhandle """
        self.fname = fname
        try:
            distutils.dir_util.mkpath(os.path.dirname(self.fname))
        except:
            raise IOError, 'Unable to create output path %s' % os.path.dirname(self.fname)
        self.fhandle=tables.openFile(self.fname, mode = "w", title = self.title)
        return
    
    def openFile(self):
        """ open file self.fname """
        self.fhandle = tables.openFile(self.fname, mode = "a")
        return
        
    def closeFile(self):
        """ close self.fhandle """
        self.fhandle.close()
        
    def createh5groups(self):
        """ creates groups """
        tvals = sorted(self.h5Paths.values())
        for v0,v1 in tvals:
            gp,gn = os.path.split(v0)     
            self.fhandle.createGroup(gp,gn,v1)
        return
    
    def createStaticArray(self,path,data,keys2do=[]):  
        """ creates a static array """
        if len(keys2do)==0:
            dp,dn = os.path.split(path)
            self.fhandle.createArray(dp,dn,data,'Static array')
        else:
            for key in keys2do:
                self.fhandle.createArray(path,key,data[key],'Static array')
        return
    
    def createDynamicArray(self,path,rec,keys2do=[]):  
        """ creates a dynamic array """
        if len(keys2do)==0:
            dp,dn = os.path.split(path)
            data = rec.copy()
            data.shape = (1,)+data.shape  ## add integration dimension to data array
            if not self.fhandle.__contains__(path):
                shape = list(data.shape)
                shape[0] = 0
                atom = tables.Atom.from_dtype(data.dtype)
                arr = self.fhandle.createEArray(dp,dn,atom,shape)
                arr.flavor='numpy'
            arr = self.fhandle.getNode(path)
            if (len(arr.shape)>2) and (data.shape[2] != arr.shape[2]):
                if data.shape[2] > arr.shape[2]:
                    # read array 
                    tarr = arr.read() 
                    # remove old node                 
                    arr.remove() 
                    tshape=list(tarr.shape); tshape[2]=data.shape[2]-tarr.shape[2]
                    tarr=scipy.append(tarr,scipy.zeros(tshape)*scipy.nan,axis=2)   
                    # create new node                 
                    shape = list(tarr.shape)
                    shape[0] = 0
                    atom = tables.Atom.from_dtype(tarr.dtype)
                    arr = self.fhandle.createEArray(dp,dn,atom,shape)
                    arr.flavor='numpy'
                    arr = self.fhandle.getNode(path)
                    # dump data
                    arr.append(tarr)
                else:
                    tshape=list(data.shape); tshape[2]=arr.shape[2]-data.shape[2]
                    data=scipy.append(data,scipy.zeros(tshape)*scipy.nan,axis=2)
            arr.append(data)
        else:
            for key in keys2do:
                data = scipy.array(rec[key])
                data.shape = (1,)+data.shape  ## add integration dimension to data array
                if not self.fhandle.__contains__(path+'/'+key):
                    shape = list(data.shape)
                    shape[0] = 0
                    atom = tables.Atom.from_dtype(data.dtype)
                    arr = self.fhandle.createEArray(path,key,atom,shape)
                    arr.flavor='numpy'
                arr = self.fhandle.getNode(path+'/'+key)
                if (len(arr.shape)>2) and (data.shape[2] != arr.shape[2]):
                    if data.shape[2] > arr.shape[2]:
                        # read array  
                        tarr = arr.read() 
                        # remove old node                 
                        arr.remove() 
                        tshape=list(tarr.shape); tshape[2]=data.shape[2]-tarr.shape[2]
                        tarr=scipy.append(tarr,scipy.zeros(tshape)*scipy.nan,axis=2)   
                        # create new node                 
                        shape = list(tarr.shape)
                        shape[0] = 0
                        atom = tables.Atom.from_dtype(tarr.dtype)
                        arr = self.fhandle.createEArray(path,key,atom,shape)
                        arr.flavor='numpy'
                        arr = self.fhandle.getNode(path+'/'+key)
                        # dump data
                        arr.append(tarr)
                    else:
                        tshape=list(data.shape); tshape[2]=arr.shape[2]-data.shape[2]
                        data=scipy.append(data,scipy.zeros(tshape)*scipy.nan,axis=2)
                arr.append(data)        
        return
    
    def setAtrributes(self):
        for key in self.h5Attribs.keys():
            for attr in self.h5Attribs[key]:
                try:  self.fhandle.setNodeAttr(key,attr[0],attr[1])
                except: ''
        return
