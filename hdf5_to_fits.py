import h5py 
import numpy as np
from astropy.table import Table

def read_hdf5_file_to_dict(file,cols=None,indices=None,path='/'):
    hf = h5py.File(file, 'r')
    
    mygroup = hf[path]

    if cols is None: cols  = list(mygroup.keys())
    if indices is None: indices = np.arange(0,len(mygroup[cols[0]]),1,dtype=np.int64)

    mydict= dict().fromkeys(cols)
    for col in cols:
        arr         = np.array(mygroup[col][:])
        mydict[col] = arr[indices]
    
    hf.close()

    return mydict

infile  = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/output/fields/copa_00000.hdf'
outfile = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/subsample/buzzard_v2_00000_copa_input_{}.fits'

#infile  = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/input/buzzard_v2.0.0_00000.hdf5'
#outfile = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/subsample/buzzard_v2.0.0_00000.fits'

pathg='members/main/'
pathc='clusters/main/'

gal = Table(read_hdf5_file_to_dict(infile,path=pathg))
cls = Table(read_hdf5_file_to_dict(infile,path=pathc))

gal.write(outfile.format('members'),format='fits',overwrite=True)
cls.write(outfile.format('cluster'),format='fits',overwrite=True)
