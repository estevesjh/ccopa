import numpy as np
import h5py
import glob

from time import time
from joblib import Parallel, delayed
import scipy.integrate as integrate
from astropy.table import Table, vstack, join
from astropy.io.fits import getdata

import sys
sys.path.append("/home/s1/jesteves/git/ccopa/python/")
from main import copacabana
from make_input_files.upload_cosmoDC2 import upload_cosmoDC2_hf5_files
from make_input_files import read_hdf5_file_to_dict


def gaussian_photoz_buzzard(zsigma,nCores=60,emulator=False):
    t0     = time()
    ## Gaussian set up
    #infile = '/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/cosmoDC2_v1.1.4_copa.hdf5' ## laod infile
    files = glob.glob('/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/output/tiles/buzzard_v2.0.0_copa_golden*')
    emulator_infile = '/home/s1/berlfein/des40a/notebooks/emuPhotoZ_bpz_dnf_random_forest.pckl'
    zw_file= '/home/s1/jesteves/git/ccopa/aux_files/emuBPZ_correction_z_buzzard.txt'

    if not emulator:
        new_group = get_name_string(zsigma) ## e.g. gauss005
    else:
        new_group = 'emuBPZ'
    
    total_time=[]
    for infile in files:
        print('\nLoad Infile')
        print('infile: %s'%(infile))
        hf     = h5py.File(infile,'a')
        group  = hf['members/main']

        indict = dict()
        indict['redshift'] = group['redshift'][:]
        indict['z_true']   = group['z_true'][:]
        indict['mag']      = group['mag'][:]
        indict['magerr']   = group['magerr'][:]
        indict['CID']      = group['CID'][:]
        indict['mid']      = group['mid'][:]
        hf.close()

        # group          = hf['members/emuBPZ_zww/']
        # indict['z']    = group['z'][:]
        # indict['zerr'] = group['zerr'][:]
        # hf.close()

        print('Running make_gaussian_photoz')
        mkPz     = make_gaussian_photoz(zsigma,infile=emulator_infile)
        outdict  = mkPz.run(indict,zw_file,emulator=emulator,parallel=True,nCores=nCores)

        print('Writing outfile')
        write_gauss_outfile(infile,outdict,new_group,overwrite=True)
        
        t1 = (time()-t0)/60
        print('partial time: %.2f min\n'%(t1))
        total_time.append(t1)

    total_time = np.array(total_time)[-1]#.sum()
    print('Total time: %.2f min'%(total_time))

def get_name_string(zsigma):
    return 'gauss0'+str(zsigma).split('.')[1]

def write_gauss_outfile(infile,outdict,group,columns=None,overwrite=True):
    path    = 'members/%s'%group

    hf      = h5py.File(infile,'a')
    if group not in hf['members'].keys():
        hf.create_group(path)
    out     = hf[path]
    
    if columns is None:
        columns = ['mid','CID','z_true','z','zerr','pz0','zoffset']
    
    if 'z' in out.keys():
        if overwrite: delete_group(hf,path)

    for col in columns:
        out.create_dataset(col,data=outdict[col])
    hf.close()

class make_gaussian_photoz:
    """ Creates a fake gaussian photo-z
    """
    def __init__(self,zwindow,infile=None):
        self.zwindow = zwindow      ## gaussian std error
        self.emu_infile = infile

    def run(self,mydict,zwindow_file,emulator=False,parallel=True,nCores=40,npoints=1000,seed=42):
        zwindow = self.zwindow

        cidxs = mydict['CID']
        zcls  = mydict['redshift']
        ztrue = mydict['z_true']
        mag   = [mydict['mag'][:,i]    for i in range(4)]
        magerr= [mydict['magerr'][:,i] for i in range(4)]

        np.random.seed(seed)
        if not emulator:
            zerr = zwindow*np.ones_like(ztrue)
            znoise= ztrue+np.random.normal(scale=zwindow,size=ztrue.size)*(1+ztrue)
            znoise= np.where(znoise<0.,0.,znoise) # there is no negative redshift
            zoffset = (znoise-zcls)/(1+zcls)
            zwindow = zwindow*np.ones_like(zcls)
        else:
            znoise, zerr = load_emulator(ztrue,mag,magerr,self.emu_infile)
            # znoise = mydict['z']-0.092
            # zerr   = mydict['zerr']
            znoise  -= 0.092
            znoise  = np.where(znoise<0.,0.,znoise)
            zoffset = (znoise-zcls)/(1+zcls)
            # zwindow = zwindow*np.ones_like(zcls)

            ## make corrections
            zres    = np.genfromtxt(zwindow_file,delimiter=',')
            zb,mean,sigma = zres[:,0],zres[:,1],zres[:,2]
            zwindow = np.interp(zcls,zb,sigma) ##np.interp(zcls,zb,sigma)
            zoffset = zoffset#-np.interp(zcls,zb,mean)

        print('Compute pz,0')
        ## parelizar
        if parallel:
            pz0     = compute_pdfz_parallel(cidxs,zoffset,zerr,zcls,zwindow,nCores=nCores)
        # else:
        #     pz0     = compute_pdfz(zoffset,zerr,zcls,zwindow)

        ## updating columns
        mydict['z']      = znoise
        mydict['zerr']   = zerr
        mydict['pz0']    = pz0
        mydict['zoffset']= zoffset

        return mydict

def check_boundaries(zmin,zcls):
    zoff_min = zcls+zmin*(1+zcls)
    if zoff_min<0:
        return zmin-zoff_min
    else:
        return zmin

zgrid = np.arange(0.005,3.01,0.01)
def compute_pdfz_bpz(pdfz,zcls,sigma):
    w,  = np.where( np.abs(zgrid-zcls) <= 1.5*sigma*(1+zcls) ) ## integrate in 1.5*sigma
    p0 = integrate.trapz(pdfz[:,w],x=zgrid[w])
    pz = np.where(p0>1., 1., p0)

    #pz = np.where(np.abs(zoffset) >= 3*sigma, 0., pz/np.max(pz))
    pz  = pz/(1.*np.max(pz))
    return pz

def compute_pdfz(zoffset,membzerr,sigma,zcls,npoints=1000):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.

        npoints=1000 # it's accurate in 2%% level
    '''  
    zmin, zmax = -5*sigma, 5*sigma
    zmin = check_boundaries(zmin,zcls)
    
    z       = np.linspace(zmin,zmax,npoints)
    zz, yy  = np.meshgrid(z,np.array(zoffset))
    zz, yy2 = np.meshgrid(z,np.array(membzerr))
    
    pdfz = gaussian(zz,yy,yy2)
    
    w,  = np.where( np.abs(z) <= 1.5*sigma) ## integrate in 1.5*sigma
    p0 = integrate.trapz(pdfz[:,w],x=zz[:,w])
    pz = np.where(p0>1., 1., p0)

    #w,  = np.where( np.abs(z) <= 1.5*sigma) ## integrate in 1.5*sigma
    #a   = np.cumsum(pdfz, axis=1)/np.sum(pdfz, axis=1)[:,np.newaxis]
    #pz  = a[:,w[-1]]-a[:,w[0]]

    ## get out with galaxies outside 5 sigma
    # pz = np.where(np.abs(zoffset) >= 3*sigma, 0., pz)
    pz = np.where(np.abs(zoffset) >= 3*sigma, 0., pz/np.max(pz))

    return pz

def compute_pdfz_old(membz,membzerr,sigma,zcls):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    zmin, zmax = (zcls-5*sigma), (zcls+5*sigma)
    if zmin<0: zmin=0.

    z = np.linspace(zmin,zmax,200)
    zz, yy = np.meshgrid(z,np.array(membz))
    zz, yy2 = np.meshgrid(z,np.array(membzerr))
    
    # if (zmin>0.001)&(zmax<=1.2):
    #     pdfc = gaussian(zz,zcls,sigma)
    #     pdfz = gaussian(zz,yy,yy2)
    # else:

    # pdfc = truncatedGaussian(zz,zcls,zmin,zmax,sigma)
    pdfz = truncatedGaussian(zz,yy,zmin,zmax,yy2,vec=True)

    pos = pdfz#*pdfc
    # norm_factor = integrate.trapz(pos,x=zz)
    # inv_factor = np.where(norm_factor[:, np.newaxis]<1e-3,0.,1/norm_factor[:, np.newaxis])

    # pdf = pos/norm_factor[:, np.newaxis] ## set pdf to unity
    # pdf[np.isnan(pdf)] = 0.
    
    pdf=pdfz
    w, = np.where( np.abs(z-zcls) <= 1.5*sigma) ## integrate in 1.5*sigma
    p0 = integrate.trapz(pdf[:,w],x=zz[:,w])
    p0 = np.where(p0>1., 1., p0)

    ## get out with galaxies outside 3 sigma
    zmin, zmax = (zcls-3*sigma), (zcls+3*sigma)
    if zmin<0: zmin=0.
    p0 = np.where((np.array(membz) < zmin )&(np.array(membz) > zmax), 0., p0)

    # p0 = np.where((yy2>0.2)&(np.abs(yy)>0.2),0.,yy)
    
    # w = np.argmin(np.abs(z-zcls))
    # pdf = pdf[:,w]#*0.01

    # pdf /= np.max(pdf)
    # pdf = np.where(pdf>1,1.,pdf)

    return p0

def compute_pdfz_bpz_parallel(cidxs,pdfz,zcls,zwindow,nCores=40):
    cids,indices = np.unique(cidxs,return_index=True)
    zcls    = zcls[indices]
    zwindow = zwindow[indices]
    
    ncls = len(cids)
    ngals= len(cidxs)

    keys   = list(chunks(cidxs,cids))
    pz_out = np.zeros((ngals,),dtype=np.float64)

    out    = Parallel(n_jobs=nCores)(delayed(compute_pdfz_bpz)(pdfz[idx,:], zcls[i], zwindow[i])for i,idx in enumerate(keys))

    for i,idx in enumerate(keys):
        if len(out[i])==idx.size:
            pz_out[idx] = out[i]
        else:
            print('error')
    return pz_out


def compute_pdfz_parallel(cidxs,zoffset,zerr,zcls,zwindow,nCores=40,npoints=1000,bpz=False,pdfz=None):
    cids,indices = np.unique(cidxs,return_index=True)
    zcls    = zcls[indices]
    zwindow = zwindow[indices]
    
    ncls = len(cids)
    ngals= len(cidxs)

    keys   = list(chunks(cidxs,cids))
    pz_out = np.zeros((ngals,),dtype=np.float64)
    if not bpz:
        out    = Parallel(n_jobs=nCores)(delayed(compute_pdfz)(zoffset[idx], zerr[idx], zwindow[i], zcls[i],
                                                            npoints=npoints) for i,idx in enumerate(keys))
    else:
        out    = Parallel(n_jobs=nCores)(delayed(compute_pdfz_bpz)(pdfz[idx], zcls[i], zwindow[i])
                                                                   for i,idx in enumerate(keys))

    for i,idx in enumerate(keys):
        if len(out[i])==idx.size:
            pz_out[idx] = out[i]
        else:
            print('error')
    return pz_out

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def truncatedGaussian(z,zcls,zmin,zmax,sigma,vec=False):
    from scipy.stats import truncnorm
    if vec:
        z0,zcls0,sigma0 = z,zcls,sigma
        s_shape = sigma.shape

        sigma = sigma.ravel()
        z = z.ravel()
        zcls = zcls.ravel()

    # user input
    myclip_a = zmin
    myclip_b = zmax
    my_mean = zcls
    my_std = sigma
    eps = 1e-9

    a, b = (myclip_a - my_mean) / (my_std+eps), (myclip_b - my_mean) / (my_std+eps) 
    # try:
    pdf = truncnorm.pdf(z, a, b, loc = my_mean, scale = (my_std+eps))
    if vec: pdf.shape = s_shape
        
    # except:
    #     print('PDFz error: ecception')
    #     pdf = gaussian(z0,zcls0,sigma0)

    return pdf

def gaussian(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def delete_group(fmaster,path):
    group   = fmaster[path]
    cols    = group.keys()
    
    for col in cols:
        del group[col]

def load_emulator(z_true,mag,magerr,infile,offset=0.,correction_file='emuBPZ_correction_z.txt'):
    ztrue = [z_true]
    colors= [(mag[0]-mag[1]),(mag[1]-mag[2]),(mag[2]-mag[3])]
    Xnew  = np.array(ztrue+mag+magerr+colors).T

    print('load emulator file')
    # import pickle
    # loaded_model = pickle.load(open(infile, 'rb'))
    # loaded_model = load_cpickle_gc(infile)
    bpz_model_mean, bpz_model_error, dnf_model_mean, dnf_model_error = loaded_model

    print('assign predictions')
    z_pred = bpz_model_mean.predict(Xnew)
    zerr = (bpz_model_error.predict(Xnew))

    # zcorr  = np.genfromtxt(correction_file,delimiter=',')
    # z_noise= z_pred*np.interp(z_true, zcorr[:,0], zcorr[:,1])
    z_noise= z_pred-offset

    return z_noise,10**zerr

def load_cpickle_gc(mypickle):
    import cPickle as pickle
    import gc
    output = open(mypickle, 'rb')

    # disable garbage collector
    gc.disable()

    mydict = pickle.load(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return mydict


if __name__ == '__main__':
    infile = '/home/s1/berlfein/des40a/notebooks/emuPhotoZ_bpz_dnf_random_forest.pckl'
    loaded_model = load_cpickle_gc(infile)
    global loaded_model

    gaussian_photoz_buzzard(0.03,nCores=60,emulator=True)
    #main_bpz()
