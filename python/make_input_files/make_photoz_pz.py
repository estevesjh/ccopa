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

# emulator_infile = '/home/s1/berlfein/des40a/notebooks/emuPhotoZ_bpz_dnf_random_forest.pckl'
# zw_file= '/home/s1/jesteves/git/ccopa/aux_files/emuBPZ_correction_z_buzzard.txt

class photoz_model:
    """ Photo-z Model
    """
    def __init__(self,zwindow=0.03,zmodel_file=None):
        self.zwindow      = zwindow      ## gaussian std error
        self.zmodel_file  = zmodel_file

    def load_data(self,fname,simulation=True):
        print('\nLoad Infile')
        print('infile: %s'%(fname))
        hf     = h5py.File(fname,'a')
        group  = hf['members/main']

        indict = dict()
        indict['z']        = group['z'][:]
        indict['zerr']     = group['zerr'][:]
        indict['zoffset']  = group['zoffset'][:]
        indict['redshift'] = group['redshift'][:]
        indict['mag']      = group['mag'][:]
        indict['magerr']   = group['magerr'][:]
        indict['CID']      = group['CID'][:]
        indict['mid']      = group['mid'][:]
        if simulation:
            indict['z_true']   = group['z_true'][:]
        hf.close()
        
        self.data = indict

    def generate_gaussian_photoz(self,seed=42):
        zwindow = self.zwindow

        ## load variables
        cidxs = self.data['CID'][:]
        zcls  = self.data['redshift'][:]
        ztrue = self.data['z_true'][:]
        
        ## make gaussian photo-z
        np.random.seed(seed)
        zerr = zwindow*np.ones_like(ztrue)*(1+ztrue)

        znoise= ztrue+np.random.normal(scale=zwindow,size=ztrue.size)*(1+ztrue)
        znoise= np.where(znoise<0.,0.,znoise) # there is no negative redshift

        ## compute zoffset
        zoffset     = (znoise-zcls)/(1+zcls)

        ## updating columns
        self.data['z']      = znoise
        self.data['zerr']   = zerr
        self.data['zoffset']= zoffset
        self.data['zwindow']= zerr

    def emulate_photoz(self,loaded_model,offset=0.092):
        mydict  = self.data.copy()
        zwindow = self.zwindow

        ## seting variables 
        zcls  = mydict['redshift']
        ztrue = [mydict['z_true']]
        mag   = [mydict['mag'][:,i]    for i in range(4)]
        magerr= [mydict['magerr'][:,i] for i in range(4)]

        colors= [(mag[0]-mag[1]),(mag[1]-mag[2]),(mag[2]-mag[3])]
        Xnew  = np.array(ztrue+mag+magerr+colors).T

        ## loading the model
        bpz_model_mean, bpz_model_error, dnf_model_mean, dnf_model_error = loaded_model

        ## prediction the photoz, photoz_error
        z_pred = bpz_model_mean.predict(Xnew)
        z_noise= z_pred-offset
        zerr   = 10**bpz_model_error.predict(Xnew)
        
        ## computing zoffset
        zoffset = (znoise-zcls)/(1+zcls)

        ## updating columns
        self.data['z']      = znoise
        self.data['zerr']   = zerr
        self.data['zoffset']= zoffset
        self.data['zwindow']= zwindow*np.ones_like(z_noise)

    def model_photoz_bias(self,exp_factor=True):
        zcls = self.data['redshift']

        ## make corrections
        if self.zmodel_file is not None:
            zres    = np.genfromtxt(self.zmodel_file,delimiter=',')
            zb,mean,sigma = zres[:,0],zres[:,1],zres[:,2]
            
            bias      = np.interp(zcls,zb,mean)
            zwindow_z = np.interp(zcls,zb,sigma) ##np.interp(zcls,zb,sigma)

            if exp_factor:
                zwindow_z *= (1+zcls)
            
            self.data['zoffset'] = self.data['zoffset']-bias
            self.data['zwindow'] = zwindow_z
        else:
            print('Error: zmodel file is None')

    def compute_photoz_probability(self,nCores=4,method=None):
        ## method default is probabilistic
        d   = self.data
        if method is None:
            pz0 = compute_pdfz_parallel(d['CID'][:],d['z'][:],d['zerr'][:],d['redshift'][:],d['zwindow'][:],nCores=nCores)
        else:
            pz0 = np.where(d['zoffset']<=3.*d['zwindow']/(1+d['redshift']),1.,0.)

        self.data['pz0']    = pz0
        pass

def get_name_string(zsigma):
    return 'gauss0'+str(zsigma).split('.')[1]

def write_gauss_outfile(infile,outdict,group,columns=None,overwrite=True):
    path    = 'members/%s'%group

    hf      = h5py.File(infile,'a')
    if group not in hf['members'].keys():
        hf.create_group(path)
    out     = hf[path]
    
    if columns is None:
        columns = ['mid','CID','z_true','z','zerr','pz0','zoffset','zwindow']
    
    if 'z' in out.keys():
        if overwrite: delete_group(hf,path)

    for col in columns:
        out.create_dataset(col,data=outdict[col])
    hf.close()

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

# def compute_pdfz(zoffset,membzerr,sigma,zcls,npoints=1000):
#     ''' Computes the probability of a galaxy be in the cluster
#         for an interval with width n*windows. 
#         Assumes that the galaxy has a gaussian redshift PDF.

#         npoints=1000 # it's accurate in 2%% level
#     '''  
#     zmin, zmax = -5*sigma, 5*sigma
#     zmin = check_boundaries(zmin,zcls)
    
#     ## photo-z floor
#     membzerr= np.where(membzerr<0.005,0.005,membzerr)

#     ## multi dymensional arrays
#     z       = np.linspace(zmin,zmax,npoints)
#     zz, yy  = np.meshgrid(z,np.array(zoffset))
#     zz, yy2 = np.meshgrid(z,np.array(membzerr))
    
#     pdfz = gaussian(zz,yy,yy2)
    
#     w,  = np.where( np.abs(z) <= 1.5*sigma) ## integrate in 1.5*sigma
#     p0 = integrate.trapz(pdfz[:,w],x=zz[:,w])
#     pz = np.where(p0>1., 1., p0)

#     #w,  = np.where( np.abs(z) <= 1.5*sigma) ## integrate in 1.5*sigma
#     #a   = np.cumsum(pdfz, axis=1)/np.sum(pdfz, axis=1)[:,np.newaxis]
#     #pz  = a[:,w[-1]]-a[:,w[0]]

#     ## get out with galaxies outside 5 sigma
#     # pz = np.where(np.abs(zoffset) >= 3*sigma, 0., pz)
#     pz = np.where(np.abs(zoffset) >= 3*sigma, 0., pz/np.max(pz))
#     return pz

def compute_pdfz(membz,membzerr,sigma,zcls,npoints=1000,correction=True):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.

        npoints=1000 # it's accurate in 2%% level
    '''  
    zmin, zmax = zcls-5*sigma, zcls+5*sigma
    #zmin = check_boundaries(zmin,zcls)
    
    ## photo-z floor
    membzerr= np.where(membzerr<0.005,0.005,membzerr)

    ## multi dymensional arrays
    z       = np.linspace(zmin,zmax,npoints)
    #zz, yy  = np.meshgrid(z,np.array(zoffset)*(1+zcls)) ## dz = z-z_cls; zoffset = (z-z_cls)/(1+z_cls) 
    zz, yy  = np.meshgrid(z,np.array(membz))
    zz, yy2 = np.meshgrid(z,np.array(membzerr))
    
    if correction:
        # pdfz = gaussian_corrected(zz,yy, np.sqrt(yy2**2+sigma**2)/(1+zcls) )
        pdfz = gaussian_corrected(zz,yy,yy2/(1+zcls))
        pdfz_max = gaussian_corrected(z,zcls,sigma/(1+zcls))
            
    else:
        pdfz = gaussian(zz,yy,yy2)
        pdfz_max = gaussian(z,np.zcls,sigma)
    
    w,  = np.where( np.abs(z-zcls) <= 2.*sigma) ## integrate in 1.5*sigma
    p0 = integrate.trapz(pdfz[:,w],x=zz[:,w])
    
    pmax = integrate.trapz(pdfz_max[w],x=z[w])
    pz   = p0/pmax

    ## get out with galaxies outside 3 sigma
    pz = np.where(np.abs(membz-zcls) >= 2.*sigma, 0., pz)

    return pz

def compute_pdfz_parallel(cidxs,z,zerr,zcls,zwindow,nCores=40,npoints=1000,pdfz=None):
    cids,indices = np.unique(cidxs,return_index=True)
    zcls    = zcls[indices]
    zwindow = zwindow[indices]
    
    ncls = len(cids)
    ngals= len(cidxs)

    keys   = list(chunks(cidxs,cids))
    pz_out = np.zeros((ngals,),dtype=np.float64)

    z_group = group_by(z,keys)
    zerr_group    = group_by(zerr,keys)

    out    = Parallel(n_jobs=nCores)(delayed(compute_pdfz)(z_group[i], zerr_group[i], zwindow[i], zcls[i],
                                                           npoints=npoints) for i in range(len(keys)))
    
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

def gaussian_corrected(x,mu,sigma):
    sigma_cor = sigma*(1+x)
    return np.exp(-(x-mu)**2/(2*sigma_cor**2))/(sigma_cor*np.sqrt(2*np.pi))

def group_by(x,keys):
    return [x[idx] for idx in keys]
    
def delete_group(fmaster,path):
    group   = fmaster[path]
    cols    = group.keys()
    
    for col in cols:
        del group[col]

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

def make_gaussian_photoz(files,zsigma,nCores=60,method=None):
    t0     = time()
    new_group = get_name_string(zsigma) ## e.g. gauss005
    
    print('Running make_gaussian_photoz')
    total_time=[]
    for infile in files:
        mkPz     = photoz_model(zwindow=zsigma)
        mkPz.load_data(infile)
        mkPz.generate_gaussian_photoz()
        # mkPz.model_photoz_bias()
        mkPz.compute_photoz_probability(nCores=nCores,method=method)

        print('Writing outfile')
        write_gauss_outfile(infile,mkPz.data,new_group,overwrite=True)
        
        t1 = (time()-t0)/60
        print('partial time: %.2f min\n'%(t1))
        total_time.append(t1)

    total_time = np.array(total_time)[-1]#.sum()
    print('Total time: %.2f min'%(total_time))

def generate_photoz_models(run_type,files,zsigma,zmodel_file=None,method=None,
                           group_name='Pz',nCores=60,emulator_file=None,offset=0.,gauss=False,emulator=False):
    t0     = time()
    if run_type=='gaussian':
        if group_name =='Pz': group_name = get_name_string(zsigma) ## e.g. gauss005
        zmodel_file= None
        gauss = True
    
    if run_type=='emulator':
        loaded_model = load_cpickle_gc(emulator_file)
        #global loaded_model
        emulator= True

    total_time=[]
    print('Generating photoz catalog: %s'%group_name)
    for infile in files:
        mkPz     = photoz_model(zwindow=zsigma,zmodel_file=zmodel_file)
        mkPz.load_data(infile)
        
        if gauss:
            mkPz.generate_gaussian_photoz()
        else:
            if emulator:
                mkPz.emulate_photoz(loaded_model,offset=offset)
            mkPz.model_photoz_bias()

        print('Computing pz0')
        mkPz.compute_photoz_probability(nCores=nCores,method=method)

        print('Writing outfile')
        write_gauss_outfile(infile,mkPz.data,group_name,overwrite=True)
        
        t1 = (time()-t0)/60
        print('partial time: %.2f min\n'%(t1))
        total_time.append(t1)

    total_time = np.array(total_time)[-1]#.sum()
    print('Total time: %.2f min'%(total_time))

if __name__ == '__main__':
    ## How to run: emulator, gaussian, photoz bias
    files = glob.glob('/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/y3/output/tiles/*.hdf5')

    root= '/home/s1/jesteves/git/ccopa'
    zw = root+'/aux_files/zwindow_model_buzzard_dnf.txt'
    
    generate_photoz_models('bias',files,0.03,zmodel_file=zw,group_name='dnf_model',nCores=60)
    generate_photoz_models('gaussian',files,0.01,zmodel_file=None,group_name=None,nCores=60,emulator_file=None,offset=0.)
    generate_photoz_models('gaussian',files,0.03,zmodel_file=None,group_name=None,nCores=60,emulator_file=None,offset=0.)
    generate_photoz_models('gaussian',files,0.05,zmodel_file=None,group_name=None,nCores=60,emulator_file=None,offset=0.)

    ###############
    ### gaussian photoz
    # make_gaussian_photoz(files, 0.03,nCores=60)
        
    ###############
    ## emulator
    # infile = '/home/s1/berlfein/des40a/notebooks/emuPhotoZ_bpz_dnf_random_forest.pckl'
    # loaded_model = load_cpickle_gc(infile)
    # global loaded_model
    # for fname in files:
    #     mkPz = photoz_model(zwindow=zsigma,zmodel_file='/home/s1/jesteves/git/ccopa/aux_files/emuBPZ_correction_z_buzzard.txt')
    #     mkPz.load_data(fname)
    #     mkPz.emulate_photoz(loaded_model,offset=0.092)
    #     mkPz.model_photoz_bias()
    #     mkPz.compute_photoz_probability(nCores=60)
    #     write_gauss_outfile(fname,mkPz.data,'emuPz',overwrite=True)

    ###############
    ## photoz bias
    # for fname in files:
    #     mkPz = photoz_model(zwindow=zsigma,zmodel_file='/home/s1/jesteves/git/ccopa/aux_files/zwindow_model_buzzard_dnf')
    #     mkPz.load_data(fname)
    #     mkPz.model_photoz_bias()
    #     mkPz.compute_photoz_probability(nCores=60)
    #     write_gauss_outfile(fname,mkPz.data,'dnf_model',overwrite=True)