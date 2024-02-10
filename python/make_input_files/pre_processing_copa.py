#!/usr/bin/env python
import numpy as np
import healpy

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from joblib import Parallel, delayed

from scipy.interpolate import interp1d
import scipy.integrate as integrate

import esutil
from time import time

Mpc2cm = 3.086e+24
rad2deg = 180/np.pi
h = 0.7
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.285)
#from makeHDF5 import outCosmoDC2

class preProcessing:
    """Produces input catalogs for copacabana"""
    def __init__(self,cdata,data,auxfile=None,dataset='cosmoDC2',h=0.7,columns_gal=None,columns_cls=None):
        self.data    = data
        self.dataset = dataset
        self.cdata   = cdata
        
        self.simulation = True
        self.auxfile = auxfile
        
        self.columns_cls = get_cluster_cols(columns_cls)
        self.columns_gal = get_galaxy_cols(columns_gal)

        ## output columns
        self.columns = ['HALOID','CID','redshift','GID','RA','DEC','z','zerr','mag','magerr','pz0','R','zoffset','dmag','Bkg']

    def make_cutouts(self,rmax=12):
        self.rmax = rmax
        cc = self.columns_cls
        cg = self.columns_gal

        rac  = np.array(self.cdata[cc['RA']][:])
        decc = np.array(self.cdata[cc['DEC']][:])
        zcls = np.array(self.cdata[cc['redshift']][:])
        
        rag  = np.array(self.data[cg['RA']][:])
        decg = np.array(self.data[cg['DEC']][:])
        
        ang_diam_dist = AngularDistance(zcls)
        mag_lim       = self.getMagLimModel_04L(self.auxfile,zcls,dm=0).T
        
        self.cdata['CID']= self.cdata[cc['HALOID']]
        self.cdata['DA'] = ang_diam_dist
        self.cdata['magLim'] = mag_lim
        
        idxg,idxc,radii = self.aperture_match(rac,decc,ang_diam_dist,rag,decg,r_aper=rmax)

        self.idxg = idxg.astype(np.int64)
        self.idxc = idxc.astype(np.int64)
        self.radii= radii

    def make_relative_variables(self,z_window=0.02,nCores=4):
        cc = self.columns_cls
        cg = self.columns_gal
        out_data = dict().fromkeys(self.columns)
        
        ## get mag limit model
        hid    = np.array(self.cdata[cc['HALOID']])[self.idxc]
        fields = np.array(self.cdata[cc['tile']])[self.idxc]
        zvec   = np.array(self.cdata[cc['redshift']])[self.idxc]
        magLim = self.getMagLimModel_04L(self.auxfile,zvec,dm=0)

        ## cluster variables
        out_data['CID']     = hid
        out_data['redshift']= zvec
        out_data['magLim']  = magLim.T
        out_data['field']   = fields

        ## galaxy variables
        out_data['HALOID']= self.data[cg['HALOID']][self.idxg]
        out_data['GID']   = self.data[cg['GID']][self.idxg]
        out_data['RA']    = self.data[cg['RA']][self.idxg]
        out_data['DEC']   = self.data[cg['DEC']][self.idxg]
        
        ## make mag variables
        out_data['mag']    = np.vstack([ np.array(self.data[mi][self.idxg]) for mi in cg['mag']]).T
        out_data['magerr'] = np.vstack([ np.array(self.data[mi][self.idxg]) for mi in cg['magerr']]).T #self.compute_mag_error_lsst(out_data['mag']).T
        #out_data['color']  = np.vstack([])
        
        ## photoz
        out_data['z']      = self.data[cg['z']][self.idxg]    #gaussian_photoz(self.data['redshift'][self.idxg],z_window)[0]
        out_data['zerr']   = self.data[cg['zerr']][self.idxg] #gaussian_photoz(self.data['redshift'][self.idxg],z_window)[1]
        
        ## relative variables
        rel_var = self.compute_relative_variables(out_data,z_window,nCores=nCores)
        
        out_data['R']       = self.radii/h
        out_data['dmag']    = rel_var[0]
        out_data['zoffset'] = rel_var[1]
        out_data['pz0']     = rel_var[2] ## empty values
        out_data['Bkg']     = (self.radii>=4.)&(self.radii<=6.)

        out = Table(out_data)
        self.out = out[self.columns]
    
    def assign_true_members(self):
        cg = self.columns_gal

        galax_halo_id = self.data[cg['HALOID']][self.idxg]
        match         = esutil.numpy_util.where1(galax_halo_id==self.out['CID'])

        true_members        = np.full((len(self.idxg),),False)
        true_members[match] = True
        
        self.out['True']        = true_members
        self.out['z_true']      = self.data[cg['z_true']][self.idxg]
        self.out['Mr']          = self.data[cg['Mr']][self.idxg]
        #self.out['stellar_mass']=self.data['stellar_mass'][self.idxg]

    def apply_mag_cut(self,dmag_cut=2):
        cut        = (self.out['dmag']<=dmag_cut)&(self.out['R']*h<=self.rmax)
        self.out   = self.out[cut]
        
    def compute_relative_variables(self,data,z_window=0.02,nCores=4):
        cidxs = np.array(data['CID'])
        zcls  = np.array(data['redshift'])
        zph   = np.array(data['z'])
        zerr  = np.array(data['zerr'])

        # dmag
        dmag= data['mag'][:,2]-data['magLim'][:,1]
        
        # zoffset
        zoffset = (zph-zcls)/(1+zcls)
        
        if zerr.size!=zoffset.size:
            print('here')

        #pz0 = np.zeros_like(zcls)
        pz0   = compute_pdfz_parallel(cidxs,zoffset,zerr,zcls,z_window*np.ones_like(zerr),nCores=nCores)

        return [dmag,zoffset,pz0]

    def compute_mag_error_lsst(self,mag):
        coefs = [[-1.28167900e-03, 9.78790967e-02,-2.11337322e+00, 1.12139087e+01],
                 [-3.12705403e-03, 1.96419803e-01,-3.80101934e+00, 2.04291313e+01],
                 [-6.75571431e-04, 4.87008215e-02,-8.51343916e-01, 1.08236535e+00],
                 [ 2.42978486e-03,-1.15899936e-01, 2.04088234e+00,-1.56178376e+01]]

        magerr = []
        for i in range(4):
            magerr.append(10**(poly_func(mag[:,i],coefs[i])))

        return np.vstack(magerr)
    
    def getMagLimModel_04L(self,auxfile,zvec,dm=0):
        '''
        Get the magnitude limit for 0.4L*

        file columns: z, mag_lim_i, (g-r), (r-i), (i-z)
        mag_lim = mag_lim + dm

        input: Galaxy clusters redshift
        return: mag. limit for each galaxy cluster and for the r,i,z bands
        output = [maglim_r(array),maglim_i(array),maglim_z(array)]
        '''
        annis=np.loadtxt(auxfile)
        jimz=[i[0] for i in annis]  ## redshift
        jimgi=[i[1] for i in annis] ## mag(i-band)
        jimgr=[i[2] for  i in annis]## (g-r) color
        jimri=[i[3] for i in annis] ## (r-i) color
        jimiz=[i[4] for i in annis] ## (i-z) color

        interp_magi=interp1d(jimz,jimgi,fill_value='extrapolate')
        interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
        interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
        interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

        mag_i,color_ri,color_iz = interp_magi(zvec),interp_ri(zvec),interp_iz(zvec)
        mag_r, mag_z = (color_ri+mag_i),(mag_i-color_iz)

        maglim_r, maglim_i, maglim_z = (mag_r+dm),(mag_i+dm),(mag_z+dm)

        magLim = np.vstack([maglim_r, maglim_i, maglim_z])

        return magLim#.transpose()

    def aperture_match(self,ra_cluster,dec_cluster,ang_diam_dist,ra_galaxy,dec_galaxy,r_aper=10):
        '''
        Get circles with rmax (Mpc) around each cluster position

        Parameters
        ----------
        ra : float, array
        cluster and galaxy, right ascension of the object in degrees.
        dec : float, array
        cluster and galaxy, declination of the object in degrees.
        ang_diam_dist: array
        angular distance in Mpc
        rmax : float
        aperture radius in Mpc

        Returns
        -------
        indicies_into_galaxies_in_aperture : array
        indices for the galaxies in the circles
        indicies_into_clusters : array
        indices for the cluster table
        radii: array
        relative distance from the center in Mpc
        '''
        #import esutil 
        depth=10
        h=esutil.htm.HTM(depth)
        #Inner match
        degrees_i=(360/(2*np.pi))*(r_aper/ang_diam_dist)
        m1i,m2i,disti=h.match(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=degrees_i,maxmatch=0)

        indicies_into_galaxies_in_aperture=[]
        indicies_into_clusters=[]
        for i in range(len(ra_cluster)):
            w_i=esutil.numpy_util.where1(m1i==i)
            indicies_into_galaxies_in_aperture_i=m2i[w_i]
            indicies_into_galaxies_in_aperture.append(indicies_into_galaxies_in_aperture_i)
            indicies_into_clusters_i = m1i[w_i]
            indicies_into_clusters.append(indicies_into_clusters_i)

        indicies_into_galaxies_in_aperture=np.concatenate(indicies_into_galaxies_in_aperture)
        indicies_into_clusters=np.concatenate(indicies_into_clusters)

        radii = np.array( disti*(np.pi/180)*ang_diam_dist[indicies_into_clusters] )

        return indicies_into_galaxies_in_aperture, indicies_into_clusters, radii

    #@vectorize(signature="(),()->()")
    
def _AngularDistance(z):
    """Angular distance calculator
    :params float z: redshift
    :returns: angular distance in Mpc
    """
    DA = ( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA

AngularDistance = np.vectorize(_AngularDistance)

def poly_func(x,coefs):
    f = np.poly1d(coefs)
    return f(x)

def compute_sigma68(zgrid,zmode,pdfz):
    nl  = len(zmode)
    a   = np.cumsum(pdfz, axis=1)/np.sum(pdfz, axis=1)[:,np.newaxis]
    
    s68 = np.zeros(nl,dtype=np.float68)
    bb  = np.zeros(nl)

    var = 1e-3
    for ii in range(nl):
        bb[ii] = np.interp(zmode[ii], zgrid, a[ii])

    bup = np.where(bb>0.83,1.-var,bb+0.17)
    blo = np.where(bb<0.17,0.+var,bb-0.17)

    for ii in range(nl):
        ql, qu = np.interp([blo[ii], bup[ii]], a[ii], zgrid)
        s68[ii] = (qu-ql)/(1+zmode[ii])
        
    return s68

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w


def check_boundaries(zmin,zcls):
    zoff_min = zcls+zmin*(1+zcls)
    if zoff_min<0:
        return zmin-zoff_min
    else:
        return zmin

def gaussian(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def compute_pdfz_parallel(cidxs,zoffset,zerr,zcls,zwindow,nCores=40,npoints=1000,bpz=False,pdfz=None):
    cids,indices = np.unique(cidxs,return_index=True)
    zcls    = zcls[indices].copy()
    zwindow = zwindow[indices].copy()
    
    ncls = len(cids)
    ngals= len(cidxs)

    keys   = list(chunks(cidxs,cids))
    pz_out = np.zeros((ngals,),dtype=np.float64)

    zoffset_group = group_by(zoffset,keys)
    zerr_group    = group_by(zerr,keys)

    out    = Parallel(n_jobs=nCores)(delayed(compute_pdfz)(zoffset_group[i], zerr_group[i], zwindow[i], zcls[i],
                                                           npoints=npoints) for i in range(len(keys)))
    
    for i,idx in enumerate(keys):
        if len(out[i])==idx.size:
            pz_out[idx] = out[i]
        else:
            print('error')
    return pz_out

def compute_pdfz(zoffset,membzerr,sigma,zcls,npoints=1000):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.

        npoints=1000 # it's accurate in 2%% level
    ''' 
    out = np.zeros_like(membzerr)

    zmin, zmax = -5*sigma, 5*sigma
    zmin = check_boundaries(zmin,zcls)

    ## photo-z floor
    zerr = membzerr.copy()
    idx = np.where(zerr<1e-3)[0]
    zerr[idx] = 1e-3

    z       = np.linspace(zmin,zmax,npoints)
    zz, yy  = np.meshgrid(z,np.array(zoffset))
    zz, yy2 = np.meshgrid(z,np.array(zerr))

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

def group_by(x,keys):
    return [x[idx] for idx in keys]

def cut_dict(indices,dicto):
    columns = list(dicto.keys())
    for col in columns:
        dicto[col] = dicto[col][indices]
    return dicto

def gaussian_photoz(ztrue,zwindow,seed=42):
    np.random.seed(seed)
    zerr = zwindow*np.ones_like(ztrue)
    znoise= ztrue+np.random.normal(scale=zwindow,size=ztrue.size)*(1+ztrue)
    znoise= np.where(znoise<0.,0.,znoise) # there is no negative redshift
    return znoise, zerr

def get_cluster_cols(columns,simulation=True):
    if columns is None:
        columns = dict()
        columns['CID'] = 'CID'
        columns['RA']  = 'RA'
        columns['DEC'] = 'DEC'
        columns['redshift'] = 'redshift'
        columns['tile'] = 'tile'
        if simulation:
            columns['M200_true']= 'M200_true'
            columns['R200_true']= 'R200_true'
    return columns

def get_galaxy_cols(columns,simulation=True):
    if columns is None:
        columns = dict()
        columns['HALOID'] = 'HALOID'
        columns['GID']    = 'GID'
        columns['RA']     = 'RA'
        columns['DEC']    = 'DEC'
        columns['z']      = 'z'
        columns['zerr']   = 'zerr'
        columns['Mr']     = 'Mr'
        columns['mag']    = ['mag_g','mag_r','mag_i','mag_z']
        columns['magerr'] = ['mag_err_g','mag_err_r','mag_err_i','mag_err_z']
        columns['tile']   = 'tile'
        
        if simulation:
            columns['z_true']= 'z_true'
    return columns
