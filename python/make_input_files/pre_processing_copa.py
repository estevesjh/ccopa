#!/usr/bin/env python
import numpy as np
import healpy

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from scipy.interpolate import interp1d
import esutil
from time import time

Mpc2cm = 3.086e+24
rad2deg = 180/np.pi
h=0.7
cosmo = FlatLambdaCDM(H0=100*h,Om0=0.285)
#from makeHDF5 import outCosmoDC2

class preProcessing:
    """Produces input catalogs for copacabana"""
    def __init__(self,cdata,data,auxfile=None,dataset='cosmoDC2',h=0.7):
        self.data    = data
        self.dataset = dataset
        self.cdata   = cdata
        
        self.simulation = True
        self.auxfile = auxfile
        
        ## output columns
        self.columns = ['HALOID','CID','redshift','GID','RA','DEC','z','zerr','mag','magerr','pz0','R','zoffset','dmag','Bkg']

    def make_cutouts(self,rmax=12):
        self.rmax = rmax

        rac  = np.array(self.cdata['RA'][:])
        decc = np.array(self.cdata['DEC'][:])
        zcls = np.array(self.cdata['redshift'][:])
        
        rag  = np.array(self.data['RA'][:])
        decg = np.array(self.data['DEC'][:])
        
        ang_diam_dist = AngularDistance(zcls)
        mag_lim       = self.getMagLimModel_04L(self.auxfile,zcls,dm=0).T
        
        self.cdata['DA'] = ang_diam_dist
        self.cdata['magLim'] = mag_lim
        
        idxg,idxc,radii = self.aperture_match(rac,decc,ang_diam_dist,rag,decg,r_aper=rmax)

        self.idxg = idxg.astype(np.int64)
        self.idxc = idxc.astype(np.int64)
        self.radii= radii

    def make_relative_variables(self,z_window=0.02):
        out_data = dict().fromkeys(self.columns)
        
        ## get mag limit model
        hid    = np.array(self.cdata['HALOID'])[self.idxc]
        fields = self.data['hpx8'][self.idxg]
        zvec   = np.array(self.cdata['redshift'])[self.idxc]
        magLim = self.getMagLimModel_04L(self.auxfile,zvec,dm=0)

        ## cluster variables
        out_data['CID']     = hid
        out_data['redshift']= zvec
        out_data['magLim']  = magLim.T
        out_data['field']   = fields

        ## galaxy variables
        out_data['HALOID']= self.data['HALOID'][self.idxg]
        out_data['GID']   = self.data['galaxy_id'][self.idxg]
        out_data['RA']    = self.data['RA'][self.idxg]
        out_data['DEC']   = self.data['DEC'][self.idxg]
        
        ## make mag variables
        out_data['mag']    = np.vstack([self.data['mag_%s_lsst'%c][self.idxg] for c in ['g','r','i','z']]).T
        out_data['magerr'] = self.compute_mag_error_lsst(out_data['mag']).T
        #out_data['color']  = np.vstack([])
        
        ## photoz
        out_data['z']      = gaussian_photoz(self.data['redshift'][self.idxg],z_window)[0]
        out_data['zerr']   = gaussian_photoz(self.data['redshift'][self.idxg],z_window)[1]
        
        ## relative variables
        rel_var = self.compute_relative_variables(out_data,z_window)
        
        out_data['R']       = self.radii/h
        out_data['dmag']    = rel_var[0]
        out_data['zoffset'] = rel_var[1]
        out_data['pz0']     = rel_var[2] ## empty values
        out_data['Bkg']     = (self.radii>=4.)&(self.radii<=6.)

        out = Table(out_data)
        self.out = out[self.columns]
    
    def assign_true_members(self):
        galax_halo_id = self.data['HALOID'][self.idxg]
        match         = esutil.numpy_util.where1(galax_halo_id==self.out['CID'])

        true_members        = np.full((len(self.idxg),),False)
        true_members[match] = True
        
        self.out['True']        = true_members
        self.out['z_true']      = self.data['redshift'][self.idxg]
        self.out['Mr']          = self.data['Mag_true_r_des_z01'][self.idxg]
        #self.out['stellar_mass']=self.data['stellar_mass'][self.idxg]

    def apply_mag_cut(self,dmag_cut=2):
        cut        = (self.out['dmag']<=dmag_cut)&(self.out['R']*h<=self.rmax)
        self.out   = self.out[cut]
        
    def compute_relative_variables(self,data,z_window=0.02):
        out = []
        
        # dmag
        dmag= data['mag'][:,2]-data['magLim'][:,1]
        
        # zoffset
        zcls = data['redshift']
        zph  = data['z']
        zoffset = (zph-zcls)/(1+zcls)
        
        pz0 = np.zeros_like(zcls)
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
