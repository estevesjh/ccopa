#!/usr/bin/env python
import numpy as np
import healpy
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d

Mpc2cm = 3.086e+24
rad2deg = 180/np.pi
h=0.7
cosmo = FlatLambdaCDM(H0=100*h,Om0=0.285)
#from makeHDF5 import outCosmoDC2

class preProcessing:
    """Produces input catalogs for copacabana"""
    def __init__(self,cdata,data,dataset='cosmoDC2',h=0.7):
        self.data    = data
        self.dataset = dataset
        self.cdata   = cdata
        
        self.simulation = True
        self.auxfile = './data/annis_mags_04_Lcut.txt'
    
    def assign_output_columns():
        self.columns = ['CID','redshift','GID','RA','DEC','z','zerr','mag','magerr','R','zoffset','dmag','pz0','Bkg']

    def make_cutouts(self,rmax=12):
        rac  = np.array(self.cdata['ra'][:])
        decc = np.array(self.cdata['dec'][:])
        zcls = np.array(self.cdata['redshift'][:])
        
        rag  = self.data['ra']
        decg = self.data['dec']
        
        ang_diam_dist = AngularDistance(zcls)
        
        idxg,idxc,radii = self.aperture_match(rac,decc,ang_diam_dist,rag,decg,r_aper=rmax)
        
        self.idxg = idxg.astype(np.int64)
        self.idxc = idxc.astype(np.int64)
        self.radii= radii

    def make_relative_variables(self,zgrid,z_window=0.02):
        out_data = dict().fromkeys(columns)
        
        ## cluster variables
        out_data['CID'] = np.array(self.cdata['halo_id'])[self.idxc]
        out_data['redshift']= np.array(self.cdata['redshift'])[self.idxc]
        out_data['magLim']= self.magModel.T
        
        ## galaxy variables
        out_data['GID']  = self.data['galaxy_id'][self.idxg]
        out_data['RA']  = self.data['ra'][self.idxg]
        out_data['DEC'] = self.data['dec'][self.idxg]
        
        ## make mag variables
        out_data['mag']    = np.vstack([self.data['mag_%s_lsst'%c][self.idxg] for c in ['g','r','i','z']]).T
        out_data['magerr'] = self.compute_mag_error_lsst(out_data['mag']).T
        #out_data['color']  = np.vstack([])
        
        ## photoz
        out_data['z']      = self.data['photoz_mode'][self.idxg]
        out_data['zerr']   = compute_sigma68(zgrid,self.data['photoz_mode'][self.idxg],self.data['photoz_pdf'][self.idxg])
        
        ## relative variables
        rel_var = self.compute_relative_variables(zgrid,self.data['photoz_pdf'][self.idxg],out_data,z_window)
        
        out_data['R']       = self.radii/h
        out_data['dmag']    = rel_var[0]
        out_data['zoffset'] = rel_var[1]
        out_data['pz0']    = rel_var[2]
        out_data['Bkg']     = (self.radii>=4.)&(self.radii<=6.)
        
        self.out = out_data
    
    def assign_true_members(self):
        galax_halo_id = self.data['halo_id'][self.idxg]
        true_members  = np.where((galax_halo_id==self.out['CID']),True,False)
        
        self.out['True']        = true_members
        self.out['z_true']      =self.data['redshift'][self.idxg]
        self.out['Mr']          =self.data['Mag_true_r_lsst_z0'][self.idxg]
        self.out['stellar_mass']=self.data['stellar_mass'][self.idxg]

    def apply_mag_cut(self,dmag_cut=2):
        zcls    = np.array(self.cdata['redshift'])[self.idxc]
        mag     = np.array(self.data['mag_i_lsst'])[self.idxg]
        
        ## mag model
        mag_model_riz = self.getMagLimModel_04L(self.auxfile,zcls,dm=0)
        dmag = mag-mag_model_riz[1] # i-band
        
        gindidces = np.arange(0,len(self.data['mag_i_lsst']),1,dtype=np.int64)
        
        ## update indices
        w, = np.where(dmag <= dmag_cut)
        self.data_idx,self.gidx = np.unique(gindidces[self.idxg[w]],return_inverse=True)
        self.idxc = self.idxc[w]
        self.idxg = self.idxg[w]
        
        self.magModel = mag_model_riz[:,w]
        
        #self.data = cut_dict(self.data_idx,self.data)
        
    def compute_relative_variables(self,zgrid,pdfz,data,z_window=0.02):
        out = []
        
        # dmag
        dmag= data['mag'][:,2]-data['magLim'][:,1]
        
        # zoffset
        zcls = data['redshift']
        zph  = data['z']
        zoffset = (zph-zcls)/(1+zcls)
        
        # pdfz
        cpdfz = np.cumsum(pdfz, axis=1)/np.sum(pdfz, axis=1)[:,np.newaxis]
        zmin  = zcls-1.5*z_window*(1+zcls)
        zmax  = zcls+1.5*z_window*(1+zcls)
        zmin = np.where(zmin<0,0.,zmin)
        zmax = np.where(zmax>3.,3.,zmax)
        pz0    = np.array([np.interp(zmax[i],zgrid,cpdfz[i])-np.interp(zmin[i],zgrid,cpdfz[i]) for i in range(len(zcls))])
        
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
        import esutil 
        depth=10
        h=esutil.htm.HTM(depth)
        #Inner match
        degrees_i=(360/(2*np.pi))*(r_aper/ang_diam_dist)
        m1i,m2i,disti=h.match(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=degrees_i,maxmatch=0)

        indicies_into_galaxies_in_aperture=[]
        indicies_into_clusters=[]
        for i in range(len(ra_cluster)):
            w_i=np.where(m1i==i)
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