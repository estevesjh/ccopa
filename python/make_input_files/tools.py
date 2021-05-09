#!/usr/bin/env python
import numpy as np
import healpy
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

Mpc2cm = 3.086e+24
rad2deg = 180/np.pi

class outCosmoDC2:
    """Make hdf5 master file to store the cosmoDC2 data"""
    def __init__(self,name_file,name_list):
        """Make the the hdf5 file: for the galaxy and photoz directories. The data is dived by healpix values of nside=32 and RING orderd"""
        import h5py
        self.fmaster = h5py.File(name_file, "w")
        #self.fmaster.create_group('galaxy')
        self.fmaster.create_group('photoz')
        self.fmaster.create_group('copa')
        
        self.name_list = name_list
        
        self._generate_groups()
    
    def _generate_groups(self):
        for name in self.name_list:
            self.fmaster['copa'].create_group(name)
            self.fmaster['photoz'].create_group(name)
    
    def write_galaxy_sample(self,hpx,table,columns):
        for col in columns:
            self.fmaster.create_dataset('galaxy/%s/%s/'%(str(hpx),col), data = table[col][:])

    def write_photoz_sample(self,cut,zbins,table,columns):
        for cix in self.name_list:
            for col in columns:
                if col != 'photoz_pdf':
                    self.fmaster['photoz/%s/'%(cix)].create_dataset(col, data = table[col][cut])
#                 else:
#                     self.fmaster['photoz/%s/'%(cix)].create_dataset('pdfz', data = np.vstack([table['photoz_pdf'][cut,:],zbins]), compression="lzf" )

    def write_copacabana_input(self,gidx,data):
        cidx = np.array(self.name_list,dtype=int)
        self.fmaster.create_dataset('cluster_ids',data=cidx)
        
        columns = list(data.keys())
        for cix in cidx: 
            ix, = np.where(data['CID']==cix)
            if ix.size>0:
                self.fmaster['copa/%i/'%(cix)].create_dataset('indices',data=gidx[ix])

                for col in columns: 
                    if col in ['mag','magerr','magLim']:
                        self.fmaster['copa/%i/'%(cix)].create_dataset('%s'%(col),data=data[col][ix,:], compression="lzf")
                    else:
                        self.fmaster['copa/%i/'%(cix)].create_dataset('%s'%(col),data=data[col][ix])
            else:
                print('%i empty cluster'%cix)
                
    def write_copacabana_extra_variables(cdata,data):
        pass
    
    def close_file(self):
        self.fmaster.close()

class healpixTools:
    """Healpix Operation Tools has a variety of functions which helps on the healpix operations for galaxy cluster science.
    """
    def __init__(self,nside,nest=False,h=0.7,Omega_m=0.283):
        """Starts a healpix map with a given nside and nest option.
        
        :param int nside: number of pixs of the map (usually a power of 2).
        :param bol nest : a boolean variable to order - TRUE (NESTED) or FALSE (RING).
        :param float h: hubble constant factor (H0 = h*100)
        :param float Omega_m: current omega matter density
        """
        self.nside = nside
        self.nest  = nest
        
        self.cosmo = FlatLambdaCDM(H0=100*h,Om0=Omega_m)
    
    def _get_healpix_cutout_radec(self,center_ra,center_dec,zcls,radius=1.):
        """ Get a healpix list which overlaps a circular cutout of a given radius centered on a given ra,dec coordinate. 

        :params float center_ra: ra coordinate in degrees.
        :params float center_dec: dec coordinate in degrees.
        :params float radius: aperture radius in Mpc.

        :returns: healpix list of the healpix values which overlaps the circular cutout.
        """
        center_ra_rad = np.radians(center_ra)
        center_dec_rad = np.radians(center_dec)

        center_vec = np.array([np.cos(center_dec_rad)*np.cos(center_ra_rad),
                            np.cos(center_dec_rad)*np.sin(center_ra_rad),
                            np.sin(center_dec_rad)])

        DA = float(self.AngularDistance(zcls))
        radii = (float(radius)/DA)*rad2deg    ## degrees

        healpix_list = healpy.query_disc(self.nside, center_vec, np.radians(radii), nest=self.nest, inclusive=True)

        return healpix_list

    def get_healpix_cutout_radec(self,ra_list,dec_list,redshift_list,radii=1):
        """ Get a healpix list which overlaps a circular cutout of a given radius centered on a given ra,dec coordinate. 
        
        :params ndarray ra_list: numpy array or list, ra coordinate in degrees.
        :params ndarray dec_list: dec coordinate in degrees.
        :params float radius: aperture radius in Mpc.
        
        :returns: healpix list of unique healpix values which overlaps all the circules defined by the list of center coordinates.
        """
        healpix_list = np.empty((0),dtype=int)
        for ra,dec,zcls in zip(ra_list,dec_list,redshift_list):
            hps_rad = self._get_healpix_cutout_radec(ra,dec,zcls,radius=radii)
            healpix_list = np.append(healpix_list,np.unique(hps_rad))
        return np.unique(healpix_list)
    
    def hpix2ang(self,pix):
        """ Convert pixels to astronomical coordinates ra and dec.
        
        :params pix: pixel values [int, ndarray or list]
        :returns: ra, dec [int, ndarray or list]
        """
        lon,lat = healpy.pix2ang(self.nside,pix,nest=self.nest)
        dec,ra=(90-(lon)*(180/np.pi)),(lat*(180/np.pi))
        return ra,dec

    def radec_pix(self,ra,dec):
        """ Convert astronomical coordinates ra and dec to pixels
        
        :params ra: ra [int, ndarray or list]
        :params dec: dec [int, ndarray or list]
        :returns: pixel values [int, ndarray or list]
        """
        return np.array(healpy.ang2pix(self.nside,np.radians(90-dec),np.radians(ra),nest=self.nest),dtype=np.int64)
    
    #@vectorize(signature="(),()->()")
    def AngularDistance(self,z):
        """Angular distance calculator
        :params float z: redshift
        :returns: angular distance in Mpc
        """
        DA = ( (self.cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
        return DA
    
    def match_with_cat(self,df,hpx_clusters,radii=8):
        healpix_list = []
        for hpx in np.unique(hpx_clusters):
            w, = np.where(hpx_clusters==hpx)
            hp_list = self.get_healpix_cutout_radec(df['ra'].iloc[w],df['dec'].iloc[w],df['redshift'].iloc[w],radii=radii)
            healpix_list.append([hpx,hp_list])
        return healpix_list

def apply_photoz_mask(data,mag_cut):
    mask = (data['photoz_mask']&(data['mag_i_lsst']<=mag_cut)).copy()
    nall = len(mask)
    
    columns = list(data.keys())
    for col in columns:
        if len(data[col])==nall:
            data[col] = data[col][mask]
    return data
