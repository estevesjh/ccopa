# !/usr/bin/env python

from astropy_healpix import HEALPix
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.stats import median_absolute_deviation


import numpy as np
import logging
from time import time
import pandas as pd
import easyaccess as ea

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
Mpc2cm = 3.086e+27


## -------------------------------
## auxiliary functions
    
def convertRA(ra):
    ra = np.where(ra>180,ra-360,ra)
    return ra

def AngularDistance(z):
    DA = float( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA
AngularDistance = np.vectorize(AngularDistance)

gaussian = lambda x,mu,sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def PhotozProbabilities(zmin,zmax,membz,membzerr):
    out = []
    for i in range(len(membz)):
        aux, err = integrate.quad(gaussian,zmin,zmax,args=(membz[i],membzerr[i]),full_output=0)
        out.append(aux)
    return np.array(out)

def getPDFz(mebz,membzerr,z_cls,window=0.1):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width 2*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    pdf = []
    for zi in z_cls:
        zmin, zmax = (zi-window*(1+zi)), (zi+window*(1+zi))
        pi = PhotozProbabilities(zmin,zmax,membz,membzerr)
        pdf.append(pi)
    return np.array(pdf)


def getAngDist(ra1, dec1, ra2, dec2):
  delt_lon = (ra1 - ra2)*np.pi/180.
  delt_lat = (dec1 - dec2)*np.pi/180.
  # Haversine formula
  dist = 2.0*np.arcsin( np.sqrt( np.sin(delt_lat/2.0)**2 + np.cos(dec1*np.pi/180.)*np.cos(dec2*np.pi/180.)*np.sin(delt_lon/2.0)**2 ) )  

  return dist/np.pi*180.

def doSomeCuts(ra_offset,dec_offset,z,flag,length=1,zrange=(0.01,1.01),NGals=0):
    mask = (ra_offset<length)&(dec_offset<length)&(z>zrange[0])&(z<zrange[1])&(flag>0)
    return mask
    
def doQuery(ra_c=0,dec_c=0,theta12Mpc=1,zrange=(0.01,1.),Nflag=0,healpixFile=None):
    Y3cols = ', gold.'.join(Y3columns)
    q1 = 'select gold.{cols} '.format(cols=Y3cols)
    q4 = 'and BPZ_ZMEAN_SOF > {zmin:.3f} and BPZ_ZMEAN_SOF < {zmax:.3f} '.format(zmin=zrange[0],zmax=zrange[1])
    q5 = 'and FLAGS_GOLD > {} '.format(Nflag)
    q6 = 'and EXTENDED_CLASS_MASH_SOF>1;'

    if healpixFile is not None:
        hp_name = healpixFile.split('.fits')[0]
        connection.load_table(healpixFile)
        connection.onecmd('CREATE INDEX idNAME ON %s (PIXEL);'%(hp_name))
        
        q2 = ', mask.CID from y3_gold_2_2 gold, '
        q3 = '%s mask where gold.hpix_4096=mask.pixel '%(hp_name)
        # query = q1+q2+q3+q6
    else:
        q2 = 'from y3_gold_2_2 gold where RA between {ra_low:6f} and {ra_high:6f} '.format(ra_low=(ra_c-theta12Mpc),ra_high=(ra_c+theta12Mpc)) ##query
        q3 = 'and DEC between {dec_low:.6f} and {dec_high:.6f} '.format(dec_low=(dec_c-theta12Mpc),dec_high=(dec_c+theta12Mpc)) ##query
    
    query = q1+q2+q3+q4+q5+q6
        
    dataFrame=connection.query_to_pandas(query)
    table = Table.from_pandas(dataFrame)

    ## just dropping these table
    if healpixFile is not None:
        cursor.execute('drop table %s'%(hp_name))

    return table

def doHealpixMatch(clusters,healPixFile,length=12):
    ra_c , dec_c = np.array(clusters['RA']), np.array(clusters['DEC'])
    cls_id, DA = np.array(clusters['CID']), np.array(clusters["DA"])
    theta12Mpc = (180/np.pi)*(length/DA) ## in degrees

    ra_c = np.where(ra_c<0,ra_c+360,ra_c)
    coords = SkyCoord(ra_c, dec_c, frame='icrs', unit='deg')
    pixel_match = np.empty(0,dtype=int)
    cid_match = np.empty(0,dtype=int)

    for coord, the, ci in zip(coords,theta12Mpc,cls_id):
        cp = hp.cone_search_skycoord(coord, radius = the * u.deg)
        if len(cp)>0:
            pixel_match = np.append(pixel_match,cp)
            cid_match = np.append(cid_match,ci*np.ones_like(cp))
    
    col0=Table.Column(data=pixel_match,name='PIXEL',format='K')
    col1=Table.Column(data=cid_match,name='CID',format='K')

    outTable = Table()
    outTable.add_columns([col0,col1])
    outTable.write(healPixFile,format='fits',overwrite=True)

def queryGalaxyData(clusters,length=12,zrange=(0.01,1.),Nflag=0,HEALPix=False):

    if HEALPix:## Find galaxies around all the cluster points
        healPixFile = 'hp_%s.fits'%(clusters['CID'][0])
        doHealpixMatch(clusters,healPixFile,length=length) ## create a healpix table for these clusters
        galTable = doQuery(zrange=zrange,Nflag=Nflag,healpixFile=healPixFile)
    
    ncls = len(clusters)
    cid = np.empty(0,dtype=int)
    radii = np.empty(0,dtype=float)
    ## divide to conquer
    gals = []
    for i in range(ncls):
        ## cut 12Mpc around each cluster
        ra_c , dec_c = clusters['RA'][i], clusters['DEC'][i]
        cls_id, DA = clusters['CID'][i], clusters["DA"][i]
        thetha12Mpc = (180/np.pi)*(length/DA) ## in degrees

        if HEALPix:
            indices = np.where(galTable['CID']==cls_id)
            galaxies = galTable[indices]
        else:## find galaxies for each cluster per time
            galaxies = doQuery(ra_c,dec_c,theta12Mpc=thetha12Mpc,zrange=zrange,Nflag=Nflag)
        
        clusterIDX = cls_id*np.ones_like(galaxies['RA'],dtype=int)
        theta_offset = getAngDist(galaxies['RA'], galaxies['DEC'], ra_c, dec_c)
        r = theta_offset*(np.pi/180)*DA ## in Mpc
        
        radii = np.append(radii,r)
        cid = np.append(cid,clusterIDX)

        gals.append(galaxies)

    galOut = gals[0]
    if len(gals)>1:
        for i in range(1,len(gals)):
            gi = gals[i]
            galOut = vstack([galOut, gi])

    return galOut, cid, radii

## -------------------------------
## Load Catalogs

def queryGalaxyCat(clusters, radius=12, zrange=(0.01,1.),Nflag=0,r_in=8,r_out=10,HEALPix=False,window=0.1):
    """
    Query the galaxy catalog at easyaccess. 
    
    Parameters
    ----------
    cluster : table
    radius : float
        Radius distance in Mpc
    ...

    Returns
    -------
    galaxyOut : table
        Galaxy Catalog within galaxy around 12 Mpc of the cluster center position
  """
    logging.debug('Starting helper.readGalaxyCat()')
    
    ## divide to conquer! (take squares around each cluster position)
    ## radius is the length of the square in Mpc
    galaxyData, cid, radii = queryGalaxyData(clusters,length=12,zrange=zrange,Nflag=Nflag,HEALPix=HEALPix)
        
    gid = np.array(galaxyData['COADD_OBJECT_ID'],dtype=int)
    ra = np.array(galaxyData['RA'])
    dec = np.array(galaxyData['DEC'])
    z = np.array(galaxyData['DEC'])

    z = np.array(galaxyData['BPZ_ZMEAN_SOF'][:])
    zerr = np.array(galaxyData['BPZ_ZSIGMA_SOF'][:])

    mag, magerr = [], []
    for ci in ['G','R','I','Z']:
        mag.append(galaxyData['MAG_AUTO_%s'%(ci)])
        magerr.append(galaxyData['MAGERR_AUTO_%s'%(ci)])
        
    mag = np.array(mag).transpose() ## 4 vector
    magerr = np.array(magerr).transpose()
    
    bkgGalaxies = (radii>r_in)&(radii<r_out)
    PDFz = getPDFz(z,zerr,cat['redshift'],window=window)
    
    galaxyOut = Table([cid, gid, ra, dec, radii, z, mag, zerr, magerr, PDFz, bkgGalaxies],
                      names=('CID', 'GID', 'RA', 'DEC', 'R', 'z', 'mag','zerr','magerr', 'PDFz', 'Bkg'))

    logging.debug('Returning from helper.readGalaxyCat()')

    return galaxyOut


if __name__ == '__main__':
    print('helperQuery.py')
    print('author: Johnny H. Esteves')
