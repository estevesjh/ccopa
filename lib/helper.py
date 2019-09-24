# !/usr/bin/env python

from astropy.table import Table, vstack
from astropy.io.fits import getdata

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.stats import median_absolute_deviation

from scipy.interpolate import interp1d
import scipy.integrate as integrate
import numpy as np
import logging
from time import time

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

def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def fastGaussianIntegration(membz,membzerr,zmin,zmax):
    zpts,zstep=np.linspace(zmin,zmax,50,retstep=True) #split redshift window for approximation
    area=[]
    for i in range(len(zpts)-1): #approximate integral using trapezoidal riemann sum
        gauss1=gaussian(zpts[i],membz,membzerr) #gauss1/2 are gaussian values for left,right points, respectively
        gauss2=gaussian(zpts[i+1],membz,membzerr)
        area1=((gauss1+gauss2)/2.)*zstep
        area.append(area1)
    area=np.array(area)
    arflip=area.swapaxes(0,1)
    prob=np.sum(arflip,axis=1)
    return prob

def PhotozProbabilities(zmin,zmax,membz,membzerr,fast=True):
    if fast:
        out = fastGaussianIntegration(membz,membzerr,zmin,zmax)
    
    else:
        out = []
        for i in range(len(membz)):
            aux, err = integrate.quad_fixed(gaussian,zmin,zmax,args=(membz[i],membzerr[i]))
            out.append(aux)
        np.array(out)

    return out

def getPDFz(membz,membzerr,zi,window=0.1):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width 2*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    zmin, zmax = (zi-window*(1+zi)), (zi+window*(1+zi))
    # pdf = PhotozProbabilities(zmin,zmax,membz,membzerr  )
    pdf = gaussian(zi,membz,membzerr)
    pdf = np.where(pdf<0.01,0.,pdf)
    return pdf

def getAngDist(ra1, dec1, ra2, dec2):
  """
    Calculate the angular distance between two coordinates.
  """  
  
  delt_lon = (ra1 - ra2)*np.pi/180.
  delt_lat = (dec1 - dec2)*np.pi/180.
  # Haversine formula
  dist = 2.0*np.arcsin( np.sqrt( np.sin(delt_lat/2.0)**2 + np.cos(dec1*np.pi/180.)*np.cos(dec2*np.pi/180.)*np.sin(delt_lon/2.0)**2 ) )  

  return dist/np.pi*180.

#color = (mag[:,0]-mag[:,1])
    #&(z>zrange[0])&(z<zrange[1])&(flag>0)&(color>-1)&(color<4)

def cutGalaxyCatalog(ra,dec,z,zerr,clusters,rmax=3,r_in=6,r_out=8,length=12,window=0.1):
    '''
       Get squares with lenth (Mpc) around each cluster position

       Parameters
    ----------
    ra : float, array
        Right ascension of the first object in degrees.
    dec : float, array
        Declination of the first object in degrees.
    z: array
        redshift of the galaxies
    cluster : astropy Table
    length : float
        square size

    Returns
    -------
    idxSubGalsCat : array
        indices for the galaxies in the cutted squares
    cid : array
        cluster id vector for the galaxy catalog
    radii : array
        radial distance from the cluster position
    '''
    ncls = len(clusters)
    cid = np.empty(0,dtype=int)
    radii = np.empty(0,dtype=float)
    pdfz = np.empty(0,dtype=float)

    ## divide to conquer
    idxSubGalsCat = np.empty(0,dtype=int)
    for i in range(ncls):
        ## cut 12Mpc around each cluster
        ra_c , dec_c = clusters['RA'][i], clusters['DEC'][i]
        cls_id, DA = clusters['CID'][i], clusters["DA"][i]
        cls_z = clusters['redshift'][i]

        thetha12Mpc = (180/np.pi)*(length/DA) ## in degrees

        mask = (np.abs(ra-ra_c)<thetha12Mpc)&(np.abs(dec-dec_c)<thetha12Mpc)
        idxSubGal, = np.where(mask)

        theta_offset = getAngDist(ra[idxSubGal], dec[idxSubGal], ra_c, dec_c)
        r = np.array(theta_offset*(np.pi/180)*DA) ## in Mpc

        idx, = np.where((r<rmax)|(r>4)|(r<r_out))
        # idx, = np.where(r<r_out)
        idxSubGal = idxSubGal[idx]

        clusterIDX = cls_id*np.ones_like(idxSubGal)
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],cls_z,window=window)
        
        idxSubGalsCat = np.append(idxSubGalsCat,idxSubGal)
        radii = np.append(radii,r)
        cid = np.append(cid,clusterIDX)
        pdfz = np.append(pdfz,pdf)
        
    return idxSubGalsCat, cid, radii, pdfz

def getMagLimModel(auxfile,zvec,dm=0):
    '''
    Get the magnitude limit for 0.4L*
    
    file columns: z, mag_lim_i, (g-r), (r-i), (i-z)
    mag_lim = mag_lim + dm
    
    input: Galaxy clusters redshift
    return: mag. limit for each galaxy cluster and for the r,i,z bands
    output = [maglim_r(array),maglim_i(array),maglim_z(array)]

    mi = mi-7 ## a correction to take the maglim as the BCG
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
    maglim_r, maglim_i, maglim_z = (mag_r-7+dm),(mag_i-7+dm),(mag_z-7+dm)

    magLim = np.array([maglim_r, maglim_i, maglim_z])
    
    return magLim.transpose()

def doSomeCuts(zg1, mag, flags_gold, AMAG, mult_niter, magMin=-19.5, crazy=[-1,4], zrange=[0.01,1.]): 
    gr = mag[:,0]-mag[:,1]
    ri = mag[:,1]-mag[:,2]
    iz = mag[:,2]-mag[:,3]

    w, = np.where( (zg1>zrange[0]) & (zg1<zrange[1]) & (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1]) & (mult_niter>0) & (AMAG<=magMin) & (flags_gold==0.) )

    return w

## -------------------------------
## Load Catalogs
def readClusterCat(catInFile, colNames, idx=None, massProxy=False):
    logging.debug('Starting helper.readClusterCat()')
    
    data = Table(getdata(catInFile))

    if idx is not None:
        data=data[idx] # JCB: to divide and conquer

    id = np.array(data[colNames[0]],dtype=int)
    ra = np.array(data[colNames[1]])
    dec = np.array(data[colNames[2]])
    z = np.array(data[colNames[3]])

    ## get RA from -180 to +180
    ra = convertRA(ra)

    ## get angular diameter
    DA = AngularDistance(z)

    print('Geting magnitude limite model')
    auxfile = './cat/auxTable/red_galaxy_El1_COSMOS_DES_filters.txt'
    mag_model_riz = getMagLimModel(auxfile,z,dm=3)
    
    if massProxy:
        mp = data[colNames[4]]
        clusterOut = Table([id, ra, dec, z, DA, mp, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA','massProxy', 'magLim'))

    else:
        clusterOut = Table([id, ra, dec, z, DA, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim'))

    logging.debug('Returning from helper.readClusterCat()')

    return clusterOut

def importTable(galaxyInFile,clusters,colNames,zrange=(0.01,1.),radius=12,window=0.1,rmax=3,r_in=6,r_out=8):
    print('Uploading data table')
    t0 = time()

    data = Table(getdata(galaxyInFile))
    
    tu = time()-t0
    print('time:',tu)
    print('\n')

    ra = convertRA(np.array(data[colNames[1]]))
    dec = np.array(data[colNames[2]])
    z = data[colNames[3]]
    zerr = data[colNames[7]]
    flag = data[colNames[8]]

    mag = []
    for i in range(4):
        mag.append(data[colNames[4+i]])
    
    mag = np.array(mag).transpose() ## 4 vector

    amag = data['Mr'][:]
    multi = data['MULT_NITER_MODEL'][:]

    ## do cuts on the galaxy catalog
    indices = doSomeCuts(z,mag,flag,amag,multi,zrange=zrange,magMin=-19.5)

    print('Cut data table')
    t0 = time()
    # cid = data['MEM_MATCH_ID']
    idx, cid, radii, pdfz = cutGalaxyCatalog(ra[indices],dec[indices],z[indices],zerr[indices],clusters,rmax=rmax,r_in=r_in,r_out=r_out,
                            length=radius,window=window)
    print('returning galaxy cut')

    tc = time()-t0
    print('time:',tc)
    print('\n')

    new_idx = indices[idx]
    
    galaxyData = data[new_idx]

    if len(galaxyData)<1:
        print('Critical Error')
        exit()

    return galaxyData, cid, radii, pdfz

def getVariables(galaxyData,colNames):

    gid = np.array(galaxyData[colNames[0]][:])

    ra = convertRA(np.array(galaxyData[colNames[1]]))
    dec = np.array(galaxyData[colNames[2]])
    
    z = np.array(galaxyData[colNames[3]])
    zerr = np.array(galaxyData[colNames[9]])

    mag, magerr = [], []
    for i in range(4):
        mag.append(galaxyData[colNames[4+i]])
        magerr.append(galaxyData[colNames[10+i]])
    
    mag = np.array(mag).transpose() ## 4 vector
    magerr = np.array(magerr).transpose() ## 4 vector

    return gid,ra,dec,z,zerr,mag,magerr

def readGalaxyCat(galaxyInFile, clusters, rmax=3,r_in=8,r_out=10, radius=12, window=0.1, zrange=(0.01,1.),Nflag=0,colNames=None):
    logging.debug('Starting helper.readGalaxyCat()')
    
    if colNames is None:
        print('Please provide the Galaxy column names - exiting the code')
        exit()

    ## divide to conquer! (take squares around each cluster position)
    ## radius is the length of the square in Mpc
    galaxyData, cid, radii, PDFz = importTable(galaxyInFile,clusters,colNames,rmax=rmax,r_in=r_in,r_out=r_out,
                                    zrange=(0.01,1.),radius=radius,window=0.1)

    gid,ra,dec,z,zerr,mag,magerr = getVariables(galaxyData,colNames)

    bkgGalaxies = (radii>r_in)&(radii<r_out)
    
    inputdata = [cid, gid, ra, dec, radii, z, mag, zerr, magerr, PDFz, bkgGalaxies]
    Cnames = ['CID', 'GID', 'RA', 'DEC', 'R', 'z', 'mag','zerr','magerr', 'PDFz', 'Bkg']

    galaxyOut = Table(inputdata,
                      names=Cnames)
    
    logging.debug('Returning from helper.readGalaxyCat()')

    return galaxyOut


if __name__ == '__main__':
    print('helper.py')
    print('author: Johnny H. Esteves')
    
    # time_gal = time()    
    # print('\n Teste')
    # root = '/home/johnny/Documents/IAG-USP/Master/catalogos/'
    # file_gal = root+"galaxias/primary/y1a1_gold_bpz_mof_xmatcha_12Mpc.fits"
    # file_cls = root+"aglomerados/primary/xmatcha.fits"
    
    # columnsLabelsCluster = ['MEM_MATCH_ID','Xra','Xdec','redshift','r500']
    # columnsLabelsGalaxy = ['COADD_OBJECTS_ID','RA','DEC','MEDIAN_Z',
    #                        'mg','mr','mi','mz', 'FLAGS_GOLD','SIGMA_Z',
    #                        'mg_err','mr_err','mi_err','mz_err']

    # idx = np.arange(0,10,1,dtype=int)
    # clusters = readClusterCat(file_cls, idx=idx, massProxy=True,
	# 								 colNames=columnsLabelsCluster)

    # # gal = readGalaxyCat(file_gal, clusters, radius=12, colNames=columnsLabelsGalaxy)
    # gal = queryGalaxyCat(clusters, radius=12, zrange=(0.01,1.), Nflag=0, HEALPix=True)
    # # gal.write('galTest')
    # end_time = time()-time_gal

    # print('Time:',end_time)