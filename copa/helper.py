# !/usr/bin/env python

from astropy.table import Table, vstack
from astropy.io.fits import getdata

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.stats import median_absolute_deviation
import astropy.io.fits as pyfits

from scipy.interpolate import interp1d
import scipy.integrate as integrate
import numpy as np
import logging
import esutil
from time import time
import os
import pandas as pd
import dask

import membAssignment as memb

h=0.7
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.286)
Mpc2cm = 3.086e+24

## -------------------------------
## auxiliary functions

def readfile(filename,columns=None):
    '''
    Read a filename with fitsio.read and return an astropy table
    '''
    import fitsio

    hdu = fitsio.read(filename, columns=columns)
    return Table(hdu)

def loadfiles(filenames, columns=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    import fitsio

    out = []
    i = 1
    print
    for f in filenames:
        # print 'File {i}: {f}'.format(i=i, f=f)
        out.append(fitsio.read(f, columns=columns))
        i += 1

    return Table(np.concatenate(out))

def readFile(file_gal,columns=None):
    galaxy_cat = pyfits.open(file_gal)
    g0 = Table(galaxy_cat[1].data)
    if columns is not None:
        g0 = g0[columns]
    return g0

def loadFiles(files,columns=None):
    import os.path as path
    allData = []

    for file_gal in files:
        if path.isfile(file_gal):
            g0 = readfile(file_gal,columns=columns)
            allData.append(g0)
        
        else:
            print('file not found %s'%(file_gal))

    g = vstack(allData)
    return g

def convertRA(ra):
    ra = np.where(ra>180,ra-360,ra)
    return ra

def AngularDistance(z):
    DA = float( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA
AngularDistance = np.vectorize(AngularDistance)

def look_up_table_photozWindow(sigma,filename='./auxTable/pz_sigma_n_optimal.csv'):
    data = Table.read(filename)
    np, sg = data['npeak'], data['sigma'][:]

    nout = interp1d(np,sg,fill_value='extrapolate',copy=False)(sigma)

    return nout

def look_up_table_photoz_model(z,filename='./auxTable/photoz_cls_model_dnf_y3_gold_2_2.fits'):
    data = Table.read(filename,format='ascii')
    zmean, bias, sigma = data['z'][:], data['mean'], data['scatter'][:]

    biasz  = interp1d(zmean,bias,fill_value='extrapolate',copy=False)(z)
    sigmaz = interp1d(zmean,sigma,fill_value='extrapolate',copy=False)(z)

    return biasz,sigmaz

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

def PhotozProbabilities(zmin,zmax,membz,membzerr,fast=False,zcls=0.1):
    if fast:
        out = fastGaussianIntegration(membz,membzerr,zmin,zmax)
    else:
        out = []
        for i in range(len(membz)):
            aux, err = integrate.fixed_quad(gaussian,zmin,zmax,args=(membz[i],membzerr[i]))
            
            out.append(aux)
        out = np.array(out)
    return out

def truncatedGaussian(z,zcls,zmin,zmax,sigma,vec=False):
    if vec:
        z0,zcls0,sigma0 = z,zcls,sigma

        z_shape = zcls.shape
        s_shape = sigma.shape
        if z_shape!=s_shape:
            print('PDFZ : error')
            print('z_shape,s_shape:',z_shape,s_shape)
            
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
    try:
        pdf = truncnorm.pdf(z, a, b, loc = my_mean, scale = (my_std+eps))
        if vec: pdf.shape = s_shape
        
    except:
        print('PDFz error: ecception')
        if vec: 
            pdf = gaussian(z0,zcls0,sigma0)
        else:
            pdf = gaussian(z,zcls,sigma)
        
    return pdf

def getPDFz(membz,membzerr,zcls,sigma):
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

zgrid2 = np.arange(0.,3.01,0.01)+0.005
def _computePz0(pdfz,zcls,bias,sigma):
    zz, _ = np.meshgrid(zgrid2,pdfz[0,:])
    zz = zz.T

    zoff = ((zgrid2-zcls)/(1+zcls)) - bias
    w, = np.where( np.abs(zoff) <= 1.5*sigma )

    pz0 = integrate.trapz(pdfz[w,:].T,x=zz[w,:].T)
    pz0 = np.where(pz0>1., 1., pz0)

    return pz0

def computePz0(gidx,pdfz,cat,sigma):
    ncls = len(cat)

    indicies = np.empty((0),dtype=int)
    pz = np.empty((0),dtype=float)
    
    if sigma < 0.:
        print('Photoz Model: bias+sigma')
        #bias, sigma = look_up_table_photoz_model(cat['redshift'],filename='auxTable/bpz_phtoz_model_cosmoDC2.csv')
        bias, sigma = look_up_table_photoz_model(cat['redshift'],filename='auxTable/y3_dnf_photoz_model.csv')
        
    else:
        sigma = sigma*(1+cat['redshift'])
        bias  = np.zeros_like(cat['redshift'])

    for i in range(ncls):
        z_cls, cls_id = cat['redshift'][i], cat['CID'][i]
        idx, = np.where(gidx==cls_id)
        
        if idx.size>0:
            pdfi = pdfz[:,idx]
            pz0 = _computePz0(pdfi,z_cls,bias[i],sigma[i])
        
            indicies = np.append(indicies,idx)
            pz  = np.append(pz,pz0)
    
    return pz, indicies

def computePDFz(z,zerr,cid,cat,sigma):
    ncls = len(cat)
    indicies = np.empty((0),dtype=int)
    pdfz = np.empty((0),dtype=float)

    if sigma < 0.:
        # bias, sigma = look_up_table_photoz_model(cat['redshift'],filename='auxTable/bpz_phtoz_model_cosmoDC2.csv')
        bias, sigma = look_up_table_photoz_model(cat['redshift'],filename='auxTable/y3_dnf_photoz_model.csv')
    else:
        sigma = sigma*(1+cat['redshift'])
        bias  = np.zeros_like(cat['redshift'])

    # results = []
    for i in range(ncls):
        z_cls, idx = cat['redshift'][i], cat['CID'][i]
        
        idxSubGal, = np.where(cid==idx)
        
        # r1 = dask.delayed(getPDFz)(z[idxSubGal],zerr[idxSubGal],z_cls,sigma)
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],z_cls+bias[i]*(1+z_cls),sigma[i]*(1+z_cls))

        # results.append(r1)
        indicies = np.append(indicies,idxSubGal)
        pdfz  = np.append(pdfz,pdf)

    # results = dask.compute(*results, scheduler='processes', num_workers=2)
    # for i in range(ncls):
    #     pdf = results[i]
    #     pdf = np.where(pdf<1e-4,0.,pdf) ## it avoids that float values gets boost by color pdfs
    #     pdfz  = np.append(pdfz,pdf)
    
    return pdfz, indicies


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

        # idx, = np.where((r<rmax)|(r>4)|(r<r_out))
        idx, = np.where(r<length)
        idxSubGal = idxSubGal[idx]

        clusterIDX = cls_id*np.ones_like(idxSubGal)
        pdf = np.zeros_like(z[idxSubGal])
        #pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],cls_z)
        
        idxSubGalsCat = np.append(idxSubGalsCat,idxSubGal)
        radii = np.append(radii,r[idx])
        cid = np.append(cid,clusterIDX)
        pdfz = np.append(pdfz,pdf)
        
    return idxSubGalsCat, cid, radii, pdfz


def aperture_match(ra_cluster,dec_cluster,ang_diam_dist,ra_galaxy,dec_galaxy,r_aper=10):
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

    return indicies_into_galaxies_in_aperture, indicies_into_clusters,disti


def cutCircle(ra_galaxy,dec_galaxy,clusters,rmax=12):
    '''
       Get circles with rmax (Mpc) around each cluster position

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
        indices for the galaxies in the circles
    cid : array
        indices for the cluster table
    '''
    ra_cluster=clusters.field('RA')
    dec_cluster=clusters.field('DEC')

    DA = clusters['DA'][:]
    indicies_into_galaxies_in_aperture, indicies_into_clusters, dist = aperture_match(ra_cluster,dec_cluster,DA,
                                                                                    ra_galaxy,dec_galaxy,r_aper=rmax)

    radii = np.array( dist*(np.pi/180)*DA[indicies_into_clusters] ) ## in Mpc

    return indicies_into_galaxies_in_aperture, indicies_into_clusters, radii

    # return idxSubGalsCat, cid, radii, pdfz

def getMagLimModel(auxfile,zvec,dm=2):
    '''
    Get the magnitude limit for Buzzard BCG galaxies
    '''
    bcgs=np.loadtxt(auxfile,delimiter=',')
    z=[i[0] for i in bcgs]  ## redshift
    mag_g=[i[1] for i in bcgs] ## g-band
    mag_r=[i[2] for  i in bcgs]## r-band
    mag_i=[i[3] for i in bcgs] ## i-band
    mag_z=[i[4] for i in bcgs] ## z-band
    
    interp_g=interp1d(z,mag_g,fill_value='extrapolate')
    interp_r=interp1d(z,mag_r,fill_value='extrapolate')
    interp_i=interp1d(z,mag_i,fill_value='extrapolate')
    interp_z=interp1d(z,mag_z,fill_value='extrapolate')

    m_g,m_r,m_i,m_z = interp_g(zvec),interp_r(zvec),interp_i(zvec),interp_z(zvec)

    magLim = np.array([m_r+dm, m_i+dm, m_z+dm])
    
    return magLim.transpose()

def getMagLimModel_04L(auxfile,zvec,dm=0):
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

    magLim = np.array([maglim_r, maglim_i, maglim_z])
    
    return magLim.transpose()

# def doSomeCuts(zg1, mag, flags_gold, AMAG, mult_niter, magMin=-19.5, crazy=[-1,4], zrange=[0.01,1.]): 
#     ''' these are the cuts from the old membAssignment version'''
#     gr = mag[:,0]-mag[:,1]
#     ri = mag[:,1]-mag[:,2]
#     iz = mag[:,2]-mag[:,3]

#     w, = np.where( (zg1>zrange[0]) & (zg1<zrange[1]) & (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1]) & (mult_niter>0) & (AMAG<=magMin) & (flags_gold==0.) )

#     return w

def do_color_redshift_cut(zg1, mag, crazy=[-0.5,4.], zrange=[0.01,1.]):
    # gcut, rcut, icut, zcut = 24.33, 24.08, 23.44, 22.69 ## Y1 SNR_10 mag cut

    gr = mag[:,0]-mag[:,1]
    gi = mag[:,0]-mag[:,2]
    ri = mag[:,1]-mag[:,2]
    rz = mag[:,1]-mag[:,3]
    iz = mag[:,2]-mag[:,3]

    w, = np.where( (zg1>=zrange[0]) & (zg1<=zrange[1]) & 
                   (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1]) )
                   #& (gi>crazy[0]) & (gi<crazy[1]) & (rz>crazy[0]) & (rz<crazy[1])) #& (amag <= magMin) ) #
    return w

## -------------------------------
## Load Catalogs
def readClusterCat(catInFile, colNames, idx=None, massProxy=False, simulation=False):
    logging.debug('Starting helper.readClusterCat()')
    
    # data = Table(getdata(catInFile))
    if simulation:
        colNames.append('r200')
        colNames.append('m200')

    data = readFile(catInFile)

    #if idx is not None:
    #    data=data[idx] # JCB: to divide and conquer

    ## excluding objects in the edge
    if 'area_frac' in data.colnames:
        data = data[data['area_frac']==1]
    print(data.colnames)
    id = np.array(data[colNames[0]],dtype=int)
    id, indices = np.unique(id, return_index=True)

    ra = np.array(data[colNames[1]])[indices]
    dec = np.array(data[colNames[2]])[indices]
    z = np.array(data[colNames[3]])[indices]

    ## get RA from -180 to +180
    ra = convertRA(ra)

    ## get angular diameter
    DA = AngularDistance(z)

    print('Geting magnitude limite model')
    # auxfile = './auxTable/buzzard_BCG_mag_model.txt'
    # mag_model_riz = getMagLimModel(auxfile,z,dm=3)
    
    # auxfile = './auxTable/buzzard_Mr_20p5_model.txt'
    # auxfile = './auxTable/buzzard_Mr_19p5_model.txt'
    # mag_model_riz = getMagLimModel(auxfile,z,dm=0)

    auxfile = './auxTable/annis_mags_04_Lcut.txt'
    mag_model_riz = getMagLimModel_04L(auxfile,z,dm=0)

    if not simulation:
        mf = data[colNames[4]][indices]
        r200 = data['R_LAMBDA'][indices]#/0.7
        m200 = data[colNames[4]][indices]
        # hpx = np.array(data['healpix_pixel'][indices])
        # clusterOut = Table([id, ra, dec, z, DA, mag_model_riz, m200], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim','M200_true'))
        clusterOut = Table([id, ra, dec, z, DA, mf, r200, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'massProxy', 'R200_true', 'magLim'))

    if simulation:
        r200 = data['r200'][indices]
        m200 = data[colNames[4]][indices]
        #hpx = np.array(data['healpix_pixel'][indices])
        #clusterOut = Table([id, ra, dec, z, DA, mag_model_riz, r200, m200, hpx], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim', 'R200_true','M200_true','hpx'))
        clusterOut = Table([id, ra, dec, z, DA, mag_model_riz, r200, m200], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim', 'R200_true','M200_true'))
        
    # else:
    #     clusterOut = Table([id, ra, dec, z, DA, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim'))

    logging.debug('Returning from helper.readClusterCat()')
    return clusterOut

# def importTable(galaxyInFile,clusters,colNames,zrange=(0.01,1.),radius=12,window=0.1,rmax=3,r_in=6,r_out=8):
#     print('Uploading data table')
#     t0 = time()

#     data = Table(getdata(galaxyInFile))    
#     # data = readfile(galaxyInFile,columns=colNames)
    
#     tu = time()-t0
#     print('time:',tu)
#     print('\n')

#     ra = convertRA(np.array(data[colNames[1]]))
#     dec = np.array(data[colNames[2]])
#     z = data[colNames[3]]
#     zerr = np.abs(data[colNames[9]])
#     flag = data[colNames[8]]
    
#     mag = []
#     for i in range(4):
#         mag.append(data[colNames[4+i]])
    
#     mag = np.array(mag).transpose() ## 4 vector

#     amag = -30.*np.ones_like(z)
#     # multi = data['MULT_NITER_MODEL'][:]

#     ## do cuts on the galaxy catalog
#     # indices = doSomeCuts(z,mag,flag,amag,multi,zrange=zrange,magMin=-19.5) ## old cuts
#     indices = doSomeCuts(z,mag,zrange=zrange)

#     print('Cut data table')
#     t0 = time()
#     # cid = data['halo_id']
#     # idx, cid, radii, pdfz = cutGalaxyCatalog(ra[indices],dec[indices],z[indices],zerr[indices],clusters,rmax=rmax,r_in=r_in,r_out=r_out,
#     #                         length=radius,window=window)
#     idx, cidx, radii = cutCircle(ra[indices],dec[indices],clusters,rmax=radius)

#     print('returning galaxy cut')
#     tc = time()-t0
#     print('time:',tc)
#     print('\n')

#     new_idx = indices[idx]
#     galaxyData = data[new_idx]

#     if len(galaxyData)<1:
#         print('Critical Error')
#         exit()

#     cid = clusters['CID'][cidx]
#     z,zerr = z[new_idx], zerr[new_idx]
#     pdfz = np.zeros_like(z,dtype=float) #getPDFzM(z,zerr,cid,clusters)

#     return galaxyData, cid, radii, pdfz

def getVariables(galaxyData,colNames,zsigma=0.05,simulation=False):
    gid = np.array(galaxyData[colNames[0]][:])

    ra = convertRA(np.array(galaxyData[colNames[1]]))
    dec = np.array(galaxyData[colNames[2]])

    np.random.RandomState(seed=42)
    z = np.array(galaxyData[colNames[3]].copy())
    zerr = np.array(galaxyData[colNames[9]])#zsigma*(1+z)

    ## photz estimations
    # z_noise = z

    ## sinthetic photoz
    z_noise = z - np.random.normal(scale=zsigma*(1+z),size=len(z))
    z_noise = np.where(z_noise<0.,0.,z_noise)
    zerr = zsigma*(1+z)

    mag, magerr = [], []
    for i in range(4):
        mag.append(galaxyData[colNames[4+i]])
        magerr.append(np.zeros_like(z))
        # magerr.append(galaxyData[colNames[9+i]])
    
    mag = np.array(mag).transpose() ## 4 vector
    magerr = np.array(magerr).transpose() ## 4 vector

    ## initialize some variables
    cid = np.zeros_like(gid)
    radii = np.zeros_like(z)
    pdfz = radii
    bkgFlag = np.full(gid.shape,False)

    return cid,gid,ra,dec,radii,z,z_noise,zerr,mag,magerr,pdfz,bkgFlag


def getRadii(galaxy,clusters,h=0.7,rmax=12,r_in=4,r_out=6,simulation=True):
    ### get a circle
    ra,dec = galaxy['RA'],galaxy['DEC']

    idx, cidx, radii = cutCircle(ra,dec,clusters,rmax=rmax)
    cid = clusters['CID'][cidx]
    bkgGalaxies = (radii>r_in)&(radii<r_out)

    galaxyData = galaxy[idx]
    galaxyData['CID'] = cid
    galaxyData['R'] = radii/h
    galaxyData['Bkg'] = bkgGalaxies
    galaxyData['dmag'] = galaxyData['mag'][:,2]-clusters['magLim'][cidx,1] ##i-band

    ## update true members
    if simulation:
        # galaxyData['True'] = np.where((galaxyData['HALOID']==cid)&(galaxyData['True']==True),True,False)
        galaxyData['True'] = np.where((galaxyData['halo_id']==cid),True,False)
        galaxyData.remove_column('halo_id')
        
    return galaxyData

def do_galaxy_cuts(galaxy,clusters,zrange=[0.,1.3]):
    ## do some cuts

    ## get unique galaxy ids
    ids, indices = np.unique(galaxy['GID'], return_index=True)
    galaxy1 = galaxy[indices]

    ## get out with crazy colors
    ## get galaxies in the redshift range
    z = galaxy1['z']
    mag = galaxy1['mag']

    indices_cut = do_color_redshift_cut(z,mag,zrange=zrange)
    galaxy2 = galaxy1[indices_cut]

    indices_cut, = np.where((galaxy2['zerr']<2.)&(galaxy2['zerr']>=0.))
    galaxy2 = galaxy2[indices_cut]

    ## get an upper cut on the magnitude (i-band)
    mag_i = galaxy2['mag'][:,2]
    upper_cut = np.max(clusters['magLim'][:,1])+10
    indices_mag = mag_i<=upper_cut

    return galaxy2[indices_mag]

def generate_sample(kde,nsample=10000):
    return np.random.choice(zgrid2, nsample, p=kde/np.sum(kde))

def conf68(pdfz):
    v = np.percentile(pdfz, [16, 50, 84])
    sigma68 = 0.5*(v[1]-v[0])
    return v[0], sigma68

def get_sigma68(pdfz,nsample=1000):
    sample = generate_sample(pdfz,nsample=nsample)
    median, s68 = conf68(sample)
    return s68

def get_sigma(pdfz,nsample=1000):
    sample = generate_sample(pdfz,nsample=nsample)
    sigma = np.std(sample)
    return sigma

def read_close_pdf(pi):
    np.set_printoptions(precision=5, suppress=True)
    x = pi.strip("[]")
    b = x.split('\n')
    b = ('         ').join(b)
    array = np.fromstring(b, dtype = np.float,  sep ='         ' ,count=301)
    return array

def get_pdfz(pdfz_list):
    # res = [dask.delayed(read_close_pdf)(pi) for pi in pdfz_list]
    # pdfz = dask.compute(*res, scheduler='processes', num_workers=12)
    pdfz = [read_close_pdf(pi) for pi in pdfz_list]
    return np.vstack(pdfz).T

def readCosmoDC2(galaxyInFile, clusters, h=0.7, rmax=3,r_in=8,r_out=10, radius=12, window=0.1, zrange=(0.01,1.),Nflag=0,colNames=None,simulation=False):
    healpix_list = list(np.unique(clusters['hpx']))
    galaxy_prefix = os.path.splitext(galaxyInFile)[0]

    # infile_list = [galaxy_prefix+'_{:0>5d}.csv'.format(hpx) for hpx in healpix_list]
    # infile_list = [infile for infile in infile_list if os.path.isfile(infile)]
    
    # res = [dask.delayed(_readCosmoDC2)(infile, clusters, h=0.7, rmax=rmax,r_in=r_in,r_out=r_out, radius=radius, window=window, zrange=zrange,colNames=colNames,simulation=simulation)
    #         for infile in infile_list]
    # data = dask.compute(*res, scheduler='processes', num_workers=32)

    data = []
    for hpx in healpix_list:
        print('hpx:',hpx)
        infile = galaxy_prefix+'_{:0>5d}b.csv'.format(hpx)
        if os.path.isfile(infile):
            gi = _readCosmoDC2(infile, clusters, h=0.7, rmax=rmax,r_in=r_in,r_out=r_out, radius=radius, window=window, zrange=zrange,colNames=colNames,simulation=simulation)
            data.append(gi)
    
    galaxy = vstack(data)
    return galaxy


def _readCosmoDC2(galaxyInFile, clusters, h=0.7, rmax=3,r_in=8,r_out=10, radius=12, window=0.1, zrange=(0.01,1.),Nflag=0,colNames=None,simulation=False):
    logging.debug('Starting helper.readGalaxyCat()')
    
    if colNames is None:
        print('Please provide the Galaxy column names - exiting the code')
        exit()

    ## get the data
    # galaxyData = Table(getdata(galaxyInFile))
    df = pd.read_csv(galaxyInFile)
    # pdfz = get_pdfz(df['photoz_pdf'][:])

    del df['photoz_pdf']

    galaxyData = Table.from_pandas(df)
    
    ## get variables
    cid,gid,ra,dec,radii,z,z_noise,zerr,mag,magerr,PDFz,bkgFlag = getVariables(galaxyData,colNames,zsigma=window,simulation=simulation) 
    
    dmag = np.zeros_like(cid)
    
    ### cosmoDC2
    cid = galaxyData['CID']
    radii = galaxyData['R']
    dmag = galaxyData['dmag']
    smass= galaxyData['stellar_mass']

    inputdata = [cid, gid, ra, dec, radii, z_noise, mag, dmag, zerr, magerr, PDFz, bkgFlag, smass]
    Cnames = ['CID', 'GID', 'RA', 'DEC', 'R', 'z', 'mag', 'dmag', 'zerr', 'magerr', 'PDFz', 'Bkg', 'stellar_mass']

    Cnames.append('halo_id'); Cnames.append('True'); Cnames.append('z_true'); Cnames.append('Mr');
    inputdata.append(galaxyData['halo_id']); inputdata.append(galaxyData['TRUE_MEMBERS']); inputdata.append(z); inputdata.append(galaxyData['Mag_true_r_lsst_z0']);

    galaxy = Table(inputdata,names=Cnames)

    ### cosmoDC2
    bkgGalaxies = (galaxy['R']>r_in)&(galaxy['R']<r_out)
    galaxy['Bkg'] = bkgGalaxies
    galaxy['True'] = np.where((galaxy['halo_id']==galaxy['CID']),True,False)

    ## cut galaxies outside the maglim
    idx, = np.where(galaxy['R']<=r_out)
    # for i in idx:
    #     galaxy['zerr'][i] = get_sigma68(pdfz[:,i],nsample=int(5e3))
    
    galaxy_cut = galaxy[idx]

    # compute Pz0 from the galaxy pdf
    # pz,idxs = computePz0(galaxy_cut['CID'],pdfz[:,idx],clusters,window)
    # galaxy_cut['PDFz'][idxs] = pz

    return galaxy_cut

def readGalaxyCat(galaxyInFile, clusters, h=0.7, rmax=3,r_in=8,r_out=10, radius=12, window=0.1, zrange=(0.01,1.),Nflag=0,colNames=None,simulation=False):
    logging.debug('Starting helper.readGalaxyCat()')
    print('reading galaxy catalog')
    
    if colNames is None:
        print('Please provide the Galaxy column names - exiting the code')
        exit()

    ## get the data
    galaxyData = Table(getdata(galaxyInFile))
    
    if not simulation:
        galaxyData = galaxyData[galaxyData['FLAGS_GOLD']==0]

    ## get variables
    cid,gid,ra,dec,radii,z,z_noise,zerr,mag,magerr,PDFz,bkgFlag = getVariables(galaxyData,colNames,zsigma=window,simulation=simulation) 
    
    # Mr = galaxyData['Mr']
    dmag = np.zeros_like(cid)
    inputdata = [cid, gid, ra, dec, radii, z_noise, mag, dmag, zerr, magerr, PDFz, bkgFlag]
    Cnames = ['CID', 'GID', 'RA', 'DEC', 'R', 'z', 'mag','dmag', 'zerr', 'magerr', 'PDFz', 'Bkg']

    if simulation:
        # Cnames.append('halo_id');Cnames.append('True');Cnames.append('amag'); Cnames.append('Mr');Cnames.append('z_true')
        # inputdata.append(galaxyData['halo_id']); inputdata.append(galaxyData['TRUE_MEMBERS']);inputdata.append(galaxyData['AMAG']);inputdata.append(galaxyData['Mr']);inputdata.append(z)

        Cnames.append('halo_id');Cnames.append('True');Cnames.append('z_true'); #Cnames.append('Mr');
        #inputdata.append(galaxyData['halo_id']); inputdata.append(galaxyData['TRUE_MEMBERS']);inputdata.append(z);inputdata.append(galaxyData['Mag_true_r_lsst_z0']);
        inputdata.append(galaxyData['halo_id']); inputdata.append(galaxyData['TRUE_MEMBERS']);inputdata.append(z);

    galaxy = Table(inputdata,names=Cnames)

    ## make cuts on the galaxy dataset
    galaxy_cut = do_galaxy_cuts(galaxy,clusters,zrange=zrange)

    ## get distance from the cluster center
    galaxyCircles = getRadii(galaxy_cut,clusters,h=h,rmax=radius,r_in=r_in,r_out=r_out,simulation=simulation)

    ## cut galaxies outside the maglim
    galaxy_maglim = galaxyCircles[galaxyCircles['dmag']<=0.]
    # galaxy_maglim = galaxyCircles[(galaxyCircles['Mr']-np.log10(h))<=-19.5]

    # zerr = np.where(galaxy_maglim['zerr']>0.3,0.3,galaxy_maglim['zerr'])
    # pz,idxs = computePDFz(galaxy_maglim['z'],zerr,galaxy_maglim['CID'],clusters,window)
    # galaxy_maglim['PDFz'][idxs] = pz

    # ids, indices = np.unique(galaxyCircles['GID','CID'], return_index=True)
    # galaxyCircles = galaxyCircles[indices]

    logging.debug('Returning from helper.readGalaxyCat()')
    return galaxy_maglim

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
