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
import esutil
from time import time

import fitsio

cosmo = FlatLambdaCDM(H0=70, Om0=0.286)
Mpc2cm = 3.086e+24

## -------------------------------
## auxiliary functions

def readfile(filename,columns=None):
    '''
    Read a filename with fitsio.read and return an astropy table
    '''
    hdu = fitsio.read(filename, columns=columns)
    return Table(hdu)

def loadfiles(filenames, columns=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    out = []
    i = 1
    print
    for f in filenames:
        # print 'File {i}: {f}'.format(i=i, f=f)
        out.append(fitsio.read(f, columns=columns))
        i += 1

    return Table(np.concatenate(out))

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
            aux, err = integrate.fixed_quad(gaussian,zmin,zmax,args=(membz[i],membzerr[i]))
            out.append(aux)
        out = np.array(out)
    return out

def getPDFz(membz,membzerr,zi):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    sigma = np.median(membzerr)
    ni = look_up_table_photozWindow(sigma)
    # ni = 1
    
    # zmin, zmax = (zi-ni*0.01*(1+zi)), (zi+ni*0.01*(1+zi))
    # pdf = PhotozProbabilities(zmin,zmax,membz,sigma*np.ones_like(membz))
    # pdf = PhotozProbabilities(zmin,zmax,membz,membzerr)
    pdf = gaussian(zi,membz,membzerr)
    # pdf = np.where(pdf<0.001,0.,pdf/np.max(pdf))
    return pdf

def getPDFzM(z,zerr,cid,cat):
    ncls = len(cat)
    pdfz = np.empty(0,dtype=float)

    for i in range(ncls):
        z_cls, idx = cat['redshift'][i], cat['CID'][i]
        idxSubGal, = np.where(cid==idx)
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],z_cls)
        pdfz = np.append(pdfz,pdf)
    
    return pdfz

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
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],cls_z,window=window)
        
        idxSubGalsCat = np.append(idxSubGalsCat,idxSubGal)
        radii = np.append(radii,r[idx])
        cid = np.append(cid,clusterIDX)
        pdfz = np.append(pdfz,pdf)
        
    return idxSubGalsCat, cid, radii, pdfz


def aperture_match(ra_cluster,dec_cluster,ang_diam_dist,ra_galaxy,dec_galaxy,r_aper=10):
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
    mag_i=[i[3] for i in bcgs] ## 
    mag_z=[i[4] for i in bcgs] ## 
    
    interp_g=interp1d(z,mag_g,fill_value='extrapolate')
    interp_r=interp1d(z,mag_r,fill_value='extrapolate')
    interp_i=interp1d(z,mag_i,fill_value='extrapolate')
    interp_z=interp1d(z,mag_z,fill_value='extrapolate')

    m_g,m_r,m_i,m_z = interp_g(zvec),interp_r(zvec),interp_i(zvec),interp_z(zvec)

    magLim = np.array([m_r+dm, m_i+dm, m_z+dm])
    
    return magLim.transpose()

def getMagLimModelHuanLi(auxfile,zvec,dm=0):
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

# def doSomeCuts(zg1, mag, flags_gold, AMAG, mult_niter, magMin=-19.5, crazy=[-1,4], zrange=[0.01,1.]): 
#     ''' these are the cuts from the old membAssignment version'''
#     gr = mag[:,0]-mag[:,1]
#     ri = mag[:,1]-mag[:,2]
#     iz = mag[:,2]-mag[:,3]

#     w, = np.where( (zg1>zrange[0]) & (zg1<zrange[1]) & (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1]) & (mult_niter>0) & (AMAG<=magMin) & (flags_gold==0.) )

#     return w

def doSomeCuts(zg1, mag, amag, flags_gold, crazy=[-1,4], zrange=[0.01,1.],magMin=-19.5):
    # gcut, rcut, icut, zcut = 24.33, 24.08, 23.44, 22.69 ## Y1 SNR_10 mag cut
    gcut, rcut, icut, zcut = 28, 28, 28, 28 ## crazy mag!

    gr = mag[:,0]-mag[:,1]
    ri = mag[:,1]-mag[:,2]
    iz = mag[:,2]-mag[:,3]

    w, = np.where( (zg1>zrange[0]) & (zg1<zrange[1]) & 
                   (gr>crazy[0]) & (gr<crazy[1]) & (ri>crazy[0]) & (ri<crazy[1]) & (iz>crazy[0]) & (iz<crazy[1]) &
                   (mag[:,2] < icut + 5) & (amag <= magMin) ) #

    return w

## -------------------------------
## Load Catalogs
def readClusterCat(catInFile, colNames, idx=None, massProxy=False, simulation=False):
    logging.debug('Starting helper.readClusterCat()')
    
    # data = Table(getdata(catInFile))
    if simulation:
        colNames.append('R200')
        colNames.append('M200')

    data = readfile(catInFile,columns=colNames)

    if idx is not None:
        data=data[idx] # JCB: to divide and conquer

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
    auxfile = './auxTable/buzzard_BCG_mag_model.txt'
    mag_model_riz = getMagLimModel(auxfile,z,dm=3)
    
    if massProxy:
        mp = data[colNames[4]][indices]
        clusterOut = Table([id, ra, dec, z, DA, mp, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA','massProxy', 'magLim'))

    if simulation:
        r200 = data['R200'][indices]
        m200 = data['M200'][indices]
        clusterOut = Table([id, ra, dec, z, DA, mag_model_riz, r200, m200], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim', 'R200_true','M200_true'))

    else:
        clusterOut = Table([id, ra, dec, z, DA, mag_model_riz], names=('CID', 'RA', 'DEC', 'redshift', 'DA', 'magLim'))

    logging.debug('Returning from helper.readClusterCat()')

    return clusterOut

def importTable(galaxyInFile,clusters,colNames,zrange=(0.01,1.),radius=12,window=0.1,rmax=3,r_in=6,r_out=8):
    print('Uploading data table')
    t0 = time()

    # data = Table(getdata(galaxyInFile))    
    data = readfile(galaxyInFile,columns=colNames)
    
    tu = time()-t0
    print('time:',tu)
    print('\n')

    ra = convertRA(np.array(data[colNames[1]]))
    dec = np.array(data[colNames[2]])
    z = data[colNames[3]]
    zerr = np.abs(data[colNames[9]])
    flag = data[colNames[8]]
    
    mag = []
    for i in range(4):
        mag.append(data[colNames[4+i]])
    
    mag = np.array(mag).transpose() ## 4 vector

    amag = -30.*np.ones_like(z)
    # multi = data['MULT_NITER_MODEL'][:]

    ## do cuts on the galaxy catalog
    # indices = doSomeCuts(z,mag,flag,amag,multi,zrange=zrange,magMin=-19.5) ## old cuts
    indices = doSomeCuts(z,mag,amag,flag,zrange=zrange)

    print('Cut data table')
    t0 = time()
    # cid = data['HALOID']
    # idx, cid, radii, pdfz = cutGalaxyCatalog(ra[indices],dec[indices],z[indices],zerr[indices],clusters,rmax=rmax,r_in=r_in,r_out=r_out,
                            # length=radius,window=window)
    idx, cidx, radii = cutCircle(ra[indices],dec[indices],clusters,rmax=radius)

    print('returning galaxy cut')
    tc = time()-t0
    print('time:',tc)
    print('\n')

    new_idx = indices[idx]
    galaxyData = data[new_idx]

    if len(galaxyData)<1:
        print('Critical Error')
        exit()

    cid = clusters['CID'][cidx]
    z,zerr = z[new_idx], zerr[new_idx]
    pdfz = np.zeros_like(z,dtype=float) #getPDFzM(z,zerr,cid,clusters)

    return galaxyData, cid, radii, pdfz

def getVariables(galaxyData,colNames,zsigma=0.05):

    gid = np.array(galaxyData[colNames[0]][:])

    ra = convertRA(np.array(galaxyData[colNames[1]]))
    dec = np.array(galaxyData[colNames[2]])
    
    z = np.array(galaxyData[colNames[3]])
    zerr = zsigma*(1+z)#np.array(galaxyData[colNames[10]])
    z = z - np.random.normal(scale=zsigma*(1+z),size=len(z))
    z= np.where(z<0.,0.001,z)

    mag, magerr = [], []
    for i in range(4):
        mag.append(galaxyData[colNames[4+i]])
        magerr.append(galaxyData[colNames[10+i]])
    
    mag = np.array(mag).transpose() ## 4 vector
    magerr = np.array(magerr).transpose() ## 4 vector

    return gid,ra,dec,z,zerr,mag,magerr

def readGalaxyCat(galaxyInFile, clusters, rmax=3,r_in=8,r_out=10, radius=12, window=0.1, zrange=(0.01,1.),Nflag=0,colNames=None,simulation=False):
    logging.debug('Starting helper.readGalaxyCat()')
    
    if colNames is None:
        print('Please provide the Galaxy column names - exiting the code')
        exit()

    if simulation:
        colNames.append('HALOID')
        colNames.append('TRUE_MEMBERS')
    
    ## divide to conquer! (take squares around each cluster position)
    ## radius is the length of the square in Mpc
    galaxyData, cid, radii, PDFz = importTable(galaxyInFile,clusters,colNames,rmax=rmax,r_in=r_in,r_out=r_out,
                                    zrange=zrange,radius=radius,window=window)

    gid,ra,dec,z,zerr,mag,magerr = getVariables(galaxyData,colNames,zsigma=window)

    bkgGalaxies = (radii>r_in)&(radii<r_out)
    
    inputdata = [cid, gid, ra, dec, radii, z, mag, zerr, magerr, PDFz, bkgGalaxies]
    Cnames = ['CID', 'GID', 'RA', 'DEC', 'R', 'z', 'mag','zerr','magerr', 'PDFz', 'Bkg']

    if simulation:
        true_members = cid<0
        w, = np.where((galaxyData['HALOID']==cid)&(galaxyData['TRUE_MEMBERS']==True))
        true_members[w] = True

        Cnames.append('z_true')
        Cnames.append('True')
        inputdata.append(galaxyData['Z'][:])
        inputdata.append(true_members)

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