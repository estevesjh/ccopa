# !/usr/bin/env python
import numpy as np
from astropy.table import Table, vstack
from astropy.io.fits import getdata
from scipy import ndimage

from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt

##
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, PowerNorm

def plotDensityMap(image,x,y,radius,pmem=0.5,title='Density Map',savename='./density_map.png'):
    """It generates density maps
    """
    # radius = (np.max(x)+np.max(y))/21
    ticks=[-radius,-radius/2,0,radius/2,radius]
    # extent=(x.min(),x.max(),y.min(),y.max())
    extent=[-radius,radius,-radius,radius]
    plt.clf()
    plt.title(title)
    plt.xlabel(r'$\Delta X $ [Mpc]')
    plt.ylabel(r'$\Delta Y $ [Mpc]')
    # plt.scatter(x,-y,c='white',s=10*(pmem)**2,alpha=0.7)
    # plt.scatter(x,y,c='white',s=10*(pmem)**2,alpha=0.7)
    plt.imshow(image,extent=extent,origin='lower',cmap=cm.inferno,norm=PowerNorm(gamma=1/2),interpolation='nearest')
    # plt.imshow(image,extent=extent,cmap=cm.inferno)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.colorbar()
    plt.savefig(savename)

## -------------------------------
## auxiliary functions

def calcNbkg(pz,theta,bkgMask,nslices=72,n_high=2.,method='pdf'):
    pz_bkg = pz[bkgMask]
    theta_bkg = theta[bkgMask]

    lista = np.empty(0,dtype=float)
    for ni in range(nslices):
        w, = np.where((theta_bkg <= (ni+1)*(360/nslices))&(theta_bkg >= (ni)*(360/nslices)))
        Nbkg = np.sum(pz_bkg[w])
        if method=='counts':
            Nbkg = len(pz_bkg[w])
        
        lista = np.append(lista,Nbkg)

    nbkg0 = np.mean(lista[lista>0.5])
    mad = np.std(lista[lista>0.5])

    # print(lista)
    nl, nh = nbkg0-2*n_high*mad,nbkg0+n_high*mad
    sectors0, = np.where((lista>nl)&(lista<nh))
    lista2 = lista[sectors0]

    ### Second Turn
    nbkg = np.mean(lista2)
    mad = np.std(lista2)

    # ## Deal with projection effects
    # nl, nh = nbkg-3*mad,nbkg+(0.5*n_high)*mad
    # sectors, = np.where(lista2<nh)
    
    idx_gal = np.empty(0,dtype=int)
    for ni in sectors0:
        w, = np.where( (theta <= (ni+1)*(360/nslices)) & (theta >= (ni)*(360/nslices)) & bkgMask )
        idx_gal = np.append(idx_gal,w)
    
    #### some arbritary conditions
    # if idx_gal.size<400:
    #     idx_gal = np.empty(0,dtype=int)
    #     for ni in sectors0:
    #         w, = np.where( (theta <= (ni+1)*(360/nslices)) & (theta >= (ni)*(360/nslices)) & bkgMask )
    #         idx_gal = np.append(idx_gal,w)
    
    # Nbkg = (nbkg*nslices) ## mean number of galaxies in the region
    Nbkg = np.sum(pz_bkg)

    return Nbkg, idx_gal

def getDensityBkg(all_gal,theta,r_aper,r_in=6,r_out=8,nslices=72,method='pdf'):
    ## get bkg
    galMask = (all_gal['R']<=r_aper)
    bkgMask = (all_gal['R']>=r_in)&(all_gal['R']<=r_out)
    
    pz = all_gal['PDFz']
    area_bkg = np.pi*( (r_out)**2 - (r_in)**2 )
    area = (np.pi*r_aper**2)

    nbkg, idx_gal = calcNbkg(pz,theta,bkgMask,nslices=nslices,n_high=2.,method=method)
    ngal, _ = calcNbkg(pz,theta,galMask,nslices=nslices,n_high=5.,method=method)

    nbkg, ngal = (nbkg/area_bkg), (ngal/area)

    ## check if the nbkg is greater than the galaxy cluster density core!
    if (nbkg>ngal) | (nbkg>100):
        print('Error: nbkg > ngal or nbkg>100 gals/Mpc2')
        nbkg, idx_gal = calcNbkg(pz,theta,bkgMask,nslices=nslices+24,n_high=3.)
        nbkg = (nbkg/area_bkg)

    if np.count_nonzero(galMask) < 10: ## at least 10 galaxies
        nbkg = -1

    return nbkg, idx_gal

def calcTheta(ra,dec,ra_c,dec_c):
    deltaX, deltaY = (ra-ra_c),(dec-dec_c)
    theta = np.degrees( np.arctan2(deltaY,deltaX) ) + 180
    return theta

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def computeDensityBkg(gals,cat,r_in=6,r_out=8,r_aper=1.,nslices=72,method='pdf'):
    ''' It computes the background galaxy density in a ring (r_in,r_out). 
        In order to avoid over and under dense regions it cuts the ring in a given number of slices (nslices)
        and computes the galaxy density in each slice (nbkg_i). It discards 3sigma around the median.
        The output is the median of the distribution.

    returns: nbkg, nbkg_magnitude_limited, maskBKG
    
    maskBKG is a boolean array. It is true for the galaxies inside the ring slices that were not discarded.
    '''
    ncls = len(cat)
    nbkg, nbkg_malLimited = [], []
    maskBkg = np.full(len(gals['Bkg']), False, dtype=bool)

    count = 0
    for idx in range(ncls):
        cls_id = cat['CID'][idx]
        ra_c, dec_c = cat['RA'][idx], cat['DEC'][idx]
        # magLim_r = cat['magLim'][idx,0] ### mr cut
        
        galIndices, = np.where(gals['CID']==cls_id)
        galIndicesL, = np.where((gals['CID']==cls_id)&(gals['dmag']<=0.))
        # galIndicesL, = np.where((gals['CID']==cls_id)&(gals['mag'][:,1]<magLim_r))
        # galIndicesL, = np.where((gals['CID']==cls_id)&(gals['amag'][:,1]<=-20.5))
       
        gal = gals[galIndices]
        gal_magLim = gals[galIndicesL]
        
        theta = calcTheta(gal['RA'],gal['DEC'],ra_c,dec_c)
        thetaL = calcTheta(gal_magLim['RA'],gal_magLim['DEC'],ra_c,dec_c)

        nbkg_i, _ = getDensityBkg(gal,theta,r_aper,r_in=r_in,r_out=r_out,nslices=nslices,method=method)
        nbkg_j, bkgIndices = getDensityBkg(gal_magLim,thetaL,r_aper,r_in=r_in,r_out=r_out,nslices=nslices,method=method)

        ## Updating the background status
        maskBkg[galIndicesL[bkgIndices]] = True

        if (nbkg_i<0.1) or (bkgIndices.size<20):
            print('nbkgi:',nbkg_i)
            nbkg_i = nbkg_j = -1
            count+=1

        nbkg.append(nbkg_i)
        nbkg_malLimited.append(nbkg_j)

    if count>0:
        print('Critical Error: there is %i galaxy clusters with not enough background galaxies'%(count))

    return np.array(nbkg), np.array(nbkg_malLimited), maskBkg

def computeGalaxyDensity(gals, cat, rmax, nbkg,nslices=72):
    galsFlag = np.full(len(gals['Bkg']), False, dtype=bool)
    ngals, keys, galIndices = [],[], []
    
    count0 = 0
    for idx in range(len(cat)):
        cls_id = cat['CID'][idx]
        ra_c, dec_c = cat['RA'][idx], cat['DEC'][idx]
        # magLim_i = cat['magLim'][idx,1] ### mi cut
        
        indices, = np.where((gals['CID']==cls_id)&(gals['R']<=rmax[idx])&(gals['dmag']<=0.))
        # indices, = np.where((gals['CID']==cls_id)&(gals['R']<=rmax[idx])&(gals['mag'][:,2]<magLim_i))
        # indices, = np.where((gals['CID']==cls_id)&(gals['R']<=4)&(gals['mag'][:,2]<magLim_i))
        
        pz = gals['PDFz'][indices]
        theta = calcTheta(gals['RA'][indices],gals['DEC'][indices],ra_c,dec_c)
        area = np.pi*rmax[idx]**2

        if len(indices)>0:
            galMask = np.full(len(pz), True, dtype=bool)
            new_idx = np.arange(count0,count0+len(indices),1,dtype=int)
            ng, _ = calcNbkg(pz,theta,galMask, nslices=nslices,n_high=2.)
            # ng = np.sum(pz)

            keys.append(new_idx)
            ngals.append(ng/area)
            galIndices.append(indices)

            count0 += len(indices)
    
    return np.array(ngals), galsFlag, keys, galIndices


if __name__ == '__main__':
    print('background.py')
    print('author: Johnny H. Esteves')
