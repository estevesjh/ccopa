# !/usr/bin/env python
import numpy as np
from astropy.table import Table, vstack
from astropy.io.fits import getdata
from scipy import ndimage

from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt
import healpy as hp
from scipy import stats

##
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, PowerNorm

## local libray
from projection_radec_to_xy import xy_to_radec,radec_to_xy,delta_radec


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

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def check_background_plot(all_gal,galMask,bkgMask,indices_ring):
    ## compute dx
    gal     = all_gal[galMask]
    gal_bkg = all_gal[bkgMask]
    gal_bkg_bad = all_gal[indices_ring]

    cid   = np.unique(all_gal['CID'])
    fname = '%i_ring_bkg'%(cid)

    plotRing(gal,gal_bkg,gal_bkg_bad,fname)
    pass

def plotRing(gal,gal_bkg,gal_bkg_bad,fname):
    plt.clf()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1,right=0.95,top=0.975,bottom=0.025)
    ax.set_aspect('equal')

    ax.scatter(gal['dx'],gal['dy'],color='r',s=2)
    ax.scatter(gal_bkg_bad['dx'],gal_bkg_bad['dy'],color='lightgray',s=2)
    ax.scatter(gal_bkg['dx'],gal_bkg['dy'],color='k',s=2)
    ax.set_xlabel(r'$\Delta X $ [Mpc]')
    ax.set_ylabel(r'$\Delta Y$ [Mpc]')
    #ax.suptitle(fname)
    plt.savefig(fname)

## -------------------------------
## auxiliary functions
def remove_iqr(values):
    q75,q25 = np.percentile(values,[75,25])
    intr_qr = q75-q25

    nmax = q75+(1.5*intr_qr)
    nmin = q25-(1.5*intr_qr)
    if nmin<0: nmin = 0.

    idx,  = np.where( (values < nmax) & (values > nmin) )
    return idx

def commonValues(values):
	idx_sort = np.argsort(values)
	sorted_values = values[idx_sort]
	vals, idx_start, count = np.unique(sorted_values, return_counts=True,
                                return_index=True)

	# sets of indices
	res = np.split(idx_sort, idx_start[1:])
	#filter them with respect to their size, keeping only items occurring more than once
	vals = vals[count > 1]
	commonValuesIndicies = [ri for ri in res if ri.size>1]
	
	return commonValuesIndicies, vals

def radec_pix(ra,dec,nside=1024,nest=True):
    return np.array(hp.ang2pix(nside,np.radians(90-dec),np.radians(ra),nest=True),dtype=np.int64)

# def check_depth(ra,dec):
#     n0 = 2048
#     nsides = [n0,int(2*n0),int(2*n0),int(2*n0),int(2*n0)]
#     for nside in nsides:
#         pixels    = radec_pix(ra,dec,nside=nside,nest=True)
#         counts    = np.unique(pixels)
#         if np.std(counts)/np.mean(counts)>0.1:
#             return nside


def get_pizza_slice(weights,theta,nslices=120,eps=1e-9):
    res = np.empty(0,dtype=float)
    for ni in range(nslices):
        w, = np.where((theta <= (ni+1)*(360/nslices))&(theta >= (ni)*(360/nslices)))
        Nbkg = np.log10(np.nansum(weights[w])+eps)
        res = np.append(res,Nbkg)
    res = np.where(res<-7.,-7.,res)
    return nslices*10**res

def remove_outliers(lista,no_cuts=False,ncut=2.):
    cond = lista>-3
    if (np.count_nonzero(cond)<3):
        return 10**-7, np.empty((0,),dtype=np.int)
    if no_cuts:
        return np.median(lista[cond]),np.where(cond)[0]
    
    low,high = np.percentile(lista[cond],[25,75])
    iqr = (high-low)
    
    nl, nh = low-1.5*iqr,high+ncut*iqr
    
    sectors0, = np.where((lista>nl)&(lista<nh))
    lista2 = lista[sectors0]
    
    nbkg0 = np.median(lista[cond])
    #nbkg  = np.median(lista2)
    print('nl and nh: %.2f , %.2f'%(10**nl,10**nh))
    return 10**nbkg0,sectors0


def calcTheta(ra,dec,ra_c,dec_c):
    dra,ddec,albers = delta_radec(ra,dec,ra_c,dec_c)    ## albers projection (delta RA, delta DEC)
    theta = np.degrees( np.arctan2(dra,ddec) ) + 180
    return theta

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def computeDensityBkg(gal,cat,r_in=6,r_aper=1,r_out=8,nslices=72):
    ''' It computes the background galaxy density in a ring (r_in,r_out). 
        In order to avoid over and under dense regions it cuts the ring in a given number of slices (nslices)
        and computes the galaxy density in each slice (nbkg_i). It discards 3sigma around the median.
        The output is the median of the distribution.

    returns: nbkg, maskBKG
    
    maskBKG is a boolean array. It is true for the galaxies inside the ring slices that were not discarded.
    '''
    ncls  = len(cat)
    cidx  = cat['CID']
    gidx  = gal['CID']
    bkg   = gal['Bkg']
    pz    = gal['pz0']
    theta = gal['theta']

    area_bkg = np.pi*( (r_out)**2 - (r_in)**2 )
    area     = np.pi*r_aper**2

    keys = list(chunks(gidx,cidx))
    bkgKeys = [idx[bkg[idx]] for idx in keys]

    ## compute n bkg galaxies per slices
    sliceList  = [get_pizza_slice(pz[idx],theta[idx],nslices=nslices) for idx in bkgKeys]
    out        = [remove_outliers(np.log10(slices)) for slices in sliceList]

    nbkg    = np.array([out[i][0] for i in range(ncls)])/area_bkg
    indices = [idx[get_sectors_indices(theta[idx],out[i][1])] for i,idx in enumerate(bkgKeys)]
    
    maskBkg = np.full(len(bkg), False, dtype=bool)
    for idx in indices:
        maskBkg[idx] = True
    
    ## check nbkg<ngal_core
    cut     = gal['R']<=r_aper
    galKeys = [idx[cut[idx]] for idx in keys]
    ngal    = np.array([np.nansum(pz[idx]) for idx in galKeys])/area

    w, = np.where(nbkg>ngal)
    if w.size>0:
        print('Error: %i clusters with nbkg > ngal'%(w.size))
        nbkg[w] = np.array([remove_outliers(np.log10(slices),ncut=0.25)[0] for slices in np.array(sliceList)[w]])/area_bkg
    
    return nbkg, maskBkg

def get_sectors_indices(theta,sectors,nslices=120):
    indices = np.empty(0,dtype=int)
    if sectors.size<1:
        return np.empty((0,),dtype=np.int)

    for ni in sectors:
        w, = np.where((theta <= (ni+1)*(360/nslices)) & (theta >= (ni)*(360/nslices)))
        indices = np.append(indices,w)
    return indices

def computeGalaxyDensity(gals, cat, rmax, nbkg,nslices=72):
    galsFlag = np.full(len(gals['Bkg']), False, dtype=bool)
    ngals, keys, galIndices = [],[], []
    
    count0 = 0
    for idx in range(len(cat)):
        cls_id = cat['CID'][idx]
        ra_c, dec_c = cat['RA'][idx], cat['DEC'][idx]
        # magLim_i = cat['magLim'][idx,1] ### mi cut
        
        indices, = np.where((gals['CID']==cls_id)&(gals['R']<=rmax[idx]))

        pz = gals['pz0'][indices]
        theta = gals['theta'][indices]#calcTheta(gals['RA'][indices],gals['DEC'][indices],ra_c,dec_c)
        area = np.pi*rmax[idx]**2

        if len(indices)>0:
            print('# effec. number of bkg galaxies:',len(indices))
            galMask = np.full(len(pz), True, dtype=bool)
            new_idx = np.arange(count0,count0+len(indices),1,dtype=int)
            ng, _ = calcNbkg(pz,theta,galMask, nslices=nslices,n_high=2.)
            # ng = np.sum(pz)
            count0 += len(indices)
        
            keys.append(new_idx)
            ngals.append(np.abs(ng)/area)
        else:
            res = -1.
            ngals.append(res)
            indices = np.array([np.nan])

        galIndices.append(indices)
    
        return np.array(ngals), galsFlag, keys, galIndices


if __name__ == '__main__':
    print('background.py')
    print('author: Johnny H. Esteves')
