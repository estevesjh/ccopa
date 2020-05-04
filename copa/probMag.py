# !/usr/bin/env python
# color-magnitude subtraction algorithm

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from time import time
from scipy.interpolate import interp1d
from scipy import integrate

from sklearn.neighbors import KernelDensity
from astropy.table import Table, vstack
import dask

import scipy.stats as st

## local libraries
import gaussianKDE as kde
from probRadial import doPr, scaleBkg

from six import string_types

def interpData(x,y,x_new):
    out = np.empty(x_new.shape, dtype=y.dtype)
    out = interp1d(x, y, kind='linear', fill_value='extrapolate', copy=False)(x_new)
    # yint = interp1d(x,y,kind='linear',fill_value='extrapolate')
    return out

gaussian = lambda x,mu,sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def kde_sklearn(x, x_assign, bandwidth=None, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    # from sklearn.grid_search import GridSearchCV
    if bandwidth is None:
        grid = GridSearchCV(KernelDensity(rtol=1e-4,kernel='gaussian'),
                        {'bandwidth': np.linspace(0.0005, 0.05, 30)},
                        cv=15) # 20-fold cross-validation
        grid.fit(x[:, np.newaxis])
        print(grid.best_params_)
        kde = grid.best_estimator_
        bandwidth = float(grid.best_params_['bandwidth'])
    else:
        kde = KernelDensity(bandwidth=bandwidth, rtol=1e-4)
        kde.fit(x[:, np.newaxis])

    log_pdf = kde.score_samples(x_assign[:, np.newaxis])
    
    return np.exp(log_pdf), bandwidth

def getSilvermanBandwidth(x,weight=None):
    size = len(x)
    neff = 1.0
    
    if weight is not None:
        neff = neff/ np.sum(weight ** 2)

    bw_silverman = np.power(neff*(size+2.0)/4.0, -1./(size+4))
    
    return bw_silverman
    
def computeColorMagnitudeKDE(x,y,bandwidth='silverman'):
    """input: x (magnitude) and y (color)
       return: PDF (probability distribution function)
    """
    values = np.vstack([x, y])

    if np.isscalar(bandwidth) and not isinstance(bandwidth, string_types):
        bandwidth = bandwidth/np.std(values,ddof=1)

    kernel = st.gaussian_kde(values,bw_method=bandwidth)

    return kernel

def computeColorKDE(x,weight=None,bandwidth='silverman',silvermanFraction=None):
    """input: x (magnitude) and y (color)
       return: PDF (probability distribution function)
    """
    if np.isscalar(bandwidth) and not isinstance(bandwidth, string_types):
        bandwidth = bandwidth/np.std(x,ddof=1)

    if silvermanFraction is not None:
        kernel = kde.gaussian_kde(x,weights=weight,silvermanFraction=silvermanFraction)
    else:
        kernel = kde.gaussian_kde(x,weights=weight,bw_method=bandwidth)

    return kernel

def monteCarloSubtraction(p_field):
    """input: field probability
       return: indices of the non field galaxies
    """
    nsize = len(p_field)
    idx_subtracted = np.empty(0,dtype=int)

    for i in range(nsize):
        n_rand = np.random.uniform(0,1)
        pi = p_field[i]

        if n_rand > pi:
            idx_subtracted = np.append(idx_subtracted,i)

    return idx_subtracted

def backgroundSubtraction(mag,mag_bkg,mag_vec,color,color_bkg,ncls,nbkg,weight=[None,None]):
    probz, probz_bkg = weight

    ## compute kde  
    # kernel = computeColorMagnitudeKDE(mag[w],color[w],bandwidth=bandwidth)
    # kernel_bkg = computeColorMagnitudeKDE(mag_bkg[w2],color_bkg[w2],bandwidth=bandwidth)

    kernel = computeColorKDE(mag,weight=probz,silvermanFraction=2)
    kernel_bkg = computeColorKDE(mag_bkg,weight=probz_bkg,silvermanFraction=2)

    nc = (ncls-nbkg)
    if nc<0: nbkg = ncls = nc = 1

    values = mag_vec
    kde = kernel(values)
    kde_bkg = kernel_bkg(values)

    Ncls_field = ncls*kde
    Nfield = nbkg*kde_bkg
    
    kde_sub = (Ncls_field-Nfield)/nc
    kde_sub = np.where(kde_sub<0,0.,kde_sub) ## we take only the excess
    kde_sub = kde_sub/integrate.trapz(kde_sub,x=values) ## set to unity
    
    # kde_sub = np.where(kde_sub<1e-4,0.,kde_sub)
    return kde_sub, kde, kde_bkg

def doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[None,None],bandwidth=0.05,choose_color=[0,1]):
    i,i2 = choose_color

    color = (mag[:,i]-mag[:,i2])
    color2 = (mag2[:,i]-mag2[:,i2])
    color_bkg = (mag_bkg[:,i]-mag_bkg[:,i2])

    if (color.size > 3)&(color_bkg.size >3):
        k1_i, k1_cf_i, k1_i_bkg = backgroundSubtraction(mag[:,i2],mag_bkg[:,i2],color_vec,color,color_bkg,n_cls_field,nb,weight=weight)
        pdf_i = interpData(color_vec,k1_i,color2)
        pdf_i_bkg = interpData(color_vec,k1_i_bkg,color2)

    else:
        pdf_i, pdf_i_bkg = np.full(color2.size,1),np.full(color2.size,1)
        k1_i, k1_cf_i, k1_i_bkg = np.full(color_vec.size,1),np.full(color_vec.size,1),np.full(color_vec.size,1)
        
    return k1_i, k1_cf_i, k1_i_bkg, pdf_i, pdf_i_bkg

def computeMagPDF(gals,cat,r200,nbkg,keys,mag_vec):
    ''' compute probability distribution function for the m_i-m*
    '''
    from scipy import integrate
    ncls = len(cat)

    galIDX = []

    kernels, kernels_cf, kernels_field = [], [], []
    for idx,_ in enumerate(keys):
        # cls_id,z_cls = cat['CID'][idx], cat['redshift'][idx]
        # r2, nb = r200[idx], nbkg[idx]
        # # n_cls_field, nb = ngals[idx], nbkg[idx]

        # galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id) & (gals["R"]<=gals['r_aper'])) 
        # galaxies2, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)) 

        # galIDX.append(galaxies)
        # bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        # mag, mag_bkg = gals['dmag'][galaxies], gals['dmag'][bkgGalaxies]
        # mag2 = gals['dmag'][galaxies2]
        # radii = gals['R'][galaxies]

        # color, color_bkg = gals['color'][galaxies,2], gals['color'][bkgGalaxies,2]

        # probz = np.array(gals['PDFz'][galaxies])
        # probz_bkg = np.array(gals['PDFz'][bkgGalaxies])

        # n_cls_field = np.sum(probz)/(np.pi*r2**2)

        # Pr = doPr(radii,r2,n_cls_field,nb)
        # nb *= scaleBkg(r2,n_cls_field,nb,r_in=4.,r_out=6)
        # probz *= Pr

        # if (len(mag)>1)&(len(mag_bkg)>1):
        #     k,k_cf,k_bkg = backgroundSubtraction(mag,mag_bkg,mag_vec,color,color_bkg,n_cls_field,nb,weight=[probz,probz_bkg])
        # else:
        k = k_cf = k_bkg = np.ones_like(mag_vec)

        kernels.append(k)
        kernels_cf.append(k_cf)
        kernels_field.append(k_bkg)

    pdfm_list = [kernels, kernels_cf, kernels_field]
    return pdfm_list