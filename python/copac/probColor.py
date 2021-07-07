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

def scaleSTD(x,weights=None):
    xmean = np.average(x,weights=weights)
    xstd  = (np.average((x-xmean)**2, weights=weights))**(1/2)
    return (x-xmean)/(xstd), xmean, xstd

def backgroundSubtraction(mag,mag_bkg,color_vec,color,color_bkg,ncls,nbkg,weight=[None,None],bandwidth=0.05,sampling=False):
    probz, probz_bkg = weight

    ## compute kde  
    # kernel = computeColorMagnitudeKDE(mag[w],color[w],bandwidth=bandwidth)
    # kernel_bkg = computeColorMagnitudeKDE(mag_bkg[w2],color_bkg[w2],bandwidth=bandwidth)

    ## scaling the distribution
    #color, u, s = scaleSTD(color,weights=probz)
    #color_bkg = (color_bkg-u)/s
    values = color_vec#(color_vec-u)/s

    try:
        kernel = computeColorKDE(color,weight=probz,silvermanFraction=5)
        kernel_bkg = computeColorKDE(color_bkg,weight=probz_bkg,silvermanFraction=5)
    except:
        print('Color Error')
        kernel = computeColorKDE(color,silvermanFraction=5)
        kernel_bkg = computeColorKDE(color_bkg,silvermanFraction=5)

    nc = (ncls-nbkg)
    if nc<0: nbkg = ncls = nc = 1
    
    kde = kernel(values)
    kde_bkg = kernel_bkg(values)

    if not sampling:
        Ncls_field = ncls*kde
        Nfield = nbkg*kde_bkg
        
        kde_sub = (Ncls_field-Nfield)/nc
        kde_sub = np.where(kde_sub<0,0.,kde_sub) ## we take only the excess
        kde_sub = kde_sub/integrate.simps(kde_sub,x=values) ## set to unity

    else:
        nsample = 100
        value_sample = np.random.choice(values, nsample, p=kde/np.sum(kde))

        Ncls_field = ncls*kernel(value_sample)
        Nfield = nbkg*kernel_bkg(value_sample)
        
        Pfield = Nfield/(Ncls_field+1e-9)
        Pfield = np.where(Ncls_field<1e-3,1.1,Pfield)

        # # ## subtract field galaxies
        idx = monteCarloSubtraction(Pfield)
        
        if len(idx)>3:
            # kernel = computeColorKDE(color[idx],weight=probz[idx],silvermanFraction=10.)
            # kernel_sub = computeColorMagnitudeKDE(mag_sample[idx],color_sample[idx],silvermanFraction=10)
            kernel_sub = computeColorKDE(value_sample[idx],silvermanFraction=10)
            kde_sub = kernel_sub(values)
            
        else:
            print('Color Error: not enough galaxies')
            kde_sub = (kde-kde_bkg)
            kde_sub = np.where(kde_sub<0,0.,kde_sub) ## we take only the excess
            kde_sub = kde_sub/integrate.simps(kde_sub,x=values)
            # kde_bkg = np.zeros_like(mag)
    
    # kde_sub = np.where(kde_sub<1e-4,0.,kde_sub)

    cbw = kernel.silverman_factor()/10

    return kde_sub, kde, kde_bkg

def doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[None,None],bandwidth=0.05,choose_color=[0,1]):
    i,i2 = choose_color

    color = (mag[:,i]-mag[:,i2])
    color2 = (mag2[:,i]-mag2[:,i2])
    color_bkg = (mag_bkg[:,i]-mag_bkg[:,i2])

    if (color.size > 4)&(color_bkg.size > 4):
        k1_i, k1_cf_i, k1_i_bkg = backgroundSubtraction(mag[:,i2],mag_bkg[:,i2],color_vec,color,color_bkg,n_cls_field,nb,weight=weight,bandwidth=10)
        pdf_i = interpData(color_vec,k1_i,color2)
        pdf_i_bkg = interpData(color_vec,k1_i_bkg,color2)

    else:
        pdf_i, pdf_i_bkg = np.full(color2.size,1),np.full(color2.size,1)
        k1_i, k1_cf_i, k1_i_bkg = np.full(color_vec.size,1),np.full(color_vec.size,1),np.full(color_vec.size,1)
        
    return k1_i, k1_cf_i, k1_i_bkg, pdf_i, pdf_i_bkg

def computeColorPDF(gals,cat,r200,nbkg,keys,color_vec,bandwidth=[0.008,0.001,0.001],parallel=True):
    ''' compute probability distribution function for the 5 sequential colors 
        0:(g-r); 1:(g-i); 2:(r-i); 3:(r-z); 4:(i-z)
    '''
    from scipy import integrate
    ncls = len(cat)

    # pdf = np.empty((1,5),dtype=float)
    # pdf_bkg = np.empty((1,5),dtype=float)
    # normalization_factor = np.empty((0),dtype=float)

    results0, results1, results2, results3, results4 = [], [], [], [], []
    galIDX = []

    kernels, kernels_cf, kernels_field = [], [], []
    for idx in range(len(cat)):
        cls_id,z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]
        # n_cls_field, nb = ngals[idx], nbkg[idx]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id) & (gals["R"]<=gals['r_aper'])) 
        # galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id) & (gals["R"]<=1.)) 
        galaxies2, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)) 

        galIDX.append(galaxies)
        
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        mag, mag_bkg = gals['mag'][galaxies], gals['mag'][bkgGalaxies]
        mag2 = gals['mag'][galaxies2]
        radii = gals['R'][galaxies]

        probz = np.array(gals['pz0'][galaxies])
        probz_bkg = np.array(gals['pz0'][bkgGalaxies])

        n_cls_field = np.sum(probz)/(np.pi*r2**2)

        # Pr = doPr(radii,r2,n_cls_field,nb)
        # nb *= scaleBkg(r2,n_cls_field,nb,r_in=4.,r_out=6)
        # probz *= Pr

        if not parallel:  
            t0 = time()
            ## 5 Color distributions
            kernel_0, kernel_cf_0, kernel_field_0, pdf_0, pdf_bkg_0 = doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,1],bandwidth=bandwidth[0]) ##(g-r)
            kernel_1, kernel_cf_1, kernel_field_1, pdf_1, pdf_bkg_1 = doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,2],bandwidth=bandwidth[0]) ##(g-i)
            kernel_2, kernel_cf_2, kernel_field_2, pdf_2, pdf_bkg_2 = doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,2],bandwidth=bandwidth[0]) ##(r-i)
            kernel_3, kernel_cf_3, kernel_field_3, pdf_3, pdf_bkg_3 = doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,3],bandwidth=bandwidth[0]) ##(r-z)
            kernel_4, kernel_cf_4, kernel_field_4, pdf_4, pdf_bkg_4 = doKDEColors(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[2,3],bandwidth=bandwidth[0]) ##(i-z)            
            print('%i - Color prob. time:'%cls_id,time()-t0)

            # c5 = np.array([pdf_0,pdf_1,pdf_2,pdf_3,pdf_4]).transpose()
            # c5_bkg = np.array([pdf_bkg_0,pdf_bkg_1,pdf_bkg_2,pdf_bkg_3,pdf_bkg_4]).transpose()

            k5 = np.array([kernel_0,kernel_1,kernel_2,kernel_3,kernel_4]).transpose()
            k5_cf = np.array([kernel_cf_0,kernel_cf_1,kernel_cf_2,kernel_cf_3,kernel_cf_4]).transpose()
            k5_bkg = np.array([kernel_field_0,kernel_field_1,kernel_field_2,kernel_field_3,kernel_field_4]).transpose()
           
            # norm_factor = integrate.trapz(kernel_0*kernel_1*kernel_2*kernel_3*kernel_4,x=color_vec)

            # pdf = np.vstack([pdf,c5])
            # pdf_bkg = np.vstack([pdf_bkg,c5_bkg])
            # normalization_factor = np.append(normalization_factor,norm_factor*np.ones_like(pdf_0))

            kernels.append(k5)
            kernels_cf.append(k5_cf)
            kernels_field.append(k5_bkg)

        else:
            t0 = time()
            ## 5 Color distributions
            y0 = dask.delayed(doKDEColors)(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,1],bandwidth=bandwidth[0])##(g-r)
            y1 = dask.delayed(doKDEColors)(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,2],bandwidth=bandwidth[0])##(g-i)
            y2 = dask.delayed(doKDEColors)(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,2],bandwidth=bandwidth[1])##(r-i)
            y3 = dask.delayed(doKDEColors)(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,3],bandwidth=bandwidth[1])##(r-z)
            y4 = dask.delayed(doKDEColors)(color_vec,mag2,mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[2,3],bandwidth=bandwidth[2])##(i-z)
            
            results0.append(y0)
            results1.append(y1)
            results2.append(y2)
            results3.append(y3)
            results4.append(y4)

    if parallel:        
        results0 = dask.compute(*results0, scheduler='processes', num_workers=2)
        results1 = dask.compute(*results1, scheduler='processes', num_workers=2)
        results2 = dask.compute(*results2, scheduler='processes', num_workers=2)
        results3 = dask.compute(*results3, scheduler='processes', num_workers=2)
        results4 = dask.compute(*results4, scheduler='processes', num_workers=2)

        for i in range(len(galIDX)):
            galaxies = galIDX[i]

            kernel_0, kernel_cf_0, kernel_field_0, pdf_0 ,pdf_bkg_0 = results0[i]
            kernel_1, kernel_cf_1, kernel_field_1, pdf_1 ,pdf_bkg_1 = results1[i]
            kernel_2, kernel_cf_2, kernel_field_2, pdf_2 ,pdf_bkg_2 = results2[i]
            kernel_3, kernel_cf_3, kernel_field_3, pdf_3 ,pdf_bkg_3 = results3[i]
            kernel_4, kernel_cf_4, kernel_field_4, pdf_4 ,pdf_bkg_4 = results4[i]

            # c5 = np.array([pdf_0,pdf_1,pdf_2,pdf_3,pdf_4]).transpose()
            # c5_bkg = np.array([pdf_bkg_0,pdf_bkg_1,pdf_bkg_2,pdf_bkg_3,pdf_bkg_4]).transpose()

            k5 = np.c_[kernel_0,kernel_1,kernel_2,kernel_3,kernel_4]#.transpose()
            k5_cf = np.c_[kernel_cf_0,kernel_cf_1,kernel_cf_2,kernel_cf_3,kernel_cf_4]#.transpose()
            k5_bkg = np.c_[kernel_field_0,kernel_field_1,kernel_field_2,kernel_field_3,kernel_field_4]#.transpose()

            # norm_factor = integrate.trapz(kernel_0*kernel_1*kernel_2*kernel_3*kernel_4,x=color_vec)

            # pdf = np.vstack([pdf,c5])
            # pdf_bkg = np.vstack([pdf_bkg,c5_bkg])
            # normalization_factor = np.append(normalization_factor,norm_factor*np.ones_like(pdf_0))

            kernels.append(k5)
            kernels_cf.append(k5_cf)
            kernels_field.append(k5_bkg)

        # print('Parallel - Color time:',round(time()-t0,2))

    pdfc_list = [kernels, kernels_cf, kernels_field]

    return pdfc_list
    
#############################################################################
### plot

def plotDoubleColor(color, pdf_all, pdf_bkg, pdf, title, nbkg, ncls_field, name_cls="Cluster", lcolor='g-r', pdf_true=None):
    Ncls_field = pdf_all*ncls_field
    N_bkg = pdf_bkg*nbkg
    N_sub = pdf*(ncls_field-nbkg)

    plt.clf()
    fig, axs = plt.subplots(1, 2, sharey=True,sharex=True, figsize=(8,6))
    fig.subplots_adjust(left=0.075,right=0.95,bottom=0.15,wspace=0.075)
    fig.suptitle(title)
    # fig.tight_layout()

    axs[0].scatter(color,Ncls_field,color='blue',linestyle='--',label=r'Cluster+Field')
    axs[0].scatter(color,N_bkg,color='r',linestyle='--',label=r'Field')
    axs[0].set_title('Cluster+Field')
    axs[0].legend(loc='upper right')

    axs[1].scatter(color,N_sub,color='blue',label=r'Cluster Model')
    if pdf_true is not None:
        axs[1].scatter(color,(ncls_field-nbkg)*pdf_true,color='gray',label=r'True members')

    axs[1].set_title('Cluster')

    # axs[0].set_xlim(-0.2,0.2)

    axs[0].set_ylabel(r'$ N $')
    axs[0].set_xlabel(lcolor)
    axs[1].set_xlabel(lcolor)
        
    axs[1].legend(loc='upper right')
    
    plt.savefig(name_cls+'_colorDistribution.png', bbox_inches = "tight")


def plotTrioColorMagDiagram(mag,color,mag_bkg,color_bkg,idx,name_cls='cluster'):
    plt.clf()
    fig, axs = plt.subplots(1, 3, sharey=True,sharex=True, figsize=(12,4))
    fig.subplots_adjust(left=0.075,right=0.95,top=0.9,bottom=0.15,wspace=0.075)
    # fig.tight_layout()

    ## RS line
    # xmin = np.where(np.min(mag)>16,16,np.min(mag))-0.3
    # a, b = -0.02,1.88
    # xmag = np.linspace(xmin,23,100)
    # ycol = a*xmag+b

    # xmin = mag[idx].min()-1.5
    # xmax = mag[idx].max()-1.5

    # mask = (mag<mag.mean())
    # ymin = color[mask[idx]].min() - np.std(color[mask[idx]])
    # ymax = color[mask[idx]].max() + 3*np.std(color[mask[idx]])

    axs[0].scatter(mag, color, color='k', s=3)
    # axs[0].plot(xmag,ycol-0.08*np.ones_like(ycol),color='r',linestyle='--')
    # axs[0].plot(xmag,ycol+0.08*np.ones_like(ycol),color='r',linestyle='--')
    axs[0].set_title('Cluster+Field')

    axs[1].scatter(mag_bkg, color_bkg, color='k', s=3)
    axs[1].set_title('Field')

    axs[2].scatter(mag[idx], color[idx], color='k', s=3)
    # axs[2].plot(xmag,ycol,color='k',linestyle='--')
    # axs[2].plot(xmag,ycol-0.08*np.ones_like(ycol),color='r',linestyle='--')
    # axs[2].plot(xmag,ycol+0.08*np.ones_like(ycol),color='r',linestyle='--')
    axs[2].set_title('Cluster')

    axs[0].set_ylabel(r'$(g-r)$')
    for i in range(3):
        axs[i].set_xlabel(r'$r$')
        # axs[i].set_xlim([xmin,xmax])
        # axs[i].set_ylim([ymin,ymax])

    plt.savefig(name_cls+'_color-magnitude_subtraction.png')
