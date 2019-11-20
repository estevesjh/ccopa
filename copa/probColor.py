# !/usr/bin/env python
# color-magnitude subtraction algorithm

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from time import time
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from astropy.table import Table, vstack
import dask

## local libraries
import gaussianKDE as kde
from six import string_types

def interpData(x,y,x_new):
    out = np.empty(x_new.shape, dtype=y.dtype)
    out = interp1d(x, y, kind='linear', copy=False)(x_new)
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
    
def computeColorMagnitudeKDE(x,y,weight=None,bandwidth='silverman'):
    """input: x (magnitude) and y (color)
       return: PDF (probability distribution function)
    """
    values = np.vstack([x, y])

    if np.isscalar(bandwidth) and not isinstance(bandwidth, string_types):
        bandwidth = bandwidth/np.std(values,ddof=1)

    kernel = kde.gaussian_kde(values,weights=weight,bw_method=bandwidth)

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

def monteCarloSubtraction(p_field,mag,color):
    """input: field probability
       return: indices of the non field galaxies
    """
    nsize = len(mag)
    idx_subtracted = np.empty(0,dtype=int)

    for i in range(nsize):
        n_rand = np.random.uniform(0,1)
        pi = p_field[i]

        if n_rand > pi:
            idx_subtracted = np.append(idx_subtracted,i)

    return idx_subtracted

def getRedSequenceWidth(color):
    color_cut_upper_level = np.percentile(color,75)
    color_cut_lower_level = np.percentile(color,25)

    cut, = np.where((color>=color_cut_lower_level)&(color<=color_cut_upper_level))

    std = np.std(color[cut])

    return std

def backgroundSubtraction(mag,color,mag_bkg,color_bkg,ncls,nbkg,weight=[None,None],bandwidth=0.05,quartile=95,tyColor=0):
    ## compute kde    
    # kernel = computeColorMagnitudeKDE(mag,color,weight=weight[0],bandwidth=bandwidth)
    # kernel_bkg = computeColorMagnitudeKDE(mag_bkg,color_bkg,weight=weight[1],bandwidth='silverman')
    # values = np.vstack([mag,color])
    
    kernel = computeColorKDE(color,weight=weight[0],bandwidth='silverman')
    kernel_bkg = computeColorKDE(color_bkg,weight=weight[1],bandwidth='silverman')
    
    # kernel = computeColorKDE(color,weight=None,bandwidth='silverman')
    # kernel_bkg = computeColorKDE(color_bkg,weight=None,bandwidth='silverman')

    values = color

    kde = kernel(values)
    kde_bkg = kernel_bkg(values)

    nc = (ncls-nbkg)
    if nc<0:
        nbkg = ncls = nc = 1

    Ncls_field = ncls*kde
    Nfield = nbkg*kde_bkg
    
    Pfield = (Nfield/Ncls_field)

    # print('Pfiled>1: ',np.count_nonzero(Pfield>1)/len(mag),'%')
    # Pfield = np.where(Pfield>1,1.,Pfield)

    ## subtract field galaxies
    idx = monteCarloSubtraction(Pfield,mag,color)
    
    off_threshold = [0.55,0.2,0.12]
    if len(idx)>5:
        color_cut_upper_level = np.percentile(color[idx],quartile)
        color_cut_lower_level = np.percentile(color[idx],3)

        off = color_cut_upper_level-np.median(color[idx])
        if off>off_threshold[tyColor]:
            std = getRedSequenceWidth(color[idx])
            color_cut_upper_level = 2*std + np.median(color[idx])
            print('color:',tyColor)
            print('std:',std)


        print('color median:',np.median(color[idx]))
        print('color offset:',off)
        print('\n')
        cut, = np.where((color[idx]>=color_cut_lower_level)&(color[idx]<=color_cut_upper_level))
        idx = idx[cut]

        if weight[0] is not None:
            weight = weight[0]
            weight = weight[idx]

        # kernel = computeColorKDE(color[idx],weight=weight,silvermanFraction=10.)
        kernel = computeColorKDE(color[idx],weight=weight,bandwidth=bandwidth)
        # kernel = computeColorMagnitudeKDE(mag[idx],color[idx],weight=weight,bandwidth=bandwidth)
        kde_sub = kernel(values)

        
    else:
        # print('Color Error: not enough galaxies')
        kde_sub = (kde-kde_bkg)
        kde_sub = np.where(kde_sub<0,0.,kde_sub) ## we take only the excess
        # kde_bkg = np.zeros_like(mag)

    return idx, kde_sub, kde_bkg

def doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[None,None],bandwidth=0.05,choose_color=[0,1]):
    quartile=98
    i,i2 = choose_color

    magi, mag_bkgi = mag[:,i2], mag_bkg[:,i2]
    color = (mag[:,i]-mag[:,i2])
    color_bkg = (mag_bkg[:,i]-mag_bkg[:,i2])

    if (magi.size > 3)&(mag_bkgi.size >3):
        idx, pdf_i, pdf_i_bkg = backgroundSubtraction(magi,color,mag_bkgi,color_bkg,n_cls_field,nb,weight=weight,bandwidth=bandwidth,quartile=quartile,tyColor=i)
    else:
        idx, pdf_i, pdf_i_bkg = np.full(magi.size,False),np.full(magi.size,1),np.full(magi.size,1)
        
    return idx, pdf_i, pdf_i_bkg

def computeColorPDF(gals,cat,r200,nbkg,testPz=False,bandwidth=[0.008,0.001,0.001],parallel=False, plot=True):
    ''' compute probability distribution function for the 5 sequential colors 
        0:(g-r); 1:(g-i); 2:(r-i); 3:(r-z); 4:(i-z)
    '''
    ncls = len(cat)

    Flag = np.full((len(gals['Bkg']),5), True, dtype=bool)
    pdf = np.empty((1,5),dtype=float)
    pdf_bkg = np.empty((1,5),dtype=float)

    results0, results1, results2, results3, results4 = [], [], [], [], []
    galIDX = []
    
    good_indices, = np.where(nbkg>0)
    for idx in good_indices:
        cls_id,z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]
        # n_cls_field, nb = ngals[idx], nbkg[idx]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)) #&(gals['mag'][2,:]<magLim[idx])
        galIDX.append(galaxies)
        
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id)) #&(gals['mag'][2,:]<magLim[idx])

        mag, mag_bkg = gals['mag'][galaxies], gals['mag'][bkgGalaxies]
        probz = np.array(gals['PDFz'][galaxies])
        probz_bkg = np.array(gals['PDFz'][bkgGalaxies])

        n_cls_field = np.sum(probz)/(np.pi*r2**2)
        
        if not parallel:  
            t0 = time()
            ## 5 Color distributions
            id0, pdf_0, pdf_bkg_0 = doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,1],bandwidth=bandwidth[0]) ##(g-r)
            id1, pdf_1, pdf_bkg_1 = doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,2],bandwidth=bandwidth[0]) ##(g-i)
            id2, pdf_2, pdf_bkg_2 = doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,2],bandwidth=bandwidth[0]) ##(r-i)
            id3, pdf_3, pdf_bkg_3 = doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,3],bandwidth=bandwidth[0]) ##(r-z)
            id4, pdf_4, pdf_bkg_4 = doKDEColors(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[2,3],bandwidth=bandwidth[0]) ##(i-z)            
            print('%i - Color prob. time:'%cls_id,time()-t0)

            c5 = np.array([pdf_0,pdf_1,pdf_2,pdf_3,pdf_4]).transpose()
            c5_bkg = np.array([pdf_bkg_0,pdf_bkg_1,pdf_bkg_2,pdf_bkg_3,pdf_bkg_4]).transpose()
            
            pdf = np.vstack([pdf,c5])
            pdf_bkg = np.vstack([pdf_bkg,c5_bkg])

            idxss = [id0,id1,id2,id3,id4]
            for i in range(5):
                Flag[galaxies[idxss[i]],i] = True

            if plot:
                lcolor = r'$ (r-i) $'
                svname = './controlPlots/color/Cluster_%i'%cls_id
                color = mag[:,1]-mag[:,2]
                
                # kernel = kde.gaussian_kde(color,bw_method='silverman')
                # kernel_true = kde.gaussian_kde(color[gals['True'][galaxies] ==True ],bw_method='silverman')

                kernel = kde.gaussian_kde(color,bw_method='silverman',weights=probz)
                kernel_true = kde.gaussian_kde(color[gals['True'][galaxies] ==True ],bw_method='silverman', weights=probz[gals['True'][galaxies] ==True])

                pdf_all_2 = kernel(color)
                pdf_all_2_true = kernel_true(color)

                plotDoubleColor(color, pdf_all_2, pdf_bkg_2, pdf_2, 'Cluster - %i at z=%.2f'%(cls_id,z_cls), nb, n_cls_field, name_cls=svname, lcolor=lcolor, pdf_true=pdf_all_2_true)

        else:
            t0 = time()
            ## 5 Color distributions
            y0 = dask.delayed(doKDEColors)(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,1],bandwidth=bandwidth[0])##(g-r)
            y1 = dask.delayed(doKDEColors)(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[0,2],bandwidth=bandwidth[0])##(g-i)
            y2 = dask.delayed(doKDEColors)(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,2],bandwidth=bandwidth[1])##(r-i)
            y3 = dask.delayed(doKDEColors)(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[1,3],bandwidth=bandwidth[1])##(r-z)
            y4 = dask.delayed(doKDEColors)(mag,mag_bkg,n_cls_field,nb,weight=[probz,probz_bkg],choose_color=[2,3],bandwidth=bandwidth[2])##(i-z)
            
            results0.append(y0)
            results1.append(y1)
            results2.append(y2)
            results3.append(y3)
            results4.append(y4)

    if parallel:        
        results0 = dask.compute(*results0, scheduler='processes', num_workers=2)
        results1 = dask.compute(*results1, scheduler='processes', num_workers=2)
        results2 = dask.compute(*results2, scheduler='processes', num_workers=2)
        results3 = dask.compute(*results2, scheduler='processes', num_workers=2)
        results4 = dask.compute(*results2, scheduler='processes', num_workers=2)

        for i in range(len(galIDX)):
            galaxies = galIDX[i]

            id0, pdf_0, pdf_bkg_0 = results0[i]
            id1, pdf_1, pdf_bkg_1 = results1[i]
            id2, pdf_2, pdf_bkg_2 = results2[i]
            id3, pdf_3, pdf_bkg_3 = results3[i]
            id4, pdf_4, pdf_bkg_4 = results4[i]

            c5 = np.array([pdf_0,pdf_1,pdf_2,pdf_3,pdf_4]).transpose()
            c5_bkg = np.array([pdf_bkg_0,pdf_bkg_1,pdf_bkg_2,pdf_bkg_3,pdf_bkg_4]).transpose()
            
            pdf = np.vstack([pdf,c5])
            pdf_bkg = np.vstack([pdf_bkg,c5_bkg])

            idxss = [id0,id1,id2,id3,id4]
            for i in range(5):
                Flag[galaxies[idxss[i]],i] = True

        print('Parallel - Color time:'%cls_id,round(time()-t0,2))
    return pdf[1:], pdf_bkg[1:], Flag
    
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
