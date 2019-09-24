# !/usr/bin/env python
# color-magnitude subtraction algorithm

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from time import time
from sklearn.neighbors import KernelDensity

from astropy.table import Table, vstack

gaussian = lambda x,mu,sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def kde_sklearn(x, x_assign, bandwidth=None, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    # from sklearn.grid_search import GridSearchCV
    if bandwidth is None:
        grid = GridSearchCV(KernelDensity(rtol=1e-5,kernel='gaussian'),
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

def computeColorMagnitudeKDE(x,y):
    """input: x (magnitude) and y (color)
       return: PDF (probability distribuition function)
    """
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values,bw_method='silverman')
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

def backgroundSubtraction(mag,color,mag_bkg,color_bkg,prior=None):
    ## compute kde
    kernel = computeColorMagnitudeKDE(mag,color)
    kernel_bkg = computeColorMagnitudeKDE(mag_bkg,color_bkg)

    values = np.vstack([mag,color])
    kde = kernel(values)
    kde_bkg = kernel_bkg(values)

    if prior is None:
        Pfield = (kde_bkg/kde) 
    else:
        Pfield = (kde_bkg/kde) * ((1-prior)/prior)

    # print('Pfiled>1: ',np.count_nonzero(Pfield>1)/len(mag),'%')
    Pfield = np.where(Pfield>1,1.,Pfield)

    ## subtract field galaxies
    idx = monteCarloSubtraction(Pfield,mag,color)
    mag_sub, color_sub = mag[idx],color[idx]

    kernel = computeColorMagnitudeKDE(mag_sub,color_sub)
    kde_sub = kernel(values)

    return idx, kde_sub, kde_bkg

def computeColorPDF(gals,cat,magLim=None,bandwidth=0.01,plot=False):
    ''' compute probability distribution function for the 3 sequential colors 
        1:(g-r) ; 2:(r-i); 3:(i-z)
        TO DO: 1) Test kde; 2) Test mag cut; 3) Test other colors (g-i),(r-z)
    '''
    ncls = len(cat)

    Flag = np.full((len(gals['Bkg']),3), False, dtype=bool)
    pdf = np.empty((1,3),dtype=float)
    pdf_bkg = np.empty((1,3),dtype=float)

    for idx in range(ncls):
        cls_id = cat['CID'][idx]

        all_gal, = np.where(gals['CID']==cls_id)
        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)) #&(gals['mag'][2,:]<magLim[idx])
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id)) #&(gals['mag'][2,:]<magLim[idx])

        mag, mag_bkg = gals['mag'][galaxies], gals['mag'][bkgGalaxies]
        mag_all = gals['mag'][all_gal]
        # probz = gals['PDFz'][galaxies]
        
        t0 = time()
        ## 3 Color distributions
        colors3, colors3_bkg = [], []
        for i in range(3):
            magi, mag_bkgi = mag[:,i], mag_bkg[:,i]
            color = (mag[:,i]-mag[:,i+1])
            color_bkg = (mag_bkg[:,i]-mag_bkg[:,i+1])
            color_all = (mag_all[:,i]-mag_all[:,i+1])

            idx, pdf_i, pdf_i_bkg = backgroundSubtraction(magi,color,mag_bkgi,color_bkg,prior=None)
            # pi = st.gaussian_kde(color[idx], bw_method='silverman')
            # pi_bkg = st.gaussian_kde(color_bkg, bw_method='silverman')
            
            # pdf_i = pi(color_all)
            # pdf_i_bkg = pi_bkg(color_all)
            # pdf_i,_ = kde_sklearn(color[idx], color, bandwidth=bandwidth)
            # pdf_i_bkg,_ = kde_sklearn(color_bkg, color, bandwidth=bandwidth)
            
            colors3.append(pdf_i)
            colors3_bkg.append(pdf_i_bkg)

            Flag[galaxies[idx],i] = True
            if (i==1)&(plot):
                plotTrioColorMagDiagram(magi,color,mag_bkgi,color_bkg,idx,name_cls='./check/probColor/Planck_%i'%cls_id)

        print('%i - Color prob. time:'%cls_id,time()-t0)
        c3 = np.array(colors3).transpose()
        c3_bkg = np.array(colors3_bkg).transpose()
        
        pdf = np.vstack([pdf,c3])
        pdf_bkg = np.vstack([pdf_bkg,c3_bkg])

    return pdf[1:], pdf_bkg[1:], Flag

#############################################################################
### plot


# def doGrid(xmin=16,xmax=24,ymin=-1,ymax=4):
#     xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:200j]
#     return xx,yy

# def getKDE(x,y,xmin=16,xmax=24,ymin=-1,ymax=4):
#     kernel = computeColorMagnitudeKDE(x,y)

#     xx, yy = doGrid()
#     positions = np.vstack([xx.ravel(), yy.ravel()])
#     kde = np.reshape(kernel(positions).T, xx.shape)

#     return kde

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

# def plotTrioColorMagPDF(mag,color,mag_bkg,color_bkg,idx,name_cls='cluster'):
#     plt.clf()
#     fig, axs = plt.subplots(1, 3, sharey=True,sharex=True, figsize=(12,4))
#     fig.subplots_adjust(left=0.075,right=0.95,top=0.9,bottom=0.15,wspace=0.075)
#     fig.tight_layout()

#     xmin = mag[idx].min()+0.5
#     xmax = mag[idx].max()+0.5

#     ymin = color[idx].min() - np.std(color[idx])
#     ymax = color[idx].max() + 3*np.std(color[idx])

#     xx,yy = doGrid()
#     kde = getKDE(mag,color)
#     kde_bkg = getKDE(mag_bkg,color_bkg)
#     kde_sub = getKDE(mag[idx],color[idx])


#     cfset = axs[0].contourf(xx, yy, kde, cmap='Blues')
#     cset = axs[0].contour(xx, yy, kde, colors='white')
#     axs[0].clabel(cset, inline=1, fontsize=10)
#     axs[0].set_title('Cluster+Field')

#     cfset = axs[1].contourf(xx, yy, kde_bkg, cmap='Blues',alpha=0.7)
#     cset = axs[1].contour(xx, yy, kde_bkg, colors='white')
#     axs[1].clabel(cset, inline=1, fontsize=10)
#     axs[1].set_title('Field')

#     cfset = axs[2].contourf(xx, yy, kde_sub, cmap='Blues')
#     cset = axs[2].contour(xx, yy, kde_sub, colors='white')
#     axs[2].clabel(cset, inline=1, fontsize=10)
#     axs[2].set_title('Cluster')

#     axs[0].set_ylabel(r'$(g-r)$')
#     for i in range(3):
#         axs[i].set_xlabel(r'$r$')
#         axs[i].set_xlim([xmin,xmax])
#         axs[i].set_ylim([ymin,ymax])

#     plt.savefig(name_cls+'_color-magnitude_subtraction_kernel.png')

# def plotRingBackground():
#     plt.clf()
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     fig.subplots_adjust(left=0.1,right=0.95,top=0.975,bottom=0.025)
#     ax.set_aspect('equal')

#     ax.scatter(gal['dx'],gal['dy'],color='r',s=2)
#     ax.scatter(gal_bkg_bad['dx'],gal_bkg_bad['dy'],color='lightgray',s=2)
#     ax.scatter(gal_bkg['dx'],gal_bkg['dy'],color='k',s=2)
#     ax.set_xlabel(r'$\Delta X $ [Mpc]')
#     ax.set_ylabel(r'$\Delta Y$ [Mpc]')
#     plt.savefig(name_cls+'_ring_bkg.png')



# ncls = len(all_gal[all_gal['Rnorm']<0.15])/(area*0.15**2/r200**2)


