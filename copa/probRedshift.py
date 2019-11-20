# !/usr/bin/env python
# redshift stuff algorithm

import numpy as np
from astropy.table import Table, vstack
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

## local libraries
import gaussianKDE as kde

def zshift(z,z_cls):
    return (z-z_cls)/(1+z_cls)

gaussian = lambda x,mu,sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def interpData(x,y,x_new):
    yint = interp1d(x,y,fill_value='extrapolate')
    return yint(x_new)

def kde_sklearn(x, x_assign, bandwidth=None, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    # from sklearn.grid_search import GridSearchCV
    from sklearn.model_selection import GridSearchCV

    if bandwidth is None:
        grid = GridSearchCV(KernelDensity(rtol=1e-5,kernel='gaussian'),
                        {'bandwidth': np.linspace(0.02, 0.1, 20)},
                        cv=15) # 20-fold cross-validation
        grid.fit(x[:, np.newaxis])
        print(grid.best_params_)
        kde = grid.best_estimator_
        bandwidth = float(grid.best_params_['bandwidth'])
        print(bandwidth)
    else:
        kde = KernelDensity(bandwidth=bandwidth, rtol=1e-4)
        kde.fit(x[:, np.newaxis])

    log_pdf = kde.score_samples(x_assign[:, np.newaxis])
    
    return np.exp(log_pdf), bandwidth

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

def redshiftDistribuitionSubtraction(z,z_bkg,nb,ncf,prior=[None,None],bw=0.01):
    ## compute kde
    # kernel = kde.gaussian_kde(z,bw_method=bw/ z.std(ddof=1),weights=prior[0])
    # kernel_bkg = kde.gaussian_kde(z_bkg,bw_method=bw/z_bkg.std(ddof=1),weights=prior[1])

    kernel = kde.gaussian_kde(z,bw_method='silverman',weights=prior[0])
    kernel_bkg = kde.gaussian_kde(z_bkg,bw_method='silverman',weights=prior[1])

    pdf = kernel(z)
    pdf_bkg = kernel_bkg(z)

    nc = (ncf-nb)
    if nc<0:
        nb = ncf = nc = 1
    
    Nf = pdf_bkg*nb
    Ncf = pdf*ncf

    Pfield = Nf/Ncf

    # print('Pfiled>1: ',np.count_nonzero(Pfield>1)/len(z),'%')
    Pfield = np.where(Pfield>1,1.,Pfield)
    # print('pfield:',Pfield)

    # nc = np.abs(ncf-nb)
    Nc = np.where((Ncf-Nf)<0,0,(Ncf-Nf)) ## We take only the galaxy excess
    pdfz = Nc/nc
    pdfz = np.where(pdfz<0.01,0.,pdfz)
    
    ## subtract field galaxies
    idx = monteCarloSubtraction(Pfield)
    
    return idx, pdfz, pdf_bkg

# def doSigmaClip(z_gal, z_bkg, nb, n_cls_field, bw=0.001, prior=[None,None]):
#     idx, pdf_i, pdf_i_bkg = redshiftDistribuitionSubtraction(z_gal, z_bkg, nb, n_cls_field, bw=bw, prior=prior)

#     z_upper_l = np.percentile(z_gal,75)
#     z_lower_l = np.percentile(z_gal,15)

#     pdf_i = np.where(z_gal>=z_upper_l,0.,pdf_i)
#     pdf_i = np.where(z_gal<=z_lower_l,0.,pdf_i)

#     return idx, pdf_i, pdf_i_bkg

def computeRedshiftPDF(gals,cat,r200,nbkg,bandwidth=0.008,plot=False,testPz=False):
    ## estimate PDFz
    pdf = np.empty(0,dtype=float)
    pdf_bkg = np.empty(0,dtype=float)

    Flag = np.full(len(gals['Bkg']), False, dtype=bool)
    
    good_indices, = np.where(nbkg>0)
    for idx in good_indices:
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]
        # n_cls_field, nb = ngals[idx], nbkg[idx]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id))
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        z_gal = zshift(gals['z'][galaxies],z_cls)
        z_bkg = zshift(gals['z'][bkgGalaxies],z_cls)
        
        probz = np.array(gals['PDFz'][galaxies])
        probz_bkg = np.array(gals['PDFz'][bkgGalaxies])

        n_cls_field = np.sum(probz)/(np.pi*r2**2)
        
        if len(z_gal)>0:
            idx, pdf_i, pdf_i_bkg = redshiftDistribuitionSubtraction(z_gal, z_bkg, nb, n_cls_field, bw=bandwidth,prior=[probz,probz_bkg])
            Flag[galaxies[idx]] = True
            
            # pdf_i = probz
            # pdf_i_bkg = pdf_i_bkg
        
        else:
            pdf_i, pdf_i_bkg = np.ones_like(z_gal),np.ones_like(z_gal)
            
        if plot:
            svname = './controlPlots/redshift/Cluster_%i'%cls_id
            kernel = kde.gaussian_kde(z_gal,bw_method='silverman',weights=probz)
            pdf_i_all = kernel(z_gal)

            plotTrioRedshift(z_gal, pdf_i_all, pdf_i_bkg, pdf_i, z_cls, nb, n_cls_field,name_cls=svname)
            
        pdf = np.append(pdf,pdf_i)
        pdf_bkg = np.append(pdf_bkg,pdf_i_bkg)
        
    return pdf, pdf_bkg, Flag  

########################################################
def plotTrioRedshift(z,pdf_all,pdf_bkg,pdf,z_cls,nbkg,ncls_field,name_cls='Cluster'):
    Ncls_field = pdf_all*ncls_field
    N_bkg = pdf_bkg*nbkg
    N_sub = pdf*(ncls_field-nbkg)

    plt.clf()
    fig, axs = plt.subplots(1, 2, sharey=True,sharex=True, figsize=(10,8))
    fig.subplots_adjust(left=0.075,right=0.95,bottom=0.15,wspace=0.075)
    fig.suptitle(r'$z_{cls}$=%.2f'%(z_cls))
    # fig.tight_layout()

    axs[0].scatter(z,Ncls_field,color='blue',linestyle='--',label=r'Cluster+Field')
    axs[0].scatter(z,N_bkg,color='r',linestyle='--',label=r'Field')
    axs[0].set_title('Cluster+Field')
    axs[0].legend(loc='upper right')

    axs[1].scatter(z,N_sub,color='blue',label=r'Cluster Model')
    axs[1].set_title('Cluster')

    axs[0].set_xlim(-0.2,0.2)

    axs[0].set_ylabel(r'$ N $')
    axs[0].set_xlabel(r'$(z-z_{cls})/(1+z_{cls})$')
    axs[1].set_xlabel(r'$(z-z_{cls})/(1+z_{cls})$')
        
    axs[1].legend()
    
    plt.savefig(name_cls+'_redshiftDistribution.png', bbox_inches = "tight")

def plotPfield(z,pz,z_cls,name):
    plt.clf()
    plt.figure(figsize=(6,6))
    
    plt.scatter(z,pz,color='k',label='z = %.3f'%(z_cls))
    plt.axvline(0.,color='r',linestyle='--')
    plt.xlabel(r'$(z - z_{cls})/(1+z_{cls})$')
    plt.ylabel(r'$P_{z}$')

    plt.ylim(0.,1.)
    plt.xlim(-0.13,0.13)

    # plt.title(cls_id)
    plt.legend()
    plt.savefig(name)
