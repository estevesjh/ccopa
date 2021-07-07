# !/usr/bin/env python
# redshift stuff algorithm

import numpy as np
from astropy.table import Table, vstack
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

## local libraries
import gaussianKDE as kde
import helper as hp

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

def scaleSTD(x):
    xmean = np.mean(x)
    xstd  = np.std(x)
    return (x-xmean)/(xstd), xmean, xstd

def redshiftDistribuitionSubtraction(z,z_gal,z_bkg,nb,ncf,prior=[None,None],bw=0.01):
    ## compute kde
    # kernel = kde.gaussian_kde(z_gal,bw_method=bw,weights=prior[0])
    # kernel_bkg = kde.gaussian_kde(z_bkg,bw_method=bw,weights=prior[1])

    ## scaling the data
    # z_gal, u, s = scaleSTD(z_gal)
    # z_bkg = (z_bkg-u)/s
    values = z#(z-u)/s

    try:
        kernel = kde.gaussian_kde(z_gal,silvermanFraction=1,weights=prior[0])
        pdf_cf = kernel(values)
    except:
        pdf_cf = np.ones_like(values)

    kernel_bkg = kde.gaussian_kde(z_bkg,silvermanFraction=1,weights=prior[1])
    pdf_bkg = kernel_bkg(values)

    nc = (ncf-nb)
    if nc<0:
        nb = ncf = nc = 1
    
    Nf = pdf_bkg*nb
    Ncf = pdf_cf*ncf

    Pfield = Nf/(Ncf+1e-6)

    # print('Pfiled>1: ',np.count_nonzero(Pfield>1)/len(z),'%')
    Pfield = np.where(Pfield>1,1.,Pfield)
    # print('pfield:',Pfield)

    # nc = np.abs(ncf-nb)
    pdfz = np.where((Ncf-Nf)<0,0,(Ncf-Nf)/nc) ## We take only the galaxy excess
    pdfz = pdfz/integrate.trapz(pdfz,x=values) ## set pdf to unity

    # pdfz = np.where(pdfz<0.01,0.,pdfz)
    
    zbw = 0.01#kernel.silverman_factor()/2
    return  pdfz, pdf_cf, pdf_bkg, zbw

def truncatedGaussian(z,zcls,zmin,zmax,sigma,vec=False):
    if vec:
        s_shape = sigma.shape
        sigma = sigma.ravel()
        z = z.ravel()
        zcls = zcls.ravel()

    # user input
    myclip_a = zmin
    myclip_b = zmax
    my_mean = zcls
    my_std = sigma

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    pdf = truncnorm.pdf(z, a, b, loc = my_mean, scale = my_std)

    if vec: pdf.shape = s_shape
    return pdf

def verifyTrunc(zcls,sigma):
    return (zcls-5*sigma)>0.

def getModel(zvec,zcls,sigma,bias):
    mean  = zcls-bias
    trunc = verifyTrunc(zcls,sigma)
    if trunc:
        pdf = truncatedGaussian(zvec,mean,0.,np.max(zvec),sigma,vec=False)
    else:
        pdf = gaussian(zvec,mean,sigma)

    zoff  = (zvec-mean)/(1+zcls)
    pdf = np.where(np.abs(zoff)>=2.*sigma,0.,pdf)
    return pdf 
    
def computeRedshiftPDF(gals,cat,r200,nbkg,keys,sigma,zfile=None,bandwidth=0.008,zvec=np.arange(0.,1.,0.005)):
    ## estimate PDFz
    pdf_cls = []
    pdf_cf = []
    pdf_field = []
    
    if sigma<0:
        # bias, sigma = hp.look_up_table_photoz_model(cat['redshift'],filename='auxTable/bpz_phtoz_model_cosmoDC2.csv')
        bias, sigma = hp.look_up_table_photoz_model(cat['redshift'],filename=zfile)
        #bias  = np.zeros_like(sigma)
        
    else:
        sigma = sigma*np.ones_like(cat['redshift'])
        bias  = np.zeros_like(cat['redshift'])
        
    for idx in range(len(cat)):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]
        # n_cls_field, nb = ngals[idx], nbkg[idx]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)& (gals["R"]<=gals['r_aper']) )
        galaxies2, = np.where((gals['Gal']==True)&(gals['CID']==cls_id))
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        z_gal = gals['z'][galaxies]
        zerr = gals['zerr'][galaxies]
        z_bkg = gals['z'][bkgGalaxies]
        
        probz = np.array(gals['pz0'][galaxies])
        probz_bkg = np.array(gals['pz0'][bkgGalaxies])
        n_cls_field = np.sum(probz)/(np.pi*r2**2)
        
        dz = np.diff(zvec)[0]
        if (len(z_gal)>2)&(len(z_bkg)>2):
            k_i, k_cf_i, k_i_bkg, dz = redshiftDistribuitionSubtraction(zvec, z_gal, z_bkg, nb, n_cls_field, bw=bandwidth)
            # pdf_i, pdf_i_bkg = redshiftDistribuitionSubtraction(z_gal2, z_gal, z_bkg, nb, n_cls_field, bw='silverman',prior=[probz,probz_bkg])

        else:
            k_i = k_i_bkg = k_cf_i = np.ones_like(zvec)
            # pdf_i = pdf_i_bkg = np.ones_like(z_gal)
            
        k_i = getModel(zvec,z_cls,sigma[idx]*(1+z_cls),bias[idx]*(1+z_cls))
        
        dz=1.
        pdf_cls.append(dz*k_i)
        pdf_cf.append(dz*k_cf_i)
        pdf_field.append(dz*k_i_bkg)
    
    pdfz_list = [pdf_cls, pdf_cf, pdf_field]
    return pdfz_list

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

    axs[0].set_xlim(-0.125,0.125)

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
