# !/usr/bin/env python
# redshift stuff algorithm

import numpy as np
from astropy.table import Table, vstack
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt

def zshift(z,z_cls):
    return (z-z_cls)/(1+z_cls)

gaussian = lambda x,mu,sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def interpData(x,y,x_new):
    yint = interp1d(x,y,fill_value='extrapolate')
    return yint(x_new)

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
        kde = KernelDensity(bandwidth=bandwidth, rtol=1e-5)
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

def redshiftDistribuitionSubtraction(z,z_bkg,prior=None,bw=0.01):
    ## compute kde
    kde,bw = kde_sklearn(z,z,bandwidth=bw)
    kde_bkg,_ = kde_sklearn(z_bkg,z,bandwidth=bw)

    Pfield = (kde_bkg/kde)
    
    if prior is not None:
        Pfield *= (1-prior)/(prior)

    # print('Pfiled>1: ',np.count_nonzero(Pfield>1)/len(z),'%')
    Pfield = np.where(Pfield>1,1.,Pfield)

    ## subtract field galaxies
    idx = monteCarloSubtraction(Pfield)

    return idx, Pfield

def computeRedshiftPDF(gals,cat,bandwidth=0.01,plot=False):
    ## estimate PDFz
    ncls = len(cat)
    pdf = np.empty(0,dtype=float)
    pdf_bkg = np.empty(0,dtype=float)

    Flag = np.full(len(gals['Bkg']), False, dtype=bool)

    for idx in range(ncls):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        magLim_i = cat['magLim'][idx,1]
        print('cls_id',cls_id, 'at redshift', z_cls)

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id))
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        z     = zshift(gals['z'][gals['CID']==cls_id],z_cls)
        z_gal = zshift(gals['z'][galaxies],z_cls)
        z_bkg = zshift(gals['z'][bkgGalaxies],z_cls)
        
        probz = gals['PDFz'][galaxies]
    
        idx, pf = redshiftDistribuitionSubtraction(z_gal, z_bkg, prior=probz/np.max(probz), bw=bandwidth)
        
        ## Deal with projections effects
        if np.std(z[idx])>0.06 or len(z[idx])<5:
            print('bad')
            # pdf_i = gaussian(z_gal,0.,0.05)
            pdf_i = probz
            # idx, pf = redshiftDistribuitionSubtraction(z_gal, z_bkg, prior=pdf_cls/np.max(pdf_cls), bw=bandwidth)
        else:
            pdf_i,_ = kde_sklearn(z[idx], z_gal, bandwidth=bandwidth)
        
        # pdf_i = gaussian(z_gal,0.,0.05)
        # pdf_i_bkg = 0.1*np.ones_like(z_gal)
        pdf_i_bkg = interpData(z_bkg, gals['PDFz'][bkgGalaxies],z_gal)
        # pdf_i_bkg,_ = kde_sklearn(z_bkg, z_gal, bandwidth=bandwidth)
        Flag[galaxies[idx]] = True
        
        if plot:
            plotTrioRedshift(z[idx],z_bkg,z_gal,z_cls,'./check/probRedshift/Planck_%i'%cls_id)

        pdf = np.append(pdf,pdf_i)
        pdf_bkg = np.append(pdf_bkg,pdf_i_bkg)
        
    return pdf, pdf_bkg, Flag  


########################################################3
# def computePzMonteCarlo(z,z_bkg,z_grid,prob=None,norm=1,bw=0.008):
#     ## compute kde
#     kde,bw = kde_sklearn(z,z_grid,bandwidth=bw)
#     kde_bkg,_ = kde_sklearn(z_bkg,z_grid,bandwidth=bw)
    
#     if prob is None:
#         Pz = norm*kde/(norm*kde+kde_bkg)
#     else:
#         Pz = norm*kde*prob/(norm*kde*prob+kde_bkg*(1-prob))
#     return Pz

# def PhotozProbabilities(zmin,zmax,membz,membzerr):
#     out = []
#     for i in range(len(membz)):
#         aux, err = integrate.quad(gaussian,zmin,zmax,args=(membz[i],membzerr[i]),full_output=0)
#         out.append(aux)
#     return np.array(out)

# def doPz(membz,membzerr,z_cls,window=0.15):
#     zmin, zmax = (z_cls -window*(1+z_cls)),(z_cls + window*(1+z_cls))
#     prob=PhotozProbabilities(zmin,zmax,membz,membzerr)#/erf(window*(1+z_cls)/np.sqrt(2*membzerr))
#     # prob = np.where(prob>1,1,prob)
#     return np.array(prob)


# # plotTrioRedshift(z_sub,z[bkgMask],z[galMask],cls_z,title,w1=prior[galMask],w2=(1/norm)*prior[bkgMask],w3=prior[idx])

def plotTrioRedshift(z_sub,z_bkg,z_gal,z_cls,name_cls,w1=None,w2=None,w3=None):
    plt.clf()
    fig, axs = plt.subplots(1, 3, sharey=True,sharex=True, figsize=(12,4))
    fig.subplots_adjust(left=0.075,right=0.95,top=0.9,bottom=0.15,wspace=0.075)
    # fig.tight_layout()

    xmin,xmax = -0.21,+0.21
    bins = np.arange(xmin,xmax,0.02)

    axs[0].hist(zshift(z_gal,z_cls), bins=bins, weights=w1, color='k',density=True)
    # axs[0].plot(bins,kde,color='r')
    axs[0].set_title('Cluster+Field')

    axs[1].hist(zshift(z_bkg,z_cls), weights=w2, bins=bins, color='k',density=True)
    axs[1].set_title('Field')

    # axs[2].hist(zshift(z_gal,z_cls), bins=bins, weights=w1, color='r',density=False)
    axs[2].hist(zshift(z_sub,z_cls), weights=w3, bins=bins, color='k',density=True)
    # axs[2].hist(zshift(z_sub,z_cls), bins=bins, color='k')
    axs[2].set_title('Cluster')
    # plt.legend()
    axs[0].set_ylabel(r'$Frequency$')
    for i in range(3):
        axs[i].set_xlabel(r'$(z-z_{cls})/(1+z_{cls})$')
    #     axs[i].set_xlim([xmin,23.])
    #     axs[i].set_ylim([-0.,2.6])
    plt.savefig(name_cls+'_redshiftDistribution.png')

# def plotProbz(deltaZ,pz,sigma,cls_z,cls_id):
#     c = pz>0.95
#     c2 = (deltaZ>-2*sigma)&(deltaZ<2*sigma)

#     plt.clf()
#     plt.figure(figsize=(6,5))
#     plt.hist(deltaZ[c2],bins=12,range=(-2*sigma,2*sigma),label=r'$ \pm 2 \sigma $')
#     plt.hist(deltaZ[c],bins=12,range=(-2*sigma,2*sigma),weights=pz[c],label='Pz > 0.95',color='lightblue')
#     plt.xlabel(r'$z_{gal}-z_{agl}/(1+z_{agl})$')
#     plt.ylabel('N')
#     plt.title(cls_id+'at redshfit %.3f'%(cls_z))
#     plt.legend()
#     plt.savefig(graphs+'photoZ/DES/%s_photoz.png'%(cls_id))