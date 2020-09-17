# !/usr/bin/env python
# color-magnitude subtraction algorithm

from time import time
import numpy as np
from scipy.stats import uniform, norm, stats
import os

from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import matplotlib

#import pandas as pd
import seaborn as sns; sns.set(color_codes=True)

# from myPlots import *

plt.rcParams.update({'font.size': 16})
sns.set_style("whitegrid")

def plot_scatter_hist(x,y,weights=None,xtrue=None,ytrue=None,xlabel='redshift',ylabel=r'$\mu_{\star}\,\,[10^{12}\,M_{\odot}]$',save='./img/bla.png'):
    compare = (xtrue is not None) and (ytrue is not None)
    
    if weights is not None:scale = 30*weights**(1/2)+1
    else: scale=50

    fig = plt.figure(figsize=(12,10))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.01
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    scatter_axes = plt.axes(rect_scatter)
    scatter_axes.tick_params(direction='in', top=True, right=True)
    x_hist_axes = plt.axes(rect_histx,sharex=scatter_axes)
    x_hist_axes.tick_params(direction='in')
    y_hist_axes = plt.axes(rect_histy,sharey=scatter_axes)
    y_hist_axes.tick_params(direction='in')

    # scatter_axes = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2)
    # x_hist_axes = plt.subplot2grid((3, 3), (0, 0), colspan=2,
    #                                sharex=scatter_axes)
    # y_hist_axes = plt.subplot2grid((3, 3), (1, 2), rowspan=2,
    #                                sharey=scatter_axes)

    xmax = np.nanmax(x)
    ymin, ymax = 0.98*np.nanmin(y), 1.02*np.nanmax(y)
    binx = np.linspace(0.0, xmax, num=50)
    biny = np.linspace(ymin, ymax, num=105)  # 1500/100.

    pal = sns.dark_palette("#3498db", as_cmap=True)
    # sns.kdeplot(x, y, ax=scatter_axes, cmap=pal, zorder=3)  # n_levels=10

    scatter_axes.set_xlabel(xlabel, fontsize=25)
    scatter_axes.set_ylabel(ylabel, fontsize=25)

    scatter_axes.scatter(x, y, s=scale, color='b', marker='o', alpha=0.25, zorder=0)
    scatter_axes.axhline(np.mean(y),linestyle='--',color='b')

    x_hist_axes.hist(x, bins=binx, weights=weights, normed=False, alpha=0.85, color='b')
    y_hist_axes.hist(y, bins=biny, weights=weights, normed=False, orientation='horizontal', alpha=0.85, color='b')
    y_hist_axes.axhline(np.mean(y),linestyle='--', color='b')

    if compare:
        scatter_axes.scatter(xtrue,ytrue, s=25, color='r', marker='o', alpha=0.25, zorder=1)
        scatter_axes.axhline(np.mean(ytrue),linestyle='--',color='r')

        x_hist_axes.hist(xtrue, bins=binx, color='r', histtype='stepfilled', alpha=0.5, normed=False)
        y_hist_axes.hist(ytrue, bins=biny, color='r', histtype='stepfilled', alpha=0.5, normed=False,orientation='horizontal')
        y_hist_axes.axhline(np.mean(ytrue),linestyle='--',color='r')

    x_hist_axes.set_yticklabels([])
    y_hist_axes.set_xticklabels([])

    if ymax<0:
        ymin,ymax=0.98*np.nanmax(y), 1.02*np.nanmin(y)

        scatter_axes.set_ylim(ymin, ymax)
        y_hist_axes.set_ylim(ymin, ymax)

    scatter_axes.xaxis.set_tick_params(labelsize=15)
    scatter_axes.yaxis.set_tick_params(labelsize=15)
    x_hist_axes.xaxis.set_tick_params(labelsize=0.05, labelcolor='white')
    y_hist_axes.yaxis.set_tick_params(labelsize=0.05, labelcolor='white')

    fig.subplots_adjust(hspace=.01, wspace=0.01)
    plt.savefig(save)
    plt.close()

def sky_plot(RA,DEC,title="Buzzard v1.6 - 1000 GC",savefig='./img/sky_plot.png'):
    ############################
    #Codigo para plotar coordenadas de objetos na esfera celeste
    #############################
    import matplotlib.pyplot as pplot
    import astropy.coordinates as coord
    from astropy import units as u
    
    ra = coord.Angle(RA*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(DEC*u.degree)

    ##############
    #Plotando os objetos
    #import astropy.coordinates as coord
    fig = pplot.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="aitoff")
    plt.title(title)
    ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
    ax.grid(True)
    ax.scatter(ra.radian, dec.radian, s=10, alpha=0.5)
    plt.subplots_adjust(top=0.9,bottom=0.0)
    # ax.set_xticklabels(['10h','8h','6h','4h','2h','0h','20h','18h','16h','14h','12h'])
    
    fig.savefig(savefig, bbox_inches = "tight")


def plot_lin_reg(x,y, ylog=False, xlabel = r'N$_{gals}$', ylabel = r'richness', save='./img/richness_ngals.png'):
    nmin, nmax = 5., 1.25*np.max([x,y])

    linreg=lin_reg(x,y)
    idx = np.argsort(x)
    xt,yh = x[idx],linreg['Yhat'][idx]

    b0 = round(linreg['b0'],3)
    b1 = round(linreg['b1'],3)
    cb_u, cb_l = linreg['cb_u'], linreg['cb_l']

    xs = np.linspace(nmin,nmax,200)

    fig = plt.figure(figsize=(8,6))
    plt.plot(xt,yh, color="gray",label='y=%.2f x + %.2f'%(b1,b0))
    plt.fill_between(xt, cb_l, cb_u, color="gray", alpha=0.25, label='_nolabel_')
    plt.plot(xt,cb_l, color="gray", label='_nolabel_')
    plt.plot(xt,cb_u, color="gray", label='_nolabel_')
    plt.scatter(x,y, s=20, alpha=0.5, label='_nolabel_')
    # plt.plot(xs,xs,color='k',linestyle='--', label='y = x')
    plt.legend(loc='lower right')
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    plt.title("S-Plus - z=[0.02,0.4); N={0:d}".format(len(x)))
    # plt.xlim(nmin,nmax)
    # plt.ylim(nmin,nmax)
    if ylog:
        plt.yscale('log')
    # plt.xscale('log')
    
    plt.savefig(save, bbox_inches = "tight")
    plt.close()

def makeBin(variable, nbins=10., xvec=None):
    width = len(variable)/nbins
    if xvec is None:
        xmin, xmax = (variable.min()), (variable.max() + width / 2)
        xvec = np.arange(xmin, xmax, width)

    idx, xbins = [], []
    for i in range(len(xvec) - 1):
        xlo, xhi = xvec[i], xvec[i + 1]
        w, = np.where((variable >= xlo) & (variable <= xhi))
        bins = (xlo + xhi) / 2
        idx.append(w)
        xbins.append(bins)

    return idx, xbins

def get_binned_variables(x,y,xedges=None):
    
    if xedges is None:
        xedges = np.linspace(np.nanmin(x),np.nanmax(x),6)

    indices, x_bin = makeBin(x, xvec=xedges)
    x_bin_err = np.diff(xedges)/ 2
    y_bin = [np.mean(y[idx]) for idx in indices]
    y_bin_err = [np.std(y[idx]) for idx in indices]

    return x_bin, x_bin_err, y_bin, y_bin_err

def plot_group_by_redshift(x, y, xtrue=None, ytrue=None, ylim=None, ylog=False, xlabel = r'redshift', ylabel = r'richness', title=None, save='./img/redshift_richness.png'):
    
    compare = (xtrue is not None) and (ytrue is not None)

    x_bin, x_bin_err, y_bin, y_bin_err = get_binned_variables(x,y)

    if compare:
        x_bin2, x_bin_err2, y_bin2, y_bin_err2 = get_binned_variables(xtrue,ytrue)

    fig = plt.figure(figsize=(8,6))
    plt.scatter(x,y, s=6, alpha=0.5, label='_nolabel_')
    plt.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err,color='b', fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2)
    
    if compare:
        plt.scatter(xtrue,ytrue, s=10, alpha=0.5, color='r', label='_nolabel_')
        plt.errorbar(x_bin2,y_bin2,xerr=x_bin_err2,yerr=y_bin_err2, color='r', fmt='o', linestyle='--', markersize=4, capsize=2, capthick=2, label='True Distrib.')

    plt.legend(loc='lower right')
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.xlim(nmin,nmax)
    
    if ylog:
        plt.yscale('log')
    # plt.xscale('log')

    plt.savefig(save)
    plt.close()

def plot_mag_evolution(zs,mag, ylims = (-27., -17.5), ylabel='Mr',save='./amag_evolution.png'):
    mmax, mmin = ylims
    fig = plt.figure(figsize=(8,6))
    plt.scatter(zs,mag,color='royalblue',s=15,alpha=0.3)
    plt.xlabel('redshift',fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    plt.ylim(mmin,mmax)
    
    plt.savefig(save)

def chunks2(x, yedges):
    for i in range(len(yedges)-1):
        w, = np.where( (x>=yedges[i]) & (x<=yedges[i+1]) )
        yield w

def binsCounts(x,n):
    """get n points per bin"""
    xs = np.sort(x,kind='mergesort')
    xedges = [ xs[0] ]
    for i in np.arange(0,len(xs),n):
        if (i+n)<len(xs)-1:
            xi = (xs[i+n+1]+xs[i+n])/2.
        else:
            xi = xs[-1]
        xedges.append(xi)

    xmean = [(xedges[i+1]+xedges[i])/2 for i in range(len(xedges)-1)]
    return np.array(xedges), np.array(xmean)


def plot_color_bin(z,mean,sigma,dz=0.025,label='RS Model',lcolor='r',axs=None,scatter_mean=True):
    b = np.arange(np.min(z)-dz/2,np.max(z)+dz,dz)
    indices_list = list(chunks2(z, b))

    y_bin = np.array([ np.nanmedian(mean[idx]) for idx in indices_list])
    x_bin = np.array([ np.median(z[idx]) for idx in indices_list])
    # std_bin = np.array([(np.nanmedian(sigma[idx])**2+np.nanstd(mean[idx])**2)**(1/2) for idx in indices_list])
   
    std_bin = np.array([np.nanmedian(sigma[idx]) for idx in indices_list])
    
    if scatter_mean:
        std_bin = np.array([np.nanstd(mean[idx]) for idx in indices_list])
        # band_bin = np.array([(np.nanmedian(sigma[idx])**2+np.nanstd(mean[idx])**2)**(1/2) for idx in indices_list])
        band_bin = np.array([np.nanmedian(sigma[idx]) for idx in indices_list])
        # plt.fill_between(x_bin,-2*band_bin,2*band_bin,color=lcolor,alpha=0.2)

    if axs is None:
        plt.errorbar(x_bin,y_bin,yerr=std_bin,color=lcolor,label=label)
    else:
        axs.errorbar(x_bin,y_bin,yerr=std_bin,color=lcolor,label=label)

def get_curves(array):
    y_bin = np.nanmedian(array,axis=0)
    cb_l = np.nanpercentile(array,75,axis=0)
    cb_h = np.nanpercentile(array,25,axis=0)

    return y_bin, cb_l, cb_h

def get_galaxy_density(radii,pmem,density=True):
    rvec = np.linspace(0.075,1.,13)#np.logspace(np.log10(0.05),np.log10(1.0), 10)
    #rvec = np.linspace(-0.15,+.15,60)
    area = np.ones_like(rvec[1:])

    if density: area = np.pi*(rvec[1:]**2-rvec[:-1]**2)

    indices, radii_bin = makeBin(radii, xvec=rvec)
    ng = np.array([np.nansum(pmem[idx]) for idx in indices])/area

    return radii_bin, ng

def get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem,pmem2,density=True):
    ng, ng2 = [], []
    for w,w2 in zip(indices,indices2):
        radii_bin, ngals = get_galaxy_density(radii[w],pmem[w],density=density)
        radii_bin2, ngals2 = get_galaxy_density(radii2[w2], pmem2[w2],density=density)

        ng.append(ngals)
        ng2.append(ngals2)
    
    return radii_bin, np.array(ng), np.array(ng2)

def get_sum_clusters(indices,indices2,x,x2,pmem,pmem2,xlims=(-0.15,0.15,30)):
    xedges = np.linspace(xlims[0],xlims[1],xlims[2]+1)
    s, s2 = [], []
    
    for w,w2 in zip(indices,indices2):
        indices_bin, x_bin = makeBin(x[w], xvec=xedges)
        indices_bin2, _ = makeBin(x2[w2], xvec=xedges)

        sum_g = np.array([np.nansum(pmem[idx]) for idx in indices_bin])
        sum_g2 = np.array([np.nansum(pmem2[idx]) for idx in indices_bin2])

        s.append(sum_g)
        s2.append(sum_g2)

    return x_bin, np.array(s), np.array(s2)

def get_pdf_clusters(indices,indices2,x,x2,pmem,xlims=(-0.15,0.15,30)):
    xedges = np.linspace(xlims[0],xlims[1],xlims[2]+1)
    _, x_bin = makeBin(x, xvec=xedges)

    x_bin = np.array(x_bin)

    p, p2 = [], []
    for w,w2 in zip(indices,indices2):
        pdf = get_pdf(x_bin,x[w],weights=pmem[w])
        pdf2 = get_pdf(x_bin,x2[w2])

        p.append(pdf)
        p2.append(pdf2)

    return x_bin, np.array(p), np.array(p2)

def get_pdf(xvec,x,weights=None):    
    if weights is not None:
        kernel = kde.gaussian_kde(x,weights=weights,bw_method='silverman')
    else:
        kernel = kde.gaussian_kde(x,bw_method='silverman')
    return kernel(xvec)

def plot_validation_pdf_redshift_triple(gal,gal2,cat,cat2,save='./img/pdf_redshift_validation_triple.png'):
    indices = list(chunks(gal['CID'],cat['CID']))
    indices2 = list(chunks(gal2['CID'],cat['CID']))

    z = zoffset(gal['z'],gal['redshift'])
    z2 = zoffset(gal2['z'],gal2['redshift'])
    
    smass = gal['mass']
    smass2 = gal2['mass']

    # pmem = gal['Pz']*gal['Pr']
    pmem = gal['Pmem']
    pmem_mu = pmem*10**(smass)/10**12

    # pmem2 = gal2['Pz']*gal2['Pr']
    pmem2 = gal2['Pmem']
    pmem_mu_2 = pmem2*10**(smass2)/10**12

    ### curve 1
    x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,z,z2,pmem,xlims=(-0.25,0.25,30))
    y_bin, yb_l, yb_h  = get_curves(pdf)
    y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

    per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
    # per_error = (pdf/pdf2)
    residual, cb_l, cb_h  = get_curves(per_error)

    ### curve 2
    x_bin2, ng, ng2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem,pmem2,xlims=(0.,0.25,30))
    label1, label2 = r'$N_{gals} $', r'$N_{gals} / N_{gals,true}$'
    ng, ng2 = np.cumsum(ng,axis=1), np.cumsum(ng2,axis=1)
    ny_bin, nyb_l, nyb_h  = get_curves(ng)
    ny_bin2, nyb_l2, nyb_h2  = get_curves(ng2)

    nper_error = np.where(ng/ng2>1000,0.,(ng-ng2)/ng2)
    # sper_error = (sg/sg2)
    nresidual, ncb_l, ncb_h  = get_curves(nper_error)

    # else:
    x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem_mu,pmem_mu_2,xlims=(0.,0.25,30))
    label3, label4 = r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$', r'$\mu_{\star} / \mu_{\star,true}$'
    
    sg, sg2 = np.cumsum(sg,axis=1), np.cumsum(sg2,axis=1)
    sy_bin, syb_l, syb_h  = get_curves(sg)
    sy_bin2, syb_l2, syb_h2  = get_curves(sg2)

    sper_error = np.where(sg/sg2>1000,0.,(sg-sg2)/sg2)
    # sper_error = (sg/sg2)
    sresidual, scb_l, scb_h  = get_curves(sper_error)

    fig = plt.figure(figsize=(10,6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.05,
                        height_ratios=[2,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[4])
    ax5 = plt.subplot(gs[2])
    ax6 = plt.subplot(gs[5])

    ## curve 1
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

    ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax1.legend()

    ax3.scatter(x_bin,residual,color='b',marker='s',s=20)
    ax3.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')

    ## curve 2
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax2.scatter(x_bin2,ny_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax2.fill_between(x_bin2, nyb_l, nyb_h, color="b", alpha=0.25, label='_nolabel_')

    ax2.scatter(x_bin2,ny_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax2.fill_between(x_bin2, nyb_l2, nyb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax4.scatter(x_bin2,nresidual,color='b',marker='s',s=20)
    ax4.fill_between(x_bin2,ncb_l,ncb_h,color="b", alpha=0.25, label='_nolabel_')

    
    ## cuve 3
    ax5.scatter(x_bin2,sy_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax5.fill_between(x_bin2, syb_l, syb_h, color="b", alpha=0.25, label='_nolabel_')

    ax5.scatter(x_bin2,sy_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax5.fill_between(x_bin2, syb_l2, syb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax6.scatter(x_bin2,sresidual,color='b',marker='s',s=20)
    ax6.fill_between(x_bin2,scb_l,scb_h,color="b", alpha=0.25, label='_nolabel_')

    ax3.set_ylim(-0.5,.5)
    # ax4.set_ylim(-0.2,1.)

    # ax3.set_xlabel(xlabel,fontsize=16)
    ax3.set_ylabel('perc. error',fontsize=16)
    xlabel=r'$(z-z_{cls})/(1+z_{cls})$'
    ax3.set_xlabel(xlabel,fontsize=16)
    ax6.set_xlabel(r'$\|z-z_{cls}\|/(1+z_{cls})$)',fontsize=16)
    ax4.set_xlabel(r'$\|z-z_{cls}\|/(1+z_{cls})$',fontsize=16)
    # ax4.set_ylabel('perc. error',fontsize=16)

    ax1.set_ylabel(r'$PDF(z)$',fontsize=16)
    ax2.set_ylabel(label1,fontsize=16)
    ax5.set_ylabel(label3,fontsize=16)

    plt.suptitle('Weighted by Pz')
    plt.savefig(save)
    plt.clf()


def plot_validation_pdf_redshift(gal,gal2,cat,cat2,method='N',save='./img/pdf_redshift_validation.png'):
    indices = list(chunks(gal['CID'],cat['CID']))
    indices2 = list(chunks(gal2['CID'],cat['CID']))

    z = zoffset(gal['z'],gal['redshift'])
    z2 = zoffset(gal2['z'],gal2['redshift'])

    # pmem = gal['Pz']*gal['Pr']
    pmem = gal['Pmem']

    # pmem2 = gal2['Pz']*gal2['Pr']
    pmem2 = gal2['Pmem']

    ### curve 1
    x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,z,z2,pmem,xlims=(-0.25,0.25,30))
    y_bin, yb_l, yb_h  = get_curves(pdf)
    y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

    per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
    # per_error = (pdf/pdf2)
    residual, cb_l, cb_h  = get_curves(per_error)

    ### curve 2
    if method=='N':
        x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem,pmem2,xlims=(0.,0.25,30))
        label1, label2 = r'$N_{gals} $', r'$N_{gals} / N_{gals,true}$'

    else:
        pmem_mu = pmem*10**(smass)/10**12
        pmem_mu_2 = pmem2*10**(smass2)/10**12

        smass = gal['mass']
        smass2 = gal2['mass']

        x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem_mu,pmem_mu_2,xlims=(0.,0.25,30))
        label1, label2 = r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$', r'$\mu_{\star} / \mu_{\star,true}$'
    
    sg, sg2 = np.cumsum(sg,axis=1), np.cumsum(sg2,axis=1)
    sy_bin, syb_l, syb_h  = get_curves(sg)
    sy_bin2, syb_l2, syb_h2  = get_curves(sg2)

    sper_error = np.where(sg/sg2>1000,0.,(sg-sg2)/sg2)
    # sper_error = (sg/sg2)
    sresidual, scb_l, scb_h  = get_curves(sper_error)

    fig = plt.figure(figsize=(8,6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.05,
                        height_ratios=[3,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ## curve 1
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

    ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax1.legend()

    ax3.scatter(x_bin,residual,color='b',marker='s',s=20)
    ax3.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')

    ## curve 2
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax2.scatter(x_bin2,sy_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax2.fill_between(x_bin2, syb_l, syb_h, color="b", alpha=0.25, label='_nolabel_')

    ax2.scatter(x_bin2,sy_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax2.fill_between(x_bin2, syb_l2, syb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax2.legend()

    ax4.scatter(x_bin2,sresidual,color='b',marker='s',s=20)
    ax4.fill_between(x_bin2,scb_l,scb_h,color="b", alpha=0.25, label='_nolabel_')

    ax3.set_ylim(-0.5,.5)
    # ax4.set_ylim(-0.2,1.)

    # ax3.set_xlabel(xlabel,fontsize=16)
    ax3.set_ylabel('perc. error',fontsize=16)
    xlabel=r'$(z-z_{cls})/(1+z_{cls})$'
    ax3.set_xlabel(xlabel,fontsize=16)
    ax4.set_xlabel(r'$\Delta (z-z_{cls})/(1+z_{cls})$',fontsize=16)
    ax4.set_ylabel('perc. error',fontsize=16)

    ax1.set_ylabel(r'$PDF(z)$',fontsize=16)
    ax2.set_ylabel(label1,fontsize=16)

    plt.suptitle('Weighted by Pmem')
    plt.savefig(save)
    plt.clf()

def plot_validation_pdf_color(gal,gal2,cat,cat2,method='N',save='./img/pdf_color_validation.png'):
    indices = list(chunks(gal['CID'],cat['CID']))
    indices2 = list(chunks(gal2['CID'],cat['CID']))

    color = gal['delta_rs']
    color2 = gal2['delta_rs']

    # pmem = gal['Pz']#*gal['Pr']
    pmem = gal['Pmem']

    # pmem2 = gal2['Pz']#*gal2['Pr']
    pmem2 = gal2['Pmem']

    ### curve 1
    x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,color,color2,pmem,xlims=(-1.5,0.5,30))
    y_bin, yb_l, yb_h  = get_curves(pdf)
    y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

    per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
    # per_error = (pdf/pdf2)
    residual, cb_l, cb_h  = get_curves(per_error)

    ### curve 2
    if method=='N':
        x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,color,color2,pmem,pmem2,xlims=(-1.5,0.5,30))
        label1, label2 = r'$N_{gals} $', r'$N_{gals} / N_{gals,true}$'

    else:
        smass = gal['mass']
        smass2 = gal2['mass']
        
        pmem_mu = pmem*10**(smass)/10**12
        pmem_mu_2 = pmem2*10**(smass2)/10**12

        x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,color,color2,pmem_mu,pmem_mu_2,xlims=(-1.5,0.5,30))
        label1, label2 = r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$', r'$\mu_{\star} / \mu_{\star,true}$'

    sg, sg2 = np.cumsum(sg,axis=1), np.cumsum(sg2,axis=1)
    sy_bin, syb_l, syb_h  = get_curves(sg)
    sy_bin2, syb_l2, syb_h2  = get_curves(sg2)

    sper_error = np.where(sg/sg2>1000,0.,(sg-sg2)/sg2)
    # sper_error = (sg/sg2)
    sresidual, scb_l, scb_h  = get_curves(sper_error)

    fig = plt.figure(figsize=(8,6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.05,
                        height_ratios=[3,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ## curve 1
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

    ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax1.legend()

    ax3.scatter(x_bin,residual,color='b',marker='s',s=20)
    ax3.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')
    ax3.axvline(0.,linestyle='--')
    
    ## curve 2
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax2.scatter(x_bin2,sy_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax2.fill_between(x_bin2, syb_l, syb_h, color="b", alpha=0.25, label='_nolabel_')

    ax2.scatter(x_bin2,sy_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax2.fill_between(x_bin2, syb_l2, syb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax2.legend()

    ax4.scatter(x_bin2,sresidual,color='b',marker='s',s=20)
    ax4.fill_between(x_bin2,scb_l,scb_h,color="b", alpha=0.25, label='_nolabel_')

    # ax3.set_ylim(-0.5,.5)
    ax3.set_ylim(-1.,1.)
    ax4.set_ylim(-0.2,1.)

    # ax3.set_xlabel(xlabel,fontsize=16)
    ax3.set_ylabel('perc. error',fontsize=16)
    xlabel=r'$\Delta(g-r)_{RS}$'
    ax3.set_xlabel(xlabel,fontsize=16)
    ax4.set_xlabel(xlabel,fontsize=16)

    # ax1.set_ylabel(r'$PDF(R,z,color)$',fontsize=16)
    ax1.set_ylabel(r'$PDF(color)$',fontsize=16)
    ax2.set_ylabel(label1,fontsize=16)

    plt.suptitle('Weighted by Pmem')
    plt.savefig(save)
    plt.clf()

def plot_validation_pdf_radial(gal,gal2,cat,cat2, xlabel = r'R $[Mpc]$', method='Mu', save='./img/pdf_radial_validation.png'):
    indices = list(chunks(gal['CID'],cat['CID']))
    indices2 = list(chunks(gal2['CID'],cat['CID']))

    radii = gal['R']
    radii2 = gal2['R']

    # pmem = gal['Pz']*gal['Pr']
    pmem = gal['Pmem']
    
    # pmem2 = gal2['Pz']*gal2['Pr']
    pmem2 = gal2['Pmem']

    ### curve 1
    radii_bin, ng, ng2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem,pmem2)
    y_bin, yb_l, yb_h  = get_curves(ng)
    y_bin2, yb_l2, yb_h2  = get_curves(ng2)

    # per_error = (ng-ng2)/ng2
    per_error = (ng/ng2)
    residual, cb_l, cb_h  = get_curves(per_error)
    x_bin = np.array(radii_bin)

    ### curve 2
    if method=='N':
        radii_bin, sg, sg2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem,pmem2,density=False)
        label1, label2 = r'$N_{gals} $', r'$N_{gals} / N_{gals,true}$'

    else:
        smass = gal['mass']
        smass2 = gal2['mass']

        pmem_mu = pmem*10**(smass)/10**12
        pmem_mu_2 = pmem2*10**(smass2)/10**12

        radii_bin, sg, sg2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem_mu,pmem_mu_2,density=False)
        label1, label2 = r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$', r'$\mu_{\star} / \mu_{\star,true}$'

    sg,sg2 = np.cumsum(sg,axis=1), np.cumsum(sg2,axis=1)
    sy_bin, syb_l, syb_h  = get_curves(sg)
    sy_bin2, syb_l2, syb_h2  = get_curves(sg2)

    # sper_error = (sg-sg2)/sg2
    sper_error = (sg/sg2)
    sresidual, scb_l, scb_h  = get_curves(sper_error)

    fig = plt.figure(figsize=(8,6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.05,
                        height_ratios=[3,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ## curve 1
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

    ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax1.legend()

    ax3.scatter(x_bin,residual,color='b',marker='s',s=20)
    ax3.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')

    ## curve 2
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax2.scatter(x_bin,sy_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
    ax2.fill_between(x_bin, syb_l, syb_h, color="b", alpha=0.25, label='_nolabel_')

    ax2.scatter(x_bin,sy_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
    ax2.fill_between(x_bin, syb_l2, syb_h2, color="r", alpha=0.25, label='_nolabel_')

    ax2.legend()

    ax4.scatter(x_bin,sresidual,color='b',marker='s',s=20)
    ax4.fill_between(x_bin,scb_l,scb_h,color="b", alpha=0.25, label='_nolabel_')

    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax3.set_xscale('log')
    ax4.set_xscale('log')

    ax1.set_xticks([0.1,0.3,0.5,0.7,1.0])
    ax3.set_xticks([0.1,0.3,0.5,0.7,1.0])
    # ax3.set_yticks([-1.,-0.5,0.,0.5,1.])
    # ax3.set_yticks([-0.25,-0.15,0.,0.15,.25])

    ax2.set_xticks([0.1,0.3,0.5,0.7,1.0])
    ax4.set_xticks([0.1,0.3,0.5,0.7,1.0])
    # ax4.set_yticks([-1.,-0.5,0.,0.5,1.])
    # ax4.set_yticks([-0.25,-0.15,0.,0.15,.25])

    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set_ylim(0.5,1.5)

    ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax4.set_ylim(0.5,1.5)

    ax3.set_xlabel(xlabel,fontsize=16)
    ax3.set_ylabel(r'$\Sigma / \Sigma_{true}$',fontsize=16)

    ax4.set_xlabel(xlabel,fontsize=16)
    ax4.set_ylabel(label2,fontsize=16)

    ax1.set_ylabel(r'$\Sigma \; [\# gals / Mpc^{2}]$',fontsize=16)
    ax2.set_ylabel(label1,fontsize=16)

    plt.savefig(save)
    plt.clf()

def validating_color_model_residual(lcolor='g-r'):
    #### Color plots
    color = gal[lcolor]
    zrs = cat['redshift']

    color2 = gal2[lcolor]
    zrs2 = cat2['redshift']

    mur = cat['rs_param_%s'%(lcolor)][:,0]
    sigr = cat['rs_param_%s'%(lcolor)][:,1]

    mub = cat['bc_param_%s'%(lcolor)][:,0]
    sigb = cat['bc_param_%s'%(lcolor)][:,1]

    mur2 = cat2['rs_param_%s'%(lcolor)][:,0]
    sigr2 = cat2['rs_param_%s'%(lcolor)][:,1]

    mub2 = cat2['bc_param_%s'%(lcolor)][:,0]
    sigb2 = cat2['bc_param_%s'%(lcolor)][:,1]


    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2, 2, wspace=0.30, hspace=0.04, height_ratios=[3,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])

    ax3 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])

    ax1.set_title('Color Mean')
    # scale = 10*pmem**(1/2)
    # plt.scatter(zcls,color,s=scale,color='k',alpha=0.08)
    # plt.scatter(zcls2,color2,color='k',alpha=0.08)
    ax1.scatter(zrs,mur,color='r',alpha=0.5,s=10,label='RS: mean')
    ax1.scatter(zrs,mub,color='b',alpha=0.5,s=10,label='BC: mean')
    
    plot_color_bin(zrs2,mur2,sigr2,dz=0.05,lcolor='#F1948A',axs=ax1)
    plot_color_bin(zrs2+0.025,mub2,sigb2,dz=0.05,label='BC Model',lcolor='#85C1E9',axs=ax1)

    ax2.scatter(zrs2,(mur-mur2),color='r',alpha=0.5,s=5)
    ax2.scatter(zrs2,(mub-mub2),color='b',alpha=0.5,s=5)    
    plot_color_bin(zrs2,(mur-mur2),sigr2,dz=0.05,lcolor='#F1948A',axs=ax2,scatter_mean=True)
    plot_color_bin(zrs2+0.025,(mub-mub2),sigb2,dz=0.05,label='BC Model',lcolor='#85C1E9',axs=ax2,scatter_mean=True)

    ax1.legend()
    ax2.set_ylim(-0.5,0.5)
    ax2.set_xlim(0.08,0.92)
    ax1.set_xlim(0.08,0.92)
    ax1.set_ylim(0.,np.max(mur)+2*np.max(sigr))

    ax2.set_xlabel('redshift',fontsize=16)
    ax1.set_ylabel(r'$(%s)$'%(lcolor),fontsize=16)
    ax2.set_ylabel('residual [mag]',fontsize=16)

    ####### Second Pannel
    ax3.set_title('Gaussian Width')
    ax3.scatter(zrs,sigr,color='r',alpha=0.5,s=10)
    ax3.scatter(zrs,sigb,color='b',alpha=0.5,s=10)
    
    plot_color_bin(zrs2,sigr2,mur2,dz=0.05,lcolor='#F1948A',axs=ax3)
    plot_color_bin(zrs2+0.025,sigb2,mub2,dz=0.05,label='BC Model',lcolor='#85C1E9',axs=ax3)

    ax4.scatter(zrs2,(sigr-sigr2)/sigr2,color='r',alpha=0.5,s=5)
    ax4.scatter(zrs2,(sigb-sigb2)/sigb2,color='b',alpha=0.5,s=5)    
    plot_color_bin(zrs2,(sigr-sigr2)/sigr2,sigr2,dz=0.05,lcolor='#F1948A',axs=ax4,scatter_mean=True)
    plot_color_bin(zrs2+0.025,(sigb-sigb2)/sigb2,sigb2,dz=0.05,lcolor='#85C1E9',axs=ax4,scatter_mean=True)

    ax3.legend()
    ax4.set_ylim(-0.75,0.75)
    # ax2.set_xlim(0.08,0.92)
    # ax1.set_xlim(0.08,0.92)
    # ax1.set_ylim(0.,np.max(mur)+2*np.max(sigr))

    ax4.set_xlabel('redshift',fontsize=16)
    ax3.set_ylabel(r'$\sigma_{%s}$'%(lcolor),fontsize=16)
    ax4.set_ylabel('frac. error',fontsize=16)
    
    fig.savefig('img/color_model_%s.png'%(lcolor))
    plt.close()

    # sns.kdeplot(zcls,color,color='b', shade=True, zorder=3, shade_lowest=False, alpha=0.6)
    # plt.hexbin(zcls, color, pmem, gridsize=100, cmap=plt.cm.get_cmap('Blues_r', 45), reduce_C_function=np.sum, vmin=-10, vmax=100) 

    # 
    # plt.plot(x_bin,cb_u,color='r')

def validating_color_model(lcolor='g-r'):
    #### Color plots
    color = gal[lcolor]
    zrs = cat['redshift']

    color2 = gal2[lcolor]
    zrs2 = cat2['redshift']

    mur = cat['rs_param_%s'%(lcolor)][:,0]
    sigr = cat['rs_param_%s'%(lcolor)][:,1]

    mub = cat['bc_param_%s'%(lcolor)][:,0]
    sigb = cat['bc_param_%s'%(lcolor)][:,1]

    mur2 = cat2['rs_param_%s'%(lcolor)][:,0]
    sigr2 = cat2['rs_param_%s'%(lcolor)][:,1]

    mub2 = cat2['bc_param_%s'%(lcolor)][:,0]
    sigb2 = cat2['bc_param_%s'%(lcolor)][:,1]

    # scale = 10*pmem**(1/2)
    # plt.scatter(zcls,color,s=scale,color='k',alpha=0.08)
    # plt.scatter(zcls2,color2,color='k',alpha=0.08)
    plt.scatter(zrs,mur,color='r',alpha=0.5,s=10,label='RS: mean')
    plt.scatter(zrs,mub,color='b',alpha=0.5,s=10,label='BC: mean')
    
    # plot_color_bin(zrs,mur,sigr,lcolor='#E74C3C')
    # plot_color_bin(zrs,mub,sigb,label='blue cloud',lcolor='#3498DB')
    
    plot_color_bin(zrs2,mur2,sigr2,label='True Distritbution',lcolor='#F1948A')
    plot_color_bin(zrs2,mub2,sigb2,label='True Distritbution',lcolor='#85C1E9')
    
    plt.legend()
    # plt.xlim(0.08,0.92)
    plt.ylim(0.,np.max(mur)+2*np.max(sigr))
    plt.xlabel('redshift',fontsize=16)
    plt.ylabel(r'$(%s)$'%(lcolor),fontsize=16)
    plt.savefig('color_model_%s.png'%(lcolor))
    plt.close()

    # sns.kdeplot(zcls,color,color='b', shade=True, zorder=3, shade_lowest=False, alpha=0.6)
    # plt.hexbin(zcls, color, pmem, gridsize=100, cmap=plt.cm.get_cmap('Blues_r', 45), reduce_C_function=np.sum, vmin=-10, vmax=100) 

    # 
    # plt.plot(x_bin,cb_u,color='r')

def validating_color_model_grid(color_list):
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace=0.03, wspace=0.25)
    
    for i,li in enumerate(color_list):
        axs = fig.add_subplot(2, 3, i+1)
        
        #### Color plots
        color = gal[li]
        zrs = cat['redshift']

        mur = cat['rs_param_%s'%(li)][:,0]
        sigr = cat['rs_param_%s'%(li)][:,1]

        mub = cat['bc_param_%s'%(li)][:,0]
        sigb = cat['bc_param_%s'%(li)][:,1]

        scale = 10*pmem**(1/2)
        # axs.scatter(zcls,color,s=scale,color='k',alpha=0.08)
        axs.scatter(zrs,mur,color='r',alpha=0.4,s=10,label='RS: mean')
        axs.scatter(zrs,mub,color='b',alpha=0.4,s=10,label='BC: mean')
        plot_color_bin(zrs,mur,sigr,lcolor='#F1948A')
        plot_color_bin(zrs,mub,sigb,label='blue cloud',lcolor='#85C1E9')
        # plt.xlim(0.08,0.92)
        axs.set_ylim(0.,np.max(mur)+1.5*np.max(sigr))

        if i>=3:
            axs.set_xlabel('redshift',fontsize=16)

        axs.set_ylabel(r'$(%s)$'%(li),fontsize=16)

    fig.savefig('./img/color_model_grid.png', bbox_inches = "tight")
    plt.close()

def estimatePDF2(z_new,z,zcls,weights,gidx):
    res, res2 = [], []
    for i in range(len(gidx)-1):
        w = np.arange(gidx[i],gidx[i+1],dtype=int)
        true_i = gal['True'][w] == True
        
        kernel2 = kde.gaussian_kde(zoffset(z[w],zcls[w]),weights=weights[w],bw_method='silverman')
        kernel = kde.gaussian_kde(zoffset(z[w[true_i]],zcls[w[true_i]]),bw_method='silverman')
        
        r1 = kernel(z_new)
        r2 = kernel2(z_new)
        
        res.append(r1)
        res2.append(r2)
   
    return res,res2

def getPDF(xnew,x,weights,gidx):
    res = []
    for i in range(len(gidx)-1):
        w = np.arange(gidx[i],gidx[i+1],dtype=int)

        if weights is not None:
            kernel = kde.gaussian_kde(x[w],weights=weights[w],bw_method='silverman')
        else:
            kernel = kde.gaussian_kde(x[w],bw_method='silverman')
        
        res.append(kernel(xnew))
    return np.array(res).transpose()

def bin_pdf(x,pdf,gidx,method='z'):
    zvec = np.arange(-0.175,0.175,0.005)
    
    if method=='color':
        zvec = np.arange(0.2,2.75,0.01)

    x_bin = (zvec[1:]+zvec[:-1])/2

    r1, r2 = [], []
    for i in range(len(gidx)-1):
        w = np.arange(gidx[i],gidx[i+1],dtype=int)
        indices_list = list(chunks2(x[w], zvec))

        pdfi = pdf[w]
        y_bin = np.array([np.nanmedian(pdfi[idx]) for idx in indices_list])
        y_bin_std = np.array([np.nanstd(pdfi[idx]) for idx in indices_list])

        r1.append(y_bin)
        r2.append(y_bin_std)
    
    r1 = np.array(r1).transpose()
    r2 = np.array(r2).transpose()

    return x_bin, r1, r2

def plot_validation_pdf(pdf,pdf2,ntrue,pz,gidx,gidx2,save='./img/pdf_z_validation.png',method='z'):
    
    if method=='z':
        x_bin, y_bin_list, y_bin_std_list = bin_pdf(zoffset(z,zcls),pdf,gidx)
        xtrue_bin, ytrue_bin_list, ytrue_bin_std_list = bin_pdf(zoffset(z2,zcls2),pdf2,gidx2)

    else:
        x_bin, y_bin_list, y_bin_std_list = bin_pdf(color,pdf,gidx,method='color')
        xtrue_bin, ytrue_bin_list, ytrue_bin_std_list = bin_pdf(color2,pdf2,gidx2,method='color')
        # y_bin_list =  getPDF(x_bin,color2,None,gidx2)

    y_bin = np.nanmedian(y_bin_list,axis=1)
    y_bin_std = np.nanmedian(y_bin_std_list,axis=1)

    ytrue_bin = 0*np.nanmedian(ytrue_bin_list,axis=1)
    ytrue_bin_std = 0*(np.nanstd(ytrue_bin_list,axis=1)**2 + np.nanmedian(ytrue_bin_std_list,axis=1)**2)**(1/2)
    
    residual = np.nanmedian(y_bin_list-ytrue_bin_list,axis=1)

    std_res = 0*np.nanstd(y_bin_list-ytrue_bin_list,axis=1)
    cb_ur, cb_lr = (std_res), (-std_res)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.05,
                        height_ratios=[2,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    cb_u, cb_l = (y_bin+y_bin_std), (y_bin-y_bin_std)
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.plot(x_bin,y_bin,color='b', label='Estimated Distrub.')
    ax1.fill_between(x_bin, cb_l, cb_u, color="b", alpha=0.25, label='_nolabel_')

    cb_u, cb_l = (ytrue_bin+ytrue_bin_std), (ytrue_bin-ytrue_bin_std)
    ax1.plot(x_bin,ytrue_bin,color='r',alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, cb_l, cb_u, color="r", alpha=0.25, label='_nolabel_')

    if method=='z':
        ax1.set_ylabel('PDF(z)')
        xlabel = r'$(z-z_{cls})/(1+z_{cls})$'
        vec='z'
    else:
        ax1.set_ylabel('PDF(color)')
        xlabel = '(gi_o)'
        vec = 'color'
    
    ax1.legend()

    residual = y_bin-ytrue_bin
    cb_u, cb_l = (ytrue_bin_std), (-ytrue_bin_std)
    # residual_err = (np.array(y_bin_std)**2+np.array(ytrue_bin_std)**2)**(1/2)
    
    ax3.plot(x_bin,residual,color='b')
    # ax3.fill_between(x_bin,-y_bin_std,+y_bin_std,color='b',alpha=0.25, label='_nolabel_')
    ax3.fill_between(x_bin,cb_lr,cb_ur,color="b", alpha=0.25, label='_nolabel_')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('residual')

    sum_pz = np.array([np.nansum(pz[np.arange(gidx[i],gidx[i+1],dtype=int)]) for i in range(len(gidx)-1)])
    # ntrue = np.array([len(pdf2[np.arange(gidx2[i],gidx2[i+1],dtype=int)] == True) for i in range(len(gidx2)-1)])

    ax2.scatter(ntrue,sum_pz,s=10,color='b',alpha=0.3)
    ax2.plot(ntrue,ntrue,color='r')
    ax2.set_ylabel(r'$\sum p(%s)$'%(vec))

    residual = np.log10((sum_pz/ntrue))
    ax4.scatter(ntrue,residual,color='b',alpha=0.3,s=10,label='scatter = %.2f'%(np.nanstd(residual)))
    ax4.axhline(np.mean(residual),linestyle='--',color='b',label='mean = %.2f'%(np.nanmean(residual)))
    ax4.set_ylabel(r'$\log (\Sigma p_{%s} / N_{true} )$'%(vec))
    ax4.set_xlabel(r'$N_{true}$')
    ax4.legend()

    plt.savefig(save)
    plt.clf()

def plot_pdf_validation(x1,x2,pz,gidx,gidx2,ntrue,method='z', save='./img/pdf_z_validation.png'):
    if method=='z':
        xbins_linspace= (-0.25,0.25,60)
    else:
        xbins_linspace= (-1.75,0.5,100)

    ### curve 1
    x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,x1,x2,pz,xlims=xbins_linspace)
    y_bin, cb_l, cb_u  = get_curves(pdf)
    y_bin2, cb_l2, cb_u2  = get_curves(pdf2)

    per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
    # per_error = (pdf/pdf2)
    residual, cb_lr, cb_ur  = get_curves(per_error)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.05,
                        height_ratios=[2,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    # cb_u, cb_l = (y_bin+y_bin_std), (y_bin-y_bin_std)
    # ax1.scatter(zoffset(z,zcls),pdf,color='b',s=2,alpha=0.2)
    ax1.plot(x_bin,y_bin,color='b', label='Estimated Distrub.')
    ax1.fill_between(x_bin, cb_l, cb_u, color="b", alpha=0.25, label='_nolabel_')

    # cb_u2, cb_l2 = (y_bin2+y_bin_std2), (y_bin2-y_bin_std2)
    ax1.plot(x_bin,y_bin2,color='r',alpha=0.9, label='True Distrub.')
    ax1.fill_between(x_bin, cb_l2, cb_u2, color="r", alpha=0.25, label='_nolabel_')

    if method=='z':
        ax1.set_ylabel('PDF(z)')
        xlabel = r'$(z-z_{cls})/(1+z_{cls})$'
        vec='z'
    else:
        ax1.set_ylabel('PDF(R,z,color)')
        xlabel = r'$\Delta(g-r)_{RS}$'
        vec = 'z,color'
    
    ax1.legend()

    # residual = y_bin-y_bin2
    # cb_u, cb_l = (y_bin2_std), (-y_bin2_std)
    # residual_err = (np.array(y_bin_std)**2+np.array(y_bin2_std)**2)**(1/2)
    
    ax3.plot(x_bin,residual,color='b')
    # ax3.fill_between(x_bin,-y_bin_std,+y_bin_std,color='b',alpha=0.25, label='_nolabel_')
    ax3.fill_between(x_bin,cb_lr,cb_ur,color="b", alpha=0.25, label='_nolabel_')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('perc. error')

    # sum_pz = np.array([np.nansum(pz[np.arange(gidx[i],gidx[i+1],dtype=int)]) for i in range(len(gidx)-1)])
    sum_pz = np.array([np.nansum(pz[idx]) for idx in gidx])
    # ntrue = np.array([len(pdf2[np.arange(gidx2[i],gidx2[i+1],dtype=int)] == True) for i in range(len(gidx2)-1)])

    ax2.scatter(ntrue,sum_pz,s=10,color='b',alpha=0.3)
    ax2.plot(ntrue,ntrue,color='r')
    ax2.set_ylabel(r'$\sum p(%s)$'%(vec))

    residual = np.log10((sum_pz/ntrue))
    ax4.scatter(ntrue,residual,color='b',alpha=0.3,s=10,label='scatter = %.2f'%(np.nanstd(residual)))
    ax4.axhline(np.mean(residual),linestyle='--',color='b',label='mean = %.2f'%(np.nanmean(residual)))
    ax4.set_ylabel(r'$\log (\Sigma p_{%s} / N_{true} )$'%(vec))
    ax4.set_xlabel(r'$N_{true}$')

    ax3.set_ylim(-1.,1.)
    ax4.legend()

    plt.savefig(save)
    plt.clf()

def computeColorKDE(x,weight=None,bandwidth='silverman',silvermanFraction=None):
    from six import string_types
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


def plotColor(color,color2,pmem,lcolor,title='',save='bla.png',xlims=(-1.,3.),xmax=3,break4000=0.):
    x = np.arange(xlims[0],xlims[-1],0.0075)

    idx = np.argsort(color[color>=xlims[0]])
    color, pmem = color[idx], pmem[idx]
    kernel = computeColorKDE(color,weight=pmem,silvermanFraction=10)
    kde = kernel(x)

    color_true = np.sort(color2)
    kernel_true = computeColorKDE(color_true,silvermanFraction=10)
    kde_true = kernel_true(x)

    fig = plt.figure(figsize=(6,6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 1, wspace=0.35, hspace=0.05,
                        height_ratios=[3,1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.plot(x,kde,color='b',linewidth=2)
    ax1.fill_between(x,0,kde,color='b',alpha=0.2)
        
    ax1.plot(x,kde_true,label='True members',color='r',linewidth=2.)
    ax1.fill_between(x,0,kde_true,color='r',alpha=0.2)
    # ax1.axvline(break4000,linestyle='--',color='k',label='4000 $\angstron$ break')
    ax1.legend(loc='upper right')

    ax1.set_xlim(xlims)
    ax2.set_xlim(xlims)

    if np.max(kde_true)>xmax: xmax = round(np.max(kde_true))+1.
    ax1.set_ylim(0.,xmax)
    # ax1.set_ylim(0.,1.05)
    
    residual = (kde-kde_true)
    ax2.plot(x,residual,color='b',linewidth=2.)
    ax2.fill_between(x,0,residual,color='b',alpha=0.2)
    ax2.set_ylim(-0.5,0.5)

    ax1.set_ylabel('PDF(color)',fontsize=16)
    ax2.set_ylabel('residual',fontsize=16)
    ax2.set_xlabel(r'$(%s)$'%(lcolor),fontsize=16)
    
    ax1.xaxis.set_tick_params(labelsize=0.05, labelcolor='white')
    ax1.set_title(title)
    plt.savefig(save)
    plt.close()

def makeColorGif(gal,gal2,zrange,pmem,lcolor='r-i',xlims=(-1.,3.)):
    print('color %s and xlims = %.2f , %.2f'%(lcolor,xlims[0],xlims[1]))
    files = []
    for i in range(zrange.size-1):
        zi,zj = zrange[i],zrange[i+1]
        w = np.where( (z>=zi)&(z<=zj))
        w2 = np.where( (z2>=zi)&(z2<=zj))

        ti = r"Buzzard: $ %.3f < z < %.3f$"%(zi,zj)
        si = './img/gifs/%s_Buzzard_color_%02i.png'%(lcolor,i)
        color = gal[lcolor]
        color2 = gal2[lcolor]
        plotColor(color[w],color2[w2],pmem[w],lcolor,title=ti,save=si,xlims=xlims,break4000=(zi+zj)/2)
        files.append(si)

    os.system('convert -delay 55 -loop 10 %s img/animated_%s.gif'%(' '.join(files),lcolor))


###########################################
def getIndices(gindices,gkeys,ckeys):
    indicies = np.empty((0),dtype=int)
    indicies_into_cluster = np.empty((0),dtype=int)

    for i in range(ckeys.size):
        idx, = np.where(gkeys==ckeys[i])
        if idx.size>0:
            w2 = np.arange(gindices[idx],gindices[idx+1], 1, dtype=int)
            w = np.full(w2.size,i,dtype=int)

            indicies = np.append(indicies,w2)
            indicies_into_cluster = np.append(indicies_into_cluster,w)

    return indicies,indicies_into_cluster

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def filter_ngals(c,g,Ngals_min=5,col_label='Ngals_true'):
    c = c[ c[col_label] >= Ngals_min ]

    c = c[ np.where(np.logical_not(np.isnan(c[col_label]))) ]

    g = g.group_by('CID')
    gidx, cidx = getIndices(g.groups.indices,g.groups.keys['CID'],c['CID'])
    g = g[gidx]
    return c,g

###########################################
test = 0
clusterPlots = True
galaxyPlots = False
colorEvolution = 1
compareDatasets = True

color_label = 'gr'
###########################################
ii=0
print('Geting Data')

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members_stellarMass.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_stellarMass.fits' 

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_color_%s_members.fits'%(color_label)
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_color_%s.fits'%(color_label)

# file_gal2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table_members_stellarMass.fits'
# file_cls2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table_stellarMass.fits' 

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed.fits'

root='/home/johnny/Documents/realData/DESY1/redmapper/out/'
file_gal2 = root+'lgt5_vlim_copa_stellarMass_final_members.fits'
file_cls2 = root+'lgt5_vlim_copa_stellarMass_final_members.fits'

# file_gal2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table_members.fits'
# file_cls2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table.fits' 

cat_in = Table(getdata(file_cls))
gal = Table(getdata(file_gal))

print('Taking only cluster within 5 true members')
mask = (gal["R"]<=1.)#&(gal["Pmem"]>0.19)#&(gal["Pz"]>0.01)#&(gal["Pc"]>0.1)
gal = gal[mask]

cat, gal = filter_ngals(cat_in,gal,Ngals_min=5)

# cat2 = Table(getdata(file_cls2))
# gal2 = Table(getdata(file_gal2))

# mask = (gal2["R"]<=1.)
# gal2 = gal2[mask]

# cat2, gal2 = filter_ngals(cat2,gal2,Ngals_min=5,col_label='Ngals')
# idx = [np.where(cat2["CID"] == cidx)[0] for cidx in cat["CID"] if len(np.where(cat2["CID"] == cidx)[0])>0]
# idx = np.stack(idx).ravel()
# cat2 = cat2[idx]

print('Defining variables')
keys = cat['CID'][:]
redshift = cat['redshift'][:]
ngals = cat['Ngals'][:]
norm = cat['Norm'][:]
nbkg = cat['Nbkg'][:]
MU = cat['MU']
sum_ssfr = cat['SSFR']
sum_smass = cat['SUM_MASS']

# keys2 = cat2['CID'][:]
# redshift2 = cat2['redshift'][:]
# ngals2 = cat2['Ngals'][:]
# norm2 = cat2['Norm'][:]
# nbkg2 = cat2['Nbkg'][:]
# MU2 = cat2['MU']
# sum_ssfr2 = cat2['SSFR']
# sum_smass2 = cat2['SUM_MASS']

# Nred = cat['Nred']
# Nblue = cat['Nblue']
# red_fraction = Nred/(Nred+Nblue)

if clusterPlots:
    sky_plot(cat['RA'], cat['DEC'],title='Buzzard Simulation v1.6')
    plot_group_by_redshift(redshift, ngals, xtrue=redshift2, ytrue=ngals2, ylog=False, ylabel = r'N$_{gals}$', xlabel = r'redshift', save='./img/redshift_ngals.png')
    plot_group_by_redshift(redshift, norm, xtrue=redshift2, ytrue=norm2, ylog=False, ylabel = r'Norm', xlabel = r'redshift', save='./img/redshift_norm.png')
    plot_group_by_redshift(redshift, nbkg, ylim=(0.,100.),ylog=False, ylabel = r'N$_{bkg}$', xlabel = r'redshift', save='./img/redshift_nbkg.png')

    plot_scatter_hist(redshift, MU/100., xtrue=redshift2, ytrue=(MU2/100), save='./img/scale_mu_star.png')
    plot_scatter_hist(redshift, sum_ssfr, xtrue=redshift2, ytrue=sum_ssfr2, ylabel = r'$\Sigma (sSFR)$', xlabel = r'redshift', save='./img/scale_ssfr.png')
    plot_scatter_hist(redshift, sum_smass, xtrue=redshift2, ytrue=sum_smass2, ylabel = r'$\Sigma (M_{*}) \,\,[10^{12}\,M_{\odot}]$', save='./img/scale_sum_smass.png')
    plot_scatter_hist(redshift, ngals, ylabel = r'N$_{gals}$', xlabel = r'redshift', save='./img/scale_ngals.png')

print('starting galaxies related plots')
if galaxyPlots:
    mask = gal['True']==True

    z = gal['z']
    zcls = gal['redshift']
    pmem = gal['Pmem']
    amag = gal['rabs'][:]
    gi_o = gal['gi_o'][:]
    smass = gal['mass'][:]
    ssfr = np.where(gal['ssfr'][:]==0,0.,np.log10(gal['ssfr'][:]))

    z2 = gal2['z']
    zcls2 = gal2['redshift']
    pmem2 = gal2['Pmem']
    amag2 = gal2['rabs'][:]
    gi_o2 = gal2['gi_o'][:]
    smass2 = gal2['mass'][:]
    ssfr2 = np.where(gal2['ssfr'][:]==0,0.,np.log10(gal2['ssfr'][:]))


    plot_scatter_hist(zcls,amag,weights=pmem, xtrue=zcls2, ytrue=amag2, ylabel=r'$M_{r}$',save='./img/amag_evolution.png')
    plot_scatter_hist(zcls,smass,weights=pmem, xtrue=zcls2, ytrue=smass2, ylabel=r'$log(M_{*}) \: [M_{\odot}]$',save='./img/smass_evolution.png')
    plot_scatter_hist(zcls,ssfr,weights=pmem, xtrue=zcls2, ytrue=ssfr2, ylabel=r'$log(sSFR)$', save='./img/ssfr_evolution.png')
    plot_scatter_hist(zcls,gi_o,weights=pmem, xtrue=zcls2, ytrue=gi_o2, ylabel=r'$(g-i)_{rest-frame}$', save='./img/gi_rest_frame_evolution.png')

print('fim')
# ptaken = gal['Ptaken']

# def cut_colors_above_RS(gal,gmm_parameters,color_list=['g-r'],nsigma=2):
#     zcls = gal['redshift']
#     new_pmem = gal['Pmem']
#     z = gmm_parameters['redshift']

#     for li in color_list:
#         color = gal[li]
#         mur = gmm_parameters['rs_param_%s'%(li)][:,0]
#         sigr = gmm_parameters['rs_param_%s'%(li)][:,1]

#         b = np.arange(0.08,0.92,0.03)
#         indices_list = list(chunks2(z, b))

#         y_bin = np.array([ np.nanmedian(mur[idx]) for idx in indices_list])
#         std_bin = np.array([(np.nanmedian(sigr[idx])**2+np.nanstd(mur[idx])**2)**(1/2) for idx in indices_list])
#         x_bin = np.array([ np.median(z[idx]) for idx in indices_list])

#         from scipy import interpolate
#         cb_u = y_bin + nsigma*std_bin
#         cb_upper = interpolate.interp1d(x_bin,cb_u,kind='cubic',fill_value='extrapolate')(zcls)
        
#         new_pmem = np.where(color>=cb_upper,0.,new_pmem)

#     return new_pmem

# # new_pmem = cut_colors_above_RS(gal,cat,color_list=['g-r','r-i'])

# # new_pmem = pmem

# pz = gal['Pz']
# pcolor = gal['Pc']
# ptaken = gal['Ptaken']
# pz *= ptaken
# pcolor *= ptaken
# pr = gal['Pr']

# gt = gal.group_by('CID')
# gidx = gt.groups.indices   

# gt2 = gal2.group_by('CID')
# gidx2 = gt2.groups.indices   

# # npmem = gal['Pmem_new']

# import copa.gaussianKDE as kde

# def zoffset(z,zcls):
#     return (z-zcls)/(1+zcls)

# def estimatePDF(z_new,z,zcls,weights,gidx):
#     res = np.empty((0,),dtype=float)
#     res2 = np.empty((0,),dtype=float)
    
#     for i in range(len(gidx)-1):
#         w = np.arange(gidx[i],gidx[i+1],dtype=int)
#         true_i = gal['True'][w] == True

#         kernel = kde.gaussian_kde(zoffset(z[w[true_i]],zcls[w[true_i]]),bw_method='silverman')
#         kernel2 = kde.gaussian_kde(zoffset(z[w],zcls[w]),weights=weights[w],bw_method='silverman')
        
#         r1 = kernel(z_new[w])
#         r2 = kernel2(z_new[w])

#         res = np.append(res,r1)
#         res2 = np.append(res2,r2)
    
#     return res,res2

# # plot_pdf_validation(zoffset(z,zcls),zoffset(z2,zcls2),pz,gidx,gidx2,ngals,save='./img/pdf_z_validation.png')
# # plot_validation_pdf(pdfz,pdfz2,ngals2,pz,gidx,gidx2)

# # gal['gi_o'] = np.where(np.isnan(gal['gi_o']),0., gal['gi_o'])
# # gal2['gi_o'] = np.where(np.isnan(gal2['gi_o']),0., gal2['gi_o'])

# gal['g-r'] = gal['mag'][:,0]-gal['mag'][:,1]
# gal2['g-r'] = gal2['mag'][:,0]-gal2['mag'][:,1]

# color = gal['g-r']
# color2 = gal2['g-r']

# pdfc = gal['pdfs'][:,2]
# pdfc2 = gal2['pdfs'][:,2]

# # plotColor(gal['gi_o'],gal2['gi_o'],pmem,'gi_o',title='Buzzard Simulations v1.6: 0.1 < z < 0.9',save='./img/pdf_c_validation.png',xlims=(0.05,1.3),xmax=6.5)

# # zrange = np.arange(0.1,0.925,0.025)
# # color_list = ['gi_o','g-r','r-i','i-z']
# # xlims_list = [(0.1,1.3),(0.,2.75),(-0.25,2.),(-0.25,1.)]
# # for (li,xl) in zip(color_list,xlims_list): makeColorGif(gal,gal2,zrange,pmem,lcolor=li,xlims=xl)

# # color_list = color_list = ['gi_o','g-i','r-z','g-r','r-i','i-z']
# color_list = color_list = ['g-i','r-z','g-r','r-i','i-z']

# # # # validating_color_model_grid(color_list)
# # for li in color_list: validating_color_model_residual(lcolor=li)
# # w, = np.where(ngals<30)
# # w2, = np.where(ngals2<30)

# # indices = list(chunks(gal['CID'],cat['CID']))
# # indices2 = list(chunks(gal2['CID'],cat['CID']))

# # def get_delta_color(gal,cat,indices,lcolor='g-r'):
# #     gal['delta_rs'] = 0.
# #     color = gal['%s'%lcolor]
# #     for i,idx in enumerate(indices):
# #         rs_param = cat['rs_param_%s'%lcolor][i]
# #         ri = (color[idx]-rs_param[0])#/rs_param[1]
# #         if np.nanmedian(ri)>0.05: ri += np.nanmedian(ri)
# #         gal['delta_rs'][idx] = ri

# #     return gal['delta_rs']

# # color = get_delta_color(gal,cat,indices)
# # color2 = get_delta_color(gal2,cat2,indices2)

# # plot_pdf_validation(color,color2,pmem,indices,indices2,ngals2,method='color',save='./img/bla_pdf_c_validation.png')
# # plot_pdf_validation(zoffset(z,zcls),zoffset(z2,zcls2),pmem,indices,indices2,ngals2,save='./img/bla_pdf_z_validation.png')

# ## with Ngals
# plot_validation_pdf_radial(gal,gal2,cat,cat2,method='N',save='./img/pdfn_radial_validation.png')
# plot_validation_pdf_redshift(gal,gal2,cat,cat2,method='N',save='./img/pdfn_redshift_validation.png')
# # plot_validation_pdf_color(gal,gal2,cat,cat2,method='N',save='./img/pdfn_color_validation.png')

# # plot_validation_pdf_radial(gal,gal2,cat,cat2,method='N',save='./testColors/%s_pdfn_radial_validation.png'%(color_label))
# # plot_validation_pdf_redshift(gal,gal2,cat,cat2,method='N',save='./testColors/%s_pdfn_redshift_validation.png'%(color_label))
# # plot_validation_pdf_color(gal,gal2,cat,cat2,method='N',save='./testColors/%s_pdfn_color_validation.png'%(color_label))

# # with mu*
# # plot_validation_pdf_radial(gal,gal2,cat,cat2,method='Mu')
# # plot_validation_pdf_redshift(gal,gal2,cat,cat2,method='Mu')
# # plot_validation_pdf_redshift_triple(gal,gal2,cat,cat2)
# # plot_validation_pdf_color(gal,gal2,cat,cat2,method='Mu')


# ## trash
# # plot_validation_pdf(pdfz,pdfz2,ngals2,pz,gidx,gidx2)
# # plot_validation_pdf(pdfc,pdfc2,ngals2,pcolor,gidx,gidx2,method='color',save='./img/pdf_c_validation.png')
# # x_bin, y_bin_list, y_bin_std_list = bin_pdf(color,pdfc,gidx,method='color')
# # xtrue_bin, ytrue_bin_list, ytrue_bin_std_list = bin_pdf(color2,pdfc2,gidx2,method='color')

# # plot_pdf_validation(color,color2,pz*pr,gidx,gidx2,ngals,method='color',save='./img/pdf_c_validation.png')
# # plot_validation_pdf(pdfc,pdfc2,ngals2,pcolor,gidx,gidx2,method='color',save='./img/pdf_c_validation.png')

# def plot_PDF2(ax1,ax2,indices,indices2,xvec,xvec2,pmem,pmem2,radial_mode=False,method='None',labels=['',''],title='',xlims=(-1.5,0.5,30)):
#     ### it works for color and redshift pdfs
#     ### for radial distribution set the radial_mode to true
    
#     if radial_mode:
#         x_bin, pdf, pdf2 = get_galaxy_density_clusters(indices,indices2,xvec,xvec2,pmem,pmem2)

#     else:
#         x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,xvec,xvec2,pmem,xlims=xlims)
    
#     y_bin, yb_l, yb_h  = get_curves(pdf)
#     y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

#     per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
#     residual, cb_l, cb_h  = get_curves(per_error)

#     ## To Do!
#     # if method=='Ngals':
#     #     x_bin, pdf, pdf2 = get_sum_clusters(indices,indices2,xvec,xvec2,pmem,pmem2,xlims=(-1.5,0.5,30))
#     #     pdf, pdf2 = np.cumsum(pdf,axis=1), np.cumsum(pdf2,axis=1)
#     #     y_bin, yb_l, yb_h  = get_curves(pdf)
#     #     y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

#     #     per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf)
#     #     # sper_error = (sg/sg2)
#     #     residual, cb_l, cb_h  = get_curves(per_error)
#     ##
    
#     ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label=labels[0])
#     ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

#     ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label=labels[1])
#     ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

#     ax1.legend()

#     ax2.scatter(x_bin,residual,color='b',marker='s',s=20)
#     ax2.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')
#     ax2.axhline(0.,linestyle='--')
    
#     ax1.set_title(title)

# def plot_PDF(ax1,ax2,indices,indices2,color,color2,pmem,method='None',labels=['',''],title='',xlims=(-1.5,0.5,30)):
#     ### it works for color and redshift pdfs

#     ### curve 1
#     x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,color,color2,pmem,xlims=xlims)
#     y_bin, yb_l, yb_h  = get_curves(pdf)
#     y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

#     per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
#     residual, cb_l, cb_h  = get_curves(per_error)

#     if method=='Ngals':
#         x_bin, pdf, pdf2 = get_sum_clusters(indices,indices2,color,color2,pmem,pmem2,xlims=(-1.5,0.5,30))
#         pdf, pdf2 = np.cumsum(pdf,axis=1), np.cumsum(pdf2,axis=1)
#         y_bin, yb_l, yb_h  = get_curves(pdf)
#         y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

#         per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf)
#         # sper_error = (sg/sg2)
#         residual, cb_l, cb_h  = get_curves(per_error)
    
#     ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label=labels[0])
#     ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

#     ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label=labels[1])
#     ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

#     ax1.legend()

#     ax2.scatter(x_bin,residual,color='b',marker='s',s=20)
#     ax2.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')
#     ax2.axhline(0.,linestyle='--')
    
#     ax1.set_title(title)

# def plot_probabilities_colorPDF(gal,gal2,cat,cat2,save='./img/prob_color_PDF.png'):
#     indices = list(chunks(gal['CID'],cat['CID']))
#     indices2 = list(chunks(gal2['CID'],cat['CID']))

#     color = gal['delta_rs']
#     color2 = gal2['delta_rs']

#     weights_label = ['Pz','Pc','Pr','Pmem']
#     weights = [gal[col] for col in weights_label]
#     lims = []

#     fig = plt.figure(figsize=(12,6))
#     import matplotlib.gridspec as gridspec
#     gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

#     axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
#     for i in range(len(weights_label)):
#         ax1, ax2 = axis_list[i]
        
#         pmem = weights[i]
#         ti = 'Weighted by %s'%(weights_label[i])
#         label=['_','_']

#         ax2.set_xlabel(r'$\Delta(g-r)_{RS}$',fontsize=16)
#         if i==0:
#             label = ['Estimated Distrib.','True Distrib.']
#             ax1.set_ylabel(r'$PDF(color)$',fontsize=16)
#             ax2.set_ylabel(r'frac. error',fontsize=16)
            

#         plot_PDF2(ax1,ax2,indices,indices2,color,color2,pmem,None,labels=label,title=ti,xlims=(-1.5,0.5,30))
#         ax2.set_ylim(-1.,1.)
#         lims.append(ax1.get_ylim())

#     for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

#     plt.savefig(save,bb_box='tight')
#     plt.clf()

# def plot_probabilities_redshiftPDF(gal,gal2,cat,cat2,save='./img/prob_redshift_PDF.png'):
#     indices = list(chunks(gal['CID'],cat['CID']))
#     indices2 = list(chunks(gal2['CID'],cat['CID']))

#     z = zoffset(gal['z'],gal['redshift'])
#     z2 = zoffset(gal2['z'],gal2['redshift'])

#     weights_label = ['Pr','Pc','Pz','Pmem']
#     weights = [gal[col] for col in weights_label]
#     lims = []

#     fig = plt.figure(figsize=(12,6))
#     import matplotlib.gridspec as gridspec
#     gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

#     axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
#     for i in range(len(weights_label)):
#         ax1, ax2 = axis_list[i]
        
#         pmem = weights[i]
#         ti = 'Weighted by %s'%(weights_label[i])
#         label=['_','_']

#         ax2.set_xlabel(r'$(z-z_{cls})/(1+z_{cls})$',fontsize=16)
#         if i==0:
#             label = ['Estimated Distrib.','True Distrib.']
#             ax1.set_ylabel(r'$PDF(z)$',fontsize=16)
#             ax2.set_ylabel(r'frac. error',fontsize=16)
            
#         plot_PDF2(ax1,ax2,indices,indices2,z,z2,pmem,None,labels=label,title=ti,xlims=(-0.20,0.20,50))
#         ax2.set_ylim(-0.5,0.5)
#         lims.append(ax1.get_ylim())

#     for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

#     plt.savefig(save,bb_box='tight')
#     plt.clf()

# def plot_probabilities_radialPDF(gal,gal2,cat,cat2,save='./img/prob_radial_PDF.png'):
#     indices = list(chunks(gal['CID'],cat['CID']))
#     indices2 = list(chunks(gal2['CID'],cat2['CID']))

#     radii = gal['R']
#     radii2 = gal2['R']

#     weights_label = ['Pr','Pc','Pz','Pmem']
#     weights = [gal[col] for col in weights_label]
    
#     pmem2 = gal2['Pmem']

#     fig = plt.figure(figsize=(12,6))
#     import matplotlib.gridspec as gridspec
#     gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

#     axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
#     lims = []
#     for i in range(len(weights_label)):
#         ax1, ax2 = axis_list[i]
        
#         pmem = weights[i]
#         ti = 'Weighted by %s'%(weights_label[i])
#         label=['_','_']

#         ax2.set_xlabel(r'R $[Mpc]$',fontsize=16)
#         if i==0:
#             label = ['Estimated Distrib.','True Distrib.']
#             ax1.set_ylabel(r'$\Sigma \; [\# gals / Mpc^{2}]$',fontsize=16)
#             ax2.set_ylabel(r'frac. error',fontsize=16)
            
#         plot_PDF2(ax1,ax2,indices,indices2,radii,radii2,pmem,pmem2,radial_mode=True,labels=label,title=ti,xlims=(-0.20,0.20,50))
#         ax2.set_ylim(-0.5,0.5)
#         lims.append(ax1.get_ylim())

#     for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

#     plt.savefig(save,bb_box='tight')
#     plt.clf()


# def plot_multiple_scaling_relations(gal,gal2,cat,cat2,save='./img/prob_scaling_relation.png'):
#     indices = list(chunks(gal['CID'],cat['CID']))
#     indices2 = list(chunks(gal2['CID'],cat2['CID']))

#     weights_label = ['Pr','Pc','Pz','Pmem']
#     ngals_list = [np.array([np.nansum(gal[col][idx]) for idx in indices]) for col in weights_label]
#     ngals2 = np.array([np.nansum(gal2['Pmem'][idx]) for idx in indices2])

#     fig = plt.figure(figsize=(20,6))
#     import matplotlib.gridspec as gridspec
#     gs = gridspec.GridSpec(2, 4, wspace=0.15, hspace=0.01, height_ratios=[3,1])

#     axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
#     for i in range(len(weights_label)):
#         ax1, ax2 = axis_list[i]
#         ti = r'$\sum{%s}$'%(weights_label[i])

#         ax2.set_xlabel(r'$N_{gals,True}$',fontsize=16)
#         # ax1.set_ylabel(r'$\sum{%s}$'%(weights_label[i]),fontsize=16)

#         if i==0:
#             ax2.set_ylabel(r'frac. error',fontsize=16)

#         plot_scaling_relations(ax1,ax2,ngals_list[i],ngals2,title=ti,nmin=5,nmax=200)

#     plt.savefig(save,bb_box='tight')
#     plt.close()

# def plot_scaling_relations(ax1,ax2,x,x2,title='Title',nmin=5,nmax=300):
#     linreg=lin_reg(x2,x)

#     idx = np.argsort(x2)
#     xt,yh = x2[idx],linreg['Yhat'][idx]

#     b0 = round(linreg['b0'],3)
#     b1 = round(linreg['b1'],3)
#     cb_u, cb_l = linreg['cb_u'], linreg['cb_l']

#     xs = np.linspace(nmin,nmax,200)

#     ax1.plot(xt,yh, color="b", label='y=%.2f+%.2fx'%(b0,b1))
#     ax1.fill_between(xt, cb_l, cb_u, color="b", alpha=0.25, label='_nolabel_')
#     ax1.plot(xt,cb_l, color="b", label='_nolabel_')
#     ax1.plot(xt,cb_u, color="b", label='_nolabel_')
#     ax1.scatter(x2,x, color="b", s=20, alpha=0.3, label='_nolabel_')
#     ax1.plot(xs,xs,color='k',linestyle='--', label='y = x')

#     ### residual
#     ax2.scatter(x2,(x-x2)/x2,s=20,alpha=0.3,color='b',label=r'$ \sigma = %.3f $'%(np.std(x-x2)))
#     ax2.axhline(0,linestyle='--',color='b')

#     ax1.legend(loc='lower right')
#     ax2.legend()
#     ax1.set_title(title)

#     ax1.set_xlim(nmin,nmax)
#     ax1.set_ylim(nmin,nmax)
#     ax2.set_xlim(nmin,nmax)
#     ax2.set_ylim(-1.,1.)

#     ax1.set_yscale('log')
#     ax1.set_xscale('log')
#     ax2.set_xscale('log')

# def plot_purity_treshold():
#     prange = np.arange(0.001,1.0,0.01)
#     pmem_bin = (prange[1:]+prange[:-1])/2

#     Ntrue_all = len(gal2)
#     pmem = gal['Pmem']

#     p_bin, c_bin = [], []
#     for i in range(len(prange)-1):
#         pi,pj=prange[i],prange[i+1]
#         idx, = np.where( (pmem>=pi) )
        
#         gal_out = gal[pmem<pi]
#         Nmissed = len(gal_out[gal_out["True"]==True])

#         p,c = get_purity_completenes(gal[idx],Nmissed)

#         p_bin.append(p)
#         c_bin.append(c)

#     p_bin = np.array(p_bin)
#     c_bin = np.array(c_bin)

#     plt.plot(pmem_bin,p_bin,label='Purity')
#     plt.plot(pmem_bin,c_bin,label='Completeness')
#     plt.axvline(pmem_bin[np.argmax(p_bin*c_bin)])
#     plt.ylim(0.,1.01)
#     plt.xlabel('Pmem')
#     plt.ylabel('P/C')
#     plt.legend()

#     plt.savefig('purity_completeness.png')

# def get_purity_completenes(gal,Nmissed,lcol='Pmem'):
#     mask = gal["True"]==True
#     Ntrue = np.count_nonzero(mask)
#     Ninterlopers = np.sum(gal[lcol][(np.logical_not(mask))])
#     Nselected = np.sum(gal[lcol])#len(gal)

#     #Nmissed = Ntrue_all - Ntrue
#     P = (Nselected-Ninterlopers)/Nselected
#     C = (Ntrue-Nmissed)/Ntrue

#     return P, C


# # plot_probabilities_colorPDF(gal,gal2,cat,cat2)
# plot_probabilities_redshiftPDF(gal,gal2,cat,cat2)
# plot_probabilities_radialPDF(gal,gal2,cat,cat2)
# plot_multiple_scaling_relations(gal,gal2,cat,cat2)

# # plot_probabilities_colorPDF(gal,gal2,cat,cat2,save='./testColors/%s_prob_color_PDF.png'%(color_label))
# # plot_probabilities_redshiftPDF(gal,gal2,cat,cat2,save='./testColors/%s_prob_redshift_PDF.png'%(color_label))
# # plot_probabilities_radialPDF(gal,gal2,cat,cat2,save='./testColors/%s_prob_radial_PDF.png'%(color_label))
# # plot_multiple_scaling_relations(gal,gal2,cat,cat2,save='./testColors/%s_prob_scaling_relation.png'%(color_label))