# !/usr/bin/env python
# color-magnitude subtraction algorithm

from time import time
import numpy as np
from scipy.stats import uniform, norm, stats

from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
#import pandas as pd
import seaborn as sns; sns.set(color_codes=True)

from myPlots import *

plt.rcParams.update({'font.size': 16})
sns.set_style("whitegrid")

def plot_scatter_hist(x,y,weights=None,xtrue=None,ytrue=None,xlabel='redshift',ylabel=r'$\mu_{\star}\,\,[10^{12}\,M_{\odot}]$',save='./img/bla.png'):
    compare = (xtrue is not None) and (ytrue is not None)
    
    if weights is not None:scale = 100*weights**(1/2)+1
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
        scatter_axes.scatter(xtrue,ytrue, s=25, color='gray', marker='o', alpha=0.35, zorder=1)
        scatter_axes.axhline(np.mean(ytrue),linestyle='--',color='gray')

        x_hist_axes.hist(xtrue, bins=binx, color='gray', histtype='stepfilled', alpha=0.65, normed=False)
        y_hist_axes.hist(ytrue, bins=biny, color='gray', histtype='stepfilled', alpha=0.65, normed=False,orientation='horizontal')
        y_hist_axes.axhline(np.mean(ytrue),linestyle='--',color='gray')

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
        x_bin, x_bin_err, y_bin, y_bin_err = get_binned_variables(xtrue,ytrue)

    fig = plt.figure(figsize=(8,6))
    plt.scatter(x,y, s=6, alpha=0.5, label='_nolabel_')
    plt.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err,color='gray', fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2)
    
    if compare:
        plt.scatter(x,y, s=6, alpha=0.5, color='red', label='_nolabel_')
        plt.errorbar(x_bin,y_bin,xerr=x_bin_err,yerr=y_bin_err, color='darkred', fmt='o', linestyle='--', markersize=8, capsize=4, capthick=2, label='True Distrib.')

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

###########################################
test = 0
colorEvolution = 1
compareDatasets = True
###########################################
ii=0
print('Geting Data')

file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members_stellarMass.fits'
file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_stellarMass.fits' 

# file_gal = '/home/johnny/Documents/Github/ccopa/out/splus_STRIPE82_master_DR2_SN_10_galaxyClusters_pmem_members_stellarMass.fits'
# file_cls = '/home/johnny/Documents/Github/ccopa/out/splus_STRIPE82_master_DR2_SN_10_galaxyClusters_pmem_stellarMass.fits'

cat = Table(getdata(file_cls))
gal = Table(getdata(file_gal))

print('Taking only cluster within 5 true members')
# cat = cat[(cat['Nbkg'] >= 0)&(cat['Nbkg'] <= 10)]

gal = gal.group_by('CID')
gidx, cidx = getIndices(gal.groups.indices,gal.groups.keys['CID'],cat['CID'])
gal = gal[gidx]

print('Defining variables')
keys = cat['CID'][:]
redshift = cat['redshift'][:]
ngals = cat['Ngals'][:]
norm = cat['Norm'][:]
nbkg = cat['Nbkg'][:]
MU = cat['MU']
sum_ssfr = cat['SSFR']
sum_smass = cat['SUM_MASS']

# Nred = cat['Nred']
# Nblue = cat['Nblue']
# red_fraction = Nred/(Nred+Nblue)

# richness = cat['richness'][:]*1e3
# sn = cat['sn'][:]

sky_plot(cat['RA'], cat['DEC'],title='Buzzard Simulation v1.6')
plot_group_by_redshift(redshift, ngals, ylog=False, ylabel = r'N$_{gals}$', xlabel = r'redshift', save='./img/redshift_ngals.png')
plot_group_by_redshift(redshift, norm, ylog=False, ylabel = r'Norm', xlabel = r'redshift', save='./img/redshift_norm.png')
plot_group_by_redshift(redshift, nbkg, ylim=(0.,100.),ylog=False, ylabel = r'N$_{bkg}$', xlabel = r'redshift', save='./img/redshift_nbkg.png')

# w, = np.where(red_fraction>=0.)
# plot_group_by_redshift(redshift[w], red_fraction[w], ylog=False, ylabel = r'$f_{red}$', xlabel = r'redshift', save='./img/redshift_nred.png')
# plot_lin_reg(ngals,richness,xlabel = r'N$_{gals}$', ylabel = r'richness [1e-3]', save='./img/richness_ngals.png')
# plot_lin_reg(ngals,sn, xlabel = r'N$_{gals}$', ylabel = r'SN', save='./img/sn_ngals.png')

plot_scatter_hist(redshift, MU/100., save='./img/scale_mu_star.png')
plot_scatter_hist(redshift, sum_ssfr, ylabel = r'$\Sigma (sSFR)$', xlabel = r'redshift', save='./img/scale_ssfr.png')
plot_scatter_hist(redshift, sum_smass, ylabel = r'$\Sigma (M_{*}) \,\,[10^{12}\,M_{\odot}]$', save='./img/scale_sum_smass.png')
# plot_scatter_hist(redshift, ngals, ylabel = r'N$_{gals}$', xlabel = r'redshift', save='./img/scale_ngals.png')


print('starting galaxies related plots')
# print('to do !')

z = gal['z']
zcls = gal['redshift']
pmem = gal['Pmem']
mag_g = gal['mag'][:,0]
mag_r = gal['mag'][:,1]
mag_i = gal['mag'][:,2]
mag_z = gal['mag'][:,3]

amag = gal['rabs'][:]
gi_o = gal['gi_o'][:]
smass = gal['mass'][:]
ssfr = np.where(gal['ssfr'][:]==0,0.,np.log10(gal['ssfr'][:]))

gr = mag_g - mag_r
ri = mag_r - mag_i
iz = mag_i - mag_z

mask = gal['True']==True

plot_scatter_hist(zcls,amag,weights=pmem, xtrue=zcls[mask], ytrue=amag[mask], ylabel=r'$M_{r}$',save='./img/amag_evolution.png')
plot_scatter_hist(zcls,smass,weights=pmem, xtrue=zcls[mask], ytrue=smass[mask], ylabel=r'$log(M_{*}) \: [M_{\odot}]$',save='./img/smass_evolution.png')
plot_scatter_hist(zcls,ssfr,weights=pmem, xtrue=zcls[mask], ytrue=ssfr[mask], ylabel=r'$log(sSFR)$', save='./img/ssfr_evolution.png')
plot_scatter_hist(zcls,gi_o,weights=pmem, xtrue=zcls[mask], ytrue=gi_o[mask], ylabel=r'$(g-i)_{rest-frame}$', save='./img/gi_rest_frame_evolution.png')

print('fim')
