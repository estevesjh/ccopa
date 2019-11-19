# !/usr/bin/env python
# color-magnitude subtraction algorithm

from time import time
import numpy as np
import scipy.stats as st
from scipy.stats import uniform, norm, stats

from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt



import sys
import os
sys.path.append(os.path.abspath("/home/johnny/Documents/Brandeis/CCOPA/lib/"))
import gaussianKDE as kde

plt.style.use('seaborn')
plt.rcParams.update({'font.size': 24})
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

def matchIndices(keys1,keys2):
    indicies = np.empty((0),dtype=int)
    indicies2 = []

    ncls = np.max([keys1.size,keys2.size])
    for i in range(ncls):
        idx, = np.where(keys1==keys2[i])
        indicies = np.append(indicies,idx)
        indicies2.append(i)

    return indicies, np.array(indicies2)

def makeBin(variable,width=0.01,xvec=None):
    if xvec is None:
        xmin, xmax = (variable.min()), (variable.max() + width/2)
        xvec = np.arange(xmin,xmax,width)
    
    idx,xbins = [], []
    for i in range(len(xvec)-1):
        xlo,xhi = xvec[i],xvec[i+1]
        w, = np.where((variable>=xlo)&(variable<=xhi))
        bins = (xlo+xhi)/2
        idx.append(w)
        xbins.append(bins)

    return idx, xbins

def computeColorMagnitudeKDE(x,y,weight=None):
    """input: x (magnitude) and y (color)
       return: PDF (probability distribuition function)
    """
    values = np.vstack([x, y])
    if weight is not None:
        kernel = kde.gaussian_kde(values,weights=weight,bw_method='silverman')
    else:
        kernel = kde.gaussian_kde(values,bw_method='silverman')
    return kernel

def doGrid(xmin=15,xmax=23,ymin=0,ymax=3.):
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:75j]
    return xx,yy

def getKDE(x,y,weight=None,xmin=15,xmax=23,ymin=0,ymax=3.):
    kernel = computeColorMagnitudeKDE(x,y,weight=weight)

    xx, yy = doGrid(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde = np.reshape(kernel(positions).T, xx.shape)

    return kde

def computeKDE(mag,color,weight,xmin=15,xmax=23,ymin=0,ymax=2.5):
    xx,yy = doGrid(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    kde = getKDE(mag,color,weight=weight,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    return kde

def plotTrioColorMagPDF(kde,kde_true,Ngal,Ngal_true, title="Buzzard Simulations", name_cls='cluster'):
    
    plt.clf()
    fig, axs = plt.subplots(1, 3, sharey=True,sharex=True, figsize=(18,4))
    # fig.subplots_adjust(left=0.075,right=0.95,bottom=0.15,wspace=0.1)
    # fig.tight_layout()

    Ngal = 1#len(mag)
    Ngal_true = 1#len(mag_true)
    kde_sub = (Ngal*kde-Ngal_true*kde_true)#/np.abs(Ngal-Ngal_true)

    # kde_sub = getKDE(mag[idx],color[idx],weight=None,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

    cmax = 0.7
    xx,yy = doGrid()

    plt.subplot(131)
    cfset0 = plt.contourf(xx, yy, kde, cmap='RdBu_r',vmin=-1*cmax, vmax=cmax)
    cset = plt.contour(xx, yy, kde, colors='white',vmin=-1*cmax, vmax=cmax)
    img = plt.clabel(cset, inline=1, fontsize=10)
    plt.title('Observed')
    plt.ylabel(r'$(g-r)$')

    # plt.colorbar(img)

    plt.subplot(132)
    cfset = plt.contourf(xx, yy, kde_true, cmap='RdBu_r',vmin=-1*cmax, vmax=cmax)
    cset = plt.contour(xx, yy, kde_true, colors='white',vmin=-1*cmax, vmax=cmax)
    plt.clabel(cset, inline=1, fontsize=10)
    plt.title('True')
    plt.colorbar(cfset0)
    plt.clim(0.,cmax)

    plt.subplot(133)
    cfset = plt.contourf(xx, yy, kde_sub, cmap='RdBu_r',vmin=-1*cmax, vmax=cmax)
    cset = plt.contour(xx, yy, kde_sub, colors='white',vmin=-1*cmax, vmax=cmax)
    plt.clabel(cset, inline=1, fontsize=10)
    plt.title('Residual')
    plt.colorbar(cfset)
    # cb.clim(-1*cmax,cmax)

    # for i in range(3):
    #     axs[i].set_xlabel(r'$r$')
    
    plt.figtext(0.22,0.2, title, fontsize=16, va="top", ha="left")
    # plt.colorbar()

    plt.savefig(name_cls+'_color-magnitude_residual_kernel.png', bbox_inches = "tight")
    plt.close()

def plotResidual(xtrue,x,y,save,kind=['Mu','z'],bins=None):
    residue = 1-(x/xtrue)
    cutOutliers, = np.where( np.abs(residue-np.median(residue)) <= 3*np.std(residue) )

    if cutOutliers.size>0:
        residue = residue[cutOutliers]
        y = y[cutOutliers]
        x,xtrue = x[cutOutliers], xtrue[cutOutliers]

    if kind[0]=='N':
        xlabel = r'$1 - \frac{N}{N_{True}}$'
        nmin, nmax = 5., 1.75*np.max([xtrue,x])

    elif kind[0]=='Mu':
        xlabel = r'$1 - \frac{\mu^{*}}{\mu^{*}_{True}}$'
        nmin, nmax = 20., 2.*np.max([xtrue,x])
    
    if kind[1]=='z':
        ylabel = r'redshift'
        save = save+'_redshift.png'
    
    elif kind[1]=='mass':
        ylabel = r'M$_{200} \; [10^{14} M_{\odot}/h]$'
        save = save+'_mass.png'
    
    elif kind[1]=='N':
        ylabel = r'N$_{True}$'
        save = save+'_ntrue.png'

    if kind[0]=='Bkg':
        xlabel = r'$\log(N_{Bkg}/N_{Bkg,True})$'
        # ylabel = r'N$_{bkg}$'
        residue = np.log10(x/xtrue)

    if bins is not None:
        indices, xbins = makeBin(y,width=2.,xvec=bins)
        ybins = [np.mean(residue[idx]) for idx in indices]
        yerr = [np.std(residue[idx]) for idx in indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(y,residue,color='royalblue',s=50,alpha=0.3)
    plt.axhline(0.,linestyle='--',color='gray',linewidth=2.)

    if bins is not None:
        plt.errorbar(xbins,ybins,yerr=yerr,xerr=np.diff(bins)/2,
                    fmt='o',markersize=4,color='gray',elinewidth=2,markeredgewidth=2,capsize=3)
    
    if kind[1]=='mass':
        print('here')
        plt.xscale('log')

    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    # plt.title('Buzzard Simulations')
    plt.savefig(save,bbox_inches = "tight")
    plt.close()

def plotN200M200():
    bins=np.histogram(ngals_true,bins=8)[1]
    indices, xbins = makeBin(ngals_true,width=2.,xvec=bins)
    ybins = [np.mean(m200[idx]) for idx in indices]
    yerr = [np.std(m200[idx]) for idx in indices]

    plt.errorbar(xbins,ybins,yerr=yerr,xerr=np.diff(bins)/2,fmt='o',markersize=4,color='gray',elinewidth=2,markeredgewidth=2,capsize=3,label='_nolabel_')
    plt.scatter(ngals_true,m200,color='royalblue',s=50,alpha=0.3,label=r'$mag_{lim}=m_{BCG}+3$')
    plt.xlabel(r'N$_{200}$')
    plt.ylabel(r'M$_{200} \; [10^{14} M_{\odot}]$')
    plt.xscale('log')
    plt.legend()
    plt.savefig('n_m200_buzzard.png',bbox_inches = "tight")
    plt.close()


    # bins=np.histogram(m200,bins=8)[1]
    bins = np.array([3.,4.,5.,6.6,8.56,11.07,13.62,26.5])
    indices, xbins = makeBin(m200,width=2.,xvec=bins)
    ybins = [np.mean(ngals_true[idx]) for idx in indices]
    yerr = [np.std(ngals_true[idx]) for idx in indices]

    plt.errorbar(xbins,ybins,xerr=yerr,yerr=np.diff(bins)/2,fmt='o',markersize=4,color='gray',elinewidth=2,markeredgewidth=2,capsize=3,label='_nolabel_')
    plt.scatter(m200,ngals_true,color='royalblue',s=50,alpha=0.3,label=r'$mag_{lim}=m_{BCG}+3$')
    plt.ylabel(r'N$_{200}$')
    plt.xlabel(r'M$_{200} \; [10^{14} M_{\odot}]$')
    plt.xscale('log')
    plt.legend()
    plt.savefig('n_m200_buzzard.png',bbox_inches = "tight")
    plt.close()


def plotIdentity(xtrue,x,save,kind='N'):
    if kind=='N':
        xlabel, ylabel = r'N$_{True}$', r'N'
        nmin, nmax = 5., 1.5*np.max([xtrue,x])

    elif kind=='Mu':
        xlabel, ylabel = r'$\mu^{*}_{True}$',r'$\mu^{*}$'
        nmin, nmax = 30., 5000.

    else:
        print('please enter a valid kind (N, Mu)')
        exit()

    linreg=lin_reg(xtrue,x)
    
    idx = np.argsort(xtrue)
    xt,yh = xtrue[idx],linreg['Yhat'][idx]

    b0 = round(linreg['b0'],3)
    b1 = round(linreg['b1'],3)
    cb_u, cb_l = linreg['cb_u'], linreg['cb_l']

    xs = np.linspace(nmin,nmax,200)

    fig = plt.figure(figsize=(8,6))
    plt.plot(xt,yh, color="lightskyblue",label='y=%.2f+%.2fx'%(b0,b1))
    plt.fill_between(xt, cb_l, cb_u, color="lightskyblue", alpha=0.25, label='_nolabel_')
    plt.plot(xt,cb_l, color="lightskyblue", label='_nolabel_')
    plt.plot(xt,cb_u, color="lightskyblue", label='_nolabel_')
    plt.scatter(xtrue,x, color="royalblue", s=20, alpha=0.5, label='_nolabel_')
    plt.plot(xs,xs,color='k',linestyle='--', label='y = x')
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Buzzard Simulations - z=[0.1,0.9); N={0:d}".format(len(x)))
    plt.xlim(nmin,nmax)
    plt.ylim(nmin,nmax)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.savefig(save, bbox_inches = "tight")
    plt.close()

def lin_reg(X,Y):
    barX=np.mean(X); barY=np.mean(Y)
    XminusbarX=X-barX; YminusbarY=Y-barY
    b1=sum(XminusbarX*YminusbarY)/sum(XminusbarX**2)
    b0=barY-b1*barX
    Yhat=b0+b1*X
    e_i=Y-Yhat
    sse=np.sum(e_i**2)
    ssr=np.sum((Yhat-barY )**2)
    n=len(X)
    MSE=sse/np.float(n-2)

    s_of_yh_hat=np.sqrt(MSE*(1.0/n+(X-barX)**2/sum(XminusbarX**2)))
    W=np.sqrt(2.0*st.f.ppf(0.95,2,n-2))

    cb_upper=Yhat+W*s_of_yh_hat
    cb_lower=Yhat-W*s_of_yh_hat
    idx=np.argsort(X)

    return {'Yhat':Yhat,'b0':b0,'b1':b1,'cb_u':cb_upper[idx], 'cb_l': cb_lower[idx]}

def plotColorEvolution(zs,colors,pmems,labely, fgsize, outname):
    '''To Do
    '''
    x1, x2 = zs
    y1, y2 = colors
    z1, z2 = pmems

    fig, axs = plt.subplots(1, 3, figsize=fgsize)
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    #axs = axs.ravel()
    plt.figure()
    plt.subplot(131)
    h1=plt.hexbin(x1, y1, z1, gridsize=40, vmin=0, vmax=400, cmap=plt.cm.RdBu_r)
    plt.colorbar()

    plt.subplot(132)
    h2=plt.hexbin(x2, y2, z2, gridsize=40, vmin=0, vmax=400, cmap=plt.cm.RdBu_r)
    plt.colorbar()

    plt.subplot(133)
    # Create dummy hexbin using whatever data..:
    h3=plt.hexbin(x1, y1, z1, gridsize=40, vmin=-50, vmax=50, cmap=plt.cm.RdBu_r)
    h3.set_array(h1.get_array()-h2.get_array())
    cb = plt.colorbar()

    cb.set_label(r'N')
    plt.savefig(outname, bbox_inches='tight')
    
def colorz(zs, colors, richness, labely, limy, fgsize, outname, plottype):
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(1, 3, figsize=fgsize)
    fig.subplots_adjust(hspace=0.15)
    # fig.subplots_adjust(hspace=0.15, left=0.07, right=0.95)
    xlim = [[-0.25,2.8],[-0.5,1.65],[-0.3,1.]]
    #axs = axs.ravel()
    for i in range(len(colors)):
        if plottype=='scatter':
            axs[i].plot(zs, colors[i],'r.', alpha=0.3)
            axs[i].set_xlabel('z')
            axs[i].set_xlim(0,1)
            axs[i].set_ylabel(labely[i])
        elif plottype=='hex_density':
            im=axs[i].hexbin(zs, colors[i], cmap=plt.cm.jet)
            axs[i].cmap=plt.cm.jet
            axs[i].set_xlabel('z')
            axs[i].set_xlim(0,1) 
            axs[i].set_ylabel(labely[i])
            #axs[i].set_ylim(limy)
        elif plottype=='hex_lambda':
            xlimi = xlim[i]
            im=axs[i].hexbin(zs, colors[i],richness, gridsize=30, cmap=plt.cm.get_cmap('Blues_r', 14), reduce_C_function=np.sum, vmin=-50, vmax=500)
            # im=axs[i].hexbin(zs, colors[i], richness, gridsize=40, cmap='inferno',vmin=0.,vmax=1.)
            axs[i].set_xlabel('redshift')
            axs[i].set_xlim(0.09,0.91)
            axs[i].set_ylim(xlimi[0],xlimi[1])
            axs[i].set_ylabel(labely[i])
            if i==2:
                # axs[i].set_yticks([-0.3,0.,0.3,0.9,1.])
                axs[i].set_yticks(np.arange(-0.3,1.,0.3))
            if i==1:
                axs[i].set_yticks(np.arange(-0.5,1.6,0.5))
                # axs[i].set_yticks([-0.5,0.,0.5,0.75,1.5])

            if 'stdcolorz' in outname:
                axs[i].axhline(np.mean(colors[i]) , color='k', linestyle='-')
                axs[i].text(0.8, np.mean(colors[i]), '$\mu$= '+str(round(np.mean(colors[i]),3)), fontsize=12, horizontalalignment='left', verticalalignment='top', fontweight='bold')
                axs[i].axhline(np.mean(colors[i]) + 3*np.std(colors[i]) , color='k', linestyle='--')
                axs[i].text(0.8, np.mean(colors[i]) + 3*np.std(colors[i]), '$3\sigma$= '+str(round(3*np.std(colors[i]),3)), fontsize=12, horizontalalignment='left', verticalalignment='bottom', fontweight='bold')
                #axs[i].axhline(np.mean(colors[i]) - np.std(colors[i]) , color='r', linestyle='--')
                idxs=(colors[i] > 3*np.std(colors[i]) )
                N_out=len(np.array(colors)[i][idxs])
                N=len(colors[i])
                frac = (N_out*100)/N
                axs[i].set_title('$frac_{out}$= '+str(round(frac,2))+'$\%$')
                #axs[i].text(0.5, 0.5, '$frac_{out}$= '+str(round(frac,2))+'$\%$', fontsize=12, horizontalalignment='center', verticalalignment='center', fontweight='bold', transform=axs[i].transAxes)

    if plottype=='hex_lambda':
        cb = plt.colorbar(im, ax=axs[2], orientation="vertical", shrink=0.9)
        cb.set_label(r'N')
    plt.savefig(outname, bbox_inches='tight')
    plt.close()


def getN200(g,keys,true_gals=False,col='Pmem'):
    ngals = []
    for idx in keys:
        
        if true_gals:
            w, = np.where((g['CID']==idx)&(g['True']==True))#&(g['Mr']<=-19.5)
            # w, = np.where((g['HALOID']==idx)&(g['R']<=r2)&(g['TRUE_MEMBERS']==True)&(g['mag'][:,2]<=mi))#&(g['Mr']<=-19.5)
            ni = len(w)
        else:
            w, = np.where((g['CID']==idx))#&(g['True']==True))#
            ni = np.sum(g[col][w])
        ngals.append(ni)

    return np.array(ngals)

def plot0(n200,richness,projection=True,nmin=500,nmax=5):
    
    if projection:
        label = 'projection in the LOS'
        color = "lightcoral"
        color2 = 'r'
    else:
        label = 'no projection in the LOS'
        color = "lightskyblue"
        # color = 'blue'
        color2 = 'royalblue'
    
    linreg=lin_reg(n200,richness)
    idx = np.argsort(n200)
    x = n200[idx]
    yh = linreg['Yhat'][idx]

    xs = np.linspace(nmin,nmax,200)

    b0 = round(linreg['b0'],3)
    b1 = round(linreg['b1'],3)
    cb_u, cb_l = linreg['cb_u'], linreg['cb_l']
    
    plt.fill_between(x, cb_l, cb_u, color=color, alpha=0.25, label='_nolegend_')
    plt.plot(x,cb_l, color=color, label='_nolegend_')
    plt.plot(x,cb_u, color=color, label='_nolegend_')
    plt.plot(xs,xs,color='k',linestyle='--', label='_nolegend_')

    plt.plot(x,yh, color=color, label = "a = {0}; b = {1}".format(b0,b1))
    plt.scatter(n200,richness, color=color2, s=20, alpha=0.6, label=label)
        
    # plt.title("z=[0.1,0.9); N={0:d}; y={1:.2f}+{2:.2f}x".format(len(richness), b0,b1 ))

def plotRichnessN200Projection(richness,n200,mask,savename='richness_n200.png'):
    nmask = np.logical_not(mask)

    plt.clf()
    plt.legend()
    fig = plt.figure(figsize=(8,6))
    nmin, nmax = 3., 2.5*np.max(n200)
    plot0(n200[mask],richness[mask],projection=True,nmin=nmin,nmax=nmax)
    plot0(n200[nmask],richness[nmask],projection=False,nmin=nmin,nmax=nmax)
    
    plt.xlim(nmin,nmax)
    plt.ylim(nmin,nmax)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$N_{True}$')
    plt.ylabel(r'$N$')
    plt.legend(loc='lower right')
    
    plt.savefig(savename, bbox_inches = "tight")

def zoffset(z,zi):
    return (z-zi)/(1+zi)

def scalePmem(x):
    return ( 50*(x)**(2) + 0.1 )

def plotPredshift(z,zcls,y,ylabel,pmem,save):
    zoff = zoffset(z,zcls)
    scale = scalePmem(pmem)

    plt.figure(figsize=(8,6))
    plt.hexbin(y,zoff, pmem, gridsize=40, cmap='inferno_r', reduce_C_function=np.sum, vmin=0,vmax=100)
    # plt.hexbin(y,zoff, pmem, gridsize=40, cmap='inferno_r', vmin=0.,vmax=1.)
    # plt.axhline(0,linestyle='--',color='k')
    plt.ylabel(r'$(z-z_{cls})/(1+z_{cls})$')
    plt.xlabel(ylabel)
    # plist = [0.1,0.5,1.]
    # for pi in plist:
    #     plt.scatter([],[],s=scalePmem(pi), color='k', alpha=0.3, label='%.1f'%pi)
    
    plt.colorbar(label='N')
    plt.savefig(save, bbox_inches = "tight")
    plt.close()