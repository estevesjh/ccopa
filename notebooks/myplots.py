import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

## colors
blue = '#2E86C1'
gray = '#A6ACAF'
red = '#A93226'

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

## colors
blue = '#2E86C1'
gray = '#A6ACAF'
red = '#A93226'

def plot_scaling_relation(x,y,title='Buzzard',xlims=(1,900),xl=r'$N_{true}$',yl=r'$N_{obs}$',fit=False):
    ## Bins
    xbins     = np.logspace( np.nanmin(np.log10(x)), np.nanmax(np.log10(x)) ,12)
    keys, xb  = makeBins(x,xbins)
    xb_std    = np.diff(xbins)/2
    
    yb     = np.array([np.nanmean(y[idx]) for idx in keys])
    yb_std = np.array([np.nanstd(y[idx]) for idx in keys])
    
    ## linear fit
    linreg=lin_reg(x,y)
    
    idx = np.argsort(x)
    xt,yh = x[idx],linreg['Yhat'][idx]

    b0 = round(linreg['b0'],3)
    b1 = round(linreg['b1'],3)
    cb_u, cb_l = linreg['cb_u'], linreg['cb_l']
    
    ## Plot
    fig = plt.figure(figsize=(10,8))
    
    if fit:
        plt.plot(xt,yh, color="r",label='y=%.2f+%.2fx'%(b0,b1))
        plt.fill_between(xt, cb_l, cb_u, color="gray", alpha=0.25, label='_nolabel_')
        plt.plot(xt,cb_l, color="r", label='_nolabel_')
        plt.plot(xt,cb_u, color="r", label='_nolabel_')

    sc = plt.scatter(x,y,s=75, alpha=0.25, color=gray)#,label='$scatter = %.1f$'%(np.std(ngals-nt))
    plt.errorbar(xb,yb,xerr=xb_std,yerr=yb_std,color=blue,linewidth=2.,fmt='o')
    plt.plot(np.linspace(xlims[0],xlims[1]),np.linspace(xlims[0],xlims[1]),linestyle='--',color='r')

    plt.xscale('log')
    plt.yscale('log')
    
    plt.ylim(xlims)
    plt.xlim(xlims)
    
    plt.xlabel(xl,fontsize=22)
    plt.ylabel(yl,fontsize=22)
    plt.legend(fontsize=14)
    
    plt.title(title,fontsize=22)
    fig.tight_layout()

def plot_residual(xvar,yvar1,yvar2, ax=None, xlabel='redshift', xbins=None,log=False):
    if ax is None: ax = plt.axes()
    
    if xbins is None:
        xbins = splitBins(xvar)
    
    residual = (1-yvar2/yvar1)
    if log:
        residual = np.log(yvar2/yvar1)
    mask2    = np.logical_not(np.isnan(residual))&np.logical_not(np.isinf(residual))
    mask     = (np.abs(residual)<0.5)&(mask2)
    if log:
        mask     = remove_outliers(residual,n=2)&(mask2)
    
    nmask    = np.logical_not(mask)
    of       = 1.-1.*np.count_nonzero(mask)/len(yvar1)
    
    keys, xvarb = makeBins(xvar[mask],xbins)
    xvarb_std = np.diff(xbins)/2
    
    residualb = np.array([np.nanmedian(residual[mask][idx]) for idx in keys])
    residualb_std = np.array([np.nanstd(residual[mask][idx]) for idx in keys])
    
    ax.scatter(xvar[nmask],residual[nmask],color=red,alpha=0.25,s=50,label='Outlier fraction: %.2f'%(of))
    ax.scatter(xvar,residual,color='#A6ACAF',alpha=0.25,s=50)
    ax.errorbar(xvarb,residualb,xerr=xvarb_std,yerr=residualb_std,color='#2E86C1',fmt='o')
    ax.set_xlabel(xlabel,fontsize=18)
    ax.legend()

def plot_triple_pannel(zcls,ntru,logm,yvar1,yvar2,title='Residuals',save=None,ymin=-1,ymax=1.5):
    ylabel=r'frac. residual'
    ngbins = np.logspace(np.log10(2),np.nanmax(np.log10(yvar2)),9)
    
    fig, ax = plt.subplots(3, 1, sharey='col', figsize=(10,14))
    fig.subplots_adjust(hspace=0.4,wspace=0.6)
    
    plot_residual(zcls,yvar1,yvar2,ax=ax[0])
    plot_residual(ntru,yvar1,yvar2,ax=ax[2],xlabel=r'$N_{true}$',xbins=ngbins)
    plot_residual(logm,yvar1,yvar2,ax=ax[1],xlabel=r'$\log{M_{200}}$ [$M_{\odot}\, h^{-1}$]')

    fig.suptitle(title,fontsize=18)

    ax[1].set_ylabel(ylabel,fontsize=24)
    ax[2].set_xscale('log')
    ax[0].set_ylim(ymin,ymax)
    ax[2].set_xlim(0.7*1,2*np.nanmax(yvar2))
    for i in range(3):
        ax[i].axhline(0.2,color='r',linestyle='--')
        ax[i].axhline(-0.2,color='r',linestyle='--')
    fig.tight_layout()
    
    if save:
        plt.savefig(save,bb_box='tight')
    #fig.clf()
    pass

def plot_four_pannel(zcls,r200,ntrue,nbkg,x1,x2,ylabel='y',ylims=(-2,2)):
    fig, ax = plt.subplots(2, 2, sharey='all', figsize=(12,8))
    fig.subplots_adjust(hspace=0.25,wspace=0.05)

    ax = ax.flatten(order='F')

    plot_residual(zcls ,x1,x2,ax=ax[0],log=True)
    plot_residual(r200 ,x1,x2,ax=ax[2],log=True)
    plot_residual(ntrue,x1,x2,ax=ax[1],log=True)
    plot_residual(nbkg ,x1,x2,ax=ax[3],log=True)

    ax[0].set_xlabel('redshift')
    ax[1].set_xlabel('Ngals')
    ax[2].set_xlabel('R200')
    ax[3].set_xlabel('Nbkg')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[0].set_ylim(ylims)
    ax[0].set_ylabel(ylabel,fontsize=22)
    ax[1].set_ylabel(ylabel,fontsize=22)


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

def makeBins(variable,xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

def remove_outliers(x,n=1.5):
    q25,q75 = np.nanpercentile(x,[25,75])
    iqr     = q75-q25
    lo, up  = q25-n*iqr, q75+n*iqr
    mask    = (x<up)&(x>lo)
    return mask

def splitBins(var):
    nmin = np.nanmin(var)
    n1 = np.percentile(var,10)
    n2 = np.percentile(var,20)
    n3 = np.percentile(var,30)
    n4 = np.percentile(var,40)
    n5 = np.percentile(var,50)
    n6 = np.percentile(var,60)
    n7 = np.percentile(var,70)
    n8 = np.percentile(var,80)
    n9 = np.percentile(var,90)
    nmax = np.max(var)
    
    return np.array([nmin,n1,n2,n3,n4,n5,n6,n7,n8,n9,nmax])