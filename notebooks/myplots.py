import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

## colors
blue = '#2E86C1'
gray = '#A6ACAF'
red = '#A93226'


def sky_plot(RA,DEC,title="Buzzard v1.6 - 1000 GC",savefig='sky_plot.png'):
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
    #plt.clf()
    #plt.close()

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
    mask     = remove_outliers(residual,n=5)&(mask2)
    if log:
        mask     = remove_outliers(residual,n=5)&(mask2)
    
    nmask    = np.logical_not(mask)
    of       = 1.-1.*np.count_nonzero(mask)/len(yvar1)
    
    keys, xvarb = makeBins(xvar[mask],xbins)
    xvarb_std = np.diff(xbins)/2
    
    residualb = np.array([np.nanmedian(residual[mask][idx]) for idx in keys])
    residualb_std = np.array([np.nanstd(residual[mask][idx]) for idx in keys])
    
    ax.scatter(xvar[nmask],residual[nmask],color='r',alpha=0.25,s=50,label='Outlier fraction: %.2f'%(of))
    ax.scatter(xvar,residual,color='#A6ACAF',alpha=0.25,s=50)
    ax.errorbar(xvarb,residualb,xerr=xvarb_std,yerr=residualb_std,color='#2E86C1',fmt='o')
    print(residualb_std)
    ax.set_xlabel(xlabel,fontsize=18)
    #ax.legend()

def plot_triple_pannel(zcls,ntru,logm,yvar1,yvar2,title='Residuals',save=None,ymin=-1,ymax=1.5):
    ylabel=r'frac. residual'
    ngbins = np.logspace(np.log10(2),1.*np.nanpercentile(np.log10(yvar2),95),9)
    
    fig, ax = plt.subplots(3, 1, sharey='col', figsize=(10,14))
    fig.subplots_adjust(hspace=0.4,wspace=0.6)
    
    plot_residual(zcls,yvar1,yvar2,ax=ax[0])
    plot_residual(ntru,yvar1,yvar2,ax=ax[2],xlabel=r'$N_{true}$',xbins=ngbins)
    plot_residual(logm,yvar1,yvar2,ax=ax[1],xlabel=r'$\log{M_{200}}$ [$M_{\odot}\, h^{-1}$]')

    #fig.suptitle(title,fontsize=18)

    ax[1].set_ylabel(ylabel,fontsize=24)
    ax[2].set_xscale('log')
    ax[0].set_ylim(ymin,ymax)
    ax[2].set_xlim(0.8*ngbins[0],1.*ngbins[-1])
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

def plot_r200_identity(r200h,r200,title='Buzzard',ylabel=r'$R_{200}$ [$Mpc\, h^{-1}$]',xlims=[1.,900],ylims=[1.,900],logy=False, li='hod'):
    r200_bins = splitBins(r200)
    if logy:
        r200_bins = np.logspace(np.log10(np.nanmin(r200)+0.01),np.log10(np.nanmax(r200)),11)
#     _, r200_bins = np.histogram(r200,bins=10)
    keys, r200b = makeBins(r200,r200_bins)
    r200b_std = np.diff(r200_bins)/2
    
    r200hb = np.array([np.mean(r200h[idx]) for idx in keys])
    r200hb_std = np.array([np.std(r200h[idx]) for idx in keys])

    fig = plt.figure(figsize=(8,6))
    sc = plt.scatter(r200,r200h,s=75, alpha=0.25, color=gray)#,label='$scatter = %.1f$'%(np.std(ngals-nt))
    plt.errorbar(r200b,r200hb,xerr=r200b_std,yerr=r200hb_std,color=blue,linewidth=2.,fmt='o')
    if not logy:
        plt.plot(np.linspace(xlims[0],xlims[1]),np.linspace(ylims[0],ylims[1]),linestyle='--',color='r')
    
    plt.ylim(ylims)
    plt.xlim(xlims)

    plt.xlabel(ylabel,fontsize=22)
    plt.ylabel(r'$R_{200,%s}$ [$Mpc\, h^{-1}$]'%li,fontsize=22)
    plt.legend(fontsize=14)
    plt.title(title,fontsize=22)
    fig.tight_layout()

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