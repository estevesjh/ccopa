import numpy as np
import scipy.stats as st
import os

from astropy.table import Table, vstack
from astropy.io.fits import getdata

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.integrate as integrate
from scipy.interpolate import interp1d

import seaborn as sns; sns.set(color_codes=True)
# plt.rcParams.update({'font.size': 16})
sns.set_style("whitegrid")

# sns.set_style("white")
# sns.set_context("talk", font_scale=1.2)

## local libraries
import sys
sys.path.append(os.path.abspath("/home/johnny/Documents/Brandeis/CCOPA/lib/"))
import gaussianKDE as kde
from probRadial import doPDF, norm_constant



def checkPath(path):
    """ check the existance of a dir """
    if not os.path.isdir(path):
        os.makedirs(path)

def makeBin(variable, xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w


def chunks2(x, yedges):
    for i in range(len(yedges)-1):
        w, = np.where( (x>=yedges[i]) & (x<=yedges[i+1]) )
        yield w


def zoffset(z,zcls):
    return (z-zcls)/(1+zcls)

def get_pdf_clusters(indices,indices2,x,x2,pmem,xlims=(-0.15,0.15,30)):
    xedges = np.linspace(xlims[0],xlims[1],xlims[2]+1)
    _, x_bin = makeBin(x, xedges)

    x_bin = np.array(x_bin)

    p, p2 = [], []
    for w,w2 in zip(indices,indices2):
        if (w.size>2)&(w2.size>2):
            pdf = get_pdf(x_bin,x[w],weights=pmem[w])
            p.append(pdf)

            pdf2 = get_pdf(x_bin,x2[w2])
            p2.append(pdf2)

    return x_bin, np.array(p), np.array(p2)

def get_pdf(xvec,x,weights=None):    
    if weights is not None:
        kernel = kde.gaussian_kde(x,weights=weights,bw_method='silverman')
    else:
        kernel = kde.gaussian_kde(x,bw_method='silverman')
    return kernel(xvec)

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

def get_purity_completenes(gal,Ntrue,lcol='Pmem'):
    mask = gal["True"]==True
    Ntrue_selected = np.count_nonzero(mask)
    
    Ninterlopers = np.count_nonzero((np.logical_not(mask)))#np.sum(gal[lcol][(np.logical_not(mask))])
    Nselected = len(gal)#np.sum(gal[lcol])

    #Ntrue_selected = Ntrue - Nmissed
    P = (Nselected-Ninterlopers)*np.where(Nselected>1e-6,1/(Nselected+1e-6),0.)
    C = Ntrue_selected*np.where(Ntrue>1e-6,1/(Ntrue+1e-6),0.)

    return P, C

def get_purity_completenes_binned(xedges,gal,gal2,variable='redshit'):
    x = gal[variable]
    x2 = gal2[variable]

    indices, xbins = makeBin(x,xedges)
    indices2, xbins2 = makeBin(x2,xedges)

    p_bin, c_bin = [], []
    for idx,idx2 in zip(indices,indices2):
        #Ntrue = np.count_nonzero(gal["True"][idx]==True)
        Ntrue = np.count_nonzero(gal2["True"][idx2]==True)
        #Nmissed = Ntrue2-Ntrue
        
        p,c = get_purity_completenes(gal[idx],Ntrue)

        p_bin.append(p)
        c_bin.append(c)

    p_bin = np.array(p_bin)
    c_bin = np.array(c_bin)

    return xbins, p_bin, c_bin

def get_purity_completenes_threshold(xedges,gal,gal2,lcol='Pmem'):
    x = gal[lcol]
    x2 = gal2[lcol]

    indices, xbins = makeBin(x,xedges)
    mask = gal['True'] == True
    Ntrue = np.count_nonzero(mask)

    p_bin, c_bin = [], []
    for xi in (xbins):
        idx, = np.where(x>=xi)
        p,c = get_purity_completenes(gal[idx],Ntrue,lcol=lcol)

        p_bin.append(p)
        c_bin.append(c)

    p_bin = np.array(p_bin)
    c_bin = np.array(c_bin)

    return xbins, p_bin, c_bin

def get_curves(array,qr=25):
    y_bin = np.nanmedian(array,axis=0)
    cb_l = np.nanpercentile(array,100-qr,axis=0)
    cb_h = np.nanpercentile(array,qr,axis=0)

    return y_bin, cb_l, cb_h

def get_galaxy_density(radii,pmem,density=True):
    rvec = np.linspace(0.075,1.,13)#np.logspace(np.log10(0.05),np.log10(1.0), 10)
    #rvec = np.linspace(-0.15,+.15,60)
    area = np.ones_like(rvec[1:])

    if density: area = np.pi*(rvec[1:]**2-rvec[:-1]**2)

    indices, radii_bin = makeBin(radii, rvec)
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
        indices_bin, x_bin = makeBin(x[w], xedges)
        indices_bin2, _ = makeBin(x2[w2], xedges)

        sum_g = np.array([np.nansum(pmem[idx]) for idx in indices_bin])
        sum_g2 = np.array([np.nansum(pmem2[idx]) for idx in indices_bin2])

        s.append(sum_g)
        s2.append(sum_g2)

    return x_bin, np.array(s), np.array(s2)

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
    plt.clf()
    plt.close()

from sklearn.utils.multiclass import unique_labels

def plot_color_bin(z,mean,sigma,dz=0,scatter_mean=False,label='RS Model',lcolor='r',axs=None):

    b = np.arange(0.08,0.92,0.03)
    indices_list = list(chunks2(z, b))

    y_bin = np.array([ np.nanmedian(mean[idx]) for idx in indices_list])
    # std_bin = np.array([(np.nanmedian(sigma[idx])**2+np.nanstd(mean[idx])**2)**(1/2) for idx in indices_list])
    std_bin = np.array([np.nanmedian(sigma[idx]) for idx in indices_list])
    if scatter_mean:
        std_bin = np.array([ np.nanstd(mean[idx]) for idx in indices_list])

    x_bin = np.array([ np.median(z[idx]) for idx in indices_list])

    if axs is None:
        plt.errorbar(x_bin,y_bin,yerr=std_bin,color=lcolor,label=label)
    else:
        axs.errorbar(dz+x_bin,y_bin,yerr=std_bin,color=lcolor,label=label)


def _plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    ## https://github.com/scikit-learn/scikit-learn/issues/12700
    """
    from sklearn.metrics import confusion_matrix
    sns.set_style("white")
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title ='Confusion matrix, without normalization'
    
    cm = confusion_matrix(y_true, y_pred)
    cm2 = cm.copy()
    cm2[0,:] = cm[1,:]
    cm2[1,:] = cm[0,:]
    
    cm = cm2
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    # We want to show all ticks...
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=(classes), yticklabels=np.flip(classes),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylim(-.5,1.5)
    fig.tight_layout()

    sns.set_style("whitegrid")
    return ax

class checkPlots:
    """ This class makes check plots for the Copacabana page

    :param galaxies: astropy table, galaxy sample
    :param clusters: astropy table, cluster sample
    """
    def __init__(self,hdf,hdf2):
        self.hdf = hdf
        self.hdf2 = hdf2
    
    def close(self):
        plt.clf()
        plt.close()

    def hdf_close(self):
        self.hdf.close()
        self.hdf2.close()

    def get_pdf(self,hdf5,name_list,pdf_name):
        data =[]
        for name in name_list:
            pdf = hdf5['%s/%s'%(name,pdf_name)][:]
            data.append(pdf)

        return np.stack(data)

    def getColorLabel(self,color_number):
        color_list = [r'$(g-i)$',r'$(g-r)$',r'$(r-i)$',r'$(r-z)$',r'$(i-z)$']
        color_save = ['gi','gr','ri','rz','iz']
        return color_list[color_number], color_save[color_number]

    def plot_check_pdf_all_colors(self,save='./img/check_pdf_color'):
        number_of_color = range(5)

        for lcolor in number_of_color:
            self.plot_check_pdf_color(lcolor,save=save)
        
    def plot_check_pdf_color(self,lcolor,save='./img/check_pdf_color'):
        name_list = list(self.hdf.keys())

        chunk1 = self.get_pdf(self.hdf,name_list,'pdf_c_cls')
        chunk2 = self.get_pdf(self.hdf,name_list,'pdf_c_truth')
        # chunk2 = self.get_pdf(self.hdf2,name_list,'pdf_c_cls')

        color_vec = chunk1[0,:,0]
        pdf_all_clusters = chunk1[:,:,lcolor+1]
        pdf_true_all_clusters = chunk2[:,:,lcolor+1]

        inv_pdf_true_all = np.where(pdf_true_all_clusters>1e-9, 1/(pdf_true_all_clusters+1e-9),0.)
        residual_all_clusters = (pdf_all_clusters-pdf_true_all_clusters)*inv_pdf_true_all
        frac_error_all_cluster = residual_all_clusters*inv_pdf_true_all

        self.xlabel = self.getColorLabel(lcolor)[0]
        self.ylabel = r'PDF%s'%(self.xlabel)

        xmin,xmax = color_vec[np.argmax(pdf_all_clusters[0,:])]-0.75, color_vec[np.argmax(pdf_all_clusters[0,:])]+0.75
        xlims = (xmin,xmax)

        save = save+'_%s.png'%(self.getColorLabel(lcolor)[1])

        self.makePlot_check_pdf(color_vec,pdf_all_clusters,pdf_true_all_clusters,frac_error_all_cluster,xlims=xlims,save=save)
        self.close()

    def interpData(self,x,y,x_new):
        out = np.empty(x_new.shape, dtype=y.dtype)
        out = interp1d(x, y, kind='linear', fill_value='extrapolate', copy=False)(x_new)
        # yint = interp1d(x,y,kind='linear',fill_value='extrapolate')
        return out

    def get_pdfz(self,zvec,hdf5,name_list,label='cls'):
        data = []
        for name in name_list:   
            zcls = hdf5['%s'%(name)].attrs['zcls']
            pdf = hdf5['%s/%s'%(name,'pdf_z_%s'%(label))][:]
            
            pdfz = pdf[:,1]
            zold = pdf[:,0]

            z_offset = zoffset(zold,zcls)
            pdf_new = self.interpData(z_offset,pdfz,zvec)
            pdf_new = np.where(pdf_new<0,0.,pdf_new)
            data.append(pdf_new)
        
        return np.stack(data)

    def plot_check_pdf_redshfit(self,save='./img/check_pdf_redshift.png'):
        name_list = list(self.hdf.keys())

        zvec = np.arange(-0.2,0.2,0.005)

        pdf_all_clusters = self.get_pdfz(zvec,self.hdf,name_list)
        pdf_true_all_clusters = self.get_pdfz(zvec,self.hdf,name_list,label='truth')
        # pdf_true_all_clusters = self.get_pdfz(zvec,self.hdf2,name_list)

        inv_pdf_true_all = np.where(pdf_true_all_clusters>1e-9, 1/(pdf_true_all_clusters+1e-9),0.)
        residual_all_clusters = (pdf_all_clusters-pdf_true_all_clusters)*inv_pdf_true_all
        frac_error_all_cluster = residual_all_clusters*inv_pdf_true_all

        self.xlabel =  r'$(z-z_{cls})/(1+z_{cls})$'
        self.ylabel = r'PDF%s'%('(z)')
        self.makePlot_check_pdf(zvec,pdf_all_clusters,pdf_true_all_clusters,frac_error_all_cluster,save=save)
        self.close()

    def makePlot_check_pdf(self,x_bin,pdf,pdf2,perc_error,xlims=None,save='img.png'):
        y_bin, yb_l, yb_h  = get_curves(pdf,qr=5)
        y_bin2, yb_l2, yb_h2  = get_curves(pdf2,qr=5)
        residual, cb_l, cb_h  = get_curves(perc_error,qr=5)

        fig = plt.figure(figsize=(6,8))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.02,
                            height_ratios=[2,1])

        # fig.tight_layout()
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
        ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

        ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
        ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

        ax1.legend()

        ax2.scatter(x_bin,residual,color='b',marker='s',s=20)
        ax2.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')

        ax1.set_xlabel(self.xlabel,fontsize=16)
        ax1.set_ylabel(self.ylabel,fontsize=16)

        ax2.set_xlabel(self.xlabel,fontsize=16)
        ax2.set_ylabel('perc. error',fontsize=16)

        ax2.set_ylim(-0.75,0.75)

        if xlims is not None:
            ax1.set_xlim(xlims); ax2.set_xlim(xlims)
        plt.savefig(save)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


class generalPlots:
    """ This class makes validation plots for the Copacabana page

    :param galaxies: astropy table, galaxy sample
    :param clusters: astropy table, cluster sample
    """

    def __init__(self,title='Buzzard v1.6',path='./img/'):
        self.path = path
        self.title = title

    def close(self):
        plt.clf()
        plt.close()

    def plot_scaling_relation(self,cluster,cluster2,kind='richness'):
        if kind=='richness':
            x = cluster['Ngals']
            xtrue = cluster2['Ngals']
            xlabel, ylabel = r'N$_{True}$', r'N'
            nmin, nmax = 5., 1.5*np.max([xtrue,x])

        elif kind=='Mu':
            x = cluster['MU']
            xtrue = cluster2['MU']
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
        plt.title(self.title+" ; N={0:d}".format(len(x)))
        plt.xlim(nmin,nmax)
        plt.ylim(nmin,nmax)
        plt.yscale('log')
        plt.xscale('log')
        
        plt.savefig(self.path+'scaling_relation_%s.png'%(kind), bbox_inches = "tight")
        self.close()

    def plotResidual(self,cluster,cluster2,kind=['Mu','z'],bins=None):
        save = self.path+'residual_scaling_relation'

        if kind[0]=='richness':
            x = cluster['Ngals']
            xtrue = cluster2['Ngals']
            
            xlabel = r'$\frac{N-Ntrue}{N_{True}}$'#r'$N-N_{true}$'#
            nmin, nmax = 5., 1.75*np.max([xtrue,x])

        elif kind[0]=='Mu':
            xlabel = r'$\mu^{*}-\mu^{*}_{True}$' #r'$1 - \frac{\mu^{*}}{\mu^{*}_{True}}$'
            nmin, nmax = 20., 2.*np.max([xtrue,x])
        
        if kind[1]=='z':
            y = cluster['redshift']
            ylabel = r'redshift'
            save = save+'_redshift.png'
        
        elif kind[1]=='mass':
            y = cluster2['M200_true']
            ylabel = r'M$_{200} \; [10^{14} M_{\odot}/h]$'
            save = save+'_mass.png'
        
        elif kind[1]=='N':
            y = cluster2['Ngals']
            ylabel = r'N$_{True}$'
            save = save+'_ntrue.png'

        residue = (x-xtrue)/xtrue
        # residue[np.isnan(residue)] = 0.

        if bins is not None:
            indices, xbins = makeBin(y,bins)
            ybins = [np.mean(residue[idx]) for idx in indices]
            yerr = [np.std(residue[idx]) for idx in indices]


        # cutOutliers, = np.where( np.abs(residue-np.median(residue)) <= 3*np.std(residue) )

        # if cutOutliers.size>0:
        #     residue = residue[cutOutliers]
        #     y = y[cutOutliers]
        #     x,xtrue = x[cutOutliers], xtrue[cutOutliers]

        plt.figure(figsize=(8, 6))
        plt.scatter(y,residue,color='royalblue',s=50,alpha=0.3)
        plt.axhline(0.,linestyle='--',color='gray',linewidth=2.)

        if bins is not None:
            plt.errorbar(xbins,ybins,yerr=yerr,xerr=np.diff(bins)/2,
                        fmt='o',markersize=4,color='gray',elinewidth=2,markeredgewidth=2,capsize=3)
        
        if kind[1]=='mass':
            print('here')
            plt.xscale('log')

        plt.ylim(-1.,1.5*np.percentile(residue,95))
        # plt.ylim(-1.,1.)
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        # plt.title('Buzzard Simulations')
        plt.savefig(save)
        plt.close()
        
    def plot_fraction_pmem(self,galaxies,plt=plt,lcol='Pmem',color='r',label='_noLabel_',save=False):
        pmem = galaxies[lcol]
        mask = galaxies['True']==True

        pmem_edges = np.arange(0.0,1.1,0.1)
        idx, x_bin = makeBin(pmem, pmem_edges)
        fraction_bin = [len(galaxies[ix][mask[ix]])/(len(galaxies[ix])+1e-9) for ix in idx ]

        # fig = plt.figure(figsize=(6,6))
        plt.scatter(x_bin,fraction_bin,color=color,s=50,label=label)
        plt.plot(np.linspace(0,1.,100),np.linspace(0,1.,100),linestyle='--',color='k')

        if save:
            plt.savefig(os.path.join(self.path,'fraction_of_true_members_%s.png'%lcol))

    def plot_grid_fractions_pmem(self,galaxies,title='Buzzard v1.6'):
        fig, axis = plt.subplots(2, 2, figsize=(8,8), sharex='col', sharey='row')
        
        labels = [['Pr','Pz'],['Pc','Pmem']]
        for i in range(2):
            for j in range(2):
                self.plot_fraction_pmem(galaxies,plt=axis[i,j],lcol=labels[i][j])
                axis[i,j].set_xlabel(labels[i][j],fontsize=16)
                axis[i,j].set_ylabel('Fraction of true members',fontsize=16)

                if i ==0 and j==0: axis[0,0].legend(fontsize=14)

        fig.suptitle(self.title)
        plt.savefig(os.path.join(self.path,'fraction_of_true_members.png'),bb_box='tight')
        
        self.close()

    def plot_grid_histograms(self,galaxies,title='Buzzard v1.6'):
        fig, axis = plt.subplots(2, 2, figsize=(8,8), sharex='col', sharey='row')
        
        labels = [['Pr','Pz'],['Pc','Pmem']]
        for i in range(2):
            for j in range(2):
                self.plot_histograms(galaxies,axis[i,j],lcol=labels[i][j])

                if i ==0 and j==0: axis[0,0].legend(fontsize=14)

        fig.suptitle(self.title)
        plt.savefig(os.path.join(self.path,'probability_histograms.png'),bb_box='tight')
        
        self.close()

    def plot_histograms(self,galaxies,axis,lcol='Pmem'):
        prob = galaxies[lcol]
        
        mask = galaxies['True'] == True
        nmask = galaxies['True'] == False

        xbins = np.linspace(0.0,1.,10)
        
        # axis.hist(prob,bins=xbins,ec='blue', fc='none', lw=1.5, histtype='step',label='All')
        axis.hist(prob[mask],bins=xbins,ec='red', fc='none', lw=1.5, histtype='step',label='True members')
        axis.hist(prob[nmask],bins=xbins,ec='green', fc='none', lw=1.5, histtype='step',label='Non members')

        # axis.legend(loc='best',fontsize=16)
        axis.set_xlabel(lcol,fontsize=14)
        axis.set_ylabel('N')
    
    def plot_purity_completeness(self,gal,gal2):    
        # Ntrue = np.count_nonzero(gal["True"]==True)
        Ntrue = np.count_nonzero(gal2["True"]==True)

        pm,cm = get_purity_completenes(gal,Ntrue,lcol='Pmem')
        pc,cc = get_purity_completenes(gal,Ntrue,lcol='Pc')
        pz,cz = get_purity_completenes(gal,Ntrue,lcol='Pz')
        pr,cr = get_purity_completenes(gal,Ntrue,lcol='Pr')

        plt.scatter(pm,cm,marker='s',label=r'$P_{mem}$')
        plt.scatter(pc,cc,marker='p',label=r'$P_{color}$')
        plt.scatter(pz,cz,marker='P',label=r'$P_{z}$')
        plt.scatter(pr,cr,marker='*',label=r'$P_{r}$')

        plt.ylim(-0.05,1.05)
        plt.xlabel('Purity')
        plt.ylabel('Completenes')
        plt.legend()

        plt.savefig(os.path.join(self.path,'purity_completeness_probs.png'))
        self.close()

    def plot_purity_completeness_threshold(self,gal,gal2,lcol):
        xedges = np.arange(-0.02,1.01,0.05)

        x_bin, p_bin, c_bin = get_purity_completenes_threshold(xedges,gal,gal2,lcol=lcol)

        plt.scatter(x_bin,p_bin,color='b',label='Purity')
        plt.scatter(x_bin,c_bin,color='r',label='Completeness')
        
        plt.axvline(x_bin[np.argmax(p_bin*c_bin)],color='k',linestyle='--',label='max(P*C)')
        
        plt.ylim(-0.05,1.05)
        plt.xlabel(lcol)
        plt.ylabel('P/C')
        plt.legend()

        plt.savefig(os.path.join(self.path,'purity_completeness_threshold_%s.png'%lcol))
        self.close()

    def plot_purity_completeness_variable(self,gal,gal2,xedges,column):
        x_bin, p_bin, c_bin = get_purity_completenes_binned(xedges,gal,gal2,variable=column)

        plt.scatter(x_bin,p_bin,color='b',label='Purity')
        plt.scatter(x_bin,c_bin,color='r',label='Completeness')
        # plt.axvline(x_bin[np.argmax(p_bin*c_bin)],color='k',linestyle='--',label='max(P*C)')
        # plt.ylim(-0.05,1.05)
        plt.xlabel(column)
        plt.ylabel('P/C')
        plt.legend()

        plt.savefig(os.path.join(self.path,'purity_completeness_%s.png'%column))
        self.close()

    def plot_confusion_matrix(self,gal,column,title=None):
        from sklearn.metrics import precision_recall_curve

        scores=gal[column]
        true_members = np.where(gal['True'],1.,0.)
        precisions, recalls, thresholds = precision_recall_curve(true_members, scores)

        idx = np.argmax(recalls*precisions)
        recall_opt_precision = recalls[idx]
        precision_opt_precision = precisions[idx]
        threshold_opt_precision = thresholds[idx]

        y_pred_opt = scores>=threshold_opt_precision

        columns = np.array(['Non members', 'members'])
        _plot_confusion_matrix(true_members, y_pred_opt,columns,normalize=True,title=title,cmap=plt.cm.Blues)

        plt.savefig(os.path.join(self.path,'confusion_matrix.png'))
        self.close()

        return threshold_opt_precision

    def plot_roc_curve(self,gal,column,threshold_opt_precision,label=None):
        from sklearn.metrics import roc_curve,roc_auc_score

        scores=gal[column]
        true_members = np.where(gal['True'],1.,0.)

        fpr, tpr, thresholds2 = roc_curve(true_members, scores)
        roc_auc_value = roc_auc_score(true_members, scores)
        
        idx, = np.where(thresholds2==threshold_opt_precision)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=3, label='AUC = %.2f'%(roc_auc_value))
        plt.plot([0, 1], [0, 1], 'k--', label='random') # dashed diagonal
        plt.axis([0, 1.02, 0, 1.02])
        plt.xlabel('False Positive Rate (Contamination)', fontsize=16)
        plt.ylabel('True Positive Rate (Completeness)', fontsize=16)
        plt.xticks(np.arange(0.1,1.01,0.1))
        plt.grid(True)                    

        plt.plot([fpr[idx], fpr[idx]], [0.,tpr[idx]], "r:")
        plt.plot([0.0, fpr[idx]], [tpr[idx],tpr[idx]], "r:") 
        plt.plot([fpr[idx]], [tpr[idx]], "ro", label='optimal')           
        plt.legend()
        plt.savefig(os.path.join(self.path,"roc_curve_plot")) 
        
        self.close()

    def plot_precision_recall_vs_threshold(self,gal,column,lcol='$P_{mem}$',title=None):
        from sklearn.metrics import precision_recall_curve
        
        scores=gal[column]
        true_members = np.where(gal['True'],1.,0.)
        precisions, recalls, thresholds = precision_recall_curve(true_members, scores)

        idx = np.argmax(recalls*precisions)
        recall_opt_precision = recalls[idx]
        precision_opt_precision = precisions[idx]
        threshold_opt_precision = thresholds[idx]

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions[:-1], "b-", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.plot([threshold_opt_precision, threshold_opt_precision], [0., recall_opt_precision], "r:")
        plt.plot([0., threshold_opt_precision], [precision_opt_precision, precision_opt_precision], "r:")
        plt.plot([0., threshold_opt_precision], [recall_opt_precision, recall_opt_precision], "r:")
        plt.plot([threshold_opt_precision], [precision_opt_precision], "ro")
        plt.plot([threshold_opt_precision], [recall_opt_precision], "ro")

        plt.legend(fontsize=16)
        plt.xlabel(r'Threshold: $%s$'%lcol, fontsize=16)
        plt.grid(True)         
        plt.ylim(0.0,1.)
        plt.xlim(0.,1.01)
        ticks0 = np.arange(0.1,1.01,0.1)
        plt.xticks(ticks0)
        plt.yticks(ticks0)

        plt.title("Precision, Recall vs Threshold",fontsize=16)
        plt.savefig(os.path.join(self.path,"precision_recall_threshold_plot"))
        self.close()
        

    def plot_validation_pdf_color(self,gal,gal2,cat,cat2,lcolor='delta_rs',method='N',save='./img/pdf_color_validation.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat['CID']))

        color = gal[lcolor]
        color2 = gal2[lcolor]

        # pmem = gal['Pz']#*gal['Pr']
        pmem = gal['Pmem']

        # pmem2 = gal2['Pz']#*gal2['Pr']
        pmem2 = gal2['Pmem']

        ### curve 1
        x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,color,color2,pmem,xlims=(-1.5,0.5,30))
        # x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,color,color2,pmem,xlims=(0.25,2.5,30))

        ### curve 2
        if method=='N':
            x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,color,color2,pmem,pmem2,xlims=(-1.5,0.5,30))
            # x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,color,color2,pmem,pmem2,xlims=(0.25,2.5,30))
            self.ylabel, self.ylabel2 = r'$PDF(color)$', r'$N_{gals} $'

        else:
            smass = gal['mass']
            smass2 = gal2['mass']
            
            pmem_mu = pmem*10**(smass)/10**12
            pmem_mu_2 = pmem2*10**(smass2)/10**12

            x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,color,color2,pmem_mu,pmem_mu_2,xlims=(-1.5,0.5,30))
            self.ylabel, self.ylabel2 = r'$PDF(z)$', r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$'
        
                
        args = x_bin,x_bin2,pdf,pdf2,sg,sg2

        self.kind = 'color'
        self.xlabel, self.xlabel2 = r'$\Delta(g-r)_{RS}$', r'$\Delta(g-r)_{RS}$'

        self.plot_grid_pdf(*args,save=save)

    def plot_validation_pdf_redshift(self,gal,gal2,cat,cat2,method='N',save='./img/pdf_redshift_validation.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat['CID']))

        z = zoffset(gal['z'],gal['redshift'])
        z2 = zoffset(gal2['z'],gal2['redshift'])

        # pmem = gal['Pz']*gal['Pr']
        pmem = gal['Pmem']

        # pmem2 = gal2['Pz']*gal2['Pr']
        pmem2 = gal2['Pmem']

        ### curve 1
        x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,z,z2,pmem,xlims=(-0.15,0.15,100))

        ### curve 2
        if method=='N':
            x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem,pmem2,xlims=(0.,0.15,50))
            self.ylabel, self.ylabel2 = r'$PDF(z)$', r'$N_{gals} $'

        else:
            smass = gal['mass']
            smass2 = gal2['mass']

            pmem_mu = pmem*10**(smass)/10**12
            pmem_mu_2 = pmem2*10**(smass2)/10**12

            x_bin2, sg, sg2 = get_sum_clusters(indices,indices2,np.abs(z),np.abs(z2),pmem_mu,pmem_mu_2,xlims=(0.,0.15,50))
            self.ylabel, self.ylabel2 = r'$PDF(z)$', r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$'
        
        # args = x_bin,y_bin,y_bin2,yb_l,yb_h, yb_l2, yb_h2, residual, cb_l, cb_h,sy_bin,sy_bin2, syb_l, syb_h,syb_l2,syb_h2,sresidual,scb_l,scb_h
        
        args = x_bin,x_bin2,pdf,pdf2,sg,sg2

        self.kind = 'redshift'
        self.xlabel, self.xlabel2 = r'$(z-z_{cls})/(1+z_{cls})$', r'$ \|z-z_{cls}\|/(1+z_{cls}$'

        self.plot_grid_pdf(*args,save=save)

    def plot_validation_pdf_radial(self, gal,gal2,cat,cat2, method='N', save='./img/pdf_radial_validation.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat['CID']))

        radii = gal['Rn']
        radii2 = gal2['Rn']

        # pmem = gal['Pz']*gal['Pr']
        pmem = gal['Pmem']
        
        # pmem2 = gal2['Pz']*gal2['Pr']
        pmem2 = gal2['Pmem']

        ### curve 1
        radii_bin, ng, ng2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem,pmem2)

        ### curve 2
        if method=='N':
            radii_bin, sg, sg2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem,pmem2,density=False)
            self.ylabel, self.ylabel2 = r'$\Sigma \; [\# gals / Mpc^{2}]$', r'$N_{gals} $'

        else:
            smass = gal['mass']
            smass2 = gal2['mass']

            pmem_mu = pmem*10**(smass)/10**12
            pmem_mu_2 = pmem2*10**(smass2)/10**12

            radii_bin, sg, sg2 = get_galaxy_density_clusters(indices,indices2,radii,radii2,pmem_mu,pmem_mu_2,density=False)
            self.ylabel, self.ylabel2 = r'$\Sigma \; [\# gals / Mpc^{2}]$', r'$\mu_{\star} \; [ 10^{12} M_{\odot}]$'

        # args = x_bin,y_bin,y_bin2,yb_l,yb_h, yb_l2, yb_h2, residual, cb_l, cb_h,sy_bin,sy_bin2, syb_l, syb_h,syb_l2,syb_h2,sresidual,scb_l,scb_h
        
        args = radii_bin,radii_bin,ng,ng2,sg,sg2

        self.kind = 'radial'
        self.xlabel, self.xlabel2 = r'R [$R_{200}$]', r'R [$R_{200}$]'

        self.plot_grid_pdf(*args,save=save)

    def plot_grid_pdf(self,*args,save='./img/pdf_radial_validation.png'):
        # x_bin,y_bin,y_bin2,yb_l,yb_h, yb_l2, yb_h2, residual, cb_l, cb_h,sy_bin,sy_bin2, syb_l, syb_h,syb_l2,syb_h2,sresidual,scb_l,scb_h = args
        x_bin,x_bin2,pdf,pdf2,sg,sg2 = args

        y_bin, yb_l, yb_h  = get_curves(pdf)
        y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

        per_error = np.where(pdf/pdf2>1e6,0.,(pdf-pdf2)/pdf2)
        residual, cb_l, cb_h  = get_curves(per_error)

        sg, sg2 = np.cumsum(sg,axis=1), np.cumsum(sg2,axis=1)
        sy_bin, syb_l, syb_h  = get_curves(sg)
        sy_bin2, syb_l2, syb_h2  = get_curves(sg2)

        sper_error = np.where(sg/sg2>1e6,0.,(sg-sg2)/sg2)
        sresidual, scb_l, scb_h  = get_curves(sper_error)

        fig = plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.05,
                            height_ratios=[3,1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])

        ## curve 1
        ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
        ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

        ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
        ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

        ax1.legend()

        ax3.scatter(x_bin,residual,color='b',marker='s',s=20)
        ax3.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')

        ## curve 2
        ax2.scatter(x_bin2,sy_bin,color='b', s=20, marker='s', label='Estimated Distrub.')
        ax2.fill_between(x_bin2, syb_l, syb_h, color="b", alpha=0.25, label='_nolabel_')

        ax2.scatter(x_bin2,sy_bin2,color='r', s=20, alpha=0.9, label='True Distrub.')
        ax2.fill_between(x_bin2, syb_l2, syb_h2, color="r", alpha=0.25, label='_nolabel_')

        ax2.legend()

        ax4.scatter(x_bin2,sresidual,color='b',marker='s',s=20)
        ax4.fill_between(x_bin2,scb_l,scb_h,color="b", alpha=0.25, label='_nolabel_')

        # ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax3.set_ylim(-0.5,0.5)

        # ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax4.set_ylim(-0.5,0.5)
        
        ax3.set_xlabel(self.xlabel,fontsize=16)
        ax3.set_ylabel('perc. error',fontsize=16)

        ax4.set_xlabel(self.xlabel2,fontsize=16)
        ax4.set_ylabel('perc. error',fontsize=16)

        ax1.set_ylabel(self.ylabel,fontsize=16)
        ax2.set_ylabel(self.ylabel2,fontsize=16)

        if self.kind=='radial':
            ax1.set_yscale('log')
            ax1.set_xscale('log')
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


        plt.savefig(save)
        self.close()

    def plot_PDF2(self,ax1,ax2,indices,indices2,xvec,xvec2,pmem,pmem2,radial_mode=False,method='None',labels=['',''],title='',xlims=(-1.5,0.5,30)):
    ### it works for color and redshift pdfs
    ### for radial distribution set the radial_mode to true   
        if radial_mode:
            x_bin, pdf, pdf2 = get_galaxy_density_clusters(indices,indices2,xvec,xvec2,pmem,pmem2)

        else:
            x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,xvec,xvec2,pmem,xlims=xlims)
        
        y_bin, yb_l, yb_h  = get_curves(pdf)
        y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

        per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf2)
        residual, cb_l, cb_h  = get_curves(per_error)

        ## To Do!
        # if method=='Ngals':
        #     x_bin, pdf, pdf2 = get_sum_clusters(indices,indices2,xvec,xvec2,pmem,pmem2,xlims=(-1.5,0.5,30))
        #     pdf, pdf2 = np.cumsum(pdf,axis=1), np.cumsum(pdf2,axis=1)
        #     y_bin, yb_l, yb_h  = get_curves(pdf)
        #     y_bin2, yb_l2, yb_h2  = get_curves(pdf2)

        #     per_error = np.where(pdf/pdf2>1000,0.,(pdf-pdf2)/pdf)
        #     # sper_error = (sg/sg2)
        #     residual, cb_l, cb_h  = get_curves(per_error)
        ##
        
        ax1.scatter(x_bin,y_bin,color='b', s=20, marker='s', label=labels[0])
        ax1.fill_between(x_bin, yb_l, yb_h, color="b", alpha=0.25, label='_nolabel_')

        ax1.scatter(x_bin,y_bin2,color='r', s=20, alpha=0.9, label=labels[1])
        ax1.fill_between(x_bin, yb_l2, yb_h2, color="r", alpha=0.25, label='_nolabel_')

        ax1.legend()

        ax2.scatter(x_bin,residual,color='b',marker='s',s=20)
        ax2.fill_between(x_bin,cb_l,cb_h,color="b", alpha=0.25, label='_nolabel_')
        ax2.axhline(0.,linestyle='--')
        
        ax1.set_title(title)
    
    def plot_probabilities_colorPDF(self,gal,gal2,cat,cat2,lcolor='delta_rs',save='./img/prob_color_PDF.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat['CID']))

        color = gal[lcolor]
        color2 = gal2[lcolor]

        weights_label = ['Pz','Pc','Pr','Pmem']
        weights = [gal[col] for col in weights_label]
        lims = []

        fig = plt.figure(figsize=(12,6))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

        axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
        for i in range(len(weights_label)):
            ax1, ax2 = axis_list[i]
            
            pmem = weights[i]
            ti = 'Weighted by %s'%(weights_label[i])
            label=['_','_']

            ax2.set_xlabel(r'$\Delta(g-r)_{RS}$',fontsize=16)
            if i==0:
                label = ['Estimated Distrib.','True Distrib.']
                ax1.set_ylabel(r'$PDF(color)$',fontsize=16)
                ax2.set_ylabel(r'frac. error',fontsize=16)
                

            self.plot_PDF2(ax1,ax2,indices,indices2,color,color2,pmem,None,labels=label,title=ti,xlims=(-1.5,0.5,30))
            # self.plot_PDF2(ax1,ax2,indices,indices2,color,color2,pmem,None,labels=label,title=ti,xlims=(0.,2.5,30))
            ax2.set_ylim(-0.5,0.5)
            lims.append(ax1.get_ylim())

        for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

        plt.savefig(save,bb_box='tight')
        plt.clf()

    def plot_probabilities_redshiftPDF(self,gal,gal2,cat,cat2,save='./img/prob_redshift_PDF.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat['CID']))

        z = zoffset(gal['z'],gal['redshift'])
        z2 = zoffset(gal2['z'],gal2['redshift'])

        weights_label = ['Pr','Pc','Pz','Pmem']
        weights = [gal[col] for col in weights_label]
        lims = []

        fig = plt.figure(figsize=(12,6))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

        axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
        for i in range(len(weights_label)):
            ax1, ax2 = axis_list[i]
            
            pmem = weights[i]
            ti = 'Weighted by %s'%(weights_label[i])
            label=['_','_']

            ax2.set_xlabel(r'$(z-z_{cls})/(1+z_{cls})$',fontsize=16)
            if i==0:
                label = ['Estimated Distrib.','True Distrib.']
                ax1.set_ylabel(r'$PDF(z)$',fontsize=16)
                ax2.set_ylabel(r'frac. error',fontsize=16)
                
            self.plot_PDF2(ax1,ax2,indices,indices2,z,z2,pmem,None,labels=label,title=ti,xlims=(-0.15,0.15,100))
            ax2.set_ylim(-0.5,0.5)
            lims.append(ax1.get_ylim())

        for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

        plt.savefig(save,bb_box='tight')
        plt.clf()

    def plot_probabilities_radialPDF(self,gal,gal2,cat,cat2,save='./img/prob_radial_PDF.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat2['CID']))

        radii = gal['Rn']
        radii2 = gal2['Rn']

        weights_label = ['Pr','Pc','Pz','Pmem']
        weights = [gal[col] for col in weights_label]
        
        pmem2 = gal2['Pmem']

        fig = plt.figure(figsize=(12,6))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 4, wspace=0.3, hspace=0.05, height_ratios=[3,1])

        axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
        lims = []
        for i in range(len(weights_label)):
            ax1, ax2 = axis_list[i]
            
            pmem = weights[i]
            ti = 'Weighted by %s'%(weights_label[i])
            label=['_','_']

            ax2.set_xlabel(r'R [$R_{200}$]',fontsize=16)
            if i==0:
                label = ['Estimated Distrib.','True Distrib.']
                ax1.set_ylabel(r'$\Sigma \; [\# gals / Mpc^{2}]$',fontsize=16)
                ax2.set_ylabel(r'frac. error',fontsize=16)
                
            self.plot_PDF2(ax1,ax2,indices,indices2,radii,radii2,pmem,pmem2,radial_mode=True,labels=label,title=ti,xlims=(-0.20,0.20,50))
            ax2.set_ylim(-0.5,0.5)
            lims.append(ax1.get_ylim())

        for axs in (axis_list): axs[0].set_ylim(-0.1,np.max(np.array(lims)[:,1]))

        plt.savefig(save,bb_box='tight')
        plt.clf()


    def plot_multiple_scaling_relations(self,gal,gal2,cat,cat2,save='./img/prob_scaling_relation.png'):
        indices = list(chunks(gal['CID'],cat['CID']))
        indices2 = list(chunks(gal2['CID'],cat2['CID']))

        weights_label = ['Pr','Pc','Pz','Pmem']
        ngals_list = [np.array([np.nansum(gal[col][idx]) for idx in indices]) for col in weights_label]
        ngals2 = np.array([np.nansum(gal2['Pmem'][idx]) for idx in indices2])

        fig = plt.figure(figsize=(20,6))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 4, wspace=0.15, hspace=0.01, height_ratios=[3,1])

        axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+4])] for i in range(4)]
        for i in range(len(weights_label)):
            ax1, ax2 = axis_list[i]
            ti = r'$\sum{%s}$'%(weights_label[i])

            ax2.set_xlabel(r'$N_{gals,True}$',fontsize=16)
            # ax1.set_ylabel(r'$\sum{%s}$'%(weights_label[i]),fontsize=16)

            if i==0:
                ax2.set_ylabel(r'frac. error',fontsize=16)

            self.plot_scaling_relations(ax1,ax2,ngals_list[i],ngals2,title=ti,nmin=5,nmax=1.5*np.max(ngals2))

        plt.savefig(save,bb_box='tight')
        plt.close()

    def plot_scaling_relations(self,ax1,ax2,x,x2,title='Title',nmin=5,nmax=300):
        linreg=lin_reg(x2,x)

        idx = np.argsort(x2)
        xt,yh = x2[idx],linreg['Yhat'][idx]

        b0 = round(linreg['b0'],3)
        b1 = round(linreg['b1'],3)
        cb_u, cb_l = linreg['cb_u'], linreg['cb_l']

        xs = np.linspace(nmin,nmax,200)

        ax1.plot(xt,yh, color="b", label='y=%.2f+%.2fx'%(b0,b1))
        ax1.fill_between(xt, cb_l, cb_u, color="b", alpha=0.25, label='_nolabel_')
        ax1.plot(xt,cb_l, color="b", label='_nolabel_')
        ax1.plot(xt,cb_u, color="b", label='_nolabel_')
        ax1.scatter(x2,x, color="b", s=20, alpha=0.3, label='_nolabel_')
        ax1.plot(xs,xs,color='k',linestyle='--', label='y = x')

        ### residual
        ax2.scatter(x2,(x-x2)/x2,s=20,alpha=0.3,color='b',label=r'$ \sigma = %.3f $'%(np.std(x-x2)))
        ax2.axhline(0,linestyle='--',color='b')

        ax1.legend(loc='lower right')
        ax2.legend()
        ax1.set_title(title)

        ax1.set_xlim(nmin,nmax)
        ax1.set_ylim(nmin,nmax)
        ax2.set_xlim(nmin,nmax)
        ax2.set_ylim(-1.,1.)

        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    

    def validating_color_model(self,gal,cat,par_label,lcolor='g-r'):
        #### Color plots
        # color = gal[lcolor]
        # pmem = gal['Pmem']

        zcls= gal['redshift']
        zrs = cat['redshift']

        mur = cat['rs_param_%s'%(par_label)][:,0]
        sigr = cat['rs_param_%s'%(par_label)][:,1]

        mub = cat['bc_param_%s'%(par_label)][:,0]
        sigb = cat['bc_param_%s'%(par_label)][:,1]

        # scale = 10*pmem**(1/2)
        # plt.scatter(zcls,color,s=scale,color='k',alpha=0.08)
        # plt.scatter(zrs,mur,color='r',alpha=0.5,s=10,label='RS: mean')
        # plt.scatter(zrs,mub,color='b',alpha=0.5,s=10,label='BC: mean')
        
        plot_color_bin(zrs,mur,sigr,lcolor='#E74C3C')
        plot_color_bin(zrs,mub,sigb,label='blue cloud',lcolor='#3498DB')
        plt.legend()
        # plt.xlim(0.08,0.92)
        plt.ylim(0.,np.max(mur)+2*np.max(sigr))
        plt.xlabel('redshift')
        plt.ylabel(r'$(%s)$'%(lcolor))
        plt.savefig(os.path.join(self.path,'color_model_%s.png'%(lcolor)))
        self.close()

        # sns.kdeplot(zcls,color,color='b', shade=True, zorder=3, shade_lowest=False, alpha=0.6)
        # plt.hexbin(zcls, color, pmem, gridsize=100, cmap=plt.cm.get_cmap('Blues_r', 45), reduce_C_function=np.sum, vmin=-10, vmax=100) 

        # 
        # plt.plot(x_bin,cb_u,color='r')

    def validating_color_model_grid(self,gal,cat,color_list,lcolor=None,sigma=False,fraction=False):
        if lcolor is None: lcolor = color_list

        plt.clf()
        fig = plt.figure(figsize=(16,10))
        fig.subplots_adjust(hspace=0.03, wspace=0.25)
        
        for i,li in enumerate(color_list):
            axs = fig.add_subplot(2, 3, i+1)
            
            # #### Color plots
            # color = gal[lcolor[i]]
            # pmem = gal['Pmem']
      
            zrs = cat['redshift']
            mur = cat['rs_param_%s'%(li)][:,0]
            sigr = cat['rs_param_%s'%(li)][:,1]
            alpr = cat['rs_param_%s'%(li)][:,2]

            mub = cat['bc_param_%s'%(li)][:,0]
            sigb = cat['bc_param_%s'%(li)][:,1]
            alpb = cat['bc_param_%s'%(li)][:,2]

            # scale = 10*pmem**(1/2)
            if (not sigma)&(not fraction):
                # axs.scatter(zcls,color,s=scale,color='k',alpha=0.08)
                axs.scatter(zrs,mur,color='r',alpha=0.4,s=10,label='RS: mean')
                axs.scatter(zrs,mub,color='b',alpha=0.4,s=10,label='BC: mean')
                plot_color_bin(zrs,mur,sigr,scatter_mean=True,lcolor='#F1948A')
                plot_color_bin(zrs,mub,sigb,scatter_mean=True,label='blue cloud',lcolor='#85C1E9')
                # plt.xlim(0.08,0.92)
                axs.set_ylim(0.,np.max(mur)+1.2*np.max(sigr))
                axs.set_ylabel(r'$(%s)$'%(lcolor[i]),fontsize=16)

            if sigma:
                axs.scatter(zrs,sigr,color='r',alpha=0.4,s=10,label='RS: mean')
                axs.scatter(zrs,sigb,color='b',alpha=0.4,s=10,label='BC: mean')
                plot_color_bin(zrs,sigr,sigr,scatter_mean=True,lcolor='#F1948A')
                plot_color_bin(zrs,sigb,sigb,scatter_mean=True,label='blue cloud',lcolor='#85C1E9')
                # plt.xlim(0.08,0.92)
                # axs.set_ylim(0.,np.max(sigr)+1.2*np.max(sigr))
                axs.set_ylabel(r'$\sigma_{%s}$'%(lcolor[i]),fontsize=16)


            if fraction:
                axs.scatter(zrs,alpr,color='r',alpha=0.4,s=10,label='RS: mean')
                axs.scatter(zrs,alpb,color='b',alpha=0.4,s=10,label='BC: mean')
                plot_color_bin(zrs,alpr,sigr,scatter_mean=True,lcolor='#F1948A')
                plot_color_bin(zrs,alpb,sigb,scatter_mean=True,label='blue cloud',lcolor='#85C1E9')
                # plt.xlim(0.08,0.92)
                # axs.set_ylim(0.,np.max(sigr)+1.2*np.max(sigr))
                axs.set_ylabel(r'$w ({%s})$'%(lcolor[i]),fontsize=16)

            if i>=3:
                axs.set_xlabel('redshift',fontsize=16)
        
        if (not sigma)&(not fraction):
            plt.suptitle('Mean Color: GMM',fontsize=16)
            plt.savefig(os.path.join(self.path,'color_model_grid.png'), bbox_inches = "tight")
            
        if sigma:
            plt.suptitle('Sigma: GMM',fontsize=16)
            plt.savefig(os.path.join(self.path,'color_model_sigma_grid.png'), bbox_inches = "tight")
            
        if fraction:
            plt.suptitle('Fraction: GMM',fontsize=16)
            plt.savefig(os.path.join(self.path,'color_model_fraction_grid.png'), bbox_inches = "tight")

        self.close()

    def validating_color_model_residual(self,gal,cat,gal2,cat2,par_label,lcolor='g-r'):
        #### Color plots
        color = gal[lcolor]
        zrs = cat['redshift']

        color2 = gal2[lcolor]
        zrs2 = cat2['redshift']

        mur = cat['rs_param_%s'%(par_label)][:,0]
        sigr = cat['rs_param_%s'%(par_label)][:,1]

        mub = cat['bc_param_%s'%(par_label)][:,0]
        sigb = cat['bc_param_%s'%(par_label)][:,1]

        mur2 = cat2['rs_param_%s'%(par_label)][:,0]
        sigr2 = cat2['rs_param_%s'%(par_label)][:,1]

        mub2 = cat2['bc_param_%s'%(par_label)][:,0]
        sigb2 = cat2['bc_param_%s'%(par_label)][:,1]


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
        
        plt.savefig(os.path.join(self.path,'color_model_residual_%s.png'%(lcolor)))
        self.close()


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

class clusterPlots:
    """ This class makes plots for individual clusters that apears on the Copacabana page

    :param galaxies: astropy table, galaxy sample
    :param clusters: astropy table, cluster sample
    :param indice: int, take only the cluster with index indice
    """

    def __init__(self,hdf,hdf2,galaxies,galaxies2,clusters,indice):
        self.idx = indice
        self.cluster = clusters[indice]

        self.name = str(self.cluster['CID'])
        self.path = './img/%s'%(self.name)
        self.title = 'CID : %s'%self.name

        self.galIdx = self.getGalaxyIndices(galaxies,clusters)
        self.galaxy = galaxies[self.galIdx]

        self.galIdx2 = self.getGalaxyIndices(galaxies2,clusters)
        self.galaxy2 = galaxies2[self.galIdx2]

        self.group = hdf['/%s'%(self.name)]
        self.group2 = hdf2['/%s'%(self.name)]

        self.color_list = ['g-r','g-i','r-i','r-z','i-z']

        checkPath(self.path)

        self.close()

    def close(self):
        plt.clf()
        plt.close()

    def getGalaxyIndices(self,galaxies,clusters):
        cid = clusters['CID'][self.idx]
        w, = np.where(galaxies['CID']==cid)
        return w

    def interpolation(self,x,y,xmin=0.,xmax=100,npoints=1000):
        bins = np.linspace(xmin,xmax,npoints)
        yint = interp1d(x,y,fill_value='extrapolate')

        return bins, yint(bins)

    def makeBin(self, variable, xedges):
        xbins = (xedges[1:]+xedges[:-1])/2
        indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
        return indices, xbins
        
    def get_galaxy_density(self,radii,pmem,density=True):
        """ it computes the galaxy surface density """
        rvec = np.linspace(0.05,1.05,11)#np.logspace(np.log10(0.05),np.log10(1.0), 10)
        area = np.ones_like(rvec[1:])

        if density: area = np.pi*(rvec[1:]**2-rvec[:-1]**2)

        indices, radii_bin = self.makeBin(radii, rvec)
        ng = np.array([np.nansum(pmem[idx]) for idx in indices])/area

        return radii_bin, ng
    
    def makePlotRadial(self, R200=1, conc=3, title=None):
        """ It plots the surface galaxy density profile

        :param galaxies: astropy table, cols: R, Pmem, True
        :param norm: float, normalization factor of the cluster

        NFW profile
        :param R200: float, units in Mpc
        :param conc: float
        """
        area = np.pi*(1)**2
        norm = self.cluster['Norm']*self.cluster['Nbkg']*area

        radii = self.galaxy['R'][:]
        pmem = self.galaxy['Pmem'][:]

        radii2= self.galaxy2['R'][:]
        pmem2 = self.galaxy2['Pmem'][:]
        
        radii_bin, ngals = self.get_galaxy_density(radii,pmem,density=True)
        radii_binT, ngalsT = self.get_galaxy_density(radii2,pmem2,density=True)

        rvec = np.linspace(0.01,R200,100)
        constant = norm_constant(R200,c=conc)
        sigma = doPDF(rvec,R200,c=conc)*constant

        norm = 2*np.pi*integrate.trapz(ngalsT*radii_binT,x=radii_binT)
        norm2 = 2*np.pi*integrate.trapz(ngals*radii_bin,x=radii_bin)
        # norm2 = np.sum(pmem[pmem>0.1])#len(radii[mask])

        plt.figure(figsize=(8,8))
        plt.scatter(radii_bin,ngals/norm2,color='blue', label='Data')
        plt.scatter(radii_binT,ngalsT/norm,color='red', label='True')
        # x,ng = np.histogram(radii,bins=rvec,weights=pmem)
        # rmed = (rvec[1:]+rvec[:-1])/2
        # plt.bar(radii_bin,ngals,width=0.1,align='center')
        # plt.bar(radii_binT,ngalsT,width=0.1,align='center')

        plt.plot(rvec, sigma, color='k', linestyle='--', label=r'NFW(R200 = %.2f Mpc, c=%.1f)'%(R200,conc))
        # plt.plot(rvec, sigma, color='gray', linestyle='--', label=r'$N_{gals}$ NFW(R200 = %.2f Mpc, c=%.1f)'%(R200,conc))
        
        plt.yscale('log')
        plt.xscale('log')
        
        # plt.ylabel(r'$\Sigma \; [gals/Mpc^{2}]$',fontsize=16)
        plt.ylabel(r'$PDF(R)$',fontsize=16)
        plt.xlabel(r'$R \; [Mpc]$',fontsize=16)
        
        # plt.xticks([0.1,0.3,0.5,0.7,0.9,1.])
        plt.xlim(0.07,1.1)

        plt.legend(loc='lower left',fontsize=16)
        if title is None: title = 'CID : %s'%self.name
        plt.title(title, fontsize=16)
        plt.savefig(os.path.join(self.path,'radial_profile_'+self.name+'.png'))
        self.close()

    def interpData(self,x,y,x_new):
        out = np.empty(x_new.shape, dtype=y.dtype)
        out = interp1d(x, y, kind='linear', fill_value='extrapolate', copy=False)(x_new)
        # yint = interp1d(x,y,kind='linear',fill_value='extrapolate')
        return out

    def radialDistribution(self,title=None):
        zcls = self.group.attrs['zcls']
        r200 = self.group.attrs['R200_true']
        norm = self.group.attrs['norm']
        nbkg = self.group.attrs['nbkg']
        
        area = np.pi*(1)**2
        ngals_true = len(self.galaxy2)/area

        ncls = norm*nbkg ## change to norm in the future

        radii = self.group['pdf_r_cls'][:,0]
        # radii2 = self.group2['pdf_r_cls'][:,0]

        pdf = self.group['pdf_r_cls'][:,1]
        pdf_cf = self.group['pdf_r_cls_field'][:,1]
        pdf_bkg = self.group['pdf_r_field'][:,1]
        pdf2 = self.group['pdf_r_truth'][:,1]
        
        # pdf2 = self.interpData(radii2,pdf2,radii)
        # norm_cf = 2*np.pi*integrate.trapz(radii*pdf_cf,x=radii)
        # norm_bkg = 2*np.pi*integrate.trapz((radii+4)*pdf_bkg,x=(radii+4))
        # norm_true = 2*np.pi*integrate.trapz(radii*pdf2,x=radii)

        Ncls = pdf*ncls
        Nfield = pdf_bkg#/norm_bkg
        Ncls_true = pdf2#/norm_true
        Ncls_field = pdf_cf#/norm_cf

        xmin,xmax = 0.,6.01
        label_radii=r'$R \; [Mpc]$'
        if title is None: title = self.title+' and $R_{200} = %.2f$ Mpc'%(r200)

        # self.plotDistritbution(color,color,Ncls,Ncls_field,Nfield,Ncls_true,xlims=(xmin,xmax),xlabel=label_color,title=title)
        self.plotDistritbution2(radii,radii,Ncls,Ncls_field,Nfield,Ncls_true,xlims=(xmin,xmax),xlabel=label_radii,title=title,line=nbkg,log=True)
        plt.savefig(os.path.join(self.path,'pdf_radial_%s.png'%(self.name)))

        self.close()

   
    def colorDistribution(self,lcolor=0,title=None):
        zcls = self.group.attrs['zcls']
        norm = self.group.attrs['norm']
        nbkg = self.group.attrs['nbkg']

        ncls = norm*nbkg ## change to norm in the future
        
        area = np.pi*(1)**2
        ngals_true = len(self.galaxy2)/area

        color = self.group['pdf_c_cls'][:,0]
        color2 = self.group['pdf_c_truth'][:,0]

        pdf = self.group['pdf_c_cls'][:,lcolor+1]
        pdf_cf = self.group['pdf_c_cls_field'][:,lcolor+1]
        pdf_bkg = self.group['pdf_c_field'][:,lcolor+1]
        pdf2 = self.group['pdf_c_truth'][:,lcolor+1]
        
        pdf2 = self.interpData(color2,pdf2,color)
        # kernel = kde.gaussian_kde( self.galaxy2[self.color_list[j]], bw_method=0.05 )
        # pdf2 = kernel(color)

        Ncls = pdf*ncls
        Nfield = pdf_bkg*nbkg
        Ncls_true = pdf2*ngals_true
        Ncls_field = pdf_cf*(nbkg+ncls)
        
        col_random = np.random.choice(color, 100, p=pdf/np.sum(pdf))
        xmin,xmax = np.mean(col_random)-3*np.std(col_random)-0.1, np.mean(col_random)+3*np.std(col_random)+0.1

        label_color = self.color_list[lcolor]
        if title is None: title = self.title+' at redshift = %.2f'%(zcls)

        # self.plotDistritbution(color,color,Ncls,Ncls_field,Nfield,Ncls_true,xlims=(xmin,xmax),xlabel=label_color,title=title)
        self.plotDistritbution2(color,color,Ncls,Ncls_field,Nfield,Ncls_true,xlims=(xmin,xmax),xlabel=label_color,title=title)
        plt.savefig(os.path.join(self.path,'pdf_color_%s_%s.png'%(label_color[0]+label_color[2],self.name)))

        self.close()

    def redshiftDistribution(self,title=None):
        zcls = self.group.attrs['zcls']
        norm = self.group.attrs['norm']
        nbkg = self.group.attrs['nbkg']
        ncls = norm*nbkg

        area = np.pi*(1)**2
        ncls_true = np.sum(self.galaxy2['Pmem'])/area
        
        ngals = np.sum(self.galaxy['Pmem'])
        # print("name,zcls,ncls:",self.name,zcls,ncls)
        # print("name,zcls,ngals:",self.name,zcls,ngals/area)
        # print("name,zcls,ncls_true:",self.name,zcls,ncls_true)
        # print()

        z = self.group['pdf_z_cls'][:,0]
        z2 = self.group['pdf_z_truth'][:,0]
        pdf = self.group['pdf_z_cls'][:,1]
        pdf_cf = self.group['pdf_z_cls_field'][:,1]
        pdf_bkg = self.group['pdf_z_field'][:,1]
        pdf2 = self.group['pdf_z_truth'][:,1]
        
        # pdf2 = pdf
        pdf2 = self.interpData(z2,pdf2,z)

        Ncls = pdf#*ncls
        Nfield = pdf_bkg#*nbkg
        Ncls_true = pdf2#*ncls_true
        Ncls_field = pdf_cf#*(ncls+nbkg)
        
        # idx_min,idx_max = np.argpercentile(Ncls_field,5),np.argpercentile(Ncls_field,95)
        # xmin, xmax = z[idx_min]-0.1, z[idx_max]+0.1
        xmin, xmax = (zcls-0.3*(1+zcls)),(zcls+0.3*(1+zcls))

        if title is None: title = self.title+' at redshift = %.2f'%(zcls)

        # self.plotDistritbution(z,z,Ncls,Ncls_field,Nfield,Ncls_true,xlabel=r'redshift',xlims=(xmin,xmax),line=zcls,labelLine='$z_{cls}$',title=title)
        self.plotDistritbution2(z,z,Ncls,Ncls_field,Nfield,Ncls_true,xlabel=r'redshift',xlims=(xmin,xmax),line=zcls,labelLine='$z_{cls}$',title=title)
        plt.savefig(os.path.join(self.path,'pdf_redshift_'+self.name+'.png'))
        self.close()

    def plotDistritbution2(self,x,x2,Ncls,Ncls_field,Nfield,Ncls_true,xlabel='$(g-r)$',line=None,xlims=(0.,2.),labelLine=None,title=None,log=False):
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(title,fontsize=16)
        
        gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.05, height_ratios=[3,1])
        # axis_list = [[plt.subplot(gs[i]),plt.subplot(gs[i+2])] for i in range(2)]
        ax1 = plt.subplot(gs[:,0]); ax3 = plt.subplot(gs[0,1]); ax4 = plt.subplot(gs[1,1])

        # fig.tight_layout()
        Ncls_field_model = (Nfield+Ncls)
        Ncls_field_model = Ncls_field_model/integrate.trapz(Ncls_field_model,x=x)
        ymax = np.max([Ncls_field,Ncls,Ncls_true])

        ## first figure
        # ax1, ax2 = axis_list[0]
        
        if not log:
            ax1.plot(x,Ncls_field,color='gray',label=r'Cluster+Field')
            ax1.plot(x,Ncls,color='blue',label=r'Cluster Model')
            ax1.plot(x,Nfield,color='k',label=r'Field (Ring)')
            # ax2.plot(x,Ncls_field-(Ncls),color='k')

        else:
            ax1.scatter(x,Ncls_field,color='gray',label=r'Cluster+Field')
            ax1.scatter(x,Ncls,color='blue',label=r'Cluster Model')
            ax1.scatter(x+4,Nfield,color='k',label=r'Field (Ring)')
            ax1.axhline(line,color='k',linestyle='--',label=labelLine)

        ax1.set_title('Cluster+Field')

        ## second figure
        # ax3, ax4 = axis_list[1]
        if not log:
            norm = integrate.trapz(Ncls,x=x)
            norm_true = integrate.trapz(Ncls_true,x=x2)

            ax3.plot(x,Ncls/norm,color='blue',label=r'Cluster Model')
            ax3.plot(x2,Ncls_true/norm_true,linestyle='--',color='red',label=r'True members')
            ax3.set_title('Cluster')

            ax4.plot(x,Ncls/norm-Ncls_true/norm_true,color='k')
            ylims = (-0.2,1.1*ymax)

            ax3.set_xlim(xlims);ax4.set_xlim(xlims)
        else:
            norm = 2*np.pi*integrate.trapz(x*Ncls,x=x)
            norm_true = 2*np.pi*integrate.trapz(x2*Ncls_true,x=x2)

            ax3.scatter(x,Ncls/norm,color='blue',label=r'Cluster Model')
            ax3.scatter(x2,Ncls_true/norm_true,linestyle='--',color='red',label=r'True members')
            ax3.set_title('Cluster')

            ax4.scatter(x,Ncls/norm-Ncls_true/norm_true,color='k')
            ax4.set_ylim(-1.,1.)

            ymax2 = np.max([Ncls/norm,Ncls_true/norm_true])
            ax3.set_ylim(1e-3,1+ymax2)

            ylims = (-0.2,1.1*ymax)
            # for axs in axis_list:
                #axs[0].set_xscale('log');axs[1].set_xscale('log')
            ax1.set_yscale('log');ax3.set_yscale('log')
            ax3.set_xlim(-0.1,3.);ax4.set_xlim(-0.1,3.)
            # ax1.set_xscale('log');ax3.set_xscale('log')

        if (line is not None)&(not log):
            ax1.axvline(line,color='k',linestyle='--',label=labelLine)
            ax3.axvline(line,color='k',linestyle='--',label=labelLine)

        ax1.set_xlim(xlims);ax1.set_ylim(ylims)

        ax1.set_ylabel(r'$ N $',fontsize=16)
        ax3.set_ylabel(r'$ PDF $',fontsize=16)
        # ax2.set_ylabel(r'Residual',fontsize=16)
        ax1.set_xlabel(xlabel,fontsize=16)
        ax4.set_xlabel(xlabel,fontsize=16)

        ax1.legend(loc='upper right')
        ax3.legend(loc='upper right')


    def plotDistritbution(self,x,x2,Ncls,Ncls_field,Nfield,Ncls_true,xlabel='$(g-r)$',line=None,xlims=(0.,2.),labelLine=None,title=None):
        fig, axs = plt.subplots(1, 2, sharey=True,sharex=True, figsize=(8,6))
        fig.subplots_adjust(left=0.075,right=0.95,bottom=0.15,wspace=0.075)
        
        fig.suptitle(title,fontsize=16)
        # fig.tight_layout()

        axs[0].plot(x,Ncls_field,color='blue',linestyle='--',label=r'Cluster+Field')
        axs[0].plot(x,Nfield,color='r',linestyle='--',label=r'Field')
        axs[0].set_title('Cluster+Field')

        axs[1].plot(x,Ncls,color='lightblue',label=r'Cluster Model')
        axs[1].plot(x2,Ncls_true,color='lightcoral',label=r'True members')
        axs[1].set_title('Cluster')

        if line is not None:
            axs[0].axvline(line,color='k',linestyle='--',label=labelLine)
            axs[1].axvline(line,color='k',linestyle='--',label=labelLine)

        axs[0].set_xlim(xlims)

        axs[0].set_ylabel(r'$ PDF $')
        axs[0].set_xlabel(xlabel,fontsize=16)
        axs[1].set_xlabel(xlabel,fontsize=16)

        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')

    def getCoordinates(self,ra,dec):
        ra_c = self.cluster['RA']
        dec_c = self.cluster['DEC']

        ra_o, dec_o = (ra-ra_c), (dec-dec_c)
        ra_o = ra_o*np.cos(dec_c*(np.pi/180))
        
        return ra_o, dec_o

    def getTheta(self,R200=None):
        angularDiameter = self.cluster['DA']
        theta = (R200/angularDiameter)*(180/np.pi) ## degrees
        # print('theta:',theta)
        return theta

    def spatialDistribution(self,lcol='Pmem',title=None):
        zcls = self.cluster['redshift']
        theta200 = self.getTheta(R200=1)

        ra, dec = self.getCoordinates(self.galaxy['RA'],self.galaxy['DEC'])
        ra2, dec2 = self.getCoordinates(self.galaxy2['RA'],self.galaxy2['DEC'])

        pmem = self.galaxy[lcol]

        fig, axs = plt.subplots(1, 2, sharey=True,sharex=True, figsize=(12,6))
        # fig.subplots_adjust(left=0.075,right=0.95,bottom=0.15,wspace=0.075)
        fig.suptitle(title,fontsize=16)
        
        ## getting norm    
        norm = mpl.colors.Normalize(vmin=0., vmax=1.)


        ## grid 
        for i in range(2):
            axs[i].axvline(0,alpha=0.7,linestyle='--',color='gray',linewidth=2)
            axs[i].axhline(0,alpha=0.7,linestyle='--',color='gray',linewidth=2)

        ## circle
        circle = plt.Circle((0, 0), theta200, linestyle='--', color='gray', fill=False)
        idx = np.flip(np.argsort(pmem))
        # sc=axs[0].hexbin(ra,dec,gridsize = 40,C=pmem,cmap='Reds',reduce_C_function=np.sum, norm=norm)
        sc=axs[0].scatter(ra[idx],dec[idx],s=95*pmem[idx]+5,c=pmem[idx],cmap='Reds', norm=norm)
        axs[0].set_title('Estimated Distribution')
        axs[0].set_xlabel('Right Ascension (deg)')
        axs[0].set_ylabel('Declination (deg)')
        axs[0].add_artist(circle)
        axs[0].text(0.4*theta200,-theta200-0.01,r'$N_{gals}$ = %.1f'%(np.sum(pmem)))

        circle = plt.Circle((0, 0), theta200, linestyle='--', color='gray', fill=False)
        # sc2=axs[1].hexbin(ra2,dec2, gridsize = 40, C=np.ones_like(ra2),cmap='Reds',reduce_C_function=np.sum,norm=norm)    
        sc2=axs[1].scatter(ra2,dec2,s=100,c=np.ones_like(ra2),cmap='Reds', norm=norm)
        axs[1].set_title('True Members')
        axs[1].set_xlabel('Right Ascension (deg)',fontsize=12)
        axs[1].set_ylabel('Declination (deg)',fontsize=12)
        axs[1].add_artist(circle)
        axs[1].axis([-theta200-0.02,theta200+0.02,-theta200-0.02,theta200+0.02])
        axs[1].text(0.5*theta200,-theta200-0.01,r'$N_{gals}$ = %i'%(len(ra2)))
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cb2 = fig.colorbar(sc2, cax=cbar_ax, fraction=0.046, pad=0.04)
        cb2.set_label(r'$P_{mem}$',rotation=90)

        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')

        if title is None: title = self.title+' at redshift = %.2f'%(zcls)
        fig.suptitle(title,fontsize=16)

        plt.savefig(os.path.join(self.path,'radec_'+self.name+'.png'))
        self.close()    