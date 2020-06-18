# !/usr/bin/env python
# purpose: this code computes general color properties of galaxy clusters in redshift bins.
import os
import numpy as np
from astropy.table import Table, join
from astropy.io.fits import getdata
import matplotlib.pyplot as plt

from six import string_types
import warnings
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

import scipy.stats as stats

## local libraries
import gaussianKDE as kde

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 16})

def initNewColumns(data,colNames,value=-99):
    for col in colNames:
        data[col] = value*np.ones_like(data['CID'])
    return data

###
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

def get_bin_indicies(x,xedges):
    idx0 = np.full(len(x),0,dtype=int)

    for i in range(len(xedges)-1):
        xlo,xhi = xedges[i], xedges[i+1]
        w, = np.where((x>=xlo)&(x<=xhi))
        idx0[w] = i
     
    return idx0

def group_table_by(table,yedges,colName='z'):
    x = table[colName]
    indices_list = list(chunks2(x, yedges))
    return indices_list

def bin_by(x,yedges):
    '''the variable x is binned using the yedges'''
    xs = np.sort(x,kind='mergesort')
    idx_list = list(chunks2(xs, yedges))
    xlist = [xs[idx] for idx in idx_list]
    return xlist

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def chunks2(x, yedges):
    for i in range(len(yedges)-1):
        w, = np.where( (x>=yedges[i]) & (x<=yedges[i+1]) )
        yield w

def chunks3(x, yedges):
    for i in range(len(yedges)-1):
        w, = np.where( (x>=yedges[i]) & (x<=yedges[i+1]) )
        yield np.full_like(w,i)

def set_color_variables(gal,color_list,ecolor_list,color_value):
    for i in range(len(color_list)):
        ci,cj = color_value[i]
        gal[color_list[i]] = gal['mag'][:,ci]-gal['mag'][:,cj]
        gal[ecolor_list[i]] = (gal['magerr'][:,ci]**2+gal['magerr'][:,cj]**2)**(1./2.)

    return gal

def computeColorKDE(x,weight=None,bandwidth='silverman',silvermanFraction=None):
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

def plotColor(color,pmem,lcolor,title,save,mask=None,xlims=(-1.,3.)):
    # ngals = np.sum(pmem)
    idx = np.argsort(color)

    color, pmem = color[idx], pmem[idx]
    kernel = computeColorKDE(color,weight=pmem,bandwidth='silverman')
    kde = kernel(color)

    norm = 1/np.max(kde)
    plt.clf()
    plt.plot(color,norm*kde,color='royalblue',linewidth=2)
    plt.fill_between(color,0,norm*kde,color='royalblue',alpha=0.2)

    if mask is not None:
        color_true = color[mask[idx]]
        kernel_true = computeColorKDE(color_true,bandwidth='silverman')
        kde_true = kernel_true(color_true)
        
        plt.plot(color_true,norm*kde_true,label='True members',color='lightcoral',linewidth=2.)
        plt.fill_between(color_true,0,norm*kde_true,color='lightcoral',alpha=0.2)
        plt.legend(loc='upper right')

    plt.xlim(xlims)
    plt.ylim(0.,1.05)
    
    plt.xlabel(r'$(%s)$'%(lcolor))
    plt.title(title)

    plt.savefig(save)
    plt.close()

def PDF_color(xvec,xmean,sigma):
    res = stats.norm.pdf(xvec,xmean,sigma)
    # norm = stats.norm.pdf(xmean,xmean,sigma)
    return res

# def plotArrayColor(gal,zb,indices_list,lcolor='g-r',xlims=(-1.,3.),gmm=None,tM=True):
#     pass

def plotArrayColor(gal,zb,indices_list,lcolor='g-r',xlims=(-1.,3.),gmm=None,tM=True):
    import math
    ### create 5x4 plots
    plt.clf()
    fig = plt.figure(figsize=(24,16))
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    ncls = len(indices_list)

    npannels = math.ceil(ncls/(6.*5.))
    j,k=0,1
    for i in range(0, len(indices_list)):
        gal_group_0 = gal[indices_list[i]]
        color_group = gal_group_0[lcolor]
        pmem_group = gal_group_0['Pmem']

        idx, = np.where((color_group<=xlims[1])&(color_group>=xlims[0]))
        color_group, pmem_group = color_group[idx], pmem_group[idx]

        
        if tM:
            mask = gal_group_0['True']==True

        if gmm is not None:
            xmax = color_group.max() 
            color_fit = np.linspace(xlims[0],xmax,100)
            mu_r, mu_b = gmm['mean'][i,0], gmm['mean'][i,1]
            s_r, s_b = gmm['sigma'][i,0], gmm['sigma'][i,1]
            w_r, w_b = gmm['alpha'][i,0], gmm['alpha'][i,1]

            # w_r, w_b = 1,1
            L_red = w_r*PDF_color(color_fit,mu_r,s_r)
            L_blue= w_b*PDF_color(color_fit,mu_b,s_b)
        
        zi,zj = zb[i], zb[i+1]
        ti = r"$%.2f < z < %.2f$"%(zi,zj)

        idx = np.argsort(color_group)
        color_group, pmem_group = color_group[idx], pmem_group[idx]
        kernel = computeColorKDE(color_group,weight=pmem_group,bandwidth='silverman')
        kde = kernel(color_group)

        # norm = 1./np.max(kde)
        norm = 1.
        if mask is not None:
            color_true = color_group[mask[idx]]
            kernel_true = computeColorKDE(color_true,bandwidth='silverman')
            kde_true = kernel_true(color_true)
            
        ax = fig.add_subplot(5, 6, j+1)
        ax.plot(color_group,norm*kde,label='estimated distribution',color='gray',linewidth=2.)
        ax.fill_between(color_group,0,norm*kde,color='gray',linewidth=2.,alpha=0.2)
        
        # bin_height,bin_boundary = np.histogram(color_group,bins=20,density=True)
        # width = bin_boundary[1]-bin_boundary[0]
        # norm = 1.#/float(max(bin_height))
        
        # bin_height = bin_height*norm
        # ax.bar(bin_boundary[:-1],bin_height,width = width,label='color distribution',color='gray',alpha=0.5)
        # ax.hist(color_group,weights=norm*np.ones_like(color_group),bins=20,label='estimated distribution',color='gray',density=True)

        # if mask is not None:
        #     ax.plot(color_true,norm*kde_true,label='True members',color='lightcoral',linewidth=2.)
        #     ax.fill_between(color_true,0,norm*kde_true,color='lightcoral',linewidth=2.,alpha=0.2)

        if gmm is not None:
            ax.plot(color_fit,norm*L_red,label='red-sequence',color='lightcoral',linewidth=2.,linestyle='--')
            ax.fill_between(color_fit,0,norm*L_red,color='lightcoral',linewidth=2.,alpha=0.5)

            ax.plot(color_fit,norm*L_blue,label='blue cloud',color='royalblue',linewidth=2.,linestyle='--')
            ax.fill_between(color_fit,0,norm*L_blue,color='royalblue',linewidth=2.,alpha=0.5)

        ax.title.set_text(ti)
        ax.set_xlim(xlims)
        
        if j==0:
            ax.legend(loc='upper left')

        if j>15:
            ax.set_xlabel(r'$(%s)$'%(lcolor))

        if i==(k*6*5-1):
            for ax in fig.get_axes():
                ax.label_outer()

            si = 'Buzzard_color_%s_%i.png'%(lcolor,k)
            fig.savefig(si, bbox_inches = "tight")
            plt.close()
            k+=1
            j=0

            plt.clf()
            fig = plt.figure(figsize=(24,16))
            fig.subplots_adjust(hspace=0.3, wspace=0.05)

        else:
            j+=1
    
    if k<npannels:
        for ax in fig.get_axes():
            ax.label_outer()

        si = 'Buzzard_color_%s_%i.png'%(lcolor,npannels)
        fig.savefig(si, bbox_inches = "tight")
        plt.close()

def makeColorGif(lcolor='r-i',xlims=(-1.,3.)):
    files = []
    for i in range(len(indices_list)):
        gal_group_0 = gal[indices_list[i]]
        color_group = gal_group_0[lcolor]

        pmem_group = gal_group_0['Pmem']
        mask = gal_group_0['True']==True
        
        zi,zj = zb[i], zb[i+1]
        ti = r"Buzzard: $ %.2f < z < %.2f$"%(zi,zj)
        si = './fig/%s_Buzzard_color_%02i.png'%(lcolor,i)

        plotColor(color_group,pmem_group,lcolor,ti,si,mask=mask,xlims=xlims)

        files.append(si)

    os.system('convert -delay 55 -loop 2 %s animated_%s.gif'%(' '.join(files),lcolor))

def plotColorDiagram(mag,color,pmem,rs_param,lmag,lcolor,title,save,xlims=(16.,24.),ylims=(-1.,3.)):
    a, b = np.min(mag)-0.5,np.max(mag)+0.5
    mag_vec = np.linspace(a,b,100)
    rs_color = rs_param[0]*mag_vec+rs_param[1]

    plt.clf()
    plt.scatter(mag,color,color='lightcoral',s=(10*np.sqrt(pmem)+0.5),alpha=0.5)
    plt.plot(mag_vec,rs_color,color='k',linestyle='--',linewidth=1.5)

    plt.ylim(ylims)
    plt.xlim(xlims)
    
    plt.ylabel(r'$(%s)$'%(lcolor))
    plt.xlabel(r'$%s$'%(lmag))

    plt.title(title)

    plt.savefig(save)
    plt.close()

def makeColorDiagramGif(gal,rs_parameters,mag_number=1,color_number=2,lcolor='r-i',ylims=(-1.,3.),rest_frame_color=False):
    files = []
    for i in range(len(indices_list)):
        gal_group_0 = gal[indices_list[i]]
        
        if not rest_frame_color:
            z_cls = float(rs_parameters['redshift'][i])
            xlims = (14.5*(1+z_cls)**(1/4),21.*(1+z_cls)**(1/5))
            lmag = lcolor.split('-')[-1]
            color_group = gal_group_0[lcolor]
            mag_group = gal_group_0['mag'][:,mag_number]
            rs_param = [rs_parameters['slope'][i,color_number], rs_parameters['intercept'][i,color_number]]

        else:
            lmag = 'M_i'
            xlims = (-20.5,-26.5)
            color_group = gal_group_0[lcolor]
            mag_group = gal_group_0[mag_number]
            rs_param = [rs_parameters['slope'][i,0], rs_parameters['intercept'][i,0]]
        
        pmem_group = gal_group_0['Pmem']
        # mask = gal_group_0['True']==True
        
        zi,zj = zb[i], zb[i+1]
        ti = r"Buzzard: $ %.2f < z < %.2f$"%(zi,zj)
        si = './fig/%s_Buzzard_color_diagram_%02i.png'%(lcolor,i)

        plotColorDiagram(mag_group,color_group,pmem_group,rs_param,lmag,lcolor,ti,si,xlims=xlims,ylims=ylims)

        files.append(si)

    os.system('convert -delay 55 -loop 2 %s color_mag_animated_%s.gif'%(' '.join(files),lcolor))

def rsFit(x1,y,err,cut=None):
    if cut is not None:
        mask = (y>cut)&(y<1.2)&(x1>=-25.)
        x1,y,err=x1[mask],y[mask],err[mask]
    else:
        mask = y <= 3.
        x1,y,err=x1[mask],y[mask],err[mask]

    if len(x1)<5:
        return [-99.,-99.,-99.,-99.]

    err = np.array(err)
    # initialize fitters
    g_init = models.Linear1D(1,intercept=np.percentile(y,95))

    fit = fitting.LevMarLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=2.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Ignore model linearity warning from the fitter
        # mask, model = or_fit(g_init, x1, y,weights=1.0/err)
        model, mask = or_fit(g_init, x1, y,weights=1.0/err)
        # filtered_data = np.ma.masked_array(y, mask=mask)
        # y_sigma = y[mask.mask]
        a,b = model._parameters
        c,d = mask.mean(), mask.std()
    return [a,b,c,d]

def computeRS_singleColor(gal,label,indices_list,rest_frame_color=False):
    # print('label0',label)
    
    if not rest_frame_color:
        band = [ gal['mag'][:,int(label[0])][idx] for idx in indices_list]
        color = [ gal[label[1]][idx] for idx in indices_list]
        ecolor = [ gal[label[2]][idx] for idx in indices_list]
        cut=None
    
    else:
        band = [ gal['iabs'][idx] for idx in indices_list]
        color = [ gal['gi_o'][idx] for idx in indices_list]
        ecolor = [ gal['gi_o_err'][idx] for idx in indices_list]
        cut = 0.8

    slope, intercept, mean_color, std_color = [],[],[],[]
    for (i, j, k) in zip(band, color, ecolor):
        # rfit,info = np.polynomial.polynomial.polyfit(i,j,deg=1,full=True,w=1./k) #i=band, j=color, k=ecolor
        a, b, c, d = rsFit(i,j,k,cut=cut)

        slope.append(a)
        intercept.append(b)
        mean_color.append(c)
        std_color.append(d)

    return np.array(slope), np.array(intercept), np.array(mean_color), np.array(std_color)

def computeRS(gal,mag_list,color_list,color_error_list,indices_list):
    ## firt rest-frame color
    slope,intercept,mean_color,std_color = computeRS_singleColor(gal,None,indices_list,rest_frame_color=True)

    colnames = []
    data_out = []
    ## observed colors
    for (label) in zip(mag_list[1:],color_list[1:],color_error_list[1:]):
        a,b,c,d = computeRS_singleColor(gal,label,indices_list)

        data_out.append(a)
        data_out.append(b)

        colnames.append("rs_slope_%s"%label[1])
        colnames.append("rs_intercept_%s"%label[1])

    #     slope = np.vstack([slope,a])
    #     intercept = np.vstack([intercept,b])
    #     mean_color = np.vstack([mean_color,c])
    #     std_color = np.vstack([std_color,d])

    # rs_parameters = Table(data=[slope.transpose(),intercept.transpose(),mean_color.transpose(),std_color.transpose()],
    #                       names=['slope','intercept','mean_color','std_rs'])
    rs_parameters = Table(data=data_out,names=colnames)

    return rs_parameters

def getDeltaColor(mag,color,rs_parameters,indices,color_number=0):
    ncls = len(indices)

    delta_colors = []
    for i in range(ncls):
        idx = indices[i]
        mag_i, color_i = mag[idx], color[idx]

        slope = rs_parameters['slope'][i,color_number]
        intercept = rs_parameters['intercept'][i,color_number]

        color_rs = slope*mag_i + intercept
        delta = color_i - color_rs
        
        delta_colors.append(delta)

    return np.array(delta_colors)

def computeDeltaColors(gal,rs_parameters,keys,color_list,mag_list):
    delta_colors = getDeltaColor(gal[mag_list[0]],gal[color_list[0]],rs_parameters,keys,color_number=0)

    count=0
    for li,clabel in zip(mag_list[1:],color_list[1:]):
        mag, color = gal['mag'][:,li], gal[clabel]
        di = getDeltaColor(mag,color,rs_parameters,keys,color_number=count)
        delta_colors = np.vstack([delta_colors,di])
        count+=1

    return delta_colors.transpose()

def fill_mask_values(a):
    for shift in (-1,1):
        a_shifted=np.roll(a,shift=shift)
        idx=~a_shifted.mask * a.mask
        a[idx]=a_shifted[idx]
    return a.data

def get_fit_filtered(x, signal, threshold=2.):
    g_init = models.Polynomial1D(2)

    fit = fitting.LevMarLSQFitter()    
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=threshold)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Ignore model linearity warning from the fitter
        mask, model = or_fit(g_init, x, signal)
        signal_filtered = signal[mask.mask]
    
    # signal_filtered = fill_mask_values(mask)
    # nmask = np.logical_not(mask.mask)
    signal_filtered = np.where(mask.mask,model(x),signal)
    
    return signal_filtered

def get_median_filtered(signal, threshold=3):
    signal = (signal).copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

def gaussian_smooth(x,signal,nbins=35,threshold=2.):
    signal_filetered = signal.copy()

    a, b = binsCounts(x,nbins)
    indices_list = list(chunks2(x, b))
    
    y_bin = np.array([ np.nanmedian(signal[idx][signal[idx]>-1.]) for idx in indices_list])
    std_bin = np.array([ np.nanstd(signal[idx][signal[idx]>-1.]) for idx in indices_list])
    x_bin = np.array([ np.median(x[idx]) for idx in indices_list])
    
    id0 = get_bin_indicies(x,b)
    difference = np.abs(signal - y_bin[id0])
    mask = (difference >= threshold*std_bin[id0]) | (signal<-1.)
    
    from scipy import interpolate
    signal_filetered[mask] = interpolate.interp1d(x_bin, y_bin, fill_value='extrapolate')(x[mask])

    return signal_filetered

## filter the data
def nbins_rs_parameters(zcls,rs_param,nbins=10):
    from scipy import interpolate
    labels = rs_param.colnames[:-1]

    colnames = ['redshift']
    out_data = [zcls]
    
    print(labels)
    for li in labels:
        rs_param[li] = gaussian_smooth(rs_param['redshift'],rs_param[li],nbins=nbins)
        bla = interpolate.interp1d(rs_param['redshift'], rs_param[li], fill_value='extrapolate')(zcls)
        
        out_data.append(bla)
        colnames.append(li)
    
    rs_new = Table(data=out_data,names=colnames)
    return rs_new

def plotColorRedshift(gal,rs_parameters,x_bin,color_list):
    x_bin_mean = rs_parameters['redshift']
    x = gal['redshift']
    z = gal['Pmem']

    y_bin = [rs_parameters['mean_color'][:,id_color] for id_color in range(6)]
    y_bin_std = [rs_parameters['std_rs'][:,id_color] for id_color in range(6)]
    
    y = [gal[lc] for lc in color_list]
    
    plt.clf()
    fig, ax = plt.subplots(2, 3, sharex='col', figsize=(20,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    count = 0
    for i in range(2):
        for j in range(3):
            img = ax[i,j].hexbin(x, y[count], z, gridsize=50,cmap=plt.cm.get_cmap('Blues_r', 14), reduce_C_function=np.sum, vmin=0, vmax=300)
            
            ax[i,j].errorbar(x_bin_mean,y_bin[count],xerr=np.diff(x_bin)/2, yerr=y_bin_std[count],color='gray')
            ax[i,j].set_ylabel(color_list[count])

            if i>0:
                ax[i,j].set_xlabel('redshift')
            
            count+=1
            # plt.colorbar(label='density')

    fig.colorbar(img, ax=ax, label='Ngals', orientation='vertical',shrink=.75, pad=.01)

    fig.savefig('color_redshift.png', bbox_inches = "tight")
    # fig.savefig('color_redshift.png')
    plt.close()

def gmmRuntimeError():
    return -99,-99,-99,-99,-99,-99,False

def gmmLabel(alpha):
    red = np.argmax(alpha)
    blue = np.argmax(-1*alpha)
    return blue, red

def gmmValues(gmm):
    try:
        mu, sigma, alpha, conv =gmm.means_, np.sqrt(gmm.covars_), gmm.weights_, gmm.converged_
    except:
        mu, sigma, alpha, conv =gmm.means_, np.sqrt(gmm.covariances_), gmm.weights_, gmm.converged_
    blue, red = gmmLabel(mu)
    mur, sigr, alphar = float(mu[red]), float(sigma[red]), float(alpha[red])
    mub, sigb, alphab = float(mu[blue]), float(sigma[blue]), float(alpha[blue])

    return mur,mub,sigr,sigb,alphar,alphab,conv
    
def gmmFit(x,weights=None):
    if len(x)<5:
        mur,mub,sigr,sigb,alphar,alphab,conv = gmmRuntimeError()
        return [mur,mub,sigr,sigb,alphar,alphab,conv]
    else:
        from sklearn import mixture
        try:
            gmm=mixture.GMM(n_components=2,tol=1e-7,n_iter=500)
        except:
            print('New GMM version')
            gmm=mixture.GaussianMixture(n_components=2,tol=1e-7)

        try:
            fit = gmm.fit(x[:, np.newaxis], data_weights=weights[:, np.newaxis])
        except:
            fit = gmm.fit(x[:, np.newaxis])
        
        mur,mub,sigr,sigb,alphar,alphab,conv = gmmValues(fit)
        
        if alphar<0.05:
            ## switch colors
            mur,mub,sigr,sigb,alphar,alphab,conv = mub,mur,sigb,sigr,alphab,alphar,conv
        
        # if alphab<0.05:
        #     mur,mub,sigr,sigb,alphar,alphab,conv = mur,0.,0.,sigr,0.,alphar,conv
        
        if not conv:
            fit = gmm.fit(x[:, np.newaxis])
            mur,mub,sigr,sigb,alphar,alphab,conv = gmmValues(fit)
            
            if not conv:
                mur,mub,sigr,sigb,alphar,alphab,conv = gmmRuntimeError()
        
        return [mur,mub,sigr,sigb,alphar,alphab,conv]

def filter_gmm_params(zcls,gmm_table):
    out = gmm_table.copy()
    for i in range(6):
        out[:,i] = gaussian_smooth(zcls,gmm_table[:,i])

    return out

def computeGMM(gal,zb_means,indices,lcolor='gr_o',filter=False,):
    out = []
    for idx in indices:
        ci,pi = gal[lcolor][idx], gal['Pmem'][idx]

        cut = ci>0.05
        gmm_parameters = gmmFit(ci[cut],weights=pi[cut])
        out.append(gmm_parameters)
        
    out = np.array(out)
    
    if filter:
        out = filter_gmm_params(zb_means,out)

    rs_param = np.array([out[:,0],out[:,2],out[:,4],out[:,6]]).transpose()
    bc_param = np.array([out[:,1],out[:,3],out[:,5],out[:,6]]).transpose()
    # conv = out[:,6]
    # mean = np.array([out[:,0],out[:,1]]).transpose()
    # sigma = np.array([out[:,2],out[:,3]]).transpose()
    # alpha = np.array([out[:,4],out[:,5]]).transpose()
    
    # gmm6 = Table(data=[zb_means,mean,sigma,alpha,conv],names=['redshift','m_rs','sigma','alpha','conv'])
    return rs_param,bc_param

def computeGMM_AllColors(gal,zb_means,ids,indices,color_list=['g-r','r-i','i-z'],filter=False):
    out_data = [ids,zb_means]
    colnames = ['CID','redshift']
    for li in color_list:
        rsi,bci = computeGMM(gal,zb_means,indices,lcolor=li,filter=filter)

        colnames.append('rs_param_%s'%(li))
        colnames.append('bc_param_%s'%(li))

        out_data.append(rsi)
        out_data.append(bci)

    gmm6 = Table(data=out_data,names=colnames)
    return gmm6

def gmmColorProb(color,pmem,param):
    mur,mub,sigr,sigb,alphar,alphab,conv = param
    
    if conv:
        color_fit = color
        
        L_blue= PDF_color(color_fit,mub,sigb)
        L_red = PDF_color(color_fit,mur,sigr)
        
        kernel = kde.gaussian_kde(color,weights=pmem)
        L_total = kernel(color_fit)
        
        #calculate red/blue probabilities (see overleaf section 3.4)
        p_red_numerator=(alphar*L_red)
        p_red_denominator=(alphab*L_blue + alphar*L_red)
        p_red=p_red_numerator/p_red_denominator
        
        p_blue_numerator=(alphab*L_blue)
        # p_blue_denominator=((alphab*L_blue))*kappa + bg_interp(color_fit)
        p_blue_denominator=(alphab*L_blue + alphar*L_red)
        # p_blue_denominator=L_total
        p_blue=p_blue_numerator/p_blue_denominator

        
        p_blue.shape=(len(p_blue),);p_red.shape=(len(p_red),)

        tmp_Pblue = p_blue
        tmp_Pred = p_red
        
    else:
        tmp_Pred, tmp_Pblue = (-99.)*np.ones_like(color),(-99.)*np.ones_like(color)
    
    tmp_Pblue = np.where(tmp_Pblue<0,0,tmp_Pblue)
    tmp_Pred = np.where(tmp_Pred<0,0,tmp_Pred)
    
    return tmp_Pred, tmp_Pblue

# ## Color Probabilites
def get_gmm_param(gmm,idx=0):
    mu_r, mu_b = gmm['mean'][idx,0], gmm['mean'][idx,1]
    s_r, s_b = gmm['sigma'][idx,0], gmm['sigma'][idx,1]
    w_r, w_b = gmm['alpha'][idx,0], gmm['alpha'][idx,1]
    conv = gmm['conv'][idx]
    return [mu_r, mu_b, s_r, s_b, w_r, w_b,conv]

def colorProb(color,pmem,param,nsigma=2):
    mur,mub,sigr,sigb,alphar,alphab,conv = param

    if conv:
        p_red = np.where( (color <= mur+nsigma*sigr ) & (color>=mur-nsigma*sigr), 1., 0. )
        p_blue = np.where( (color<=mur-nsigma*sigr), 1., 0. )

    else:
        p_red = 0.*pmem
        p_blue = 0.*pmem

    return p_red, p_blue

def computeColorProbabilties(gal,gmm_parameters,indices,lcolor='gr_o',hard_cut=False):
    pr,pb = [],[]
    count=0
    for i in indices:
        ci,pi = gal[lcolor][i], gal['Pmem'][i]
        param = get_gmm_param(gmm_parameters,idx=count)

        if not hard_cut:
            pred, pblue = gmmColorProb(ci,pi,param)
        else:
            pred, pblue = colorProb(ci,pi,param,nsigma=2)
        
        pr.append(pred)
        pb.append(pblue)
        
        # gal['Pred'][i] = pred
        # gal['Pblue'][i] = pblue
        
        count+=1
    return gal,pr,pb

def checkPcolors(i=0):
    zi,zj = zb[i], zb[i+1]
    ti = r"Buzzard: $%.2f < z < %.2f$"%(zi,zj)
    
    plt.clf()
    color_group_0 = gal['gr_o'][indices_list[i]]
    plt.scatter(color_group_0,pred[i],color='r',label=r'$P_{red}$')
    plt.scatter(color_group_0,pblue[i],color='b',label=r'$P_{blue}$')

    # plt.scatter(color_group_0,pred_hard_cut[i],color='r',label=r'$N_{red}$')
    # plt.scatter(color_group_0,pblue_hard_cut[i],color='b',label=r'$N_{blue}$')

    plt.title(ti)
    plt.savefig('./fig/pred_pblue_%i.png'%(i))

def makePcolorGif():
    nbins = len(indices_list)

    savelist = []
    for j in range(nbins):
        checkPcolors(i=j)
        savelist.append('./fig/pred_pblue_%i.png'%(j))
    
    os.system('convert -delay 55 -loop 2 %s pcolors_animated.gif'%(' '.join(savelist)))

def getColorFractions(pred,pblue):
    N_red = np.array([np.sum(pred[idx]) for idx in range(len(pred))])
    N_blue = np.array([np.sum(pblue[idx]) for idx in range(len(pblue))])

    fred = N_red/(N_red+N_blue)
    fblue = N_blue/(N_red+N_blue)

    return fred, fblue, N_red, N_blue

def get_columns_selected(color_lis):
    out = ['CID']
    for li in color_lis:
        out.append('rs_param_%s'%li)
        out.append('bc_param_%s'%li)
    return out
# output_rs = getConfigFile()


# plots = False
# fit = False

# # print('getting data')
# file_gal = './data/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members_stellarMass.fits'
# file_cls = './data/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_stellarMass.fits'
# output_rs = 'buzzard_color_model_rs_parameters.fits'
def get_RS_flag(gal,gmm_parameters,color_list=['g-r'],nsigma=2):
    zcls = gal['redshift']
    new_pmem = np.zeros_like(gal['Pmem'])
    z = gmm_parameters['redshift']

    for li in color_list:
        color = gal[li]
        mur = gmm_parameters['rs_param_%s'%(li)][:,0]
        sigr = gmm_parameters['rs_param_%s'%(li)][:,1]

        b = np.arange(0.08,0.92,0.03)
        indices_list = list(chunks2(z, b))

        y_bin = np.array([ np.nanmedian(mur[idx]) for idx in indices_list])
        std_bin = np.array([(np.nanmedian(sigr[idx])**2+np.nanstd(mur[idx])**2)**(1/2) for idx in indices_list])
        x_bin = np.array([ np.median(z[idx]) for idx in indices_list])

        from scipy import interpolate
        # cb_u = y_bin + nsigma*std_bin
        # cb_upper = interpolate.interp1d(x_bin,cb_u,kind='cubic',fill_value='extrapolate')(zcls)
        
        # cb_l = y_bin - nsigma*std_bin
        # cb_lower = interpolate.interp1d(x_bin,cb_l,kind='cubic',fill_value='extrapolate')(zcls)
        cb_upper = mur+nsigma*sigr
        cb_lower = mur-nsigma*sigr

        indices = list(chunks(gal['CID'], gmm_parameters['CID']))
        for i,idx in enumerate(indices):
            new_pmem[idx] = np.where((color[idx]<=cb_upper[i])&(color[idx]>=cb_lower[i]),1.,0.)

    return new_pmem

def cut_colors_above_RS(gal,gmm_parameters,color_list=['g-r'],nsigma=2):
    zcls = gal['redshift']
    new_pmem = gal['Pmem']
    z = gmm_parameters['redshift']

    for li in color_list:
        color = gal[li]
        mur = gmm_parameters['rs_param_%s'%(li)][:,0]
        sigr = gmm_parameters['rs_param_%s'%(li)][:,1]

        b = np.arange(0.08,0.92,0.03)
        indices_list = list(chunks2(z, b))

        y_bin = np.array([ np.nanmedian(mur[idx]) for idx in indices_list])
        std_bin = np.array([(np.nanmedian(sigr[idx])**2+np.nanstd(mur[idx])**2)**(1/2) for idx in indices_list])
        x_bin = np.array([ np.median(z[idx]) for idx in indices_list])

        from scipy import interpolate
        cb_u = y_bin + nsigma*std_bin
        cb_upper = interpolate.interp1d(x_bin,cb_u,kind='cubic',fill_value='extrapolate')(zcls)
        
        new_pmem = np.where(color>=cb_upper,0.,new_pmem)

    return new_pmem

def colorModel(file_cls,file_gal,output_rs=None,file_gmm=None,cutColorsAboveRS=False,tMembers=False):
    import logging

    cat = Table(getdata(file_cls))
    gal = Table(getdata(file_gal))

    print('making stellar mass cut')
    cut2 = (gal['mass']>=10)&(gal['gi_o']>0.05)
    ngals = len(gal)
    
    ngals_cut = len(gal[cut2])
    logging.debug('there is ',round(100*((ngals-ngals_cut)/ngals),2),' percent of galaxies with bad stellar rest frame colors')

    if tMembers:
        cut, = np.where(gal['True']==True)#&(gal['gi_o']>0.05)#&(gal['Pmem']>0.7)
        gal = gal[cut]
        gal['Pmem'] = 1.

        file_cls = file_cls.split('.fits')[0]+'truth.fits'
        file_gal = file_gal.split('.fits')[0]+'truth.fits'

    z_cls = cat['redshift']
    ids_cls = cat['CID']
    ids_cls_gal = gal['CID']
    indicies_unique = list(chunks(ids_cls_gal,ids_cls))         ## per cluster

    ### defining color variables
    color_list = ['gi_o','gr','gi','ri','rz','iz']
    ecolor_list = [ci+'_err' for ci in color_list]

    color_value = [(0,1),(0,2),(1,2),(1,3),(2,3)]
    mag_list = ['iabs',1,2,2,3,3]
    gal = set_color_variables(gal,color_list[1:],ecolor_list[1:],color_value)

    print('Gaussian Mixture Modeling')
    gmm_parameters = computeGMM_AllColors(gal,z_cls,ids_cls,indicies_unique,color_list=color_list,filter=True)
    # plotArrayColor(gal,zb,indicies_unique,lcolor=li,xlims=(0.,1.2),gmm=gmm_parameters)

    col_selected = get_columns_selected(color_list)
    cat = join(cat,gmm_parameters[col_selected],keys='CID',join_type='outer')

    ## get red sequence(RS) parameters
    if output_rs is not None:
        ncls_per_bin = 20
        zb2, zb2_means = binsCounts(z_cls,ncls_per_bin)
        # zb2 = np.arange(z_cls.min()-0.01,z_cls.max()+0.01,0.025)
        # zb2_means = (zb2[1:]+zb2[:1])/2

        ### group the table
        indices_list2 = group_table_by(gal,zb2,colName='redshift')

        ## rs_parameters - astropy Table
        ## slope(6), intercept(6), mean_color, std_rs(6)
        ## 6 colors: gi_o, g-r, g-i, r-i, r-z, i-z
        rs_parameters = computeRS(gal,mag_list,color_list,ecolor_list,indices_list2)
        rs_parameters['redshift'] = zb2_means
        
        ## take the mean in bins of 6 points
        rs_parameters = nbins_rs_parameters(cat['redshift'],rs_parameters,nbins=5)
        rs_parameters['CID'] = cat['CID']

        # ## save red sequence model
        rs_parameters = join(rs_parameters,gmm_parameters[col_selected],keys='CID',join_type='outer')
        rs_parameters.write('./out/rs_parameters.fits',format='fits',overwrite=True)

    print('Cut colors above the rs')
    if cutColorsAboveRS:
        new_pmem = cut_colors_above_RS(gal,gmm_parameters,color_list=['gr'])
        gal['Pmem_new'] = new_pmem

    # print('Get RS galaxies')
    rs_flag1 = get_RS_flag(gal,gmm_parameters,color_list=['gi'])
    rs_flag2 = get_RS_flag(gal,gmm_parameters,color_list=['ri'])
    gal['flag_rs_gi'] = rs_flag1
    gal['flag_rs_ri'] = rs_flag2

    print('output catalogos')
    if file_gmm is not None:
        gmm_parameters.write(file_gmm,format='fits',overwrite=True)

    cat.write(file_cls,format='fits',overwrite=True)
    gal.write(file_gal,format='fits',overwrite=True)
    
    pass


def colorModel_per_bin(file_cls,file_gal,output_rs=None,ncls_per_bin=34):
    import logging

    cat = Table(getdata(file_cls))
    gal = Table(getdata(file_gal))

    print('making stellar mass cut')
    cut2 = (gal['mass']>=10)&(gal['gi_o']>0.05)
    ngals = len(gal)
    
    ngals_cut = len(gal[cut2])
    logging.info('there is ',round(100*((ngals-ngals_cut)/ngals),2),' percent of galaxies with bad stellar rest frame colors')

    # cut = (gal['mass']>=10)&(gal['True']==True)&(gal['gi_o']>0.05)#&(gal['Pmem']>0.7)
    # gal = gal[cut]

    print('creating redshift bins')
    z_cls = cat['redshift']
    # ratio = cat['Ngals']/cat['Ngals_true']

    ### get redshift bins with ncls_per_bin
    # ncls_per_bin = 10
    zb, zb_means = binsCounts(z_cls,ncls_per_bin)
    idxb = get_bin_indicies(z_cls,zb)

    ### group the table
    indices_list = group_table_by(gal,zb,colName='redshift')    ## per redshift bin

    ids_cls = cat['CID']
    ids_cls_gal = gal['CID']
    indicies_unique = list(chunks(ids_cls_gal,ids_cls))         ## per cluster

    ### defining color variables
    color_list = ['gi_o','g-r','g-i','r-i','r-z','i-z']
    ecolor_list = [ci+'_err' for ci in color_list]

    color_value = [(0,1),(0,2),(1,2),(1,3),(2,3)]
    mag_list = ['iabs',1,2,2,3,3]
    gal = set_color_variables(gal,color_list[1:],ecolor_list[1:],color_value)

    print('Gaussian Mixture Modeling')
    li='gi_o'
    gmm_parameters = computeGMM(gal,zb_means,color_list,indices_list,lcolor=li)
    # plotArrayColor(gal,zb,indices_list,lcolor=li,xlims=(0.,1.2),gmm=gmm_parameters)

    mur = gmm_parameters['mean'][:,0]
    sigr = gmm_parameters['sigma'][:,0]
    alpr = gmm_parameters['alpha'][:,0]

    mub = gmm_parameters['mean'][:,1]
    sigb = gmm_parameters['sigma'][:,1]
    alpb = gmm_parameters['alpha'][:,1]

    rs_param = np.vstack(( zb_means[idxb], mur[idxb], sigr[idxb], alpr[idxb] )).transpose()
    bc_param = np.vstack(( zb_means[idxb], mub[idxb], sigb[idxb], alpb[idxb] )).transpose()

    indicies_cls = list(chunks3(cat['redshift'],zb))
    indicies_cls = [item for sublist in indicies_cls for item in sublist]

    gal,pred,pblue = computeColorProbabilties(gal,gmm_parameters[indicies_cls],indicies_unique,lcolor=li)
    _,pred_hard_cut,pblue_hard_cut = computeColorProbabilties(gal,gmm_parameters[indicies_cls],indicies_unique,hard_cut=True,lcolor=li)

    ### Quiescent and star-forming galaxies
    # star_forming = np.log10(gal['ssfr'])>=-10.5
    # quiescent = np.log10(gal['ssfr'])<=-10.

    # p_quiescent = [gal['Pmem'][quiescent[idx]] for idx in indicies_unique]
    # p_star_forming = [gal['Pmem'][star_forming[idx]] for idx in indicies_unique]

    fred, fblue, N_red, N_blue = getColorFractions(pred,pblue)
    fred_hard_cut, fblue_hard_cut, N_red_hard_cut, N_blue_hard_cut = getColorFractions(pred_hard_cut,pblue_hard_cut)
    # fquiescent, f_star_forming, N_quiescent, N_star_forming = getColorFractions(p_quiescent,p_star_forming)

    print('output catalogos')
    # gal['starForming'] = star_forming

    new_columns = ['Pred','Pblue','Pred_hc','Pblue_hc']
    new_data = [pred,pblue,pred_hard_cut,pblue_hard_cut]

    gal = initNewColumns(gal,new_columns,value=0.)

    for ci,ndi in zip(new_columns,new_data):
        i = 0
        for idx in indicies_unique:
            gal[ci][idx] = ndi[i]
            i += 1

    gal.write(file_gal,overwrite=True,format='fits')

    new_columns = ['Nred','Nblue','Nred_hc','Nblue_hc']
    new_data = [N_red,N_blue,N_red_hard_cut, N_blue_hard_cut]

    cat = initNewColumns(cat,new_columns,value=0.)
    for ci,ndi in zip(new_columns,new_data):
        cat[ci] = ndi

    cat['rs_param'] = rs_param
    cat['bc_param'] = bc_param
    cat['color_conv'] = gmm_parameters['conv'][idxb]

    cat.write(file_cls,format='fits',overwrite=True)

    print('computing rs parameters')
    ## get red sequence(RS) parameters
    if output_rs is not None:
        ncls_per_bin = 10
        zb2, zb2_means = binsCounts(z_cls,ncls_per_bin)

        ### group the table
        indices_list2 = group_table_by(gal,zb2,colName='redshift')

        ## rs_parameters - astropy Table
        ## slope(6), intercept(6), mean_color, std_rs(6)
        ## 6 colors: gi_o, g-r, g-i, r-i, r-z, i-z
        rs_parameters = computeRS(gal,mag_list,color_list,ecolor_list,indices_list2)
        rs_parameters['redshift'] = zb2_means

        ## take the mean in bins of 6 points
        rs_parameters = nbins_rs_parameters(zb2_means,rs_parameters,nbins=5)

        # ## save red sequence model
        rs_parameters.write(output_rs,format='fits',overwrite=True)

# ## produce color offset
# delta_colors = computeDeltaColors(gal,rs_parameters,indices_list,color_list,mag_list)

# ## gmm_parameters
# ## mu_r, sig_r, alpha_r, mu_b, sig_b, alpha_b (6)

# plt.clf()
# plt.figure(figsize=(10,8))
# plt.plot(zb_means,fred,color='r',label='GMM')
# plt.plot(zb_means,fblue,color='b')
# plt.plot(zb_means,fred_hard_cut,color='lightcoral',label='hard cut',linestyle='--')
# plt.plot(zb_means,fblue_hard_cut,color='royalblue',linestyle='--')

# plt.scatter(zb_means,fquiescent,color='gray',label='Quiescent')
# plt.scatter(zb_means,f_star_forming,color='lightgray',label='Star Forming')

# plt.legend()
# plt.xlabel('redshift')
# plt.ylabel('fraction')
# plt.ylim(0.,1.)
# plt.title('Buzzard: (%s)'%li)
# # plt.title('Buzzard: (g-i) rest-frame color')
# plt.savefig('red_blue_fraction_%s.png'%(li),bbox_inches='tight')
# plt.close()

# ## Compute color fractions
# fblue, fred = computeColorFractions(pblue,pred,keys=indices_list)
    # y_bin = [ np.mean(y[idx]) for idx in indices]
    # y_bin_std = [ np.std(y[idx]) for idx in indices]
    
    # plt.errorbar(zb_means,y_bin,xerr=np.diff(zb)/2, yerr=y_bin_std)
    # plt.show()

### plot color distritbution
# if plots:
#     plotArrayColor(lcolor='g-r',xlims=(0.,3.),tM=True)
#     plotArrayColor(lcolor='r-i',xlims=(-0.5,1.8),tM=True)
#     plotArrayColor(lcolor='i-z',xlims=(-0.25,1.),tM=True)
#     plotArrayColor(lcolor='gi_o',xlims=(0.,1.3),tM=True)

#     makeColorGif(lcolor='g-r',xlims=(0.,3.))
#     makeColorGif(lcolor='r-i',xlims=(-0.25,2.))
#     makeColorGif(lcolor='i-z',xlims=(-0.25,1.))
#     makeColorGif(lcolor='gi_o',xlims=(0.,1.3))

#     plotColorRedshift(gal,rs_parameters,zb,color_list)
#     makeColorDiagramGif(gal,rs_parameters,mag_number='iabs',lcolor='gi_o',ylims=(0.,1.2),rest_frame_color=True)
#     makeColorDiagramGif(gal,rs_parameters,mag_number=2,color_number=3,lcolor='r-i',ylims=(-0.25,1.85),rest_frame_color=False)
#     makeColorDiagramGif(gal,rs_parameters,mag_number=1,color_number=1,lcolor='g-r',ylims=(0.,3.),rest_frame_color=False)

#     plot_rs_parameters(zb,zb_means,rs_parameters)
#     plot_color_magnitude(gal,rs_parameters,color)
