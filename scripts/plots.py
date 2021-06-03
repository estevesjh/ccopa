 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 2021
@author: estevesjh
"""
from astropy.table import Table, vstack
from astropy.io.fits import getdata

import matplotlib
import numpy as np

from collections import defaultdict
from matplotlib import pylab
import matplotlib.pyplot as plt

import sklearn

from scipy import stats

import sys
sys.path.append("/home/s1/jesteves/git/ccopa/python/")
from main import copacabana

matplotlib.rcParams['figure.dpi'] = 80
params = {'figure.figsize': (7, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large'}
pylab.rcParams.update(params)


import sys
sys.path.append("/home/s1/jesteves/git/ccopa/python/")

from main import copacabana

class viewClusters:
    def __init__(self):
        '/home/s1/jesteves/perl5/bin'
        cfg = '/home/s1/jesteves/git/ccopa/config_files/config_buzzard_v2.yaml'
        self.copa = copacabana(cfg,dataset='buzzard_v2')

        self.models  = defaultdict(dict)
        self.metrics = defaultdict(dict)

        self.predictors = ['Ngals','MU','R200']
        self.regressors = ['Ngals_true','MU_TRUE','R200_true']
        self.aux_vars   = ['Ngals_true','MU_TRUE','R200_true','redshift','M200_true']
        
        self.npredictors = len(self.predictors)
        self.nauxialiars = len(self.aux_vars)
        pass

    def load_data(self,run_name):
        cat = self.copa.load_copa_out('cluster',run_name)
        self.df  = cat.to_pandas()
        
        self.models[run_name]['predictors'] = np.array(cat[self.predictors])
        self.models[run_name]['regressors'] = np.array(cat[self.regressors])
        self.models[run_name]['aux_vars']   = np.array(cat[self.aux_vars])
        self.compute_residuals(run_name)

    def compute_residuals(self,run_name):
        self.models[run_name]['residual']     = np.full_like(self.models[run_name]['predictors'],-99.)
        self.models[run_name]['log_residual'] = np.full_like(self.models[run_name]['predictors'],-99.)

        for colx,coly in zip(self.regressors,self.predictors):
            x = self.models[run_name]['regressors'][colx]
            y = self.models[run_name]['predictors'][coly]
            res, log_res = get_frac_residual(x,y)
            self.models[run_name]['residual'][coly] = res
            self.models[run_name]['log_residual'][coly] = log_res

    def make_bins(self, run_name, ngals_bins, mu_star_bins, r200_bins, zcls_bins, mass_bins):
        mybins     = [ngals_bins, mu_star_bins, r200_bins, zcls_bins, mass_bins]
        runs = list(self.models.keys())
        for run in runs:
            labels,values = self._make_bins(mybins,run)
            self.models[run_name]['bins_idx'] = labels
            self.models[run_name]['bins_val'] = values
        
    def _make_bins(self,bins,run_name):
        labels = np.full_like(self.models[run_name]['aux_vars'],-99.)
        values = np.full_like(self.models[run_name]['aux_vars'],-99.)
        for xbin,col in zip(bins,self.aux_vars):
            x = self.models[run_name]['aux_vars'][col]
            keys,xbins = get_bins(x,xbin)
            labels[col]= keys
            values[col]= xbins
        return labels,values

    ### compute metrics
    def compute_bin_statstics(self,run_name):
        ## out: dict('xbins','xmean','nobjs')
        out = defaultdict(dict)
        self.metrics[run_name] = out
        for jj in self.aux_vars:
            xs_true = self.models[run_name]['aux_vars'][jj]
            id_bins = self.models[run_name]['bins_idx'][jj].astype(np.int)
            xs_bins = self.models[run_name]['bins_val'][jj]

            indices  = np.sort(np.unique(id_bins))
            keys     = [np.where(id_bins==i)[0] for i in indices]

            nobjs    = np.array([idx.size for idx in keys])
            xbins    = np.array([xs_bins[idx[0]] for idx in keys])
            xmean    = np.array([np.mean(xs_true[idx]) for idx in keys])
            
            dtypes      = [(col,'<f8') for col in self.predictors]
            ymean       = np.full(xmean.size,-99.,dtype=dtypes)
            resmean     = ymean
            for ii in self.predictors:
                ys_true  = self.models[run_name]['predictors'][ii]
                res      = self.models[run_name]['residual'][ii]
                ymean[ii]= np.array([np.mean(ys_true[idx]) for idx in keys])
                resmean[ii]= np.array([np.mean(res[idx]) for idx in keys])
                
            mydict   = {'bins':indices,'xbins':xbins,'xmean':xmean,'ymean':ymean,'res_mean':resmean,
                        'nobjs':nobjs,'keys':keys}            
            self.metrics[run_name][jj] = mydict

    def eval_metrics(self, run_name, metric):
        #error_message = f"{run_name} not yet trained"
        #assert run_name in self.models, error_message

        ys_predict = self.models[run_name]['predictors']
        for jj in self.aux_vars:
            id_bins = self.models[run_name]['bins_idx'][jj]
            xbins   = self.metrics[run_name][jj]['xbins']
            
            dtypes      = [(col,'<f8') for col in self.predictors]
            scores      = np.full_like(xbins,-99.,dtype=dtypes)
            for ii,kk in zip(self.predictors,self.regressors):
                ys_true = self.models[run_name]['regressors'][kk]
                scores[ii]  = self.metric_bin(metric,ys_true,ys_predict[ii], id_bins)
            self.metrics[run_name][jj][metric] = scores

    def metric_bin(self,metric,ytrue,ypred,bins_id):
        metric_funcs = {"r2": r2_score,
                        "bias": bias_log_res,
                        "scatter_stdev": fractional_error_stdev,
                        "scatter_percentile": fractional_error_percentile,
                        "scatter_nmad": get_sigmaNMAD,
                        "outlier_frac": get_outlier_frac}
        error_message = ('{} not recognized! options are: {}'
                         ''.format(metric, metric_funcs.keys()))
        assert metric in metric_funcs, error_message

        indices  = np.sort(np.unique(bins_id))
        keys     = [np.where(bins_id==i)[0] for i in indices]

        ytrue_bin= [ytrue[idx] for idx in keys]
        ypred_bin= [ypred[idx] for idx in keys]
        
        nbins    = len(indices)
        scores   = np.full(nbins,-99.,dtype=np.float64)
        for i,yt,yp in zip(range(nbins),ytrue_bin,ypred_bin):
            scores[i] = metric_funcs[metric](yt,yp)
        return scores


def get_bins(variable,xedges):
    nbins   = len(xedges)-1
    indices = np.full_like(variable,-99,dtype=np.int)
    xbins   = np.full_like(variable,-99,dtype=np.float)

    means = (xedges[1:]+xedges[:-1])/2.
    for i in range(nbins):
        idx = np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0]
        xbins[idx]   = means[i]
        indices[idx] = i
    return indices, xbins

def get_log(x):
    xlog = np.log10(x)
    xlog[np.isinf(xlog)] = -99
    xlog[np.isnan(xlog)] = -99
    return xlog

def get_frac_residual(x,y):
    res = y/(x+1e-6)
    log_res = get_log(res)
    return res,log_res

def mad(data, axis=None):
    return np.median(np.abs(data - np.median(data)))

def median_absolute_dev(x,y):
    res, log_res = get_frac_residual(x,y)
    return mad(res[log_res>-99])

def bias_log_res(x,y):
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    return 1-10**np.median(log_res[mask])

def fractional_error_stdev(x, y):
    res, log_res = get_frac_residual(x,y)
    score = np.std(res[log_res>-99])
    return score

def fractional_error_percentile(x, y):
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    p16 = np.percentile(res[mask], 16)
    p84 = np.percentile(res[mask], 84)
    score = 0.5*(p16+p84)
    return score

def get_sigmaNMAD(x,y):
    sigmaNMAD = 1.4*median_absolute_dev(x,y)
    return sigmaNMAD

def get_outlier_frac(x,y):
    res, log_res = get_frac_residual(x,y)
    sigmaNMAD = 1.4*mad(log_res[log_res>-99])
    bias      = np.median(log_res[log_res>-99])
    out       = np.where(np.abs((log_res-bias)>=3.*sigmaNMAD))[0]
    frac      = 1.*out.size/x.size
    return frac

def r2_score(x,y):
    """ returns non-aggregate version of r2 score.

    based on r2_score() function from sklearn (http://sklearn.org)
    """
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    return sklearn.metrics.r2_score(x[mask],y[mask]) 

def plot_scaling_relation_run(xcol,ycol,fit=False,title='',log_scale=True):
    runs = list(vc.models.keys())
    
    fig = plt.figure(figsize=(10,8))
    ax  = plt.axes()
    for run in runs:
        plot_scaling_relation(xcol,ycol,run,axs=ax,fit=False,title='',log_scale=True)    

def plot_scaling_relation(xcol,ycol,run,axs=None,fit=False,title='',log_scale=True):
    if axs is None: axs = plt.axes()

    ## get variables
    x,y,xb,yb,xb_err,yb_err,xlims = get_binned_variables(xcol,ycol,run)
    
    ## linear fit
    linreg=lin_reg(x,y)
    
    idx = np.argsort(x)
    xt,yh = x[idx],linreg['Yhat'][idx]

    b0 = round(linreg['b0'],3)
    b1 = round(linreg['b1'],3)
    cb_u, cb_l = linreg['cb_u'], linreg['cb_l']
    
    ## Plot
    if fit:
        axs.plot(xt,yh, color="r",label='y=%.2f+%.2fx'%(b0,b1))
        axs.fill_between(xt, cb_l, cb_u, color="gray", alpha=0.25, label='_nolabel_')
        axs.plot(xt,cb_l, color="r", label='_nolabel_')
        axs.plot(xt,cb_u, color="r", label='_nolabel_')

    sc = axs.scatter(x,y,s=75, alpha=0.25, color=gray,label='run:'+run)
    axs.errorbar(xb,yb,xerr=xb_err,yerr=yb_err,color=blue,linewidth=2.,fmt='o')
    axs.plot(np.linspace(xlims[0],xlims[1]),np.linspace(xlims[0],xlims[1]),linestyle='--',color='r')
    
    if log_scale:
        xlims = np.where(xlims<1.,2.,xlims)
        axs.set_xscale('log')
        axs.set_yscale('log')

    axs.set_ylim(xlims)
    axs.set_xlim(xlims)
    
    axs.set_xlabel(xcol,fontsize=22)
    axs.set_ylabel(ycol,fontsize=22)
    axs.legend(fontsize=14)
    
    axs.set_title(title,fontsize=22)
    #fig.tight_layout()