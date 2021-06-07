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
sys.path.append(os.path.abspath("/home/s1/jesteves/git/ccopa/python/"))
import copac.gaussianKDE as kde
from copac.probRadial import doPDF, norm_constant

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
        #plt.savefig(os.path.join(self.path,'probability_histograms.png'),bb_box='tight')
        
        #self.close()

    def plot_histograms(self,galaxies,axis,lcol='Pmem'):
        prob = galaxies[lcol]
        
        mask = galaxies['True'] == True
        nmask = galaxies['True'] == False

        xbins = np.linspace(0.01,1.,10)
        
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

    def plot_grid_pdf(self,args,save='./img/pdf_radial_validation.png'):
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

        fig = plt.figure(figsize=(10,8))
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
        ax3.set_ylabel('frac. error',fontsize=16)

        ax4.set_xlabel(self.xlabel2,fontsize=16)
        ax4.set_ylabel('frac. error',fontsize=16)

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


        #plt.savefig(save)
        #self.close()


    def plot_validation_pdf_color(self,gal,gal2,keys,lcolor='delta_rs',method='N',save='./img/pdf_color_validation.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

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

        self.plot_grid_pdf(args,save=save)

    def plot_validation_pdf_redshift(self,gal,gal2,keys,method='N',save='./img/pdf_redshift_validation.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

        z = gal['zoffset']
        z2 = gal2['zoffset']

        # pmem = gal['Pz']*gal['Pr']
        pmem = gal['Pmem']

        # pmem2 = gal2['Pz']*gal2['Pr']
        pmem2 = gal2['Pmem']

        ### curve 1
        x_bin, pdf, pdf2 = get_pdf_clusters(indices,indices2,z,z2,pmem,xlims=(-0.1,0.1,100))

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

        self.plot_grid_pdf(args,save=save)

    def plot_validation_pdf_radial(self, gal,gal2,keys, method='N', save='./img/pdf_radial_validation.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

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

        self.plot_grid_pdf(args,save=save)

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
    
    def plot_probabilities_colorPDF(self,gal,gal2,keys,lcolor='delta_rs',save='./img/prob_color_PDF.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

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

    def plot_probabilities_redshiftPDF(self,gal,gal2,keys,save='./img/prob_redshift_PDF.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

        z = gal['zoffset']#zoffset(gal['z'],gal['redshift'])
        z2 = gal['zoffset']#zoffset(gal2['z'],gal2['redshift'])

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

    def plot_probabilities_radialPDF(self,gal,gal2,keys,save='./img/prob_radial_PDF.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

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


    def plot_multiple_scaling_relations(self,gal,gal2,keys,save='./img/prob_scaling_relation.png'):
        indices = list(chunks(gal['CID'],keys))
        indices2 = list(chunks(gal2['CID'],keys))

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

## added on Jun 6th, 2021
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.append("/home/s1/jesteves/git/ccopa/python/")
from main import copacabana

class viewMembershipSelection:
    def __init__(self,cfg='../config_files/config_copa_dc2.yaml',dataset='cosmoDC2'):
        self.copa = copacabana(cfg,dataset=dataset)

        self.datasets   = defaultdict(dict)
        self.curves     = defaultdict(dict)

        self.cluster_vars  = ['CID','redshift']
        self.galaxy_vars   = ['dmag','zoffset','Rn']
        self.probabilities = ['Pmem','Pz','Pr','Pc','pz0','True']
        
        self.columns = self.cluster_vars+self.galaxy_vars+self.probabilities

    def load_data(self,run_name):
        gal = self.copa.load_copa_out('members',run_name)
        gal['True'] = gal['True'].astype(np.int)
        self.gal = gal
        self.datasets[run_name] = gal[self.columns]
    
    def compute_precision_recall_curves(self,run_name,prob='Pmem',th='True'):
        scores= self.datasets[run_name][prob]
        true  = self.datasets[run_name][th]
        
        precisions, recalls, thresholds = precision_recall_curve(true, scores)
        
        idx = np.argmax(recalls*precisions)
        
        optimal = {'idx':idx,'purity':precisions[idx],'completeness':recalls[idx],'thresholds':thresholds[idx]}
        
        self.curves[run_name]['purity'] = precisions[:-1]
        self.curves[run_name]['completeness'] = recalls[:-1]
        self.curves[run_name]['thresholds'] = thresholds
        self.curves[run_name]['optimal'] = optimal

    def plot_purity_completeness(self,run,ax=None,color='b',optimal=True):
        if ax is None: ax = plt.axes()

        y = self.curves[run]['purity']
        x = self.curves[run]['completeness']

        opt_y = self.curves[run]['optimal']['purity']
        opt_x = self.curves[run]['optimal']['completeness']

        label = "%s: P=%.2f,  C=%.2f"%(run,opt_y,opt_x)

        ymin, ymax = np.min(y)-0.05,np.max(y)+0.05
        ## plot purity and completeness curves 
        ax.plot(x,y, color=color, linewidth=2, ls='solid', label=label)

        ## draw red lines on the optimal points
        if optimal:
            self._plot_optimal(run,'completeness','purity',ax=ax,color=color)

        ax.grid(True)                      
        ticks0 = np.arange(0.,1.01,0.1)
        ax.set_xticks(ticks0)
        ax.set_yticks(ticks0)
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(-0.01,1.01)

    def plot_precision_recall_vs_threshold(self,run,ls='.',ax=None,c1='b',optimal=False):
        if ax is None: ax = plt.axes()

        y1 = self.curves[run]['purity']
        y2 = self.curves[run]['completeness']
        x  = self.curves[run]['thresholds']

        ## plot purity and completeness curves 
        ax.plot(x,y1, color=c1, label="Purity"      , linewidth=2, ls='--')
        ax.plot(x,y2, color=c1, label="Completeness", linewidth=2, ls='solid')

        ## draw red lines on the optimal points
        if optimal:
            self._plot_optimal(run,'thresholds','purity'      ,ax=ax,color=color)
            self._plot_optimal(run,'thresholds','completeness',ax=ax,color=color)

        ax.grid(True)                      
        ax.set_ylim(0.0,1.)
        ax.set_xlim(-0.01,1.01)
        ticks0 = np.arange(0.,1.01,0.1)
        ax.set_xticks(ticks0)
        ax.set_yticks(ticks0)

    def _plot_optimal(self,run_name,xcol,ycol,ax=None,color='r'):
        if ax is None: ax = plt.axes()

        opt_x = self.curves[run_name]['optimal'][xcol]
        opt_y = self.curves[run_name]['optimal'][ycol]

        ax.plot([opt_x, opt_x], [0., opt_y], "k:",label='_')
        ax.plot([0., opt_x], [opt_y, opt_y], "k:",label='_')
        ax.scatter([opt_x], [opt_y], color=color,s=100,label='_')


class viewClusters:
    def __init__(self,cfg='../config_files/config_copa_dc2.yaml',dataset='cosmoDC2'):
        self.copa = copacabana(cfg,dataset=dataset)

        self.models  = defaultdict(dict)
        self.metrics = defaultdict(dict)

        self.predictors = ['Ngals','MU','R200']
        self.regressors = ['Ngals_true','MU_TRUE','R200_true']
        self.aux_vars   = ['Ngals_true','MU_TRUE','R200_true','redshift','M200_true']
        
        self.npredictors = len(self.predictors)
        self.nauxialiars = len(self.aux_vars)
        
        self.metric_funcs = {"bias": bias_log_res,
                             "scatter_stdev": fractional_error_stdev,
                             "scatter_percentile": fractional_error_percentile,
                             "scatter_nmad": get_sigmaNMAD,
                             "outlier_frac": get_outlier_frac}
        pass

    def load_data(self,run_name):
        cat = self.copa.load_copa_out('cluster',run_name)
        self.df  = cat.to_pandas()
        
        self.models[run_name]['predictors'] = np.array(cat[self.predictors])
        self.models[run_name]['regressors'] = np.array(cat[self.regressors])
        self.models[run_name]['aux_vars']   = np.array(cat[self.aux_vars])
        self.compute_residuals(run_name)
    
    def compute_resiudal_all(self):
        runs = self.models.keys()
        for run_name in runs:
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

    def make_bins(self, runs, ngals_bins, mu_star_bins, r200_bins, zcls_bins, mass_bins):
        mybins             = [ngals_bins, mu_star_bins, r200_bins, zcls_bins, mass_bins]
        self.metrics['bins']= dict()
        for jj,xbin in zip(self.aux_vars,mybins):
            self.metrics['bins'][jj]  = xbin
            self.metrics['nbins'][jj] = xbin.size - 1
        
        for run_name in runs:
            labels,values = self._make_bins(mybins,run_name)
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
        for jj in self.aux_vars:
            id_bins = self.models[run_name]['bins_idx'][jj].astype(np.int)    
            xs_vals = self.models[run_name]['aux_vars'][jj]
            ys_true = self.models[run_name]['predictors']
            res     = self.models[run_name]['residual']
                
            nbins    = self.metrics['nbins'][jj]
            xbins    = self.metrics['bins'][jj]
            
            indices  = np.arange(nbins,dtype=np.int)
            xmean    = get_binned_mean(xs_vals,xs_vals,xbins)#0.5*(xbins[1:]+xbins[:-1])
            nobjs    = np.histogram(xs_vals,bins=xbins)[0]
            
            ymean       = np.full_like(ys_true,np.nan)[:nbins]
            resmean     = ymean.copy()
            
            for ii in self.predictors:
                ymean[ii]     = get_binned_mean(xs_vals,ys_true[ii],xbins)
                resmean[ii]   = get_binned_mean(xs_vals,res[ii],xbins)
                
            mydict   = {'bins':indices,'xbins':xbins,'xmean':xmean,'ymean':ymean,'res_mean':resmean,
                        'nobjs':nobjs}            
            self.metrics[run_name][jj] = mydict
    
    def eval_metrics(self,run_name,metric):
        ys_predict = self.models[run_name]['predictors']
        ys_true    = self.models[run_name]['regressors']
        
        dtypes      = [(col,'<f8') for col in self.predictors]
        scores      = np.full((1,),-99.,dtype=dtypes)
        for ii,kk in zip(self.predictors,self.regressors):
            scores[ii]  = self.metric_funcs[metric](ys_true[kk],ys_predict[ii])
        self.metrics[run_name][metric] = scores
    
    def eval_all_metrics(self,run_name,binned=False):
        ### func can be: eval_metrics_bin, eval_metrics
        metrics= self.metric_funcs.keys()
        if not binned:
            for metric in metrics:
                self.eval_metrics(run_name, metric)
        else:
            for metric in metrics: 
                self.eval_metrics_bin(run_name, metric)

    def eval_metrics_bin(self, run_name, metric):
        #error_message = f"{run_name} not yet trained"
        #assert run_name in self.models, error_message

        ys_predict = self.models[run_name]['predictors']
        ys_true    = self.models[run_name]['regressors']
        xs_vals    = self.models[run_name]['aux_vars']
        
        for jj in self.aux_vars:
            xbins   = self.metrics[run_name][jj]['xbins']
            dtypes      = [(col,'<f8') for col in self.predictors]
            scores      = np.full_like(xbins[1:],-99.,dtype=dtypes)
            for ii,kk in zip(self.predictors,self.regressors):
                scores[ii]  = self.metrics_bin(metric, ys_true[kk],ys_predict[ii], xs_vals[jj], xbins)
            self.metrics[run_name][jj][metric] = scores

    def metrics_bin(self,metric,ytrue,ypred,xvar,xbins):
        error_message = ('{} not recognized! options are: {}'
                         ''.format(metric, self.metric_funcs.keys()))
        assert metric in self.metric_funcs, error_message
        
        keys     = get_bins_group_indices(xvar,xbins)
        ytrue_bin= group_by(ytrue,keys)#get_bins_group(x,ytrue,xbins)
        ypred_bin= group_by(ypred,keys)#get_bins_group(x,ypred,xbins)#[ypred[idx] for idx in keys]
        
        nbins    = len(xbins)-1
        scores   = np.full_like(xbins[:-1],-99.,dtype=np.float64)
        
        for i,yt,yp in zip(range(nbins),ytrue_bin,ypred_bin):
            scores[i] = self.metric_funcs[metric](yt,yp)
        return scores

    def show_metrics_table_all(self,run):
        metrics= self.metric_funcs.keys()
        mydict = dict.fromkeys(metrics)
        for ycol in self.predictors:
            for col in metrics:
                mydict[col] = np.append(mydict[col],self.metrics[run][col][ycol])
        df = pd.DataFrame(mydict,index=[None]+self.predictors)[1:]
        #df.set_index('index')
        return df
            
    def get_residual_metrics(self,run,ycol,log_residual=False):
        bias = self.metrics[run]['bias'][ycol]+1
        sigma= self.metrics[run]['scatter_percentile'][ycol]
        sigma_nmad = self.metrics[run]['scatter_nmad'][ycol]
        of   = self.metrics[run]['outlier_frac'][ycol]

        if log_residual:
            bias,sigma = np.log10(bias), np.log10(sigma+1)
            sigma_nmad = np.log10(sigma_nmad+1)
        return bias,sigma,sigma_nmad,of
    
    def get_residual_metrics_binned(self,run,xcol,ycol,log_residual=False,metric='scatter_nmad'):
        xbins = self.metrics[run][xcol]['xbins']
        xmean = self.metrics[run][xcol]['xmean']
        bias  = self.metrics[run][xcol]['bias'][ycol]+1
        sigma = self.metrics[run][xcol][metric][ycol]
        of    = self.metrics[run][xcol]['outlier_frac'][ycol]
        
        lower  = xmean-xbins[:-1]
        upper  = xbins[1:]-xmean
        return xmean,bias,sigma,of,lower,upper
        
    def plot_residual_distribution(self,run,ycol,axs=None,xlims=None,log_residual=False):
        if axs is None: axs=plt.axes()
        residual    = filter_nan_inf(self.models[run]['residual'][ycol])
        log_res     = filter_nan_inf(self.models[run]['log_residual'][ycol])
        mask        = get_oulier_mask(log_res)

        xlabel   = r'$%s/%s_{true}$'%(ycol,ycol)
        units    = '\n'
        if log_residual:
            residual = log_res
            xlabel   = r'Log(%s)'%xlabel
            units    = 'dex\n'

        bias,sigma,sigma_nmad,of = self.get_residual_metrics(run,ycol,log_residual=log_residual)
        xmin,xmax   = np.nanpercentile(residual[residual>-99],[0,100])
        xlo,xup     = bias-2*sigma_nmad, bias+2*sigma_nmad

        if xlims is not None:
            xmin,xmax = xlims

        l1 = r'bias  : %.2f %s$\sigma_{NMAD}$: %.2f %s'%(bias,units,sigma_nmad,units)
        l2 = r'outlier: %.2f'%(of)

        xbins = np.linspace(xmin,xmax,50)
        _ = axs.hist(residual,bins=xbins,label=l1,histtype='step',lw=3)
        _ = axs.hist(residual[mask],bins=xbins,label=l2,histtype='step',lw=3)
        axs.axvline(bias,color='k',ls='--')
        axs.axvline(xup,color='r',ls='--')
        axs.axvline(xlo,color='r',ls='--')

        #plt.yscale('log')
        axs.set_xlabel(xlabel,fontsize=20)
        axs.legend(fontsize=12)
        axs.set_title(run,fontsize=14)
        
    def get_binned_variables(self,xcol,ycol,run,metric='scatter_nmad'):
        ## get variables
        x = self.models[run]['regressors'][xcol]
        y = self.models[run]['predictors'][ycol]
        
        xbins = self.metrics[run][xcol]['xbins']
        xb= self.metrics[run][xcol]['xmean']#0.5*(xbins[1:]+xbins[:-1])#self.metrics[run][xcol]['xmean']
        yb= self.metrics[run][xcol]['ymean'][ycol]
        
        lower,upper = xb-xbins[:-1],xbins[1:]-xb
        
        yb_err= self.metrics[run][xcol][metric][ycol]
        xb_err= [lower,upper]
        
        xmin,xmax = 0.8*(np.min(xbins)-2*yb_err[0]), 1.2*(np.max(xbins))
        xlims = (xmin,xmax)
        
        mask  = np.logical_not(np.isnan(x)|np.isnan(y))
        x = x[mask]
        y = y[mask]
        return x,y,xb,yb,xb_err,yb_err*xb,xlims
        
    def plot_scaling_relation(self,xcol,ycol,run,color='r',axs=None,xlims=None,points=False,metric='scatter_nmad',fit=False,title='',log_scale=True):
        ## get variables
        x,y,xb,yb,xb_err,yb_err,xlims0 = self.get_binned_variables(xcol,ycol,run,metric=metric)
        if xlims is None: xlims=xlims0
        _plot_scaling_relation(x,y,xb,yb,xb_err,yb_err,xlims,xcol,ycol,li=run,points=points,axs=axs,fit=fit,title=title,log_scale=log_scale,color=color)
    
    def plot_residual(self,run,xcol,ycol,ax=None,xlog=False,color='r',points=True,units='',shift=0.):
        if ax is None: ax = plt.axes()
        xmean,ymean,sigma,of,lower,upper = self.get_residual_metrics_binned(run,xcol,ycol,log_residual=False)
        ymin,ymax = np.mean(ymean-3*sigma),np.mean(ymean+3*sigma)
        xmin,xmax = 0.8*np.min(xmean-lower),1.2*np.max(xmean+upper)
        ylabel = r'$%s/%s_{true}$'%(ycol,ycol)

        if points:
            x  = self.models[run]['aux_vars'][xcol]
            res= self.models[run]['residual'][ycol]
            ax.scatter(x,res,color=color,alpha=0.3,s=40)

        ax.plot([xmin,xmax],[1.,1.],'k--')
        ax.errorbar(shift+xmean,ymean,xerr=[lower,upper],yerr=sigma,color=color, fmt='o', capsize=4, capthick=2,label=run)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel(xcol+units,fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)
        ax.legend()
        if xlog:
            ax.set_xscale('log')

########################################
########################################

def group_by(x,keys):
    return [x[idx] for idx in keys]

def get_bins_group_indices(x,bins):
    idx  = np.argsort(x)
    ## to avoid the boundary condition of the digitize function as xlow <= x < xup
    mybins = bins.copy()
    mybins[-1] += 0.1
    inds = np.digitize(x,mybins)
    return np.split(idx, np.unique(inds[idx], return_index=True)[1][1:])
    
def get_bins_group(x,y,bins):
    idx  = np.argsort(x)
    ## to avoid the boundary condition of the digitize function as xlow <= x < xup
    mybins = bins.copy()
    mybins[-1] += 0.1
    inds = np.digitize(x,mybins)
    return np.split(y[idx], np.unique(inds[idx], return_index=True)[1][1:])
    
def get_binned_mean(x,y,bins):
    y     = filter_nan_inf(y)
    y     = np.where(y==-99,0.,y)
    x     = filter_nan_inf(x)
    sum_y = np.histogram(x, bins, weights=y)[0]
    nobjs = np.histogram(x, bins)[0]
    return sum_y/nobjs
    
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
    return 10**mad(log_res[log_res>-99])

def bias_log_res(x,y):
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    return 10**np.median(log_res[mask])-1.

def fractional_error_stdev(x, y):
    res, log_res = get_frac_residual(x,y)
    score = np.std(log_res[log_res>-99])
    return 10**score-1.

def fractional_error_percentile(x, y):
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    p16 = np.percentile(log_res[mask], 16)
    p84 = np.percentile(log_res[mask], 84)
    score = 0.5*(p84-p16)
    return 10**score-1.

def get_sigmaNMAD(x,y):
    sigmaNMAD = 1.4*(median_absolute_dev(x,y)-1.)
    return sigmaNMAD

def get_outlier_frac(x,y):
    res, log_res = get_frac_residual(x,y)
    sigmaNMAD = 1.4*mad(log_res[log_res>-99])
    bias      = np.nanmedian(log_res[log_res>-99])
    out       = np.where(np.abs((log_res-bias)>=3.*sigmaNMAD))[0]
    frac      = 1.*out.size/x.size
    return frac

def get_oulier_mask(log_res):
    sigmaNMAD = 1.4*mad(log_res[log_res>-99])
    bias      = np.nanmedian(log_res[log_res>-99])
    out       = np.where(np.abs((log_res-bias)>=3.*sigmaNMAD))[0]
    return out
    
def r2_score(x,y):
    """ returns non-aggregate version of r2 score.

    based on r2_score() function from sklearn (http://sklearn.org)
    """
    res, log_res = get_frac_residual(x,y)
    mask = log_res>-99.
    return sklearn.metrics.r2_score(x[mask],y[mask]) 

def filter_nan_inf(x):
    mask = np.isinf(x)|np.isnan(x)
    x[mask] = -99.
    return x

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
