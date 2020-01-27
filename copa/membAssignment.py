# !/usr/bin/env python

import numpy as np
import logging

import scipy.integrate as integrate

from time import time
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

## -------------------------------
## local libraries
import helper as helper
import probRadial as radial
import probRedshift as probz
import probColor as probc
import background as backSub


## -------------------------------
## auxiliary functions

def initNewColumns(data,colNames,value=-1):
    for col in colNames:
        data[col] = value*np.ones_like(data['CID'])
    return data

def computeNorm(gals,cat,r200,nbkg):
    norm = []
    
    good_indices, = np.where(nbkg>0)
    for idx in good_indices:
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]

        print('cls_id',cls_id, 'at redshift', z_cls)

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)&(gals['R']<=r2))
        probz = gals['PDFz'][galaxies]
        
        n_cls_field = np.sum(probz)/(np.pi*r2**2)    
        n_gals = n_cls_field-nb

        ni = n_gals/nb
        if ni<0:
            print('norm:',ni)

        norm.append(ni)

    return np.array(norm)

def doProb(ngals,nbkg,norm,normed=False):
    
    if normed:
        ngals /= ngals.sum()
        nbkg /= nbkg.sum()

    prob = norm*ngals/(norm*ngals+nbkg)
    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    return prob

def computeProb(keys, pdfs, pdfs_bkg, norm):
    prob = np.empty((1,4),dtype=float)

    pdfr, pdfz, pdfc = pdfs
    pdfr_bkg, pdfz_bkg, pdfc_bkg = pdfs_bkg
    
    ncls = len(keys)
    for i in range(ncls):
        key_i, ni = keys[i], norm[i]

        Ngals =  pdfr[key_i]*pdfz[key_i]*pdfc[key_i] 
        Ngals_bkg = pdfr_bkg[key_i]*pdfz_bkg[key_i]*pdfc_bkg[key_i] 

        pr = doProb(pdfr[key_i],pdfr_bkg[key_i],ni)
        pz = doProb(pdfz[key_i],pdfz_bkg[key_i],ni)
        pc = doProb(pdfc[key_i],pdfc_bkg[key_i],ni)

        pmem = doProb(Ngals,Ngals_bkg,ni)
  
        pi = np.array([pr,pz,pc,pmem]).transpose()
        prob = np.vstack([prob,pi])

    return prob[1:,0], prob[1:,1], prob[1:,2], prob[1:,3]

def computeContamination(gal,keys,r200,magLim):
    bkgFlag = np.full(len(gal['Bkg']), False, dtype=bool)
    ratio = []
    
    ncls = len(keys)
    for i in range(ncls):
        r2 = r200[i]
        # contaminants = (gi['R']>=r2)&(gi['True']==False)&(gi['R']<=4*r2)
        contaminants, = np.where((gal['CID'] == keys[i])&(gal['R']<=r2)&(gal['True']==False)& (gal['mag'][:,2]<=magLim[i,2]) )
        
        pz_cont = gal['PDFz'][contaminants]
        
        area = np.pi*(r2)**2
        ncon = np.sum(pz_cont)
        
        ratio_i = ncon/area
        ratio.append(ratio_i)

        bkgFlag[contaminants] = True

    return np.array(ratio), bkgFlag

def computeNgals(g,keys,r_aper=2.,true_gals=False,col='Pmem'):
    ngals = []
    
    for idx in keys:
        if true_gals:
            w, = np.where((g['CID']==idx)&(g['True']==True)&(g['R']<=r_aper))
            ni = len(w)
        else:
            w, = np.where((g['CID']==idx)&(g['R']<=r_aper))
            ni = np.sum(g[col][w])
        ngals.append(ni)

    return np.array(ngals)

def fastGaussianIntegration(membz,membzerr,zmin,zmax):
    zpts,zstep=np.linspace(zmin,zmax,50,retstep=True) #split redshift window for approximation
    area=[]
    for i in range(len(zpts)-1): #approximate integral using trapezoidal riemann sum
        gauss1=gaussian(zpts[i],membz,membzerr) #gauss1/2 are gaussian values for left,right points, respectively
        gauss2=gaussian(zpts[i+1],membz,membzerr)
        area1=((gauss1+gauss2)/2.)*zstep
        area.append(area1)
    area=np.array(area)
    arflip=area.swapaxes(0,1)
    prob=np.sum(arflip,axis=1)
    return prob

def PhotozProbabilities(zmin,zmax,membz,membzerr,fast=True):
    if fast:
        out = fastGaussianIntegration(membz,membzerr,zmin,zmax)
    
    else:
        out = []
        for i in range(len(membz)):
            aux, err = integrate.fixed_quad(gaussian,zmin,zmax,args=(membz[i],membzerr[i]))
            out.append(aux)
        out = np.array(out)
    return out

def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def getPDFz(membz,membzerr,zi,method='pdf'):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    if method=='old':
        sigma = np.median(membzerr)
        
        zmin, zmax = (zi-1.5*sigma*(1+zi)), (zi+1.5*sigma*(1+zi))
        pdf = PhotozProbabilities(zmin,zmax,membz,membzerr)
        pdf = np.where(pdf<0.001,0.,pdf/np.max(pdf))
    
    elif method=='pdf':
        pdf = gaussian(zi,membz,membzerr)
        pdf = np.where(pdf<0.001,0.,pdf/np.max(pdf))

    return pdf

def computePDFz(z,zerr,cid,cat,method='pdf'):
    ncls = len(cat)
    indicies = np.empty((0),dtype=int)
    pdfz = np.empty((0),dtype=float)

    z= np.where(z<0.,0.,z)
    for i in range(ncls):
        z_cls, idx = cat['redshift'][i], cat['CID'][i]
        idxSubGal, = np.where(cid==idx)
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],z_cls,method=method)
        pdf = np.where(pdf<0.01,0.,pdf) ## it avoids that float values gets boost by color pdfs

        indicies = np.append(indicies,idxSubGal)
        pdfz  = np.append(pdfz,pdf)
    
    return pdfz, indicies

def plotFactor(x,y,r_in=6,r_out=8):
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.scatter(x,np.log10(y),s=50,color='r')
    plt.axhline(0.,linestyle='--',color='k')
    plt.xlabel('redshift')
    plt.ylabel(r'log( N(%i - %i Mpc)/N($<R_{200}$) )'%(r_in,r_out))
    plt.title('Buzzard Background; N=%i'%(len(y)))
    plt.ylim(-1,+1)
    plt.savefig('background.png', bbox_inches = "tight")

def getIndices(IDs,keys):
    indicies = np.empty((0),dtype=int)
    indicies_into_cluster = np.empty((0),dtype=int)

    for i in range(keys.size):
        w, = np.where(IDs==keys[i])
        w2 = np.full(w.size,i,dtype=int)
        indicies = np.append(indicies,w)
        indicies_into_cluster = np.append(indicies_into_cluster,w2)

    return indicies,indicies_into_cluster

## -------------------------------
## main function
def clusterCalc(gal, cat, member_outfile=None, cluster_outfile=None,
                r_in=4, r_out=6, M200=1e14, p_low_lim=0.01, simulation=True, computeR200=False):
    ##############
    rmax = 3 #Mpc
    method='pdf'
    ##############

    print('estimate PDFz for each galaxy')
    pz,idxs = computePDFz(gal['z'],gal['zerr'],gal['CID'],cat,method=method)
    gal['PDFz'][idxs] = pz

    print('Computing Galaxy Density')
    ## Compute nbkg
    _, nbkg, BkgFlag = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,r_aper=1.,nslices=72)
    _, nbkg2,_ = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,r_aper=1.,nslices=72,method='counts')

    ## updating galaxy status
    gal['Bkg'] = BkgFlag    ## all galaxies inside the good backgrund ring's slice
    
    print('Computing R200')
    # r200 = radial.computeR200(gal, cat, nbkg2, rmax=rmax, defaultMass=M200, compute=computeR200)
    r200 = 1.*np.ones_like(cat['CID'])
    
    if simulation:
        nbkg0, BkgFlag0 = computeContamination(gal, cat['CID'], r200, np.array(cat['magLim']))

    # ngals, galFlag, keys = backSub.computeGalaxyDensity(gal, cat, rmax*np.ones_like(r200), nbkg, nslices=72)
    ngals, galFlag, keys = backSub.computeGalaxyDensity(gal, cat, r200, nbkg, nslices=72)
    gal['Gal'] = galFlag    ## all galaxies inside R200

    print('Computing PDFs \n')
    print('-> Radial Distribution')
    pdfr, pdfr_bkg = radial.computeRadialPDF(gal, cat, r200, nbkg, c=3.53, plot=False)
    # pdfr, pdfr_bkg = np.ones_like(gal['R']), np.ones_like(gal['R'])
    print('Check size array: pdfr, pdfr_bkg',len(pdfr),len(pdfr_bkg),'\n')

    print('-> Redshift Distribution')
    pdfz, pdfz_bkg, flagz = probz.computeRedshiftPDF(gal, cat, r200, nbkg, plot=False)
    print('Check size array: pdfz, pdfz_bkg',len(pdfz),len(pdfz_bkg),'\n')

    print('-> Color Distribution')
    # area_bkg = np.pi*(r_out**2-r_in**2)
    pdfc, pdfc_bkg, flagc = probc.computeColorPDF(gal, cat, r200, nbkg, bandwidth=[0.003,0.001,0.001],parallel=True,plot=False)
    print('Check size array: pdfc, pdfc_bkg',len(pdfc),len(pdfc_bkg),'\n')
    
    # (g-r), (g-i), (r-i), (r-z), (i-z)
    pdfc = pdfc[:,0]*pdfc[:,1]*pdfc[:,2]*pdfc[:,3]*pdfc[:,4]
    pdfc_bkg = pdfc_bkg[:,0]*pdfc_bkg[:,1]*pdfc_bkg[:,2]*pdfc_bkg[:,3]*pdfc_bkg[:,4]

    # (g-r), (g-i), (r-i), (i-z)
    # pdfc = pdfc[:,0]*pdfc[:,1]*pdfc[:,2]*pdfc[:,4]
    # pdfc_bkg = pdfc_bkg[:,0]*pdfc_bkg[:,1]*pdfc_bkg[:,2]*pdfc_bkg[:,4]

    print('\n Compute Probabilities')
    pdfs = [pdfr,pdfz,pdfc]
    pdfs_bkg = [pdfr_bkg,pdfz_bkg,pdfc_bkg]

    norm = computeNorm(gal, cat, r200, nbkg)
    Pr, Pz, Pc, Pmem = computeProb(keys,pdfs,pdfs_bkg,norm)

    print('Writing Output Catalogs')
    galCut = gal[(gal['Gal']==True)]

    print('Check size array: Pr',len(Pr),'\n')
    print('Check size array: Pc',len(Pc),'\n')
    print('Check size array: Gal',len(galCut))
    
    galCut['Pr'] = Pr
    galCut['Pz'] = Pz
    galCut['Pc'] = Pc
    galCut['Pmem'] = Pmem
    
    galCut['pdfs'] = np.vstack(pdfs).transpose()
    galCut['pdfs_bkg'] = np.vstack(pdfs_bkg).transpose()

    if method=='old':
        galCut['Pz'] = galCut['PDFz']
        galCut['Pmem'] = galCut['Pz']*galCut['Pr']

    galCut['FLAG_Z']= flagz[gal['Gal']==True]
    galCut['FLAG_C']= flagc[gal['Gal']==True]
    
    print('Writing Galaxy Output','\n')
    gidx,cidx = getIndices(galCut['CID'],cat['CID'])
    galCut['redshift'] = 0.
    galCut['redshift'][gidx] = cat['redshift'][cidx]

    Colnames=['CID', 'redshift', 'GID', 'RA', 'DEC', 'R', 'z', 'zerr', 'mag', 'magerr',
              'Pr', 'Pz', 'Pc', 'Pmem','FLAG_C','FLAG_Z']

    if simulation:
        Colnames.append('z_true')
        Colnames.append('True')

        true_members = galCut['True']
        Pmem = np.where(true_members==True,1.,Pmem)
        Pz = np.where(true_members==True,1.,Pz)       

    Pmask = (Pmem > p_low_lim)&(Pz>p_low_lim)
    galOut = galCut[Colnames][Pmask]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output','\n')
    if simulation:
        newColumns = ['R200','Ngals','Norm','Nbkg','Ngals_true','Nbkg_true']
    else:    
        newColumns = ['R200','Ngals','Norm','Nbkg']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
    good_indices, = np.where(nbkg>0)
    Ngals = computeNgals(galOut,cat['CID'][good_indices],r_aper=rmax,true_gals=False,col='Pmem')

    cat['R200'] = r200
    cat['Nbkg'] = nbkg
    cat['Ngals'][good_indices] = Ngals
    cat['Norm'][good_indices] = norm
    
    if simulation:
        Ngals_true = computeNgals(galOut,cat['CID'],true_gals=True,r_aper=rmax)
        cat['Nbkg_true'] = nbkg0
        cat['Ngals_true'] = Ngals_true
        
    if cluster_outfile is not None:
        cat.write(cluster_outfile,format='fits', overwrite=True)

    print('end!')

    return galOut, cat

if __name__ == '__main__':
    print('membAssignment.py')
    print('author: Johnny H. Esteves')
