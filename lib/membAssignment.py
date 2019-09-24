# !/usr/bin/env python


import numpy as np
import logging

from time import time
from astropy.table import Table, vstack

## -------------------------------
## local libraries
import helper as helper
import probRadial as radial
import probRedshift as probz
import probColor as probc
import background as backSub


## -------------------------------
## auxiliary functions

def computeNorm(keys,pdfs,pdfs_bkg):
    ncls = len(keys)
    norm = []
    
    pdfr, pdfz, pdfc = pdfs
    pdfr_bkg, pdfz_bkg, pdfc_bkg = pdfs_bkg

    for idx in range(ncls):
        key_i = keys[idx]

        Ngals = np.sum( pdfr[key_i]*pdfz[key_i]*pdfc[key_i] )
        Ngals_bkg = np.sum( pdfr_bkg[key_i]*pdfz_bkg[key_i]*pdfc_bkg[key_i] )

        if Ngals_bkg==0:
            print('Error: Ngals bkg')
            Ngals_bkg = -1

        # Ngals2 = np.sum( pdfr[key_i]*pdfz[key_i])
        # Ngals_bkg2 = np.sum( pdfr_bkg[key_i]*pdfz_bkg[key_i])

        ratio = (Ngals)/(Ngals_bkg)
        
        # ratio2 = (Ngals2/area)/(Ngals_bkg2/area_bkg)
        print(ratio)
        if ratio<1:
            ratio=1
        norm.append(ratio)

    return np.array(norm)

def doProb(ngals,nbkg,norm):
    prob = norm*ngals/(norm*ngals+nbkg)
    prob = np.where(prob>1,1.,prob)
    return prob

def computeProb(keys, pdfs, pdfs_bkg, norm):
    ncls = len(keys)
    prob = np.empty((1,4),dtype=float)

    pdfr, pdfz, pdfc = pdfs
    pdfr_bkg, pdfz_bkg, pdfc_bkg = pdfs_bkg
    
    for idx in range(ncls):
        key_i, ni = keys[idx], norm[idx]

        Ngals = ( pdfr[key_i]*pdfz[key_i]*pdfc[key_i] )
        Ngals_bkg = ( pdfr_bkg[key_i]*pdfz_bkg[key_i]*pdfc_bkg[key_i] )

        pr = doProb(pdfr[key_i],pdfr_bkg[key_i],ni)
        pz = doProb(pdfz[key_i],pdfz_bkg[key_i],ni)
        pc = doProb(pdfc[key_i],pdfc_bkg[key_i],1)

        pmem = doProb(Ngals,Ngals_bkg,ni)
        print('Check size array: pr, pz, pc, pmem:')
        print(len(pr),len(pz),len(pc),len(pmem),'\n')
  
        pi = np.array([pr,pz,pc,pmem]).transpose()
        
        prob = np.vstack([prob,pi])
    print(len(prob))
    return prob[1:,0], prob[1:,1], prob[1:,2], prob[1   :,3]

## -------------------------------
## main function

def clusterCalc(gal,cat,galaxyOutFile, clusterOutFile,
         r_in=8,r_out=10,M200=None,bandwidth=0.01,p_low_lim=0.01):
         
    rmax = 3 #Mpc
    
    print('Computing Background Galaxies')
    ## Compute nbkg
    nbkg, nbkgMagLimited, BkgFlag = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,nfatias=64,plot=False)
    
    print('Computing R200')
    ## if you don't provide M200, estimate R200
    r200 = radial.computeR200(gal, cat, nbkg, rmax=rmax, M200=M200)
    N200, galFlag = radial.computeN200(gal, cat, r200, nbkgMagLimited)

    ## updating galaxy status
    gal['Bkg'] = BkgFlag    ## all galaxies inside the good backgrund ring's slice
    gal['Gal'] = galFlag    ## all galaxies inside R200

    print('Computing PDFs \n')

    print('-> Radial Distribution')
    pdfr, pdfr_bkg, keys = radial.computeRadialPDF(gal, cat, r200, N200, nbkg, c=3.53, plot=True) ## for all galaxies
    print('Check size array: pdfr, pdfr_bkg',len(pdfr),len(pdfr_bkg),'\n')

    print('-> Redshift Distribution')
    pdfz, pdfz_bkg, flagz = probz.computeRedshiftPDF(gal,cat,bandwidth=0.01 ,plot=True)
    print('Check size array: pdfz, pdfz_bkg',len(pdfz),len(pdfz_bkg),'\n')

    print('-> Color Distribution')
    pdfc, pdfc_bkg, flagc = probc.computeColorPDF(gal,cat,magLim=None,plot=False)
    print('Check size array: pdfc, pdfc_bkg',len(pdfc),len(pdfc_bkg),'\n')
    
    print('Compute Normalization Factor')
    
    pdfs = [pdfr,pdfz,pdfc[:,1]]    ## pdf_color: (r-i)
    pdfs_bkg = [pdfr_bkg,pdfz_bkg,pdfc_bkg[:,1]]

    norm = computeNorm(keys,pdfs,pdfs_bkg)
    
    print('\n Compute Probabilities')
    Pr, Pz, Pc, Pmem = computeProb(keys,pdfs,pdfs_bkg,norm)

    print('Writing Output Catalogs')
    galCut = gal[gal['Gal']==True]

    galCut['Pr'] = Pr
    galCut['Pz'] = Pz
    galCut['Pc'] = Pc
    galCut['Pmem'] = Pmem
    
    galCut['FLAG_Z']= flagz[gal['Gal']==True]
    galCut['FLAG_C']= flagc[gal['Gal']==True]

    Colnames=['CID', 'GID', 'RA', 'DEC', 'R', 'z', 'zerr', 'mag', 'magerr', 'Pr', 'Pz', 'Pc', 'Pmem','FLAG_C','FLAG_Z']
    
    Pmask = Pmem > p_low_lim
    # galOut = gal[Colnames][galFlag&Pmask]
    galOut = galCut[Colnames][Pmask]

    print(np.unique(galOut['CID']))
    galOut.write(galaxyOutFile,format='fits', overwrite=True)

    cat['R200'] = r200
    cat['N200'] = N200
    cat['Norm'] = norm
    cat.write(clusterOutFile,format='fits', overwrite=True)

if __name__ == '__main__':
    print('membAssignment.py')
    print('author: Johnny H. Esteves')
