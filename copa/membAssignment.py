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

class outPDF:
    """ it creates an hdf5 file to store the pdfs """
    def __init__(self,name_file,cluster_ids):
        import h5py
        self.f = h5py.File(name_file, "w")
        self.name_list = [str(cid) for cid in cluster_ids]

        self._generate_groups()
    
    def _generate_groups(self):
        for name in self.name_list:
            self.f.create_group(name)

    def pdf_write(self,var1,var2,label='z'):
        for i,name in enumerate(self.name_list):
            self.pdfs_create(name,var1,var2[i],label=label)

    def pdfs_write(self,var1,var2,var3,var4,label='z',label2='c'):
        for i,name in enumerate(self.name_list):
            self.pdfs_create(name,var1,var2[i],label=label)
            self.pdfs_create(name,var3,var4[i],label=label2)

    def pdfs_create(self,name,var1,var2,label='z'):
        self.f.create_dataset('/%s/pdf_%s'%(name,label), data = np.column_stack([var1,var2]), compression="gzip" )

    def var_create(self,name,var,label='col1'):
        self.f.create_dataset('/%s/%s'%(name,label), data=var)

    def var_write(self,var1,label='col1'):
        for i,name in enumerate(self.name_list):
            self.var_create(name,var1[i],label=label)

    def attrs_create(self,name,var,label='title'):
        self.f[name].attrs[label] = var

    def attrs_write(self,var1,label='title'):
        for i,name in enumerate(self.name_list):
            self.attrs_create(name,var1[i],label=label)

    def close_file(self):
        self.f.close()

def initNewColumns(data,colNames,value=-1):
    for col in colNames:
        data[col] = value*np.ones_like(data['CID'])
    return data

def computeNorm(gals,cat,r200,nbkg,keys):
    norm = []
    
    for idx,_ in enumerate(keys):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]

        print('cls_id',cls_id, 'at redshift', z_cls)
        # subGals = gals[galaxies2]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)&(gals['R']<=r2))
        # cut, = np.where(subGals['R']<=r2)
        probz = gals["PDFz"][galaxies]
        
        n_cls_field = np.nansum(probz)/(np.pi*r2**2)    
        n_gals = n_cls_field-nb

        ni = n_gals/nb
        if ni<0:
            print('norm less than zero:',ni)

        norm.append(ni)

    return np.array(norm)


def doProb(ngals,nbkg,norm,normed=False):
    
    if normed:
        ngals /= ngals.sum()
        nbkg /= nbkg.sum()

    prob = norm*ngals/(norm*ngals+nbkg)
    
    prob[np.isnan(prob)] = 0.
    
    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    
    return prob

def computeProb(keys, pdfs, pdfs_bkg, norm):
    prob = np.empty((1,4),dtype=float)

    pdfr, pdfz, pdfc = pdfs
    pdfr_bkg, pdfz_bkg, pdfc_bkg = pdfs_bkg
    
    for i,key_i in enumerate(keys):
        ni = norm[i]

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
        contaminants, = np.where((gal['CID'] == keys[i]) & (gal['R']<=r2)&(gal['True']==False)& (gal['mag'][:,2]<=magLim[i,1]) )
        
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

def getPDFz(membz,membzerr,zcls,method='pdf'):
    ''' Computes the probability of a galaxy be in the cluster
        for an interval with width n*windows. 
        Assumes that the galaxy has a gaussian redshift PDF.
    '''
    if method=='old':
        sigma = np.median(membzerr)
        
        zmin, zmax = (zcls-1.5*sigma*(1+zcls)), (zcls+1.5*sigma*(1+zcls))
        pdf = PhotozProbabilities(zmin,zmax,membz,membzerr)
        pdf = np.where(pdf<0.001,0.,pdf/np.max(pdf))
    
    elif method=='pdf2':
        pdf = gaussian(zcls,membz,membzerr)
        pdf = np.where(pdf<0.001,0.,pdf/np.max(pdf))

    elif method=='pdf':
        sigma = np.median(membzerr)

        delta_z = 0.0025
        zmin, zmax = (zcls-3*sigma), (zcls+3*sigma)
        if zmin<0: zmin=0

        z = np.arange(zmin,zmax,delta_z)
        zz, yy = np.meshgrid(z,np.array(membz))
        zz, yy2 = np.meshgrid(z,membzerr)
        
        pdfc = gaussian(zz,zcls,sigma)
        pdfz = gaussian(zz,yy,yy2)
        
        pos = pdfc*pdfz
        norm_factor = integrate.trapz(pos,x=zz)
        # inv_factor = np.where(norm_factor[:, np.newaxis]<1e-3,0.,1/norm_factor[:, np.newaxis])

        pdf = pos*norm_factor[:, np.newaxis] ## set pdf to unity
        pdf[np.isnan(pdf)] = 0.

        # pdf = integrate.trapz(pdf,x=zz)

        w = np.argmin(np.abs(z-zcls))
        pdf = pdf[:,w]

        # pdf = np.where(pdf>1,1.,pdf)

    return pdf

def computePDFz(z,zerr,cid,cat,method='pdf'):
    ncls = len(cat)
    indicies = np.empty((0),dtype=int)
    pdfz = np.empty((0),dtype=float)

    z = np.where(z<0.,0.,z)
    for i in range(ncls):
        z_cls, idx = cat['redshift'][i], cat['CID'][i]
        
        idxSubGal, = np.where(cid==idx)
        
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],z_cls,method=method)
        pdf = np.where(pdf<1e-4,0.,pdf) ## it avoids that float values gets boost by color pdfs

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

    indicies_list = []
    for i in range(keys.size):
        w, = np.where(IDs==keys[i])
        w2 = np.full(w.size,i,dtype=int)
        indicies = np.append(indicies,w)
        indicies_list.append(w)
        indicies_into_cluster = np.append(indicies_into_cluster,w2)

    return indicies,indicies_into_cluster,indicies_list

def getColorComb(pdfc_list):
    # (g-r), (g-i), (r-i), (r-z), (i-z)
    return pdfc_list[:,0]*pdfc_list[:,1]*pdfc_list[:,2]*pdfc_list[:,3]*pdfc_list[:,4]
    

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

# keys = list(chunks(gal['CID'],cidx))

## -------------------------------
## main function
def clusterCalc(gal, cat, outfile_pdfs=None, member_outfile=None, cluster_outfile=None,
                r_in=4, r_out=6, M200=1e14, p_low_lim=0.01, simulation=True, computeR200=False):
    ##############
    rmax = 1. #Mpc
    method='pdf'
    ##############
    print('estimate PDFz for each galaxy')
    pz,idxs = computePDFz(gal['z'],gal['zerr'],gal['CID'],cat,method=method)
    gal['PDFz'][idxs] = pz

    print('Computing Galaxy Density')
    ## Compute nbkg
    _, nbkg, BkgFlag = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,r_aper=1.,nslices=72)

    ## updating galaxy status
    gal['Bkg'] = BkgFlag    ## all galaxies inside the good backgrund ring's slice
    
    print('Computing R200')
    # r200 = radial.computeR200(gal, cat, nbkg2, rmax=rmax, defaultMass=M200, compute=computeR200) ## uncomment in the case to estimate a radius
    r200 = rmax*np.ones_like(cat['CID']) ## fixed aperture radii
    
    if simulation:
        nbkg0, BkgFlag0 = computeContamination(gal, cat['CID'], r200, np.array(cat['magLim']))

    ## get keys
    good_indices, = np.where(nbkg>=0.)
    
    # ngals, galFlag, keys = backSub.computeGalaxyDensity(gal, cat, rmax*np.ones_like(r200), nbkg, nslices=72)
    ngals, galFlag, keys = backSub.computeGalaxyDensity(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], nslices=72)
    gal['Gal'] = galFlag    ## all galaxies inside R200 within mag_i <mag_lim

    print('Computing PDFs \n')
    print('-> Radial Distribution')
    pdfr, pdfr_bkg = radial.computeRadialPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], keys, c=3.53, plot=False)
    # pdfr, pdfr_bkg = np.ones_like(gal['R']), np.ones_like(gal['R'])
    print('Check size array: pdfr, pdfr_bkg',len(pdfr),len(pdfr_bkg),'\n')

    print('-> Redshift Distribution')
    zvec = np.arange(0.,1.2,0.0005)            ## vec for the pdfz_cls
    pdfz, pdfz_bkg, pdfz_cls, pdfz_field = probz.computeRedshiftPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], keys, zvec=zvec, bandwidth=10, plot=False)
    print('Check size array: pdfz, pdfz_bkg',len(pdfz),len(pdfz_bkg),'\n')

    print('-> Color Distribution')
    color_vec = np.arange(-1.,4.,0.0025)        ## vec for the pdfc_cls
    pdfc, pdfc_bkg, pdfc_cls, pdfc_field = probc.computeColorPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], keys, color_vec, bandwidth=[0.01,0.01,0.01],parallel=True,plot=False)
    print('Check size array: pdfc, pdfc_bkg',len(pdfc),len(pdfc_bkg),'\n')
    
    # (g-r), (g-i), (r-i), (r-z), (i-z)
    pdfc = getColorComb(pdfc)
    pdfc_bkg = getColorComb(pdfc_bkg)

    print('\n Compute Probabilities')
    pdfs = [pdfr,pdfz,pdfc]
    pdfs_bkg = [pdfr_bkg,pdfz_bkg,pdfc_bkg]

    norm = computeNorm(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], keys)
    Pr, Pz, Pc, Pmem = computeProb(keys,pdfs,pdfs_bkg,norm)

    print('Writing Output: Catalogs')
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

    print('Writing cluster PDFs \n')
    ### creating pdfs outputs
    if outfile_pdfs is not None:
        makePDFs = outPDF(outfile_pdfs,cat['CID'][good_indices])
        makePDFs.attrs_write(norm,label='norm')
        makePDFs.attrs_write(nbkg[good_indices],label='nbkg')
        makePDFs.attrs_write(cat['redshift'][good_indices],label='zcls')

        makePDFs.pdfs_write(zvec,pdfz_cls,color_vec,pdfc_cls,label='z',label2='c')
        makePDFs.pdfs_write(zvec,pdfz_field,color_vec,pdfc_field,label='bkg_z',label2='bkg_c')

        makePDFs.close_file()

    print('Writing Galaxy Output \n')
    gidx,cidx,_ = getIndices(galCut['CID'],cat['CID'])
    galCut['redshift'] = 0.
    galCut['redshift'][gidx] = cat['redshift'][cidx]

    Colnames=['CID', 'redshift', 'GID', 'RA', 'DEC', 'R', 'z', 'zerr', 'mag', 'magerr',
              'Pr', 'Pz', 'Pc', 'Pmem','pdfs','pdfs_bkg']
    
    if simulation:
        Colnames.append('z_true')
        Colnames.append('True')

        # true_members = galCut['True']
        # Pmem = np.where(true_members==True,1.,Pmem)
        # Pz = np.where(true_members==True,1.,Pz)

    galOut = galCut[Colnames]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output \n')
    ### writing cluster catalogs
    if simulation:
        newColumns = ['R200','Ngals','Norm','Nbkg','Ngals_true','Nbkg_true']
    else:    
        newColumns = ['R200','Ngals','Norm','Nbkg']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
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

def clusterCalcTruthTable(gal, cat, outfile_pdfs=None, member_outfile=None, cluster_outfile=None,
                r_in=4, r_out=6, M200=1e14, p_low_lim=0.01, simulation=True, computeR200=False):
    ##############
    rmax = 1 #Mpc
    method='pdf'
    ##############
    print('estimate PDFz for each galaxy')
    pz,idxs = computePDFz(gal['z'],gal['zerr'],gal['CID'],cat,method=method)
    gal['PDFz'][idxs] = pz

    print('Computing R200')
    # r200 = radial.computeR200(gal, cat, nbkg2, rmax=rmax, defaultMass=M200, compute=computeR200)
    r200 = 1.*np.ones_like(cat['CID'])

    print('Computing Galaxy Density')
    ## Compute nbkg
    nbkg0, BkgFlag0 = computeContamination(gal, cat['CID'], r200, np.array(cat['magLim']))

    nbkg = np.zeros_like(r200)
    bkgFlag = (gal['R'] >= r_in) & (gal['R']<=r_out)

    ## updating galaxy status
    gal['Bkg'] = bkgFlag    ## all galaxies inside the good backgrund ring's slice

    print('Getting Truth Table')
    members = (gal['True']==True) | (gal['Bkg']==True)
    gal = gal[members]

    gidx,keys_vec,keys = getIndices(gal['CID'],cat['CID'])

    galFlag =(gal['mag'][:,2]<=cat['magLim'][keys_vec,1])#&(gal['R']<= r200[keys])
    gal['Gal'] = galFlag
    
    print('Number of galaxies:',np.count_nonzero(galFlag))

    print('Computing PDFs \n')
    print('-> Radial Distribution')
    pdfr, pdfr_bkg = radial.computeRadialPDF(gal, cat, r200, nbkg, keys, c=3.53, plot=False)
    # pdfr, pdfr_bkg = np.ones_like(gal['R']), np.ones_like(gal['R'])
    print('Check size array: pdfr, pdfr_bkg',len(pdfr),len(pdfr_bkg),'\n')

    print('-> Redshift Distribution')
    zvec = np.arange(0.,1.2,0.0005)            ## vec for the pdfz_cls
    pdfz, pdfz_bkg, pdfz_cls, pdfz_field = probz.computeRedshiftPDF(gal, cat, r200, nbkg, keys, zvec=zvec, bandwidth=10, plot=False)
    print('Check size array: pdfz, pdfz_bkg',len(pdfz),len(pdfz_bkg),'\n')

    print('-> Color Distribution')
    color_vec = np.arange(-1.,4.,0.0025)        ## vec for the pdfc_cls
    pdfc, pdfc_bkg, pdfc_cls, pdfc_field = probc.computeColorPDF(gal, cat, r200, nbkg, keys, color_vec, bandwidth=[0.01,0.01,0.01],parallel=True,plot=False)
    print('Check size array: pdfc, pdfc_bkg',len(pdfc),len(pdfc_bkg),'\n')

    # (g-r), (g-i), (r-i), (r-z), (i-z)
    pdfc = getColorComb(pdfc)
    pdfc_bkg = getColorComb(pdfc_bkg)

    print('\n Compute Probabilities')
    pdfs = [pdfr,pdfz,pdfc]
    pdfs_bkg = [pdfr_bkg,pdfz_bkg,pdfc_bkg]

    # norm = computeNorm(gal, cat, r200, nbkg)
    Pr = np.ones_like(pdfr)
    Pz = Pc = Pmem = Pr

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

    print('Writing Galaxy Output','\n')
    gidx,cidx,_ = getIndices(galCut['CID'],cat['CID'])
    galCut['redshift'] = 0.
    galCut['redshift'][gidx] = cat['redshift'][cidx]

    Colnames=['CID', 'redshift', 'GID', 'RA', 'DEC', 'R', 'z', 'zerr', 'mag', 'magerr',
              'Pr', 'Pz', 'Pc', 'Pmem','pdfs','pdfs_bkg']

    if simulation:
        Colnames.append('z_true')
        Colnames.append('True')

    galOut = galCut[Colnames]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output','\n')
    newColumns = ['R200','Ngals','Norm','Nbkg']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
    Ngals = computeNgals(galOut,cat['CID'],r_aper=rmax)
    norm =  (Ngals/(np.pi*r200**2)) / nbkg0

    cat['R200'] = r200
    cat['Nbkg'] = nbkg0
    cat['Ngals'] = Ngals
    cat['Norm'] = norm

    if cluster_outfile is not None:
        cat.write(cluster_outfile,format='fits', overwrite=True)

    print('Writing cluster PDFs \n')
    ### creating pdfs outputs
    makePDFs = outPDF(outfile_pdfs,cat['CID'])
    makePDFs.attrs_write(norm,label='norm')
    makePDFs.attrs_write(nbkg,label='nbkg')
    makePDFs.attrs_write(cat['redshift'],label='zcls')

    makePDFs.pdfs_write(zvec,pdfz_cls,color_vec,pdfc_cls,label='z',label2='c')
    makePDFs.pdfs_write(zvec,pdfz_field,color_vec,pdfc_field,label='bkg_z',label2='bkg_c')

    makePDFs.close_file()

    print('end!')
    return galOut, cat

if __name__ == '__main__':
    print('membAssignment.py')
    print('author: Johnny H. Esteves')
