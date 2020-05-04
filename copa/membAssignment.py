# !/usr/bin/env python

import numpy as np
import logging

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

import dask

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
import probMag as probm
import background as backSub

import gaussianKDE
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

    def pdfs_write(self,var_list,pdfr_list,pdfz_list,pdfc_list,pdfm_list,simulation=False):
        rvec,zvec,cvec,mvec = var_list
        for i,name in enumerate(self.name_list):
            self.pdfs_create(name,rvec,pdfr_list[0][i],label='r_cls')
            self.pdfs_create(name,rvec,pdfr_list[1][i],label='r_cls_field')
            self.pdfs_create(name,rvec,pdfr_list[2][i],label='r_field')

            self.pdfs_create(name,zvec,pdfz_list[0][i],label='z_cls')
            self.pdfs_create(name,zvec,pdfz_list[1][i],label='z_cls_field')
            self.pdfs_create(name,zvec,pdfz_list[2][i],label='z_field')

            self.pdfs_create(name,cvec,pdfc_list[0][i],label='c_cls')
            self.pdfs_create(name,cvec,pdfc_list[1][i],label='c_cls_field')
            self.pdfs_create(name,cvec,pdfc_list[2][i],label='c_field')

            self.pdfs_create(name,mvec,pdfm_list[0][i],label='m_cls')
            self.pdfs_create(name,mvec,pdfm_list[1][i],label='m_cls_field')
            self.pdfs_create(name,mvec,pdfm_list[2][i],label='m_field')

            if simulation:
                self.pdfs_create(name,rvec,pdfr_list[3][i],label='r_truth')
                self.pdfs_create(name,zvec,pdfz_list[3][i],label='z_truth')
                self.pdfs_create(name,cvec,pdfc_list[3][i],label='c_truth')
                self.pdfs_create(name,mvec,pdfm_list[3][i],label='m_truth')

    def pdfs_create(self,name,var1,var2,label='z'):
        self.f.create_dataset('%s/pdf_%s'%(name,label), data = np.column_stack([var1,var2]), compression="lzf" )

    def var_create(self,name,var,label='col1'):
        self.f.create_dataset('%s/%s'%(name,label), data=var)

    def var_write(self,var1,label='col1'):
        for i,name in enumerate(self.name_list):
            self.var_create(name,var1[i],label=label)

    def attrs_create(self,name,var,label='title'):
        self.f[name].attrs[label] = var

    def attrs_write(self,var1,label='title'):
        # self.f.create_group(label)
        for i,name in enumerate(self.name_list):
            self.attrs_create(name,var1[i],label=label)

    def close_file(self):
        # print(self.f.keys())
        self.f.close()

def initNewColumns(data,colNames,value=-1):
    for col in colNames:
        data[col] = value*np.ones_like(data['CID'])
    return data

def computeNorm(gals,cat,r200,nbkg):
    norm = []
    norm_vec = []
    for idx in range(len(cat)):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]

        print('cls_id',cls_id, 'at redshift', z_cls)
        # subGals = gals[galaxies2]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)&(gals['R']<=r2))
        probz = gals["PDFz"][galaxies]
        
        n_cls_field = np.nansum(probz)
        n_gals = n_cls_field-nb*(np.pi*r2**2)

        ni = n_gals/(nb*(np.pi*r2**2))
        if ni<0:
            print('norm less than zero:',ni)
        
        norm_vec_i = _computeNorm(gals[galaxies],r2,nb)

        norm.append(ni)
        norm_vec.append(norm_vec_i)

    return np.array(norm), np.array(norm_vec)


def doProb(ngals,nbkg,norm,normed=True):
    
    if normed:
        ngals /= ngals.sum()
        nbkg /= nbkg.sum()

    prob = norm*ngals/(norm*ngals+nbkg)
    prob[np.isnan(prob)] = 0.
    
    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    
    return prob

def _computeNorm(gal,r200,nbkg):
    rbins = np.linspace(0.,1.,4)
    pz = gal['PDFz']
    rg = gal['Rn']

    Ngals, rmed = np.histogram(rg,bins=rbins,weights=pz)
    
    area = np.pi*(rbins[1:]**2-rbins[:-1]**2)
    Nbkg = nbkg*area
    norm = (Ngals-Nbkg)/Nbkg
    
    w, = np.where(norm>0)
    if w.size>0:
        norm = np.where(norm<0.,np.mean(norm[norm>0]),norm)
    else:
        norm = -1.*np.ones_like(Nbkg)
    return norm

def computeProb(gal, keys, norm):
    gal = set_new_columns(gal,['Pmem','Pc','Pz','Pr'],val=0.)
    
    # pdfr, pdfz, pdfc, pdf = gal['pdfr'], gal['pdfz'], np.mean(gal['pdfc'],axis=1), gal['pdf']
    # pdfr_bkg, pdfz_bkg, pdfc_bkg, pdf_bkg = gal['pdfr_bkg'], gal['pdfz_bkg'], np.mean(gal['pdfc_bkg'],axis=1), gal['pdf_bkg']

    pdfr, pdfz, pdfc, pdf = gal['pdfr'], gal['pdfz'], gal['pdfc'][:,2], gal['pdf']
    pdfr_bkg, pdfz_bkg, pdfc_bkg, pdf_bkg = gal['pdfr_bkg'], gal['pdfz_bkg'], gal['pdfc_bkg'][:,2], gal['pdf_bkg']

    zcls = gal['redshift']
    pdfc = np.where(zcls<0.35, gal['pdfc'][:,1], gal['pdfc'][:,3])
    pdfc_bkg = np.where(zcls<0.35, gal['pdfc_bkg'][:,1], gal['pdfc_bkg'][:,3])
    probz = gal['PDFz']
    rbins = np.linspace(0.,1.,4)
    rmed = (rbins[1:]-rbins[:-1])/2.
    for i,idx in enumerate(keys):
        # ni = gal['norm'][idx]
        ni = norm[i]
        # radii = gal['Rn'][idx]
        # ni = interpData(rmed,ni,radii)

        radii = gal['R'][idx]
        radi2 = 0.15*(np.trunc(radii/0.15)+1) ## bins with 0.125 x R200 width
        areag2 = 1#np.pi*radi2**2#((radi2+0.25)**2-radi2**2)

        pz = probz[idx]
        ng = np.sum(pz)

        ## scaling for pz [0,1]
        pzmax = np.percentile(pz,99)
        pz = pz/pzmax
        pzb= 1#np.where(pz>pzmax,0.,(1-pz))
        
        pdfri, pdfzi, pdfci = pdfr[idx], pdfz[idx], pdfc[idx]
        pdfr_bkg_i, pdfz_bkg_i, pdfc_bkg_i = pdfr_bkg[idx], pdfz_bkg[idx], pdfc_bkg[idx]
            
        # Ngals =  pdf[idx] 
        # # Ngals_bkg = pdf_bkg[idx] 
        # pdfsi = np.array([pdfri, pdfzi, pdfci])
        # pdfs_bkg_i = np.array([pdfr_bkg_i, pdfz_bkg_i, pdfc_bkg_i])

        # normi = np.sum(pdfsi,axis=0)/np.product(pdfsi,axis=1)
        # norm_bkgi = np.sum(pdfs_bkg_i,axis=0)/np.product(pdfs_bkg_i,axis=1)

        pn = np.sum(pdfri),np.sum(pdfzi),np.sum(pdfci)
        pn_bkg = np.sum(pdfr_bkg_i),np.sum(pdfz_bkg_i),np.sum(pdfc_bkg_i)

        Ngals =  pdfri*pdfzi*pdfci*(np.sum(pn))/np.sum(pdfri*pdfzi*pdfci)
        Ngals_bkg = pdfr_bkg_i*pdfz_bkg_i*pdfc_bkg_i*(np.sum(pn_bkg))/np.sum(pdfr_bkg_i*pdfz_bkg_i*pdfc_bkg_i)#(np.product(pn_bkg))

        pr = doProb(pz*pdfri,pdfr_bkg_i,ni/areag2,normed=True)
        pz = doProb(pdfzi,pdfz_bkg_i,ni/areag2,normed=True)
        pc = doProb(pz*pdfci,pdfc_bkg_i,ni/areag2,normed=True)

        # pn = np.sum(pdfri),np.sum(pdfci)
        # pn_bkg = np.sum(pdfr_bkg_i),np.sum(pdfc_bkg_i)

        # Ngals =  pz*pdfri*pdfci*(np.sum(pn))/np.sum(pdfri*pdfci)
        # Ngals_bkg = (1-pz)*pdfr_bkg_i*pdfc_bkg_i*(np.sum(pn_bkg))/np.sum(pdfr_bkg_i*pdfc_bkg_i)#(np.product(pn_bkg))

        # pz = doProb(pdfz[idx],pdfz_bkg[idx],ni)
        # pc = doProb(pdfc[idx],pdfc_bkg[idx],ni)

        pmem = doProb(Ngals,Ngals_bkg,ni,normed=False)
  
        gal['Pmem'][idx] = pmem
        gal['Pc'][idx] = pc
        gal['Pz'][idx] = pz
        gal['Pr'][idx] = pr

    return gal

def computeContamination(gal,keys,r200,magLim):
    bkgFlag = np.full(len(gal['Bkg']), False, dtype=bool)
    ratio = []
    
    ncls = len(keys)
    for i in range(ncls):
        r2 = r200[i]
        # contaminants = (gi['R']>=r2)&(gi['True']==False)&(gi['R']<=4*r2)
        # contaminants, = np.where((gal['CID'] == keys[i]) & (gal['R']<= r2)&(gal['True']==False)& (gal['mag'][:,2]<=magLim[i,1]) )
        contaminants, = np.where((gal['CID'] == keys[i]) & (gal['R']<= r2)&(gal['True']==False) & (gal['dmag']<=0.) )
        
        pz_cont = gal['PDFz'][contaminants]
        
        area = np.pi*(r2)**2
        ncon = np.sum(pz_cont)
        
        ratio_i = ncon/area
        ratio.append(ratio_i)

        bkgFlag[contaminants] = True

    return np.array(ratio), bkgFlag

def computeNgals(g,keys,r_aper,true_gals=False,col='Pmem'):
    ngals = []
    
    for idx,r2 in zip(keys,r_aper):
        if true_gals:
            w, = np.where((g['CID']==idx)&(g['True']==True)&(g['R']<=r2))
            ni = len(w)
            
        else:
            w, = np.where((g['CID']==idx)&(g['R']<=r2))
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

def truncatedGaussian(z,zcls,zmin,zmax,sigma,vec=False):
    if vec:
        s_shape = sigma.shape
        sigma = sigma.ravel()
        z = z.ravel()
        zcls = zcls.ravel()

    # user input
    myclip_a = zmin
    myclip_b = zmax
    my_mean = zcls
    my_std = sigma

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    pdf = truncnorm.pdf(z, a, b, loc = my_mean, scale = my_std)

    if vec: pdf.shape = s_shape
    return pdf

def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def getPDFz(membz,membzerr,zcls,sigma,method='pdf'):
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
        # sigma = np.median(membzerr)

        # delta_z = 0.025
        zmin, zmax = (zcls-5*sigma), (zcls+5*sigma)
        if zmin<0: zmin=0.

        z = np.linspace(zmin,zmax,200)
        zz, yy = np.meshgrid(z,np.array(membz))
        zz, yy2 = np.meshgrid(z,np.array(membzerr))
        
        if (zmin>0.001)&(zmax<=1.2):
            pdfc = gaussian(zz,zcls,sigma)
            pdfz = gaussian(zz,yy,yy2)
        else:
            pdfc = truncatedGaussian(zz,zcls,zmin,zmax,sigma)
            pdfz = truncatedGaussian(zz,yy,zmin,zmax,yy2,vec=True)

        pos = pdfc*pdfz
        norm_factor = integrate.trapz(pos,x=zz)
        # inv_factor = np.where(norm_factor[:, np.newaxis]<1e-3,0.,1/norm_factor[:, np.newaxis])

        pdf = pos/norm_factor[:, np.newaxis] ## set pdf to unity
        pdf[np.isnan(pdf)] = 0.
        
        w, = np.where( np.abs(z-zcls)<= 1.5*sigma) ## integrate in 1.5*sigma
        pdf = integrate.trapz(pdf[:,w],x=zz[:,w])
        pdf = np.where(pdf>1., 1., pdf)

        ## get out with galaxies outside 3 sigma
        zmin, zmax = (zcls-5*sigma), (zcls+5*sigma)
        if zmin<0: zmin=0.
        pdf = np.where((np.array(membz) < zmin )&(np.array(membz) > zmax), 0., pdf)

        # w = np.argmin(np.abs(z-zcls))
        # pdf = pdf[:,w]#*0.01

        # pdf /= np.max(pdf)
        # pdf = np.where(pdf>1,1.,pdf)

    return pdf

def computePDFz(z,zerr,cid,cat,sigma,method='pdf'):
    ncls = len(cat)
    indicies = np.empty((0),dtype=int)
    pdfz = np.empty((0),dtype=float)

    results = []
    z = np.where(z<0.,0.,z)
    for i in range(ncls):
        z_cls, idx = cat['redshift'][i], cat['CID'][i]
        
        idxSubGal, = np.where(cid==idx)
        
        # r1 = dask.delayed(getPDFz)(z[idxSubGal],zerr[idxSubGal],z_cls,sigma*(1+z_cls),method=method)
        pdf = getPDFz(z[idxSubGal],zerr[idxSubGal],z_cls,sigma*(1+z_cls),method=method)
        # pdf = np.where(pdf<1e-4,0.,pdf) ## it avoids that float values gets boost by color pdfs

        # results.append(r1)
        indicies = np.append(indicies,idxSubGal)
        pdfz  = np.append(pdfz,pdf)

    # results = dask.compute(*results, scheduler='processes', num_workers=2)
    # for i in range(ncls):
    #     pdf = results[i]
    #     pdf = np.where(pdf<1e-4,0.,pdf) ## it avoids that float values gets boost by color pdfs
    #     pdfz  = np.append(pdfz,pdf)

    return pdfz, indicies

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

def getColorComb(color_vec,pdfc):    
    # (g-r), (g-i), (r-i), (r-z), (i-z)
    return pdfc[:,0]*pdfc[:,1]*pdfc[:,2]*pdfc[:,3]*pdfc[:,4]#/normalization ## pdf set to unity
    

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def set_new_columns(table,columns,val=0.):
    for col in columns:
        table[col] = val
    return table

def set_color_columns(table):
    #################  g-r , g-i , r-i , r-z , i-z
    colors_indices = [[0,1],[0,2],[1,2],[1,3],[2,3]]
    mag = table['mag']

    color = np.empty_like(table['z'])
    for choose_color in colors_indices:
        i,i2 = choose_color
        colori = (mag[:,i]-mag[:,i2])
        color = np.c_[color,colori]

    table['color'] = color[:,1:]
    return table

def interpData(x,y,x_new):
    out = np.empty(x_new.shape, dtype=y.dtype)
    out = interp1d(x, y, kind='linear', fill_value='extrapolate', copy=False)(x_new)
    # yint = interp1d(x,y,kind='linear',fill_value='extrapolate')
    return out

def getPDFs(gal,galIndices,vec_list,pdf_list,nbkg,mag_pdf=False):
    rvec, zvec, cvec, mvec = vec_list
    pdfr, pdfz, pdfc, pdfm = pdf_list

    gal = set_new_columns(gal,['pdf','pdfr','pdfz','pdfm','norm'],val=0.)
    gal = set_new_columns(gal,['pdf_bkg','pdfr_bkg','pdfz_bkg','pdfm_bkg'],val=0.)

    gal['pdfc'] = np.zeros_like(gal['color'])
    gal['pdfc_bkg'] = gal['pdfc']

    for i, idx in enumerate(galIndices):
        nb = nbkg[i]
        ggal = gal[idx] ## group gal

        ## getting pdfs for a given cluster i
        pdfri, pdfr_cfi  = pdfr[0][i], pdfr[1][i]
        pdfzi, pdfzi_bkg = pdfz[0][i], pdfz[2][i]
        pdfci, pdfci_bkg = pdfc[0][i], pdfc[2][i]
        pdfmi, pdfmi_bkg = pdfm[0][i], pdfm[2][i]

        ## setting galaxies variable columns
        r2 = ggal['R200'] 
        radii = ggal['R']
        zgal  = ggal['z']
        color5 = ggal['color']
        mag = ggal['dmag']
        areag = np.pi*r2**2

        radi2 = 0.15*(np.trunc(radii/0.15)+1) ## bins with 0.125 x R200 width
        # areag = np.pi*radi2**2#((radi2+0.25)**2-radi2**2)

        gal['pdfr'][idx] = interpData(rvec,pdfri,radii)
        gal['pdfz'][idx] = interpData(zvec,pdfzi,zgal)
        gal['pdfm'][idx] = interpData(mvec,pdfmi,mag)

        gal['pdfr_bkg'][idx] = np.ones_like(radi2)/areag #interpData(rvec,np.ones_like(gal['pdfr'][idx]),radii)
        gal['pdfz_bkg'][idx] = interpData(zvec,pdfzi_bkg,zgal)
        gal['pdfm_bkg'][idx] = interpData(mvec,pdfmi_bkg,mag)

        for j in range(5):
            gal['pdfc'][idx,j] = interpData(cvec,pdfci[:,j],color5[:,j])
            gal['pdfc_bkg'][idx,j] = interpData(cvec,pdfci_bkg[:,j],color5[:,j])
        
        gal['pdf'][idx] = gal['pdfr'][idx]*gal['pdfz'][idx]
        gal['pdf_bkg'][idx] = gal['pdfr_bkg'][idx]*gal['pdfz_bkg'][idx]
        
        if mag_pdf:
            gal['pdf'][idx] *= gal['pdfm'][idx]
            gal['pdf_bkg'][idx] *= gal['pdfm_bkg'][idx]

        gal['pdfc'][idx,:] = np.where(gal['pdfc'][idx,:]<0.,0.,gal['pdfc'][idx,:])
        gal['pdfc_bkg'][idx,:] = np.where(gal['pdfc_bkg'][idx,:]<0.,0.,gal['pdfc_bkg'][idx,:])

        # colors: (g-r),(g-i),(r-i),(r-z),(i-z)
        # pick_colors = [0,1,2,3,4]
        # for j in pick_colors:
        #     gal['pdf'][idx] *= gal['pdfc'][idx,j]
        #     gal['pdf_bkg'][idx] *= gal['pdfc_bkg'][idx,j]

        gal['pdf'][idx] *= np.mean(gal['pdfc'][idx],axis=1)
        gal['pdf_bkg'][idx] *= np.mean(gal['pdfc_bkg'][idx],axis=1)
        
        ng_profile = interpData(rvec,pdfr_cfi,radi2)
        gal['norm'][idx] = (ng_profile - nb)/nb

    gal['pdf'] = np.where(gal['pdf']<0.,0.,gal['pdf'])
    gal['pdfr'] = np.where(gal['pdfr']<0.,0.,gal['pdfr'])
    gal['pdfz'] = np.where(gal['pdfz']<0.,0.,gal['pdfz'])
    gal['pdfc'] = np.where(gal['pdfc']<0.,0.,gal['pdfc'])
    gal['pdfm'] = np.where(gal['pdfm']<0.,0.,gal['pdfm'])

    # gal['pdfc'] = np.where(gal['pdfc']<1e-4,0.,gal['pdfc'])
    # gal['pdfc_bkg'] = np.where(gal['pdfc_bkg']<1e-4,0.,gal['pdfc_bkg'])

    return gal

def getTruthPDFs(gal,cat,r200,indices,rvec,zvec,color_vec,mag_vec,c=3.53,bwz='silverman',bwc=[0.01,0.01,0.1]):
    cids = cat['CID'][:]

    pdfr, pdfz, pdfc, pdfm = [], [], [], []
    results1, results2, results3,results4 = [],[],[],[]
    for i,_ in enumerate(indices):
        r2 = (r200[i]) #radii = ggal['R']
        idx, = np.where((gal['CID']==cids[i])&(gal['True']==True)&(gal['Gal']==True)&(gal['R']<=r2))
        
        ggal = gal[idx]
        z = ggal['z']
        color = ggal['color'][:]
        mag = ggal['dmag'][:]

        r1 = dask.delayed(get_pdf_radial)((rvec[1:]+rvec[:-1])/2,r2,c=c)
        r2 = dask.delayed(get_pdf_redshift)(z,zvec,bw=bwz)
        r3 = dask.delayed(get_pdf_color)(color,color_vec)
        r4 = dask.delayed(computeKDE)(mag,mag_vec,silvermanFraction=2)

        results1.append(r1)
        results2.append(r2)
        results3.append(r3)
        results4.append(r4)

    ## do the loop in parallel with 2 cores
    results1 = dask.compute(*results1, scheduler='processes', num_workers=2)
    results2 = dask.compute(*results2, scheduler='processes', num_workers=2)
    results3 = dask.compute(*results3, scheduler='processes', num_workers=2)
    results4 = dask.compute(*results4, scheduler='processes', num_workers=2)

    ## append the result
    for i,idx in enumerate(indices):
        pdfr.append(results1[i])
        pdfz.append(results2[i])
        pdfc.append(results3[i])
        pdfm.append(results4[i])

    return pdfr, pdfz, pdfc, pdfm
    
def get_pdf_radial(rmed,r200,c=3.53):
    norm = radial.norm_constant(r200,c=c)
    pdfRadial = norm*radial.doPDF(rmed,r200,c=c)
    return pdfRadial

def computeKDE(x,xvec,bw=0.1,silvermanFraction=None):
    if len(x)>1:
        if silvermanFraction is None:
            kernel = gaussianKDE.gaussian_kde(x,bw_method=bw)
        else:
            kernel = gaussianKDE.gaussian_kde(x,silvermanFraction=silvermanFraction)        
    
        kde = kernel(xvec)
    else:
        kde = np.zeros_like(xvec)
    return kde

def get_pdf_redshift(z,zvec,bw):
    pdfRedshift = computeKDE(z,zvec,bw=bw)
    return pdfRedshift

def get_pdf_color(colors,color_vec):
    ## 5 Color distributions
    y0 = computeKDE(colors[:,0],color_vec,silvermanFraction=10)##(g-r)
    y1 = computeKDE(colors[:,1],color_vec,silvermanFraction=10)##(g-i)
    y2 = computeKDE(colors[:,2],color_vec,silvermanFraction=10)##(r-i)
    y3 = computeKDE(colors[:,3],color_vec,silvermanFraction=10)##(r-z)
    y4 = computeKDE(colors[:,4],color_vec,silvermanFraction=10)##(i-z)
    
    pdfColor = np.c_[y0,y1,y2,y3,y4]
    pdfColor = np.where(pdfColor<0.,0.,pdfColor)
    return pdfColor

def define_new_columns(gal,cat):
    gal = set_color_columns(gal) ## set columns for color

    gal = gal.group_by('CID')
    gidx,cidx,_ = getIndices(gal['CID'],cat['CID'])
    gal = set_new_columns(gal,['redshift','R200','dmag','zoffset'],val=0.)
    
    gal['redshift'][gidx] = cat['redshift'][cidx]
    gal['dmag'][gidx] = gal['mag'][gidx,2]-cat['magLim'][cidx,1]

    gal['zoffset'][gidx] = (gal['z']-gal['redshift'])/(1+gal['redshift'])

    gal = gal[gidx]
    return gal,cat,gidx,cidx

## -------------------------------
## main function
def clusterCalc(gal, cat, outfile_pdfs=None, member_outfile=None, cluster_outfile=None,
                r_in=4, r_out=6, sigma_z=0.05, M200=1e14, p_low_lim=0.01, simulation=True, computeR200=False):
    ##############
    rmax = 1. #Mpc
    method='pdf'
    colorBW = [0.05,0.025,0.02]
    zBW = 'silverman'
    ##############
    ids, indices = np.unique(gal['GID','CID'], return_index=True)
    gal = gal[indices] ## getting away with repeated objects
    gal,cat,gidx,cidx = define_new_columns(gal,cat)

    # print('estimate PDFz for each galaxy')
    # pz,idxs = computePDFz(gal['z'],gal['zerr'],gal['CID'],cat,sigma_z,method=method)
    # gal['PDFz'][idxs] = pz

    print('Computing Galaxy Density')
    ## Compute nbkg
    _, nbkg, BkgFlag = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,r_aper=1.,nslices=72)

    ## updating galaxy status
    gal['Bkg'] = BkgFlag    ## all galaxies inside the good backgrund ring's slice
    
    print('Computing R200')
    r200, raper = radial.computeR200(gal, cat, nbkg, rmax=3, defaultMass=M200, compute=True) ## uncomment in the case to estimate a radius
    # r200 = cat['R200_true']
    raper= 1.*r200
    # r200 = rmax*np.ones_like(cat['CID']) ## fixed aperture radii
    gal['r_aper'] = raper[cidx]
    gal['R200'] = r200[cidx]
    gal['Rn'] = gal['R']/gal['R200']

    if simulation:
        gidx,keys_vec,keys = getIndices(gal['CID'],cat['CID'])
        # galFlag = (gal['True']==True)&(gal['mag'][:,2]<=cat['magLim'][keys_vec,1])
        galFlag = (gal['True']==True)&(gal['dmag']<=0.)&(gal['R']<=gal['r_aper']) ## r-band cut
        nbkg0, BkgFlag0 = computeContamination(gal, cat['CID'], raper, np.array(cat['magLim']))
        Ngals_true = computeNgals(gal[galFlag],cat['CID'], raper,true_gals=True)

    ## get keys
    good_indices, = np.where(nbkg>=0.)
    
    galFlag = (gal['dmag']<=0.)&(gal['R']<=gal['R200'])
    # galFlag = (gal['mag'][:,1]<=cat['magLim'][keys_vec,0])&(gal['R']<=gal['R200'])
    # galFlag = (gal['amag'][:,1]<=-20.5)&(gal['R']<=gal['R200']) ## Mr<=-19.5

    ngals, _, keys, galIndices = backSub.computeGalaxyDensity(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], nslices=72)
    gal['Gal'] = galFlag    ## all galaxies inside R200 within mag_i <mag_lim

    print('Computing PDFs \n')
    print('-> Radial Distribution')
    rvec = np.linspace(0.,4.,40)
    pdfr_list = radial.computeRadialPDF(gal, cat[good_indices], r200[good_indices], raper[good_indices], nbkg[good_indices], galIndices, rvec, c=3.53)

    print('-> Redshift Distribution')
    zvec = np.arange(0.,1.2,0.005)        ## vec for the pdfz_cls
    pdfz_list = probz.computeRedshiftPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], galIndices, sigma_z, zvec=zvec, bandwidth=zBW)

    print('-> Color Distribution')
    color_vec = np.arange(-1.,4.5,0.0025) ## vec for the pdfc_cls
    pdfc_list = probc.computeColorPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], galIndices, color_vec, bandwidth=colorBW,parallel=True)

    print('-> Mag Distribution')
    mag_vec = np.arange(-10.,0.,0.01) ## vec for the pdfc_cls
    pdfm_list = probm.computeMagPDF(gal, cat[good_indices], r200[good_indices], nbkg[good_indices], galIndices, mag_vec)

    if simulation:
        print('--> Getting truth distributions')
        pdfr_true, pdfz_true, pdfc_true, pdfm_true = getTruthPDFs(gal,cat[good_indices],r200[good_indices],galIndices,rvec,zvec,color_vec,mag_vec,bwz=zBW,bwc=colorBW)
        pdfr_list.append(pdfr_true); pdfz_list.append(pdfz_true); pdfc_list.append(pdfc_true); pdfm_list.append(pdfm_true)

    rmed = (rvec[1:]+rvec[:-1])/2
    var_list = [rmed,zvec,color_vec,mag_vec]
    pdf_list = [pdfr_list,pdfz_list,pdfc_list,pdfm_list]

    norm, norm_vec = computeNorm(gal, cat[good_indices], raper[good_indices], nbkg[good_indices])

    print('\n Compute Probabilities')
    gidx,cidx,_ = getIndices(gal['CID'],cat['CID'])
    gal = set_new_columns(gal,['redshift','R200'],val=0.)
    gal['redshift'][gidx] = cat['redshift'][cidx]
    gal['R200'][gidx] = r200[cidx]

    galCut = gal[(gal['Gal']==True)&(gal['R']<=gal['r_aper'])]

    # galIndices = list(chunks(galCut['CID'],cat['CID'][good_indices]))
    gidx,cidx,galIndices = getIndices(galCut['CID'],cat['CID'][good_indices])
    galCut = getPDFs(galCut,galIndices,var_list,pdf_list,nbkg[good_indices],mag_pdf=False)
    galCut = computeProb(galCut, galIndices, norm)

    print('Writing Output: Catalogs')
    if method=='old':
        galCut['Pz'] = galCut['PDFz']
        galCut['Pmem'] = galCut['Pz']*galCut['Pr']

    print('Writing Galaxy Output \n')
    Colnames=['CID', 'redshift', 'R200', 'norm', 'GID', 'RA', 'DEC', 'R', 'Rn', 'z', 'zerr', 'zoffset', 'mag', 'magerr','color','dmag',
              'PDFz','Pr', 'Pz', 'Pc', 'Pmem', 'pdfr', 'pdfz', 'pdfc', 'pdfm', 'pdf', 'pdfr_bkg', 'pdfz_bkg', 'pdfc_bkg', 'pdfm_bkg', 'pdf_bkg']
    
    if simulation:
        Colnames.append('Mr')
        Colnames.append('amag')
        Colnames.append('z_true')
        Colnames.append('True')

    galOut = galCut[Colnames]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output \n')
    ### writing cluster catalogs
    if simulation:
        newColumns = ['R200','RAPER','Ngals','Norm','Nbkg','Ngals_true','Nbkg_true']
    else:    
        newColumns = ['R200','RAPER','Ngals','Norm','Nbkg']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
    Ngals = computeNgals(galOut,cat['CID'][good_indices],r200[good_indices],true_gals=False,col='Pmem')

    cat['R200'] = r200
    cat['RAPER'] = raper
    cat['Nbkg'] = nbkg
    cat['Ngals'][good_indices] = Ngals
    cat['Norm'][good_indices] = norm

    cat['NormVec'] = np.zeros((len(cat),norm_vec.shape[1]))
    cat['NormVec'][good_indices,:] = norm_vec
    
    if simulation:
        #Ngals_true = computeNgals(galOut,cat['CID'],r200[good_indices],true_gals=True)
        cat['Nbkg_true'] = nbkg0
        cat['Ngals_true'] = Ngals_true
        
    if cluster_outfile is not None:
        cat.write(cluster_outfile,format='fits', overwrite=True)

    print('Writing cluster PDFs \n')
    ### creating pdfs outputs
    if outfile_pdfs is not None:
        makePDFs = outPDF(outfile_pdfs,cat['CID'][good_indices])
        makePDFs.attrs_write(norm,label='norm')
        makePDFs.attrs_write(nbkg[good_indices],label='nbkg')
        makePDFs.attrs_write(cat['redshift'][good_indices],label='zcls')
        makePDFs.attrs_write(r200,label='R200')
        if simulation: makePDFs.attrs_write(cat['R200_true'][good_indices],label='R200_true')
        if simulation: makePDFs.attrs_write(cat['M200_true'][good_indices],label='M200_true')

        makePDFs.pdfs_write(var_list,pdfr_list,pdfz_list,pdfc_list,pdfm_list,simulation=simulation)
        makePDFs.close_file()

    print('end!')

    return galOut, cat

def clusterCalcTruthTable(gal, cat, outfile_pdfs=None, member_outfile=None, cluster_outfile=None,
                r_in=4, r_out=6, sigma_z=0.05, M200=1e14, p_low_lim=0.01, simulation=True, computeR200=False):
    ##############
    rmax = 1. #Mpc
    method='pdf'
    colorBW = [0.05,0.025,0.02]
    zBW = 'silverman'
    ##############
    rvec = np.linspace(0.,4.,400)
    zvec = np.arange(0.,1.2,0.005)        ## vec for the pdfz_cls
    color_vec = np.arange(-1.,4.5,0.0025) ## vec for the pdfc_cls
    mag_vec = np.arange(-4.,3.,0.01) ## vec for the pdfc_cls
    #############

    ids, indices = np.unique(gal['GID','CID'], return_index=True)
    gal = gal[indices] ## getting away with repeated objects
    gal,cat,gidx,cidx = define_new_columns(gal,cat)

    print('estimate PDFz for each galaxy')
    pz,idxs = computePDFz(gal['z'],gal['zerr'],gal['CID'],cat,sigma_z,method=method)
    gal['PDFz'][idxs] = pz

    print('Computing R200')
    # r200 = radial.computeR200(gal, cat, nbkg2, rmax=rmax, defaultMass=M200, compute=computeR200)
    r200 = cat['R200_true']

    print('Computing Galaxy Density')
    ## Compute nbkg
    nbkg0, bkgFlag = computeContamination(gal, cat['CID'], r200, np.array(cat['magLim']))

    ## updating galaxy status
    gidx,keys_vec,keys = getIndices(gal['CID'],cat['CID'])
    galFlag = (gal['True']==True)#&(gal['mag'][:,2]<=cat['magLim'][keys_vec,1]#&(gal['R']<= r200[keys])
    gal['Gal'] = galFlag

    print('Getting Truth Table')
    members = (gal['Gal'] == True)
    galCut = gal[members]

    print('Computing PDFs \n')
    galIndices = list(chunks(galCut['CID'],cat['CID']))
    pdfr_true, pdfz_true, pdfc_true, pdfm_true = getTruthPDFs(galCut,cat,r200,galIndices,rvec,zvec,color_vec,mag_vec,bwz=zBW,bwc=colorBW)

    print('\n Compute Probabilities')
    rmed = (rvec[1:]+rvec[:-1])/2
    var_list = [rmed,zvec,color_vec,mag_vec]
    pdf_list = [[pdfr_true,pdfr_true,pdfr_true],[pdfz_true,pdfz_true,pdfz_true],[pdfc_true,pdfc_true,pdfc_true],[pdfm_true,pdfm_true,pdfm_true]]
    galCut = getPDFs(galCut,galIndices,var_list,pdf_list,mag_pdf=False)

    Pr = np.ones_like(galCut['z'])
    Pz = Pc = Pmem = Pr

    print('Writing Output Catalogs')
    galCut['Pr'] = Pr
    galCut['Pz'] = Pz
    galCut['Pc'] = Pc
    galCut['Pmem'] = Pmem

    if method=='old':
        galCut['Pz'] = galCut['PDFz']
        galCut['Pmem'] = galCut['Pz']*galCut['Pr']

    print('Writing Galaxy Output','\n')
    gidx,cidx,_ = getIndices(galCut['CID'],cat['CID'])
    galCut['redshift'] = 0.
    galCut['redshift'][gidx] = cat['redshift'][cidx]

    Colnames=['CID', 'redshift', 'R200', 'GID', 'RA', 'DEC', 'R', 'z', 'zerr', 'mag', 'magerr','color','dmag',
              'PDFz','Pr', 'Pz', 'Pc', 'Pmem', 'pdfr', 'pdfz', 'pdfc', 'pdfm', 'pdf']

    Colnames.append('amag')
    Colnames.append('z_true')
    Colnames.append('True')

    galOut = galCut[Colnames]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output','\n')
    newColumns = ['R200','Ngals','Norm','Nbkg']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
    Ngals = computeNgals(galOut,cat['CID'],r200)
    norm =  (Ngals/(np.pi*r200**2)) / nbkg0

    cat['R200'] = r200
    cat['Nbkg'] = nbkg0
    cat['Ngals'] = Ngals
    cat['Norm'] = norm

    if cluster_outfile is not None:
        cat.write(cluster_outfile,format='fits', overwrite=True)

    print('end!')
    return galOut, cat

if __name__ == '__main__':
    print('membAssignment.py')
    print('author: Johnny H. Esteves')
