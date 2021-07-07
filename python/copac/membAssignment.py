# !/usr/bin/env python
from __future__ import print_function
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

h=0.7
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

def computeNorm(gals,cat,r200,nbkg,maskfrac):
    norm = []
    norm_vec = []
    area_vec = []
    for idx in range(len(cat)):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2, nb = r200[idx], nbkg[idx]
        fm = maskfrac[idx]

        # print('cls_id',cls_id, 'at redshift', z_cls)
        # subGals = gals[galaxies2]

        galaxies, = np.where((gals['Gal']==True)&(gals['CID']==cls_id)&(gals['R']<=r2))
        probz = gals["pz0"][galaxies]
        
        area = (1-fm)*(np.pi*r2**2)

        n_cls_field = np.nansum(probz)
        n_gals = n_cls_field-nb*area

        ni = n_gals#/(nb*(np.pi*r2**2))
        if ni<0:
            print('norm less than zero:',ni)
        
        norm_vec_i = _computeNorm(gals[galaxies],r2,nb*(1-fm))

        area_vec.append(area)
        norm.append(ni)
        norm_vec.append(norm_vec_i)

    return np.array(norm), np.array(norm_vec), np.array(area_vec)

def doProb(Pgals,Pbkg,Ngals,Nbkg, normed=True,eps=1e-12):
    ratio = (Ngals+Nbkg)/( np.sum(Ngals*Pgals) + np.sum(Nbkg*Pbkg) )
    Pgals *= ratio
    Pbkg  *= ratio
    
    if normed:
        Pgals /= (Pgals.sum()+eps)
        Pbkg  /= (Pbkg.sum()+eps)
    
    prob = (Ngals*Pgals)/(Ngals*Pgals+Nbkg*Pbkg+eps)
    # prob[np.isnan(prob)] = 0.

    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    return prob

def _computeNorm(gal,r200,nbkg):
    rbins = np.linspace(0.,1.,4)
    pz = gal['pz0']
    rg = gal['Rn']

    Ngals, rmed = np.histogram(rg,bins=rbins,weights=pz)
    
    area = np.pi*(rbins[1:]**2-rbins[:-1]**2)
    Nbkg = nbkg*area
    norm = (Ngals-Nbkg)#/Nbkg
    
    w, = np.where(norm>0)
    if w.size>0:
        norm = np.where(norm<0.,np.mean(norm[norm>0]),norm)
    else:
        norm = -1.*np.ones_like(Nbkg)
    return norm

def computeProb(gal, keys, norm, nbkg, area_vec, eps=1e-12):
    gal = set_new_columns(gal,['Pmem','Pc','Pz','Pr'],val=0.)

    pdfr, pdfz, pdfc, pdf = gal['pdfr'], gal['pdfz'], gal['pdfc'], gal['pdf']
    pdfr_bkg, pdfz_bkg, pdfc_bkg, pdf_bkg = gal['pdfr_bkg'], gal['pdfz_bkg'], gal['pdfc_bkg'], gal['pdf_bkg']

    zcls = gal['redshift']
    pdfc = np.where(zcls<0.35, gal['pdfc'][:,0], gal['pdfc'][:,2])
    pdfc_bkg = np.where(zcls<0.35, gal['pdfc_bkg'][:,0], gal['pdfc_bkg'][:,2])

    probz = gal['pz0']
    rbins = np.linspace(0.,1.,4)
    rmed = (rbins[1:]-rbins[:-1])/2.
    for i,idx in enumerate(keys):
        # ni = gal['norm'][idx]
        ni = norm[i]
        nb = nbkg[i]
        # radii = gal['Rn'][idx]

        radii = gal['R'][idx]
        radi2 = 0.25*(np.trunc(radii/0.25)+1) ## bins with 0.125 x R200 width
        # area  = np.pi*radi2**2#((radi2+0.25)**2-radi2**2)

        area = area_vec[i]#float( np.unique(np.pi*(gal['R200'][idx])**2)[0] )
        Nb = nb*area

        pz0 = 1#probz[idx]
        pz0b= 1#(1-pz0)

        pdfri, pdfzi, pdfci = pdfr[idx], pdfz[idx], pdfc[idx]
        pdfr_bkg_i, pdfz_bkg_i, pdfc_bkg_i = pdfr_bkg[idx], pdfz_bkg[idx], pdfc_bkg[idx]
        
        # pdfz_bkg_i = pdfz_bkg_i*np.max(pdfz_bkg_i)/np.max(pdfzi)
        
        pn = np.sum(pdfri),np.sum(pdfzi),np.sum(pdfci)
        pn_bkg = np.sum(pdfr_bkg_i),np.sum(pdfz_bkg_i),np.sum(pdfc_bkg_i)

        Lgals     =  pdfri*pdfzi*pdfci*(np.sum(pn))/np.sum(pdfri*pdfzi*pdfci + eps)#np.product(pn)#
        Lgals_bkg = pdfr_bkg_i*pdfz_bkg_i*pdfc_bkg_i*(np.sum(pn_bkg))/np.sum(pdfr_bkg_i*pdfz_bkg_i*pdfc_bkg_i +eps)#(np.product(pn_bkg))##(np.product(pn_bkg))
        
        Lgals     = np.where(Lgals>1e6,0.,Lgals)
        Lgals_bkg = np.where(Lgals_bkg>1e6,0.,Lgals_bkg)

        # factor = np.sum(pz0)/(np.sum(pz0)+np.sum(pz0b))
        # pz0 *= factor

        # print('factor: %.3f'%factor)
        pr = doProb(pz0*pdfri,pz0b*pdfr_bkg_i, ni, Nb)
        pz = doProb(    pdfzi, pdfz_bkg_i, ni, Nb)
        pc = doProb(pz0*pdfci,pz0b*pdfc_bkg_i, ni, Nb)

        pmem = doProb(pz0*Lgals, pz0b*Lgals_bkg, ni, Nb, normed=False) #normed is used to normalize the prob. for the 3 PDFs

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
        contaminants, = np.where((gal['CID'] == keys[i]) & (gal['True']==False))
        
        pz_cont = gal['pz0'][contaminants]
        
        area = np.pi*(r2)**2
        ncon = np.sum(pz_cont)
        
        ratio_i = ncon/area
        ratio.append(ratio_i)

        bkgFlag[contaminants] = True

    return np.array(ratio), bkgFlag

def compute_ngals(g,cidx,r_aper,true_gals=False,col='Pmem'):
    keys  = list(chunks(g['CID'],cidx))
    if true_gals:
        ngals = [np.count_nonzero(g['True'][idx]) for idx in keys]
    else:
        ngals = [np.sum(g[col][idx]) for idx in keys]
    return np.array(ngals)

def compute_ptaken(g0):
	"""
	it computes p_taken
	"""
	print('compute ptaken')
	# g0.sort('Pmem')
	pmem = g0['Pmem']
	ptaken = np.ones_like(pmem,dtype=float)

	## find common galaxies
	gid = np.array(g0['GID'])#.astype(np.int)
	commonGroups = commonValues(gid)

	for indices in commonGroups:
		pm_group = np.array(pmem[indices])
		pt_group = np.array(ptaken[indices])

		idx_sort = np.argsort(-1*pm_group) ## -1* to reverse order

		pm_group_s = pm_group[idx_sort]
		pt_group_s = pt_group[idx_sort]

		new_pm = 0
		toto = 1
		pm = []
		for i in range(indices.size):
			toto *= (1-new_pm)
			new_pm = toto*pm_group_s[i]
			pt_group_s[i] = toto
			pm.append(new_pm)

		pmem[indices[idx_sort]] = np.array(pm)
		ptaken[indices[idx_sort]] = pt_group_s

	g0['Pmem'] = pmem
	g0['Ptaken'] = ptaken

	return g0

def commonValues(values):
	idx_sort = np.argsort(values)
	sorted_values = values[idx_sort]
	vals, idx_start, count = np.unique(sorted_values, return_counts=True,
                                return_index=True)

	# sets of indices
	res = np.split(idx_sort, idx_start[1:])
	#filter them with respect to their size, keeping only items occurring more than once

	vals = vals[count > 1]
	commonValuesIndicies = [ri for ri in res if ri.size>1]
	
	return commonValuesIndicies

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

def interpData(x,y,x_new,extrapolate=False):
    # if not extrapolate:
    # out = np.interp(x_new,x,y,left=0,right=0)
    # else:
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

        radi2 = 0.25*(np.trunc(radii/0.25)+1) ## bins with 0.125 x R200 width
        # areag = np.pi*radi2**2#((radi2+0.25)**2-radi2**2)

        gal['pdfr'][idx] = interpData(rvec,pdfri,radii,extrapolate=True)*areag
        gal['pdfz'][idx] = interpData(zvec,pdfzi,zgal)
        gal['pdfm'][idx] = interpData(mvec,pdfmi,mag)

        gal['pdfr_bkg'][idx] = np.ones_like(radi2)#/areag #interpData(rvec,np.ones_like(gal['pdfr'][idx]),radii)
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
        
        ng_profile = interpData(rvec,pdfr_cfi,radi2,extrapolate=True)
        gal['norm'][idx] = (ng_profile - nb*areag)#/nb

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
    for i in range(len(cat)):
        cls_id,z_cls = cat['CID'][i], cat['redshift'][i]
        r2 = r200[i]
        idx, = np.where( (gal['Gal']==True) &(gal['CID']==cls_id) & (gal["R"]<=gal['r_aper']) &(gal['True']==True) )
            
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
    for i in range(len(cat)):
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
    if len(x)>2:
        try:
            if silvermanFraction is None:
                kernel = gaussianKDE.gaussian_kde(x,bw_method=bw)
            else:
                kernel = gaussianKDE.gaussian_kde(x,silvermanFraction=silvermanFraction)        
        
            kde = kernel(xvec)
        except:
            kde = np.zeros_like(xvec)
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

def check_input_catalogs(gal,cat):
    """ check if you have enough galaxies"""
    galMask    = gal["Gal"]
    bkgMask    = gal["Bkg"]

    galIndices = list(chunks(gal['CID'],cat['CID']))

    n_all = [idx.size                       for idx in galIndices]
    n_gal = [np.count_nonzero(galMask[idx]) for idx in galIndices]
    n_bkg = [np.count_nonzero(bkgMask[idx]) for idx in galIndices]

    return np.array(n_all), np.array(n_gal), np.array(n_bkg)

def group_and_sort_tables(gal,cat):
    cat.sort('CID')
    gal.sort('CID')
    
    gal = gal.group_by('CID')
    gidx,cidx,_   = getIndices(gal['CID'],cat['CID'])
    gal = gal[gidx]
    return gal, cat, gidx, cidx

## -------------------------------
## main function
def clusterCalc(gal, cat, outfile_pdfs=None, member_outfile=None, cluster_outfile=None,pixelmap=None,
                r_in=4, r_out=6, sigma_z=0.05, zfile=None, p_low_lim=0.01, simulation=True, r_aper_model='hod'):
    ##############
    colorBW = [0.05,0.025,0.02]
    zBW = 'silverman'
    ##############

    ## grouping the clusters
    gal, cat, gidx, cidx = group_and_sort_tables(gal,cat)

    if (len(cat)==0)|(len(gal)==0):
        print('Critical Error')
        return [np.nan,np.nan]

    good_clusters0 = np.arange(0,len(cat),1,dtype=np.int64)

    if good_clusters0.size==0:
        print('Critical Error')
        return [np.nan,np.nan]
    
    ## taking out h factor
    # gal['R'] *= 0.7

    print('Computing Galaxy Density')
    ## Compute nbkg
    nbkg, BkgFlag = backSub.computeDensityBkg(gal,cat,r_in=r_in,r_out=r_out,r_aper=0.75/0.7,nslices=120)

    ## updating galaxy status
    gal['Bkg'] = BkgFlag    ## all galaxies inside the good backgrund ring's slice
    
    print('Computing R200')
    if r_aper_model=='hod':
        r200, raper = radial.computeR200(gal, cat, nbkg, rmax=3/0.7, compute=True) ## uncomment in the case to estimate a radius
    else:
        r200 = cat['R200_true'][:]
        raper= 1.*r200

    gal['r_aper'] = raper[cidx]#[np.in1d(cidx,good_clusters0)]]
    gal['R200']   = r200[cidx]#[np.in1d(cidx,good_clusters0)]]
    gal['Rn']     = gal['R']/gal['R200']

    ## get keys
    good_clusters = good_clusters0[nbkg[good_clusters0]>=0.]
    galFlag = (gal['Rn']<=1.)

    #ngals, _, keys, galIndices = backSub.computeGalaxyDensity(gal, cat[good_clusters], raper[good_clusters], nbkg[good_clusters], nslices=72)
    toto = list(chunks(gal['CID'],cat['CID'][good_clusters]))
    galIndices = [idx[galFlag[idx]] for idx in toto]
    gal['Gal'] = galFlag    ## all galaxies inside R200 within mag_i <mag_lim

    if simulation:
        galCut = gal[galFlag].copy()
        nbkg0, BkgFlag0 = computeContamination(galCut, cat['CID'], raper, np.array(cat['magLim']))
        Ngals_true      = compute_ngals(galCut, cat['CID'], raper,true_gals=True)
        
    nbkg = nbkg0

    print('Computing PDFs \n')
    print('-> Radial Distribution')
    rvec = np.linspace(0.,4.,40)
    pdfr_list = radial.computeRadialPDF(gal, cat[good_clusters], r200[good_clusters], raper[good_clusters], nbkg[good_clusters], galIndices, rvec, c=3.53)

    print('-> Redshift Distribution')
    zvec = np.arange(0.,1.2,0.005)        ## vec for the pdfz_cls
    pdfz_list = probz.computeRedshiftPDF(gal, cat[good_clusters], r200[good_clusters], nbkg[good_clusters], galIndices, sigma_z, 
                                        zfile=zfile, zvec=zvec, bandwidth=zBW)

    print('-> Color Distribution')
    color_vec = np.arange(-1.,4.5,0.0025) ## vec for the pdfc_cls
    pdfc_list = probc.computeColorPDF(gal, cat[good_clusters], r200[good_clusters], nbkg[good_clusters], galIndices, color_vec, bandwidth=colorBW,parallel=True)

    print('-> Mag Distribution')
    mag_vec = np.arange(-10.,0.,0.01) ## vec for the pdfc_cls
    pdfm_list = probm.computeMagPDF(gal, cat[good_clusters], r200[good_clusters], nbkg[good_clusters], galIndices, mag_vec)

    if simulation:
        print('--> Getting truth distributions')
        pdfr_true, pdfz_true, pdfc_true, pdfm_true = getTruthPDFs(gal,cat[good_clusters],r200[good_clusters],galIndices,rvec,zvec,color_vec,mag_vec,bwz=zBW,bwc=colorBW)
        pdfr_list.append(pdfr_true); pdfz_list.append(pdfz_true); pdfc_list.append(pdfc_true); pdfm_list.append(pdfm_true)

    rmed = (rvec[1:]+rvec[:-1])/2
    var_list = [rmed,zvec,color_vec,mag_vec]
    pdf_list = [pdfr_list,pdfz_list,pdfc_list,pdfm_list]

    print('Compute MaskFraction and Norm')
    maskFraction = radial.computeMaskFraction(pixelmap, gal, cat[good_clusters], r200[good_clusters], pdfr_list[0], pdfz_list[0], rmed, zvec)
    if pixelmap is not None: del pixelmap
    
    norm, norm_vec, area_vec = computeNorm(gal, cat[good_clusters], raper[good_clusters], nbkg[good_clusters], maskFraction)

    print('\n Compute Probabilities')
    galCut = gal[(gal['Gal']==True)&(gal['Rn']<=1.)].copy()

    # galIndices = list(chunks(galCut['CID'],cat['CID'][good_clusters]))
    gidx,cidx,galIndices = getIndices(galCut['CID'],cat['CID'][good_clusters])
    galCut = getPDFs(galCut,galIndices,var_list,pdf_list,nbkg[good_clusters],mag_pdf=False)
    galCut = computeProb(galCut, galIndices, norm, nbkg[good_clusters], area_vec)

    print('Writing Output: Catalogs  \n')
    print('Writing Galaxy Output')
    Colnames=['tile','mid', 'GID', 'CID', 'norm', 'Pr', 'Pz', 'Pc', 'Pmem', 'pdfr', 'pdfz', 'pdfc', 'pdfm', 
              'pdf', 'pdfr_bkg', 'pdfz_bkg', 'pdfc_bkg', 'pdfm_bkg', 'pdf_bkg','z', 'zerr','zoffset','pz0','Rn','theta','dx','dy','R200']
    galOut = galCut[Colnames]
    
    if member_outfile is not None:
        galOut.write(member_outfile,format='fits', overwrite=True)
    
    print('Writing Cluster Output \n')
    ### writing cluster catalogs
    if simulation:
        newColumns = ['R200','RAPER','Ngals','Norm','Nbkg','MASKFRAC','Ngals_true','Nbkg_true']
    else:    
        newColumns = ['R200','RAPER','Ngals','Norm','Nbkg','MASKFRAC']
    
    cat = initNewColumns(cat,newColumns,value=-1.)
    Ngals = compute_ngals(galOut,cat['CID'][good_clusters],r200[good_clusters],true_gals=False,col='Pmem')

    cat['R200']  = r200
    cat['RAPER'] = raper
    cat['Nbkg']  = nbkg
    cat['Ngals'][good_clusters] = Ngals
    cat['Norm'][good_clusters]  = norm
    cat['MASKFRAC'][good_clusters] = maskFraction

    cat['magLim_i'] = cat['magLim'][:,1]
    cat.remove_column('magLim')
        
    if simulation:
        #Ngals_true = compute_ngals(galOut,cat['CID'],r200[good_clusters],true_gals=True)
        cat['Nbkg_true'] = nbkg0
        cat['Ngals_true'] = Ngals_true
        
    if cluster_outfile is not None:
        cat.write(cluster_outfile,format='fits', overwrite=True)

    print('Writing cluster PDFs \n')
    ### creating pdfs outputs
    if outfile_pdfs is not None:
        makePDFs = outPDF(outfile_pdfs,cat['CID'][good_clusters])
        makePDFs.attrs_write(norm,label='norm')
        makePDFs.attrs_write(nbkg[good_clusters],label='nbkg')
        makePDFs.attrs_write(cat['redshift'][good_clusters],label='zcls')
        makePDFs.attrs_write(r200,label='R200')
        if simulation: makePDFs.attrs_write(cat['R200_true'][good_clusters],label='R200_true')
        if simulation: makePDFs.attrs_write(cat['M200_true'][good_clusters],label='M200_true')
        makePDFs.pdfs_write(var_list,pdfr_list,pdfz_list,pdfc_list,pdfm_list,simulation=simulation)
        makePDFs.close_file()

    print('end!')

    return galOut, cat
    
if __name__ == '__main__':
    print('membAssignment.py')
    print('author: Johnny H. Esteves')
