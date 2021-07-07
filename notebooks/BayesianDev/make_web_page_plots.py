import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d
import os

from astropy.table import Table, vstack
from astropy.io.fits import getdata

import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns; sns.set(color_codes=True)
plt.rcParams.update({'font.size': 16})
sns.set_style("whitegrid")

## local libraries
from plotsLibrary import generalPlots,clusterPlots,checkPlots, sky_plot

#######################
individual_plots=True

#######################

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def splitBins(var):
    nmin = np.nanmin(var)
    n25 = np.percentile(var,25)
    n50 = np.nanmedian(var)
    n75 = np.percentile(var,75)
    nmax = np.max(var)
    
    return np.array([nmin,n25,n50,n75,nmax])

def getIndices(gindices,gkeys,ckeys):
    indicies = np.empty((0),dtype=int)
    indicies_into_cluster = np.empty((0),dtype=int)

    for i in range(ckeys.size):
        idx, = np.where(gkeys==ckeys[i])
        if idx.size>0:
            w2 = np.arange(gindices[idx],gindices[idx+1], 1, dtype=int)
            w = np.full(w2.size,i,dtype=int)

            indicies = np.append(indicies,w2)
            indicies_into_cluster = np.append(indicies_into_cluster,w)

    return indicies,indicies_into_cluster

def zoffset(z,zcls):
    return (z-zcls)/(1+zcls)

def get_delta_color(gal,cat,indices,lcolor='g-r'):
    gal['delta_rs'] = 0.
    color = gal['%s'%lcolor]
    for i,idx in enumerate(indices):
        rs_param = cat['rs_param_%s'%lcolor][i]
        ri = (color[idx]-rs_param[0])#/rs_param[1]
        if np.nanmedian(ri)>0.05: ri += np.nanmedian(ri)
        gal['delta_rs'][idx] = ri

    return gal['delta_rs']

def set_new_variables(gal, gal2, cat, cat2):
    ## get the same clustes
    cat = cat.group_by('CID')
    cidx, cidx2 = getIndices(cat.groups.indices,cat.groups.keys['CID'],cat['CID'])
    cat2 = cat2[cidx2]

    gal = gal.group_by('CID')
    gindices, gkeys = gal.groups.indices, gal.groups.keys['CID']
    gidx, cidx = getIndices(gindices, gkeys, cat['CID'])
    gal = gal[gidx]

    gal['Ngals'] = cat['Ngals_true'][cidx]
    gal['M200'] = cat['M200_true'][cidx]
    gal['Rnorm'] = gal['R']/cat['R200_true'][cidx]

    gal2 = gal2.group_by('CID')
    gidx, cidx = getIndices(gal2.groups.indices,gal2.groups.keys['CID'],cat['CID'])
    gal2 = gal2[gidx]

    gal2['Ngals'] = cat2['Ngals'][cidx]
    gal2['M200'] = cat2['M200_true'][cidx]
    gal2['Rnorm'] = gal2['R']/cat2['R200_true'][cidx]

    gal['z_offset'] = zoffset(gal['z'],gal['redshift'])
    gal2['z_offset'] = zoffset(gal2['z'],gal2['redshift'])

    # lcolor: 0,1,2,3,4
    color_list = ['g-r','g-i','r-i','r-z','i-z']
    color_index = [[0,1],[0,2],[1,2],[1,3],[2,3]]

    for i,pair_idx in enumerate(color_index):
        i0, i1 = pair_idx
        gal[color_list[i]] = gal['mag'][:,i0]-gal['mag'][:,i1]
        gal2[color_list[i]] = gal2['mag'][:,i0]-gal2['mag'][:,i1]

    indices = list(chunks(gal['CID'],cat['CID']))
    indices2 = list(chunks(gal2['CID'],cat['CID']))

    # gal['delta_rs'] = get_delta_color(gal,cat,indices)
    # gal2['delta_rs'] = get_delta_color(gal2,cat,indices2)

    return gal, gal2, cat, cat2

def doProb(ngals,nbkg,norm,normed=False):
    
    if normed:
        ngals /= ngals.sum()
        nbkg /= nbkg.sum()

    prob = norm*ngals/(norm*ngals+nbkg)
    prob[np.isnan(prob)] = 0.
    
    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    
    return prob
    
def interpData(x,y,x_new):
    out = np.empty(x_new.shape, dtype=y.dtype)
    out = interp1d(x, y, kind='linear', fill_value='extrapolate', copy=False)(x_new)
    # yint = interp1d(x,y,kind='linear',fill_value='extrapolate')
    return out


def get_hdf_pdfs(pdf,ggroup,mode='radii',color_idx=[0,1,2,3,4,5]):
    
    if mode=='radii':
        radii = ggroup['R']
        pdf_new = interpData(pdf[:,0],pdf[:,1],radii)
        return pdf_new

    if mode=='z':
        zoff = (ggroup['z']-ggroup['redshift'][0])/(1+ggroup['redshift'][0])
        pdf_new = interpData(pdf[:,0],pdf[:,1],zoff)
        return pdf_new

    if mode=='color':
        pdf_new = np.empty_like(ggroup['z'])

        for ix in color_idx:
            color = ggroup[getColorLabel(ix)]
            pdf_i = interpData(pdf[:,0],pdf[:,ix],color)
            pdf_new = np.c_[pdf_new,pdf_i]
        
        pdf_new = pdf_new[:,1:]

        return pdf_new

def getColorLabel(color_number):
    color_list = ['g-i','g-r','r-i','r-z','i-z']
    return color_list[color_number]


def open_hdf_get_pdfs(gal,keys,name_list,color_idx=[0,1,2,3,4,5],field=False):
    pdfz = pdfc = pdfr = np.zeros_like(gal['CID'])
    
    label = 'cls'
    if field: label = 'cls_field'
    
    for idx,name in zip(keys,name_list):
        group = hdf['%s'%(name)]
        ggroup = gal[idx]

        pdfr[idx] = get_hdf_pdfs(group['pdf_r_%s'%label],ggroup,mode='radii')
        pdfz[idx] = get_hdf_pdfs(group['pdf_z_%s'%label],ggroup,mode='z')
        pdfc[idx] = get_hdf_pdfs(group['pdf_c_%s'%label],ggroup,mode='color')

    return [pdfr,pdfz,pdfc]

# if __name__ == "__main__":
color_label = 'gr'

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members_stellarMass.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_stellarMass.fits' 

file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members.fits'
file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed.fits'

# file_gal  = 'out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_members.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed.fits'

# file_gal2  = 'out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_truth_table_members.fits'
# file_cls2 = './out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_truth_table.fits'

# pdf_file = "./out/pdfs/pdfs_50.hdf5"
# pdf_file2= "./out/pdfs/pdfs_50_truth.hdf5"

file_gal2 = './out/buzzard0_members.fits'
file_cls2 = './out/buzzard0.fits'

# file_gal = './out/buzzard_old_members.fits'
# file_cls = './out/buzzard_old.fits' 

pdf_file = "./out/pdfs/pdfs_1000.hdf5"
pdf_file2= "./out/pdfs/pdfs_1000_truth.hdf5"

# file_gal2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table_members_stellarMass.fits'
# file_cls2 = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_truth_table_stellarMass.fits' 


cat = Table(getdata(file_cls))
gal = Table(getdata(file_gal))

cat2 = Table(getdata(file_cls2))
gal2 = Table(getdata(file_gal2))

import h5py
hdf = h5py.File(pdf_file,'r')
hdf2 = h5py.File(pdf_file2,'r')

idx = [np.where(cat2["CID"] == cidx)[0] for cidx in cat["CID"] if len(np.where(cat2["CID"] == cidx)[0])>0]
idx = np.stack(idx).ravel()
cat2 = cat2[idx]

# gal = gal[(gal['R']<=1.)&(gal["Pz"]>0.4)&(gal["Pz"]>0.2)]
gal = gal[(gal['R']<=1.)]
gal2 = gal2[(gal2['R']<=1.)]

## get some columns variables
print('defining some variables')
gal, gal2, cat, cat2 = set_new_variables(gal, gal2, cat, cat2)

print('defining bins')
massBins = splitBins(cat2['M200_true'])
zBins = splitBins(cat['redshift'])
nbins = splitBins(cat2['Ngals'])

radialBin = np.linspace(0.01,1.01,8)
colorBin = np.linspace(-1.5,0.5,12)
zOffsetBin = np.linspace(-0.2,0.2,20)

print('Playing with nbkg')
# newPm = get_new_pmem(gal,hdf,mode='nbkg')
cids = np.unique(cat['CID'])
keys = list(chunks(gal['CID'],cids))

## get pdf_all
pdr, pdfz, pdfc = open_hdf_get_pdfs(gal,keys,cids)

## get pdf_bkg
pdr_field, pdfz_field, pdfc_field = open_hdf_get_pdfs(gal,keys,cids,field=True)

## get norm
ngals = cat['norm']*cat['nbkg']

## compute Pmem

Pm = doProb(pdfs,pdfs_field,ngals)

# if individual_plots:
print('Plotting General plots')
allPlots = generalPlots()

print('Probability Histograms')
allPlots.plot_grid_histograms(gal)
allPlots.plot_grid_fractions_pmem(gal)
allPlots.plot_multiple_scaling_relations(gal,gal2,cat,cat2)