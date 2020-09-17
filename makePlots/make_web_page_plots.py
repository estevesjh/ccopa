import numpy as np
import scipy.stats as st
import os

from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns; sns.set(color_codes=True)
# plt.rcParams.update({'font.size': 16})
sns.set_context("paper", font_scale=1.3)
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
    n25 = np.percentile(var,35)
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
    cat = cat[cat['Ngals_true']>=2]
    
    ## get the same clustes
    cat = cat.group_by('CID')
    cidx, cidx2 = getIndices(cat.groups.indices,cat.groups.keys['CID'],cat2['CID'])
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

def getTruthTable(gal,cat):
    gal2 = gal[gal['True']==True].copy()

    gal2['Pmem'] = 1.

    indices = list(chunks(gal2['CID'],cat['CID']))
    ngals = np.array([np.sum(gal2['Pmem'][idx]) for idx in indices])

    cat2 = cat.copy()
    cat2['Ngals'] = ngals
    cat2['R200'] = cat['R200_true'][:]
    cat2['Nbkg'] = cat['Nbkg_true'][:]

    return gal2,cat2

# if __name__ == "__main__":
color_label = 'gr'

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members_stellarMass.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_stellarMass.fits' 

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed.fits'

# file_gal = './out/buzzard_v1.6_pz_005_RHOD_magCut_0p4Lstar_members.fits'
# file_cls = './out/buzzard_v1.6_pz_005_RHOD_magCut_0p4Lstar.fits'

# file_gal = './out/CY1a_RHOD_1000_members.fits'
# file_cls = './out/CY1a_RHOD_1000.fits'

file_gal = './out/CY1a_1e13Msun_RHODa_pz003_members.fits'
file_cls = './out/CY1a_1e13Msun_RHODa_pz003.fits'

# file_gal2 = './out/buzzard_v1.6_pz_005_Rfixed_magCut_BCG_truth_table_members.fits'
# file_cls2 = './out/buzzard_v1.6_pz_005_Rfixed_magCut_BCG_truth_table.fits'

# file_gal  = 'out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_members.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed.fits'

# file_gal2  = 'out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_truth_table_members.fits'
# file_cls2 = './out/Chinchilla-0Y1a_v1.6_truth_50_ccopa_pz_005_Rfixed_truth_table.fits'

# pdf_file = "./out/pdfs/pdfs_50.hdf5"
# pdf_file2= "./out/pdfs/pdfs_50_truth.hdf5"

# file_gal2 = './out/buzzard0_members.fits'
# file_cls2 = './out/buzzard0.fits'

# file_gal = './out/buzzard1_members.fits'
# file_cls = './out/buzzard1.fits' 

pdf_file = "./out/pdfs/CY1a_1e13Msun_RHODa_pz003.hdf5"
# pdf_file2= "./out/pdfs/pdfs_1000_truth.hdf5"

cat = Table(getdata(file_cls))
gal = Table(getdata(file_gal))

# if file_gal2 is None:
gal2, cat2 = getTruthTable(gal,cat)

# cat2 = Table(getdata(file_cls2))
# gal2 = Table(getdata(file_gal2))

idx = [np.where(cat2["CID"] == cidx)[0] for cidx in cat["CID"] if len(np.where(cat2["CID"] == cidx)[0])>0]
idx = np.stack(idx).ravel()
cat2 = cat2[idx]

# gal = gal[(gal['R']<=1.)&( gal["Pmem"] >= 0.4)]
# gal = gal[(gal['R']<=1.)&( gal["Pz"] >= 0.4)]
# gal = gal[(gal['R']<=1.)]
# gal2 = gal2[(gal2['R']<=1.)]

## get some columns variables
print('defining some variables')
gal, gal2, cat, cat2 = set_new_variables(gal, gal2, cat, cat2)

# zoff = (gal['z']-gal['redshift'])/(1+gal['redshift'])
# idx, = np.where(np.abs(zoff)<0.15)
# gal = gal[idx]

# gal = gal[(gal['Rnorm']<=1.)]
# gal2 = gal2[(gal2['Rnorm']<=1.)]

print('defining bins')
massBins = splitBins(cat2['M200_true'])
zBins = splitBins(cat['redshift'])
nbins = splitBins(cat2['Ngals'])

radialBin = np.linspace(0.01,1.01,8)
colorBin = np.linspace(-1.5,0.5,12)
zOffsetBin = np.linspace(-0.2,0.2,20)

# if individual_plots:
print('Plotting General plots')
allPlots = generalPlots()

print('Sky Plot')
sky_plot(cat['RA'], cat['DEC'],title='Buzzard Simulation v1.6')
plt.clf()

print('Scaling Relations')
allPlots.plot_scaling_relation(cat,cat2,kind='richness')

allPlots.plotResidual(cat,cat2,kind=['richness','z'],bins=zBins)
allPlots.plotResidual(cat,cat2,kind=['richness','mass'],bins=massBins)
allPlots.plotResidual(cat,cat2,kind=['richness','N'],bins=nbins)

print('Probability Histograms')
allPlots.plot_grid_histograms(gal)
allPlots.plot_grid_fractions_pmem(gal)

# print('Purity and Completeness')
# scores = np.array(gal['Pmem'])

opt_tr = allPlots.plot_confusion_matrix(gal,'Pmem',title=None)
print('otp th: %.2f'%opt_tr)
# allPlots.plot_roc_curve(gal,'Pmem',opt_tr)
# allPlots.plot_precision_recall_vs_threshold(gal,'Pmem',lcol='P_{mem}',title=None)

# allPlots.plot_purity_completeness(gal,gal2)

# allPlots.plot_purity_completeness_threshold(gal,gal2,'Pmem')
# allPlots.plot_purity_completeness_threshold(gal,gal2,'Pz')

# allPlots.plot_purity_completeness_variable(gal,gal2,radialBin,'R')
# # allPlots.plot_purity_completeness_variable(gal,gal2,colorBin,'delta_rs')
# allPlots.plot_purity_completeness_variable(gal,gal2,zOffsetBin,'z_offset')

# allPlots.plot_purity_completeness_variable(gal,gal2,zBins,'redshift')
# allPlots.plot_purity_completeness_variable(gal,gal2,massBins,'M200')
# allPlots.plot_purity_completeness_variable(gal,gal2,nbins,'Ngals')

# print('PDFs')
# allPlots.plot_validation_pdf_radial(gal,gal2,cat,cat2)
# allPlots.plot_validation_pdf_redshift(gal,gal2,cat,cat2)
# #allPlots.plot_validation_pdf_color(gal,gal2,cat,cat2)

# print('Probabilities')
# allPlots.plot_probabilities_radialPDF(gal,gal2,cat,cat2)
# allPlots.plot_probabilities_redshiftPDF(gal,gal2,cat,cat2)
# # allPlots.plot_probabilities_colorPDF(gal,gal2,cat,cat2)

# galc = gal[gal['R']<= 1.]
# gal2c = gal2[gal2['R']<= 1.]
gal = gal[gal['Pmem']>=opt_tr]
gal2, cat2 = getTruthTable(gal,cat)

allPlots.plot_multiple_scaling_relations(gal,gal2,cat,cat2)
print('done')
# print('COPA')
# color_list = ['g-r','g-i','r-i','r-z','i-z']
# par_list = ['gr','gi','ri','rz','iz']
# allPlots.validating_color_model_grid(gal,cat,par_list,lcolor=color_list)
# allPlots.validating_color_model_grid(gal,cat,par_list,lcolor=color_list,sigma=True)
# allPlots.validating_color_model_grid(gal,cat,par_list,lcolor=color_list,fraction=True)
# for li,pi in zip(color_list,par_list): allPlots.validating_color_model_residual(gal,cat,gal2,cat2,pi,lcolor=li)
# 

# print('Check PDFs')
# import h5py
# hdf = h5py.File(pdf_file,'r')
# # hdf2 = h5py.File(pdf_file2,'r')

# check = checkPlots(hdf,hdf)
# check.plot_check_pdf_redshfit()
# check.plot_check_pdf_all_colors()
# check.hdf_close()

### Individual Plots
# if individual_plots:
#     print('Plotting indidual plots')
#     import h5py
#     hdf = h5py.File(pdf_file,'r')
#     hdf2 = h5py.File(pdf_file2,'r')
#     # indices = list(chunks(gal['CID'],cat['CID'][0:5]))
#     for i in range(10):
#         titulo = 'CID ='+str(cat['CID'][i])+r';  $M_{200,c} = %.2f \; 10^{14}M_{\odot}$'%(cat['M200_true'][i]/1E14)

#         plot = clusterPlots(hdf,hdf2,gal,gal2,cat,i)
#         plot.radialDistribution()        
#         # plot.makePlotRadial(R200=1,title=titulo)
#         for j in range(5):
#             plot.colorDistribution(lcolor=j) ## color g-r
#         plot.redshiftDistribution()
#         plot.spatialDistribution()

    # hdf.close()
    # hdf2.close()
