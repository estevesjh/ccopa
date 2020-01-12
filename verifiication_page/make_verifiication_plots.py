# !/usr/bin/env python

from time import time
import numpy as np
from scipy.stats import uniform, norm, stats

from astropy.table import Table, vstack
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import dask

from libraryPlots import *

plt.rcParams.update({'font.size': 16})
plt.style.context('fivethirtyweight')

###########################################
test = 0
indetity_plots = 1
residual_plot = 1
background = 1
projection_plot = 0

colorEvolution = 1
redshiftEvolution = 1
color_check_bin = 0
root = './img'
###########################################
ii=0
print('Geting Data')

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_old_members_stellarMass.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_old_stellarMass.fits'

# file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_R_members_stellarMass.fits'
# file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_R_stellarMass.fits'

file_gal = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed_members.fits'
file_cls = './out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed.fits'

cat = Table(getdata(file_cls))
gal = Table(getdata(file_gal))

print('Taking only cluster within 5 true members')
cat = cat[(cat['Ngals_true'] >= 5)]
# cat = cat[(cat['Ngals_true'] >= 5)&(cat['MU']>1.)]

gal = gal.group_by('CID')
gidx, cidx = getIndices(gal.groups.indices,gal.groups.keys['CID'],cat['CID'])
gal = gal[gidx]

print('Defining variables')
keys = cat['CID'][:]
redshift = cat['redshift'][:]
m200 = cat['M200_true'][:]*1e-14

# mu_star = cat['MU'][:]
# mu_star_true = cat['MU_TRUE'][:]

ngals = cat['Ngals'][:]
ngals_true = cat['Ngals_true'][:]

nbkg = cat['Nbkg'][:]
nbkg_true = cat['Nbkg_true'][:]

z = gal['z']
zcls = gal['redshift']
ztrue = gal['z_true']

pmem = gal['Pmem']


mag_g = gal['mag'][:,0]
mag_r = gal['mag'][:,1]
mag_i = gal['mag'][:,2]
mag_z = gal['mag'][:,3]

gr = mag_g - mag_r
ri = mag_r - mag_i
iz = mag_i - mag_z

mask = gal['True']==True
if indetity_plots:
    print('%i. Plot R200 Identity'%(ii+1))
    s1 = root+'/img/n_identity_%i.png'%(test)
    s2 = root+'/img/mu_identity_%i.png'%(test)
    
    plotIdentity(ngals_true,ngals,s1,kind='N')
    # plotIdentity(mu_star_true,mu_star,s2,kind='Mu')

    ii+=1
    print('-> saved at %s'%root)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

if residual_plot:
    print('%i. Plot Residual'%(ii+1))
    s1 = root+'/img/n_residual_%i'%(test)
    s2 = root+'/img/mu_residual_%i'%(test)
    
    # massBins = histedges_equalN(m200, 100)
    # print(massBins)

    massBins = np.array([3.,4.,5.,6.6,8.56,11.07,13.62,26.5])
    zBins = np.arange(0.1,0.9+0.05,0.075)
    nbins = np.arange(1.,ngals.max(),20.)

    plotResidual(ngals_true,ngals,redshift,s1,kind=['N','z'],bins=zBins)
    plotResidual(ngals_true,ngals,m200,s1,kind=['N','mass'],bins=massBins)

    # plotResidual(mu_star_true,mu_star,redshift,s2,kind=['Mu','z'],bins=zBins)
    # plotResidual(mu_star_true,mu_star,m200,s2,kind=['Mu','mass'],bins=massBins)
    # plotResidual(mu_star_true,mu_star,ngals_true,s2,kind=['Mu','N'],bins=nbins)

    ii+=1
    print('-> saved at %s'%root)

if projection_plot:
    print('%i. Plot projections effects'%(ii+1))
    gal['N'] = 1.
    ptaken = np.where(gal['Ptaken']<1e-6,1e-6,gal['Ptaken'])
    gal['old_Pmem'] = np.where(gal['Pmem']/ptaken>1,1,gal['Pmem']/ptaken)

    sumPt = getN200(gal,keys,col='Ptaken')
    ncluster = getN200(gal,keys,col='N')
    richness_old = getN200(gal,keys,col='old_Pmem')

    ratio = sumPt/ncluster
    projected = ratio<=0.9

    cut = ngals>0
    r1, n1 = ngals[cut], ngals_true[cut]
    p1 = projected[cut]
    np1 = np.logical_not(p1)
    plotRichnessN200Projection(r1,n1,p1,savename='./n_identity_projections.png')

    cut = richness_old>0
    r1, n1 = richness_old[cut], ngals_true[cut]
    p1 = projected[cut]
    plotRichnessN200Projection(r1,n1,p1,savename='./n_identity_projections_old_pmem.png')

print('starting galaxies related plots')
print('to do !')

if colorEvolution:
    root = './img/'
    print('%i. Plot Color Evolution'%(ii+1))
    colorz(zcls, [gr, ri, iz], pmem, ['g - r', 'r - i', 'i - z'], (0.25,2), (20,5), root+'colorz_cluster_test_hex_density.png', 'hex_lambda')
    colorz(zcls[mask], [gr[mask], ri[mask], iz[mask]], np.ones_like(pmem[mask]), ['g - r', 'r - i', 'i - z'], (0.25,2), (20,5), root+'colorz_cluster_test_hex_density_true.png', 'hex_lambda')
    ii+=1
    print('-> saved at %s'%root)

print('fim')

