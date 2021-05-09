import numpy as np
from astropy.table import Table 
from astropy.io.fits import getdata

############### SETUP ###############
infile='/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/temp_file/emuBPZ-rhod-zw_copa_test_gal.fits'
outfile='/home/s1/jesteves/git/ccopa/aux_files/zwindow_cosmoDC2_emuBPZ.txt'

z_col     = 'z'
ztrue_col = 'z_true'

zmin,zmax = 0.09,1.0
dx        = 0.020

dz_max    = 0.3
Rmax      = 1.0
#####################################

def makeBins(variable,xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

def remove_outliers(x):
    q25,q75 = np.nanpercentile(x,[25,75])
    iqr     = q75-q25
    lo, up  = q25-1.5*iqr, q75+1.5*iqr
    mask    = (x<up)&(x>lo)
    return mask

def fit_gauss(z):
    mask = remove_outliers(z)
    zfit = z[mask]
    zmin = np.nanpercentile(zfit,16)
    zmax = np.nanpercentile(zfit,84)
    res  = np.array([np.nanmean(zfit),np.nanstd(zfit),zmin,zmax,1-1.*np.count_nonzero(mask)/len(z)])
    return res

### Loading Catalog
gal     = Table(getdata(infile))
gal     = gal[gal['R']<=Rmax]

z       = gal[z_col]
ztrue   = gal[ztrue_col]

zoffset = (z-ztrue)/(1+ztrue)
dz      = np.abs(zoffset)

ztrue   = np.where(np.abs(zoffset)>dz_max,-1.,ztrue)

zbins   = np.arange(zmin,zmax+dx,dx)
keys, zb  = makeBins(ztrue,zbins)

## Computing model variables
res  = np.array([fit_gauss(zoffset[idx]) for i,idx in enumerate(keys)])
bias = res[:,0]
sigma= res[:,1]
zlow = res[:,2]
zhigh= res[:,3]
of   = res[:,4]

# bias      = np.array([np.nanmedian(zoffset[idx]) for idx in keys])
# sigma     = np.array([np.nanmedian(dz[idx]) for idx in keys])

# zlow      = np.array([np.nanpercentile(zoffset[idx],16) for idx in keys])
# zhigh     = np.array([np.nanpercentile(zoffset[idx],84) for idx in keys])

w, = np.where(np.logical_not(np.isnan(bias)))

## Saving the output
savefile = open(outfile, "w")
with open(outfile, "w") as savefile:
    header = '#z   ,bias  ,sigma ,zlow ,zhigh, outlierFrac  \n'
    savefile.write(header)
    for i in w:
        line = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n'%(zb[i],bias[i],sigma[i],zlow[i],zhigh[i],of[i])
        savefile.write(line)
savefile.close()