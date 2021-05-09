import numpy as np
from astropy.table import Table 
from astropy.io.fits import getdata

### The predicted cosmoDC2 BPZ photoz has a bias with z
### This codes computes the bias factor as a function of true z
### After the correction z_pre = z_pre*ratio(ztrue) we have an unbiased estimator
############### SETUP ###############
infile='/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/temp_file/emuBPZ-r200_copa_test_gal.fits'
outfile='emuBPZ_correction_z.txt'

z_col     = 'z'
ztrue_col = 'z_true'

zmin,zmax = 0.,3.
dx        = 0.02

Rmax      = 8.
#####################################

def makeBins(variable,xedges):
    xbins = (xedges[1:]+xedges[:-1])/2
    indices = [ np.where((variable >= xedges[i]) & (variable <= xedges[i + 1]))[0] for i in range(len(xedges)-1)]
    return indices, xbins

### Loading Catalog
gal     = Table(getdata(infile))
gal     = gal[gal['R']<=Rmax]

z       = gal[z_col]
ztrue   = gal[ztrue_col]

zoffset = (z-ztrue)/(1+ztrue)
dz      = np.abs(zoffset)

zbins     = np.arange(zmin,zmax+dx,dx)
keys, zb  = makeBins(ztrue,zbins)

## Computing model variables
x         = ztrue/z
ratio     = np.array([np.nanmedian(x[idx]) for idx in keys])

w, = np.where(np.logical_not(np.isnan(ratio)))

## Saving the output
savefile = open(outfile, "w")
with open(outfile, "w") as savefile:
    header = '#z     , ratio\n'
    savefile.write(header)
    for i in w:
        line = '%.5f,%.5f\n'%(zb[i],ratio[i])
        savefile.write(line)
savefile.close()