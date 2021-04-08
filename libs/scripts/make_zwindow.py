import numpy as np
from astropy.table import Table 
from astropy.io.fits import getdata

############### SETUP ###############
infile='/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/temp_file/bpz-r200_copa_test_gal.fits'
outfile='zwindow_cosmoDC2_bpz.txt'

z_col     = 'z'
ztrue_col = 'z_true'

zmin,zmax = 0.09,1.0
dx        = 0.02

dz_max    = 0.3
Rmax      = 1.
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

ztrue   = np.where(np.abs(zoffset)>dz_max,-1.,ztrue)

zbins   = np.arange(zmin,zmax+dx,dx)
keys, zb  = makeBins(ztrue,zbins)

## Computing model variables
bias      = np.array([np.nanmedian(zoffset[idx]) for idx in keys])
sigma     = np.array([np.nanmedian(dz[idx]) for idx in keys])

zlow      = np.array([np.nanpercentile(zoffset[idx],16) for idx in keys])
zhigh     = np.array([np.nanpercentile(zoffset[idx],84) for idx in keys])

w, = np.where(np.logical_not(np.isnan(bias)))

## Saving the output
savefile = open(outfile, "w")
with open(outfile, "w") as savefile:
    header = '#z   ,bias  ,sigma ,zlow ,zhigh  \n'
    savefile.write(header)
    for i in w:
        line = '%.3f,%.3f,%.3f,%.3f,%.3f\n'%(zb[i],bias[i],sigma[i],zlow[i],zhigh[i])
        savefile.write(line)
savefile.close()