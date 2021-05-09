
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from astropy.table import Table, vstack
from astropy.io.fits import getdata

import numpy as np

cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Msol = 1.98847e33
Mpc2cm = 3.086e+24
rad2deg= 180/np.pi
h=0.7

#--- Critical universe density
def rhoc(z):
    try:
        rho_c = float(cosmo.critical_density(z)/(u.g/u.cm**3)) # in g/cm**3
    except:
        rho_c = [float(cosmo.critical_density(zi)/(u.g/u.cm**3)) for zi in z]
        rho_c = np.array(rho_c)
    
    rho_c = rho_c*(Mpc2cm**3)/Msol # in Msol/Mpc**3
    return rho_c

def convertM200toR200(M200,z,nc=200):
    ## M200 in solar masses
    ## R200 in Mpc
    rho = rhoc(z)
    R200 = ( M200/(nc*4*np.pi*rho/3) )**(1/3.)
    return R200/h

file_cls = '/data/des61.a/data/johnny/CosmoDC2/sample2021/cosmoDC2_v1.1.4_2000_GC.fits'

cat = Table(getdata(file_cls))

zcls = cat['redshift']
m200 = cat['halo_mass']
r200 = convertM200toR200(m200,zcls)

cat['R200_true'] = r200
cat.write(file_cls,overwrite=True,format='fits')
