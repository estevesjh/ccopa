__author__ = 'NataliaDelCoco'


#Projecao 

# tirado da biblioteca skymapper
# https://github.com/pmelchior/skymapper
# usa a projecao Albers

# criado: 21/03/19

import numpy as np

DEG2RAD=np.pi/180.0
h = 0.7

#==  funcoes  =======================================
def init_params(ra,dec,pos0=None):
  ra_ = np.array(ra)
  ra_[ra_ > 180] -= 360
  ra_[ra_ < -180] += 360
  # weigh more towards the poles because that decreases distortions
  if pos0 is None:
    ra0 = (ra_ * dec).sum() / dec.sum()
    if ra0 < 0:
        ra0 += 360
    dec0 = np.median(dec)
  else:
    ra0, dec0 = pos0
  # determine standard parallels
  dec1, dec2 = dec.min(), dec.max()
  delta_dec = (dec0 - dec1, dec2 - dec0)
  dec1 += delta_dec[0]/6
  dec2 -= delta_dec[1]/6

  return ra0, dec0, dec1, dec2


def nC(dec_1,dec_2):
  n = (np.sin(dec_1 * DEG2RAD) + np.sin(dec_2 * DEG2RAD)) / 2
  C = np.cos(dec_1 * DEG2RAD)**2 + 2 * n * np.sin(dec_1 * DEG2RAD)
  return n, C


def rho_f(n,C,dec):
  aux=np.sqrt(C - 2 * n * np.sin(dec * DEG2RAD)) / n

  return aux


def _toArray(x):
  """Convert x to array if needed
  Returns:
      array(x), boolean if x was an array before
  """
  if isinstance(x, np.ndarray):
    return x, True
  if hasattr(x, '__iter__'):
    return np.array(x), True
  return np.array([x]), False

def _unstandardize(lon,lon0):
  """Revert `_standardize`"""
  # no copy needed since all lons have been altered/transformed before
  lon *= -1 # left-handed
  lon += lon0
  lon [lon < 0] += 360
  lon [lon > 360] -= 360
  return lon

def wrapRA(ra_0, ra):
    """Normalize rectascensions to -180 .. 180, with reference `ra_0` at 0"""
    ra_, isArray = _toArray(ra)
    ra_ = ra_0 - ra_ # inverse for RA
    # check that ra_aux is between -180 and 180 deg
    ra_[ra_ < -180 ] += 360
    ra_[ra_ > 180 ] -= 360
    if isArray:
        return ra_
    return ra_[0]

#== corpo ===========================================

#considera que ra e dec sao dois arrays
def proj(ra,dec,pos0=None):

  #pra achar os parametros iniciais
  #ra_0 =RA that maps onto x = 0
  #dec_0 = Dec that maps onto y = 0 
  #dec_1 = lower standard parallel
  #dec_2 = upper standard parallel (must not be -dec_1)
  ra_0, dec_0, dec_1, dec_2 = init_params(ra,dec,pos0=pos0)
  
  DEG2RAD=np.pi/180.0

  #passo 1 => constantes
  n, C = nC(dec_1,dec_2)
  rho_0=rho_f(n,C,dec_0)

  #passo 2
  ra_ = wrapRA(ra_0,ra)
  theta = n*ra_
  rho = rho_f(n,C,dec)

  #passo 3
  Xc = rho*np.sin(theta * DEG2RAD) 
  Yc = rho_0 - rho*np.cos(theta * DEG2RAD)
  
  albers = [ra_0, dec_0, dec_1, dec_2]
  return Xc/DEG2RAD,Yc/DEG2RAD, albers

### Johnny's added at 20 July, 2020
def inv_proj(x,y,proj):
  # lon/lat actually x/y
  # Snyder 1987, eq 14-8 to 14-11
  # proj = init_params(ra,dec,pos0=pos0)
  ra_0, dec_0, dec_1, dec_2 = proj
  
  DEG2RAD=np.pi/180.0
  x,y = x*DEG2RAD, y*DEG2RAD
  #passo 1 => constantes
  n, C = nC(dec_1,dec_2)
  rho_0=rho_f(n,C,dec_0)

  rho = np.sqrt(x**2 + (rho_0 - y)**2)
  if n >= 0:
    theta = np.arctan2(x, rho_0 - y) / DEG2RAD
  else:
    theta = np.arctan2(-x, -(rho_0 - y)) / DEG2RAD
  lon = _unstandardize(theta/n,ra_0)
  lat = np.arcsin((C - (rho * n)**2)/(2*n)) / DEG2RAD
  return lon, lat

#### Added by Johnny
#### Projection Functions
def doDistAngle(x,y):
    dist = np.sqrt((x)**2+(y)**2)
    return dist

def delta_radec(ra,dec,ra0,dec0):
    return proj(ra,dec,pos0=[ra0,dec0])

def xy_to_radec(x,y,albers,Mpc2theta):
    _ra,_dec = x*h/Mpc2theta, y*h/Mpc2theta ## deg
    ra,dec = inv_proj(_ra,_dec,albers)
    return ra,dec

def radec_to_xy(ra,dec,ra0,dec0,Mpc2theta):
    dra,ddec,albers = delta_radec(ra,dec,ra0,dec0)    ## albers projection (delta RA, delta DEC)
    x = dra *Mpc2theta/h
    y = ddec*Mpc2theta/h
    return x,y,albers

def radec_to_theta(ra,dec,ra_c,dec_c):
    dra,ddec,albers = delta_radec(ra,dec,ra_c,dec_c)    ## albers projection (delta RA, delta DEC)
    theta = np.degrees( np.arctan2(dra,ddec) ) + 180
    return theta