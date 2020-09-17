import numpy as np
from astropy.io import fits
import logging

def read_fits(filename):
    logging.debug('Starting helperFunctions.read_fits()')
    h=fits.open(filename)
    d=h[1].data
    id = d['COADD_OBJECTS_ID']
    haloid = d['HOST_HALOID']
    GR_P_COLOR=d['GR_P_COLOR']
    RI_P_COLOR=d['RI_P_COLOR']
    IZ_P_COLOR=d['IZ_P_COLOR']
    GR_P_MEMBER=d['GR_P_MEMBER']
    RI_P_MEMBER=d['RI_P_MEMBER']
    IZ_P_MEMBER=d['IZ_P_MEMBER']
    DIST_TO_CENTER=d['DIST_TO_CENTER']
    GRP_RED=d['GRP_RED']
    GRP_BLUE=d['GRP_BLUE']
    RIP_RED=d['RIP_RED']
    RIP_BLUE=d['RIP_BLUE']
    IZP_RED=d['IZP_RED']
    IZP_BLUE=d['IZP_BLUE']
    g = d['MAG_AUTO_G']
    gerr = d['MAGERR_AUTO_G']
    r = d['MAG_AUTO_R']
    rerr = d['MAGERR_AUTO_R']
    i = d['MAG_AUTO_I']
    ierr = d['MAGERR_AUTO_I']
    z = d['MAG_AUTO_Z']
    zerr = d['MAGERR_AUTO_Z']
    zed = d['Z']
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'GR_P_MEMBER':GR_P_MEMBER,'RI_P_MEMBER':RI_P_MEMBER,'IZ_P_MEMBER':IZ_P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    logging.debug('Returning from helperFunctions.read_fits()')
    return inputDataDict


def readJohnnyCat(filename, a, b):
    logging.debug('Starting helperFunctions.read_afterburner()')

    h=fits.open(filename)
    d=h[1].data
    d=d[a:b] # JCB: to divide and conquer
    id = d['COADD_OBJECTS_ID']
    haloid = d['MEM_MATCH_ID']
    zed = d['redshift'] # New afterburner

    GR_P_COLOR=np.where(d['type_gr']!=0,1,0)
    RI_P_COLOR=np.where(d['type_ri']!=0,1,0)
    IZ_P_COLOR=np.where(d['type_iz']!=0,1,0)
    
    P_RADIAL=d['Pr']
    P_REDSHIFT=d['Pz']
    P_MEMBER=d['Pmem']
    
    DIST_TO_CENTER=d['R']
    
    GRP_RED=np.where(d['type_gr']==1,1,0)
    GRP_BLUE=np.where(d['type_gr']==-1,1,0)
    RIP_RED=np.where(d['type_ri']==1,1,0)
    RIP_BLUE=np.where(d['type_ri']==-1,1,0)
    IZP_RED=np.where(d['type_iz']==1,1,0)
    IZP_BLUE=np.where(d['type_iz']==-1,1,0)

    g = d['mg']
    gerr = d['mg_err']
    r = d['mr']
    rerr = d['mr_err']
    i = d['mi']
    ierr = d['mi_err']
    z = d['mz']
    zerr = d['mz_err']
    
    #zed = d['Z_CLUSTER'] # Old afterburner
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'P_RADIAL':P_RADIAL,'P_REDSHIFT':P_REDSHIFT,'P_MEMBER':P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    logging.debug('Returning from helperFunctions.read_afterburner()')

    return inputDataDict

def readMaster_DES_SDSS_Cat(filename, a, b):
    logging.debug('Starting helperFunctions.read_afterburner()')

    h=fits.open(filename)
    d=h[1].data
    d=d[a:b] # JCB: to divide and conquer
    ID_DES = d['COADD_OBJECTS_ID']
    ID_SDSS = d['objid']
    id = np.where(ID_SDSS==999999,ID_DES,ID_SDSS)

    haloid = d['mid']
    zed = d['z_cls'] # New afterburner

    GR_P_COLOR=np.where(d['type_gr']!=0,1,0)
    RI_P_COLOR=np.where(d['type_ri']!=0,1,0)
    IZ_P_COLOR=np.where(d['type_iz']!=0,1,0)
    
    P_RADIAL=d['Pr']
    P_REDSHIFT=d['Pz']
    P_MEMBER=d['Pmem']
    
    DIST_TO_CENTER=d['R']
    
    GRP_RED=np.where(d['type_gr']==1,1,0)
    GRP_BLUE=np.where(d['type_gr']==-1,1,0)
    RIP_RED=np.where(d['type_ri']==1,1,0)
    RIP_BLUE=np.where(d['type_ri']==-1,1,0)
    IZP_RED=np.where(d['type_iz']==1,1,0)
    IZP_BLUE=np.where(d['type_iz']==-1,1,0)

    g = d['mg']
    gerr = d['mg_err']
    r = d['mr']
    rerr = d['mr_err']
    i = d['mi']
    ierr = d['mi_err']
    z = d['mz']
    zerr = d['mz_err']
    
    #zed = d['Z_CLUSTER'] # Old afterburner
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'P_RADIAL':P_RADIAL,'P_REDSHIFT':P_REDSHIFT,'P_MEMBER':P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    logging.debug('Returning from helperFunctions.read_afterburner()')

    return inputDataDict

coefs = [[-1.28167900e-03, 9.78790967e-02,-2.11337322e+00, 1.12139087e+01],
         [-3.12705403e-03, 1.96419803e-01,-3.80101934e+00, 2.04291313e+01],
         [-6.75571431e-04, 4.87008215e-02,-8.51343916e-01, 1.08236535e+00],
         [ 2.42978486e-03,-1.15899936e-01, 2.04088234e+00,-1.56178376e+01]]

def get_mag_error(mag,i):
    pk = coefs[i]
    f = np.poly1d(pk)
    logmerr= f(mag[:,i])
    merr = 10**(logmerr)/2.
    return np.where(merr>0.3,0.3,merr)

####
def readCCOPA(filename, a, b):
    logging.debug('Starting helperFunctions.read_afterburner()')

    h=fits.open(filename)
    d=h[1].data
    d=d[a:b] # JCB: to divide and conquer

    indice = np.arange(a,b,1,dtype=int)

    ids = d['GID'][:]
    haloid = d['CID'][:]
    zed = d['redshift'][:] # New afterburner
    # zed = d['z_true'][:]   # true redshift

    g = d['mag'][:,0]
    gerr = d['magerr'][:,0]
    #gerr = get_mag_error(d['mag'],0)

    r = d['mag'][:,1]
    rerr = d['magerr'][:,1]
    # rerr = get_mag_error(d['mag'],1)

    i = d['mag'][:,2]
    ierr = d['magerr'][:,2]
    # ierr = get_mag_error(d['mag'],2)

    z = d['mag'][:,3]
    # zerr = get_mag_error(d['mag'],3)
    zerr = d['magerr'][:,3]

    #zed = d['Z_CLUSTER'] # Old afterburner
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'indices':indice,'id':ids,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}
        # 'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'P_RADIAL':P_RADIAL,'P_REDSHIFT':P_REDSHIFT,'P_MEMBER':P_MEMBER,\
        # 'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE
        

    logging.debug('Returning from helperFunctions.read_afterburner()')
    return inputDataDict
    
def readBuzzard(filename, a, b):
    logging.debug('Starting helperFunctions.read_afterburner()')

    h=fits.open(filename)
    d=h[1].data
    d=d[a:b] # JCB: to divide and conquer
    
    id = d['ID'][:]
    haloid = d['HALOID'][:]
    zed = d['Z'][:] # New afterburner

    DIST_TO_CENTER=d['RHALO'][:]

    GR_P_COLOR=np.ones_like(haloid)
    RI_P_COLOR=np.ones_like(haloid)
    IZ_P_COLOR=np.ones_like(haloid)
    
    P_RADIAL=np.ones_like(haloid)
    P_REDSHIFT=np.ones_like(haloid)
    P_MEMBER=np.ones_like(haloid)
    
    GRP_RED=np.ones_like(haloid)
    GRP_BLUE=np.ones_like(haloid)
    RIP_RED=np.ones_like(haloid)
    RIP_BLUE=np.ones_like(haloid)
    IZP_RED=np.ones_like(haloid)
    IZP_BLUE=np.ones_like(haloid)

    g = d['MAG_AUTO_G'][:]
    gerr = d['MAGERR_AUTO_G'][:]
    
    r = d['MAG_AUTO_R'][:]
    rerr = d['MAGERR_AUTO_R'][:]
    
    i = d['MAG_AUTO_I'][:]
    ierr = d['MAGERR_AUTO_I'][:]
    
    z = d['MAG_AUTO_Z'][:]
    zerr = d['MAGERR_AUTO_Z'][:]
    
    #zed = d['Z_CLUSTER'] # Old afterburner
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'P_RADIAL':P_RADIAL,'P_REDSHIFT':P_REDSHIFT,'P_MEMBER':P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    logging.debug('Returning from helperFunctions.read_afterburner()')

    return inputDataDict
####

def read_afterburner(filename, a, b):
    logging.debug('Starting helperFunctions.read_afterburner()')

    h=fits.open(filename)
    d=h[1].data
    d=d[a:b] # JCB: to divide and conquer
    id = d['COADD_OBJECTS_ID']
    haloid = d['HOST_HALOID']
    GR_P_COLOR=d['GR_P_COLOR']
    RI_P_COLOR=d['RI_P_COLOR']
    IZ_P_COLOR=d['IZ_P_COLOR']
    #GR_P_MEMBER=d['GR_P_MEMBER']
    #RI_P_MEMBER=d['RI_P_MEMBER']
    #IZ_P_MEMBER=d['IZ_P_MEMBER']
    P_RADIAL=d['P_RADIAL']
    P_REDSHIFT=d['P_REDSHIFT']
    P_MEMBER=d['P_MEMBER']
    DIST_TO_CENTER=d['DIST_TO_CENTER']
    GRP_RED=d['GRP_RED']
    GRP_BLUE=d['GRP_BLUE']
    RIP_RED=d['RIP_RED']
    RIP_BLUE=d['RIP_BLUE']
    IZP_RED=d['IZP_RED']
    IZP_BLUE=d['IZP_BLUE']
    g = d['MAG_AUTO_G']
    gerr = d['MAGERR_AUTO_G']
    r = d['MAG_AUTO_R']
    rerr = d['MAGERR_AUTO_R']
    i = d['MAG_AUTO_I']
    ierr = d['MAGERR_AUTO_I']
    z = d['MAG_AUTO_Z']
    zerr = d['MAGERR_AUTO_Z']
    zed = d['HOST_REDSHIFT'] # New afterburner
    #zed = d['Z_CLUSTER'] # Old afterburner
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5

    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'P_RADIAL':P_RADIAL,'P_REDSHIFT':P_REDSHIFT,'P_MEMBER':P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    logging.debug('Returning from helperFunctions.read_afterburner()')

    return inputDataDict


def read_fits_sdss(filename):
    h=fits.open(filename)
    d=h[1].data
    id = d['COADD_OBJECTS_ID']
    haloid = d['HOST_HALOID']
    GR_P_COLOR=d['GR_P_COLOR']
    RI_P_COLOR=d['RI_P_COLOR']
    IZ_P_COLOR=d['IZ_P_COLOR']
    GR_P_MEMBER=d['GR_P_MEMBER']
    RI_P_MEMBER=d['RI_P_MEMBER']
    IZ_P_MEMBER=d['IZ_P_MEMBER']
    DIST_TO_CENTER=d['DIST_TO_CENTER']
    GRP_RED=d['GRP_RED']
    GRP_BLUE=d['GRP_BLUE']
    RIP_RED=d['RIP_RED']
    RIP_BLUE=d['RIP_BLUE']
    IZP_RED=d['IZP_RED']
    IZP_BLUE=d['IZP_BLUE']
    u = d['DERED_U']
    uerr = d['ERR_U']
    g = d['DERED_G_1']
    gerr = d['ERR_G']
    r = d['DERED_R_1']
    rerr = d['ERR_R']
    i = d['DERED_I_1']
    ierr = d['ERR_I']
    z = d['DERED_Z_1']
    zerr = d['ERR_Z']
    zed = d['Z']
    ugerr = (gerr**2+uerr**2)**0.5
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'haloid':haloid,'ug':u-g,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'ugerr':ugerr,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'GR_P_MEMBER':GR_P_MEMBER,'RI_P_MEMBER':RI_P_MEMBER,'IZ_P_MEMBER':IZ_P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}
         
    return inputDataDict

def read_fits_smassonly(filename):
    h=fits.open(filename)
    d=h[1].data
    id = d['ID']
    g = d['MAG_MOF_G']
    gerr = d['MAGERR_G']
    r = d['MAG_MOF_R']
    rerr = d['MAGERR_R']
    i = d['MAG_MOF_I']
    ierr = d['MAGERR_I']
    z = d['MAG_MOF_Z']
    zerr = d['MAGERR_Z']
    zed = d['MEAN_Z']
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict


def read_fits_redmagic(filename):
    h=fits.open(filename)
    d=h[1].data
    id = d['COADD_OBJECTS_ID']
    g = d['MODEL_MAG'][:,0]
    gerr =  d['MODEL_MAGERR'][:,0]
    r =  d['MODEL_MAG'][:,1]
    rerr = d['MODEL_MAGERR'][:,1]
    i =  d['MODEL_MAG'][:,2]
    ierr = d['MODEL_MAGERR'][:,2]
    z =  d['MODEL_MAG'][:,3]
    zerr = d['MODEL_MAGERR'][:,3]
    zed = d['ZREDMAGIC']
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict


def read(filename):
    d = np.genfromtxt(filename)
    id = d[:,0]
    g = d[:,1]
    gerr = d[:,2]
    r = d[:,3]
    rerr = d[:,4]
    i = d[:,5]
    ierr = d[:,6]
    z = d[:,7]
    zerr = d[:,8]
    zed = d[:,12]
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict


def read_GMM(filename):
    d = np.genfromtxt(filename)
    id = d[:,0]
    g = d[:,4]
    gerr = d[:,26]
    r = d[:,5]
    rerr = d[:,27]
    i = d[:,6]
    ierr = d[:,28]
    z = d[:,7]
    zerr = d[:,29]
    zed = d[:,30]
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    haloid = d[:,1]
    GR_P_COLOR=d[:,10]
    RI_P_COLOR=d[:,11]
    IZ_P_COLOR=d[:,12]
    GR_P_MEMBER=d[:,13]
    RI_P_MEMBER=d[:,14]
    IZ_P_MEMBER=d[:,15]
    DIST_TO_CENTER=d[:,16]
    GRP_RED=d[:,17]
    GRP_BLUE=d[:,18]
    RIP_RED=d[:,20]
    RIP_BLUE=d[:,21]
    IZP_RED=d[:,23]
    IZP_BLUE=d[:,24]
    inputDataDict = {'id':id,'haloid':haloid,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed, \
        'GR_P_COLOR':GR_P_COLOR,'RI_P_COLOR':RI_P_COLOR,'IZ_P_COLOR':IZ_P_COLOR,'GR_P_MEMBER':GR_P_MEMBER,'RI_P_MEMBER':RI_P_MEMBER,'IZ_P_MEMBER':IZ_P_MEMBER,\
        'DIST_TO_CENTER':DIST_TO_CENTER,'GRP_RED':GRP_RED,'GRP_BLUE':GRP_BLUE,'RIP_RED':RIP_RED,'RIP_BLUE':RIP_BLUE,'IZP_RED':IZP_RED,'IZP_BLUE':IZP_BLUE}

    #inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict

def read_delucia(filename):
    d = np.genfromtxt(filename)
    id = d[:,0]
    g = d[:,8]
    gerr = d[:,12]
    r = d[:,9]
    rerr = d[:,13]
    i = d[:,10]
    ierr = d[:,14]
    z = d[:,11]
    zerr = d[:,15]
    zed = d[:,4]
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict

def read_xmm(filename):
    d = np.genfromtxt(filename)
    id = d[:,0]
    g = d[:,4]
    gerr = d[:,32]
    r = d[:,5]
    rerr = d[:,33]
    i = d[:,6]
    ierr = d[:,34]
    z = d[:,7]
    zerr = d[:,35]
    zed = d[:,36]
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict

	
def read_chandra(filename):
    d = np.genfromtxt(filename)
    id = d[:,0]
    g = d[:,6]
    gerr = d[:,34]
    r = d[:,7]
    rerr = d[:,35]
    i = d[:,8]
    ierr = d[:,36]
    z = d[:,9]
    zerr = d[:,37]
    zed = d[:,38]
    grerr = (gerr**2+rerr**2)**0.5
    rierr = (rerr**2+ierr**2)**0.5
    izerr = (ierr**2+zerr**2)**0.5
    inputDataDict = {'id':id,'i':i,'ierr':ierr,'gr':g-r,'ri':r-i,'iz':i-z,'grerr':grerr,'rierr':rierr,'izerr':izerr,'zed':zed}

    return inputDataDict
