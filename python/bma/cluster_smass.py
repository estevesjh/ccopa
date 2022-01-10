import numpy as np
import h5py
from joblib import Parallel, delayed
from astropy.table import Table, join

def compute_mu_star_true(fname,run,ngals=True,nCores=12):
    fmaster = h5py.File(fname,'a')

    cluster = fmaster['clusters/copa/%s'%run]
    cid     = cluster['CID'][:]

    cidx = fmaster['members/copa/%s/CID'%run][:]    
    midx = fmaster['members/copa/%s/mid'%run][:]

    tcid = fmaster['members/main/CID' ][:]    
    tmid = fmaster['members/main/mid' ][:]
    tmem = fmaster['members/main/True'][:]
    
    if run in fmaster['members/bma/'].keys():
        path = 'members/bma/%s/'%run
        bmid = fmaster['members/bma/indices/%s'%run][:]

    else:
        path = 'members/bma/'
        bmid = fmaster[path+'mid'][:]
        
    bcid = fmaster[path+'CID'][:]
    mass = fmaster[path+'mass'][:]

    fmaster.close()

    # print 'Matching BMA with Copa output'
    idx = np.sort(midx)
    copa = Table([midx,cidx],names=['mid','CID'])
    main = Table([tmid[idx],tcid[idx],tmem[idx]],names=['mid','CID','True'])
    bma  = Table([bmid,bcid,mass],names=['mid','CID','mass'])
    
    if (len(copa)>0)&(len(main)>0)&(len(bma)>0):
        toto = join(copa,main)
        match= join(toto,bma)

        tm    = np.where(match['True'],1.,0.)
        mass  = match['mass']

        keys  = list(chunks_id(match['CID'],cid))
        ntrue = np.array([np.sum(tm[ix]) for ix in keys])

        out   = Parallel(n_jobs=nCores)(delayed(_compute_muStar)(tm[ix],mass[ix]) for ix in keys)
        mu_star     = np.array([res[0] for res in out])
        mu_star_err = np.array([res[1] for res in out])
    else:
       mu_star = np.zeros(1)
       mu_star_err = np.zeros(1)
       ntrue   = mu_star

    # res   = np.array([_compute_muStar(tm[ix],mass[ix]) for ix in keys])
    # mu_star     = res[:,0]
    # mu_star_err = res[:,1]

    # print 'mu star output size:', mu_star.size

    fmaster = h5py.File(fname,'a')
    cluster = fmaster['clusters/copa/%s'%run]
    if 'MU_TRUE' not in cluster.keys():
        cluster.create_dataset('MU_TRUE',data=mu_star)
        cluster.create_dataset('MU_TRUE_ERR_JK',data=mu_star_err)
    else:
        cluster['MU_TRUE'][:] = mu_star
        cluster['MU_TRUE_ERR_JK'][:] = mu_star_err
    
    if ngals:
        if 'Ngals_true' not in cluster.keys():
            cluster.create_dataset('Ngals_true',data=ntrue)
        else:
            cluster['Ngals_true'][:] = ntrue

    fmaster.close()


def compute_mu_star(fname,run,nCores=12):
    fmaster = h5py.File(fname,'a')

    cluster = fmaster['clusters/copa/%s'%run]
    cid     = np.unique(cluster['CID'][:])

    cidx = fmaster['members/copa/%s/CID'%run][:]    
    midx = fmaster['members/copa/%s/mid'%run][:]
    pmem = fmaster['members/copa/%s/Pmem'%run][:]
    
    tcid = fmaster['members/main/CID' ][:]    
    tmid = fmaster['members/main/mid' ][:]
    tmem = fmaster['members/main/True'][:]
    
    if run in fmaster['members/bma/'].keys():
        path = 'members/bma/%s/'%run
    else:
        path = 'members/bma/'

    bmid = fmaster[path+'mid'][:]
    bcid = fmaster[path+'CID'][:]
    mass = fmaster[path+'mass'][:]
    fmaster.close()

    # print 'Matching BMA with Copa output'
    idx = np.sort(midx)
    copa = Table([midx,cidx,pmem],names=['mid','CID','Pmem'])
    main = Table([tmid[idx],tcid[idx],tmem[idx]],names=['mid','CID','True'])
    bma  = Table([bmid,bcid,mass],names=['mid','CID','mass'])
    
    if (len(copa)>0)&(len(main)>0)&(len(bma)>0):
        toto = join(copa,main)
        match= join(toto,bma)

        mass  = match['mass']
        pmem  = match['Pmem']

        keys  = list(chunks_id(match['CID'],cid))

        out   = Parallel(n_jobs=nCores)(delayed(_compute_muStar)(pmem[ix],mass[ix]) for ix in keys)
        mu_star     = np.array([res[0] for res in out])
        mu_star_err = np.array([res[1] for res in out])
    else:
       mu_star = np.empty(0)
       mu_star_err = np.empty(0)

    # res   = np.array([_compute_muStar(nbma['Pmem'][ix],nbma['mass'][ix]) for ix in keys])
    # mu_star     = res[:,0]
    # mu_star_err = res[:,1]

    # print 'mu star output size:', mu_star.size

    fmaster = h5py.File(fname,'a')
    cluster = fmaster['clusters/copa/%s'%run]

    if 'MU' not in cluster.keys():
        cluster.create_dataset('MU',data=mu_star)
        cluster.create_dataset('MU_ERR_JK',data=mu_star_err)
    else:
        del cluster['MU']
        del cluster['MU_ERR_JK']
        cluster.create_dataset('MU',data=mu_star)
        cluster.create_dataset('MU_ERR_JK',data=mu_star_err)
    fmaster.close()


def _compute_muStar(pmem,mass):
    if len(pmem)==0:
        return np.array([np.nan,np.nan])
    else:
        linear_mass_weight = pmem*10**mass
        mu_star = (linear_mass_weight.sum())/10.**10.
        mu_star_err_jk = (jackknife_var(linear_mass_weight, lambda_star_jk))**0.5
        return np.array([mu_star,mu_star_err_jk])

def chunks_id(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)

def jackknife_var(x, func):
    """Jackknife estiamte of the variance of the estimator func."""
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return (n-1)/(n + 0.0) * np.sum((func(x[idx!=i]) - j_est)**2.0 for i in range(n))

def lambda_star_jk(weightedmass):
    return np.nansum(weightedmass)/10.**10.

def delete_group(fname,path):
    try:
        fmaster = h5py.File(fname,'a')
        group   = fmaster[path]
        cols    = group.keys()
        if len(cols)>0:
            for col in cols: del group[col]
        fmaster.close()
    except:
        fmaster.close()
        print('Error: failed to delete group %s'%path)
        return