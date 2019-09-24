# !/usr/bin/env python
import numpy as np
from astropy.table import Table, vstack
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt

## -------------------------------
## auxiliary functions

def calcNbkg(area_bkg,pz_bkg,theta_bkg,nfatias=60):
    lista = np.empty(0,dtype=float)
    for ni in range(nfatias):
        mask = (theta_bkg <= (ni+1)*(360/nfatias))&(theta_bkg >= (ni)*(360/nfatias))
        Nbkg = pz_bkg[mask].sum()/(area_bkg/nfatias)
        # Nbkg = np.count_nonzero(pz_bkg[mask])/(area_bkg/nfatias)
        lista = np.append(lista,Nbkg)

    nbkg = np.median(lista)
    mad = np.std(lista)

    nl, nh = nbkg-1.5*mad,nbkg+1.5*mad
    sectors, = np.where((lista>nl)&(lista<nh))
    
    for ni in sectors:
        mask = (theta_bkg <= (ni+1)*(360/nfatias))&(theta_bkg >= (ni)*(360/nfatias))
        Nbkg = pz_bkg[mask].sum()/(area_bkg/nfatias)
        # Nbkg = np.count_nonzero(pz_bkg[mask])/(area_bkg/nfatias)
        lista = np.append(lista,Nbkg)

    nbkg = np.median(lista)
    mad = np.std(lista)

    nl, nh = nbkg-2*mad,nbkg+2*mad
    sectors, = np.where((lista>nl)&(lista<nh))

    # nume = np.random.randn()
    # plt.clf()
    # plt.hist(lista,bins=20)
    # plt.axvline(nbkg,color='r')
    # plt.savefig("bla_%.3f.png"%(nume))
    return nbkg, sectors

def getDensityBkg(all_gal,theta,r_in=6,r_out=8,nfatias=60):
    ## get bkg
    bkgMask = (all_gal['Bkg']==True)
    
    pz = all_gal['PDFz']
    area_bkg = np.pi*( (r_out)**2 - (r_in)**2 )
    
    nbkg, sectors = calcNbkg(area_bkg,pz[bkgMask],theta[bkgMask],nfatias=nfatias)
    
    if (nbkg<0.1):
        print('Background subtraction failure')
        print('decreasing the inner and outer radius ring to 4Mpc and 8Mpc')
        bkgMask = (all_gal['R']>=4)&(all_gal['R']<=8)
        area_bkg = np.pi*( (8)**2 - (4)**2 )
        
        nbkg, sectors = calcNbkg(area_bkg,pz[bkgMask],theta[bkgMask],nfatias=48)
    
    idx_gal = np.empty(0,dtype=int)
    for ni in sectors:
        w, = np.where( (theta <= (ni+1)*(360/nfatias)) & (theta >= (ni)*(360/nfatias)) & (all_gal['Bkg'] == True) )
        idx_gal = np.append(idx_gal,w)
        
    return nbkg, idx_gal

def calcTheta(ra,dec,ra_c,dec_c):
    deltaX, deltaY = (ra-ra_c),(dec-dec_c)
    theta = np.degrees( np.arctan2(deltaY,deltaX) ) + 180
    return theta

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def plotRingBackground(gal,theta,maskBkg,name_cls):
    galaxies, = np.where(gal['R']<3)
    x, y = pol2cart(gal['R'],theta)

    plt.clf()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1,right=0.95,top=0.975,bottom=0.025)
    ax.set_aspect('equal')

    # ax.scatter(gal_bkg_bad['RA'],gal_bkg_bad['DEC'],color='lightgray',s=2)
    ax.scatter(x[galaxies],y[galaxies],color='r',s=2)
    ax.scatter(x[maskBkg],y[maskBkg],color='k',s=2)
    ax.set_xlabel(r'$\Delta X $ [Mpc]')
    ax.set_ylabel(r'$\Delta Y$ [Mpc]')
    plt.savefig(name_cls+'_ring_bkg.png')

def computeDensityBkg(gals,cat,r_in=6,r_out=8,nfatias=60,plot=False):
    ncls = len(cat)
    nbkg, nbkg_malLimited = [], []
    maskBkg = np.full(len(gals['Bkg']), False, dtype=bool)

    for idx in range(ncls):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        ra_c, dec_c = cat['RA'][idx], cat['DEC'][idx]
        magLim_i = cat['magLim'][idx,1] ### mi cut
        
        galIndices, = np.where(gals['CID']==cls_id)
        gal = gals[galIndices]

        mask = gal['mag'][:,2]<magLim_i
        gal_magLim = gal[ mask ]
        
        theta = calcTheta(gal['RA'],gal['DEC'],ra_c,dec_c)
        nbkg_i, bkgIndices = getDensityBkg(gal,theta,r_in=r_in,r_out=r_out,nfatias=nfatias)
        nbkg_j, _ = getDensityBkg(gal_magLim,theta[mask],r_in=r_in,r_out=r_out,nfatias=nfatias)

        ## Updating the background status
        maskBkg[galIndices[bkgIndices]] = True

        nbkg.append(nbkg_i)
        nbkg_malLimited.append(nbkg_j)

        if plot:
            plotRingBackground(gal,theta,bkgIndices,'./check/probRadial/Planck_%i'%(cls_id))

    return np.array(nbkg), np.array(nbkg_malLimited), maskBkg


if __name__ == '__main__':
    print('helper.py')
    print('author: Johnny H. Esteves')
    
    # time_gal = time()    
    # print('\n Teste')
    # root = '/home/johnny/Documents/IAG-USP/Master/catalogos/'
    # file_gal = root+"galaxias/primary/y1a1_gold_bpz_mof_xmatcha_12Mpc.fits"
    # file_cls = root+"aglomerados/primary/xmatcha.fits"
    
    # columnsLabelsCluster = ['MEM_MATCH_ID','Xra','Xdec','redshift','r500']
    # columnsLabelsGalaxy = ['COADD_OBJECTS_ID','RA','DEC','MEDIAN_Z',
    #                        'mg','mr','mi','mz', 'FLAGS_GOLD','SIGMA_Z',
    #                        'mg_err','mr_err','mi_err','mz_err']

    # idx = np.arange(0,10,1,dtype=int)
    # clusters = readClusterCat(file_cls, idx=idx, massProxy=True,
	# 								 colNames=columnsLabelsCluster)

    # # gal = readGalaxyCat(file_gal, clusters, radius=12, colNames=columnsLabelsGalaxy)
    # gal = queryGalaxyCat(clusters, radius=12, zrange=(0.01,1.), Nflag=0, HEALPix=True)
    # # gal.write('galTest')
    # end_time = time()-time_gal

    # print('Time:',end_time)