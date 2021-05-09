##################LOCAL BACKGROUND SUBTRACTION##################
import numpy as np
#import pyfits
from fitsio import FITS
import fitsio
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import mixture
import random
import cosmolopy.distance as cd
import cosmolopy.density
import esutil
import healpy as hp
import glob
from scipy import spatial,integrate,optimize
from scipy.interpolate import interp1d
from scipy.special import erf
from collections import Counter
from astropy.table import Table, vstack
from astropy.io import fits as pyfits
import os
import configparser as ConfigParser
#import line_profiler

from joblib import Parallel, delayed
from time import time

def old_memb(cat,gal,member_outfile=None,cluster_outfile=None,dataset='chandra',sigma_z=0.05,zfile=None,r_aper_model='hod',zmin_cls=0.0,zmax_cls=2.,zmin_gal=0.001,zmax_gal=1.1,p_lim=0):
    ## Get Label Columns
    # if dataset=='redmapper' or dataset=='mock':
    ra_label, dec_label, z_label = 'RA', 'DEC', 'z_lambda'
    
    if dataset=='chandra':
        ra_label, dec_label, z_label = 'RA', 'DEC', 'redshift'
        # ra_label, dec_label, z_label = 'Xra', 'Xdec', 'redshift'

    if dataset=='copa':
        ra_label, dec_label, z_label = 'RA', 'DEC', 'z_lambda'
        g,c = copa_to_old_memb_columns(gal,cat)
        
    ## Cluster Variables
    rac=c[ra_label][:]
    decc=c[dec_label][:]
    cid = c['MEM_MATCH_ID'][:]
    ix=rac>180; rac[ix]=rac[ix]-360

    # if dataset=='redmapper':
    z=c[z_label][:]
    
    ## Galaxy Variables
    rag1=g['RA'][:]
    ix=rag1>180; rag1[ix]=rag1[ix]-360

    decg1=g['DEC'][:]
    zg1=g['MEDIAN_Z'][:]
    AMAG=g['Mr'][:]
    if 'dmag' in g.colnames:
        DMAG = g['dmag'][:]

    mult_niter=g['MULT_NITER_MODEL'][:]
    flags_gold=g['FLAGS_GOLD'][:]
    galid=g['COADD_OBJECTS_ID'][:]

    magg=g['MAG_AUTO_G'][:]
    magr=g['MAG_AUTO_R'][:]
    magi=g['MAG_AUTO_I'][:]
    magz=g['MAG_AUTO_Z'][:]

    if dataset=='mock':
        Ngals_input = findMockGals(galid,cid)
        ngals = Ngals_input
        print 'Ngals input:', Ngals_input

    ########### make cuts ###########
    new_rac=rac
    new_decc = decc
    iy=np.argsort(new_rac)

    ra1=new_rac.min()-0.1
    ra2=new_rac.max()+0.1
    dec1=new_decc.min()-0.1
    dec2=new_decc.max()+0.1

    #w, = np.where((z>zmin) & (z<zmax) & (rac>=ra1) & (rac<=ra2) & (decc>=dec1) & (decc<=dec2) & (ngals!=0))
    w, = np.where((z>zmin_cls) & (z<zmax_cls) & (rac>=ra1) & (rac<=ra2) & (decc>=dec1) & (decc<=dec2) )#& (ngals>=70)) # cut on >9 members per MSS

    c1=c[w]
    rac=c1[ra_label][:]
    ix=rac>180; rac[ix]=rac[ix]-360 #JCB
    decc=c1[dec_label][:]
    
    ra1=min(rac)-1
    ra2=max(rac)+1
    dec1=min(decc)-1
    dec2=max(decc)+1
    w,=np.where(rag1>ra1)
    w2,=np.where(rag1[w]<ra2)
    w=w[w2]

    w2,=np.where((decg1[w]>dec1) & (decg1[w]<dec2))
    w=w[w2]

    #'crazy color' cut - don't use galaxies with colors less than -1 or greater than 4
    crazy1=-1
    crazy2=4 #2.5
    gr=magg[w]-magr[w]
    ri=magr[w]-magi[w]
    iz=magi[w]-magz[w]
    
    # w2, = np.where((zg1[w]>zmin_gal) & (zg1[w]<zmax_gal) & (mult_niter[w]>0) & (flags_gold[w]==0) & (AMAG[w]<=-19.5) & (gr>crazy1) & (gr<crazy2) & (ri>crazy1) & (ri<crazy2) & (iz>crazy1) & (iz<crazy2))
    
    # mag cut - line added on Oct 6, 2020
    w2, = np.where((zg1[w]>zmin_gal) & (zg1[w]<zmax_gal) & (mult_niter[w]>0) & (flags_gold[w]==0) & (DMAG[w]<=0.) & (gr>crazy1) & (gr<crazy2) & (ri>crazy1) & (ri<crazy2) & (iz>crazy1) & (iz<crazy2))
    
    w=w[w2]
    g1=g[w]
    
    print 'total clusters: ',len(c1)
    print 'total galaxies: ',len(g1)        
    print 

    cid=c1['MEM_MATCH_ID'][:]
    rac=c1[ra_label][:]
    ix=rac>180; rac[ix]=rac[ix]-360
    decc=c1[dec_label][:]
    z=c1[z_label][:]
    
    if dataset=='redmapper':
        zcl_err=c1['z_lambda_e'][:]
        NGALS=c1['LAMBDA_CHISQ'][:]
        
    if dataset=='mock':
        Ngals_input = findMockGals(galid,cid)
        print 'Ngals input:', Ngals_input

    rag=g1['RA'][:]
    ix=rag1>180; rag1[ix]=rag1[ix]-360
    decg=g1['DEC'][:]
    zg=g1['MEDIAN_Z'][:]
    zgerr=g1['Z_SIGMA'][:]
    galid=g1['COADD_OBJECTS_ID'][:]
    amagr=g1['Mr'][:]
    # kcorr=g1['Kir'][:]
    # gr0=g1['gr0'][:]

    magg=g1['MAG_AUTO_G'][:]
    magr=g1['MAG_AUTO_R'][:]
    magi=g1['MAG_AUTO_I'][:]
    magz=g1['MAG_AUTO_Z'][:]
    magerr_g=g1['MAGERR_AUTO_G'][:]
    magerr_r=g1['MAGERR_AUTO_R'][:]
    magerr_i=g1['MAGERR_AUTO_I'][:]
    magerr_z=g1['MAGERR_AUTO_Z'][:]

    if dataset=='mock':
        trueMembers = g1['True']
    else:
        trueMembers = np.zeros(len(g1),dtype=np.int)

    ###########BACKGROUND GALAXY DENSITY CALCULATION###############
    print
    print 'calculating background densities'

    #unique_galid,inds=np.unique(galid,return_index=True)
    #galid=list(galid)
    #zgtemp=zg[inds]
    cosmo={'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
    cosmo=cd.set_omega_k_0(cosmo)

    if zfile is not None:
        zmedian,zsigma=np.loadtxt(zfile,unpack=True)
        zfunc=interp1d(zmedian,zsigma)
    else:
        zfunc = lambda z: sigma_z*(1+z)

    print 'calculating local background galaxy densities'
    central_ra=rac
    central_dec=decc
    central_z=z
    cluster_id=cid
    galid=np.array(galid)

    ang_diam_dist=cd.angular_diameter_distance(central_z,z0=0,**cosmo)

    indices_into_galaxies,indices_into_clusters=annulus_match(rac,decc,ang_diam_dist,rag,decg,r_in=4.,r_out=6.)

    data=make_annuli_quantities(indices_into_galaxies,indices_into_clusters,galid,rag,decg,zg,zgerr,magg,magr,magi,magz)

    bgID = data['galaxyID']
    gmag=data['gmag']
    rmag=data['rmag']
    imag=data['imag']
    zmag=data['zmag']
    z_gal=data['zg']
    z_gal_err=data['zgerr']
    host_id=data['host_id']

    bg_probz,bg_total=background_photoz_probs(z,host_id,z_gal,z_gal_err,zfunc)
    bg_gal_density=background_galaxy_density(bg_total,4.,6.)

    ###########CALCULATING NGALS PER CLUSTER#############
    print 
    print 'matching galaxies within test radii'


    rmax=3.0  #maximum test radius in mpc. rmax will always be included as a test radius regardless of rmin,step
    rmin=0.1  #minimum test radius in mpc. rmin always included as test radius
    step=0.1  #step size, stepping from rmin to rmax, in mpc
    radii=np.r_[rmin:rmax:step,rmax]

    totalgals=[]
    backgroundgals=[]
    ngals=[]
    density=[]
    background_dense=[]
    #backgroundgals2=[]
    #ngals2=[]
    #density2=[]
    #background_dense2=[]
    vols=[]
    zmatch=[]
    depth=10
    h=esutil.htm.HTM(depth)
    for j in radii:
        degrees=(360/(2*np.pi))*(float(j)/ang_diam_dist) #convert radii to angular sizes
        #match cluster centers to surrounding galaxies
        m1,m2,adist=h.match(central_ra,central_dec,rag,decg,radius=degrees,maxmatch=0)
        m1uniq0=np.unique(m1)#find unique clusters (m1 has 1 cluster index for each matched galaxy)
        m1uniq=np.arange(central_ra.size).astype(int)
        clusterid=cluster_id[m1uniq]
        truez=central_z[m1uniq]
        window=zfunc(truez)
        zmin=truez-1.5*window#0.5
        zmax=truez+1.5*window#0.5
        total=[]
        totalerr=[]
        probz=[]
        for x in range(len(m1uniq)): #get the total number of matched galaxies for each cluster
            w0=np.where(m1==m1uniq[x]) #find points where gals matched to specific cluster
            w=m2[w0]
            membz=zg[w]
            membzerr=zgerr[w]
            intmin=zmin[x] #Integration window minimum
            intmax=zmax[x] #Integration window maximum
            zpts,zstep=np.linspace(intmin,intmax,20,retstep=True) #split redshift window for approximation
            area=[]
            for i in range(len(zpts)-1): #approximate integral using trapezoidal riemann sum
                zpt1=zpts[i] #zpt1,zpt2 are left,right points for trapezoid, respectively
                zpt2=zpts[i+1]
                gauss1=gaussian(zpt1,membz,membzerr) #gauss1/2 are gaussian values for left,right points, respectively
                gauss2=gaussian(zpt2,membz,membzerr)
                area1=((gauss1+gauss2)/2.)*zstep
                area.append(area1)
            area=np.array(area)
            arflip=area.swapaxes(0,1)
            prob=np.sum(arflip,axis=1)
            # prob = np.ones_like(membz)

            probz.append(prob)
            total.append(np.sum(prob))
        if 'maskfrac' in globals():
            maskfrac1=maskfrac[m1uniq] #find fraction of the cluster that is masked
        else:
            maskfrac1=np.zeros_like(m1uniq) #set zero maskfraction if not provided
        mf=1-maskfrac1
        volume=(4./3.)*np.pi*(j**3)*(mf) #calculate total volume of cluster (assuming spherical with radius j)
        area=np.pi*(j**2)*mf
        total_density=total/area#volume #calculate total galaxy density
        ztemp=central_z[m1uniq]
        #find background densities for each redshift
        #background_density=background_func(ztemp)
        #BDW - local background subtraction
        background_density=bg_gal_density[m1uniq]
        background=background_density*area#*volume
        n=total-background #calculate background subtracted richness
        n_density=total_density-background_density
        #save it all
        totalgals.append(total)
        backgroundgals.append(background)
        background_dense.append(background_density)
        ngals.append(n)
        density.append(n_density)
        vols.append(volume)


    print
    print 'Making R200/M200 measurements'

    ngals=np.array(ngals)
    density=np.array(density)
    volume=np.array(vols)

    ####params=[11.6,12.45,1.0,12.25,-0.69]#parameters for mass conversion - see table 4 in Tinker paper
    params=[11.59,12.94,1.01,12.48,-0.69]#parameters for mass conversion - see table 4 in Tinker paper

    ngalsT=np.transpose(ngals)
    massT=hod_mass_z(ngalsT,z,params) #calculate mass given ngals (see above functions)
    mass=np.transpose(massT)

    mass_density=mass/volume
    test=ngals[0]
    ngals=ngals.swapaxes(0,1)
    density=density.swapaxes(0,1)
    mass_density=mass_density.swapaxes(0,1)
    mass=mass.swapaxes(0,1)
    background_dense=np.array(background_dense)
    background_dense=background_dense.swapaxes(0,1)
    rho_crit,rho_0=cosmolopy.density.cosmo_densities(**cosmo) #current critical density
    print 'Critical Density:',rho_crit
    pc=200*np.ones_like(radii)

    R200_measure=[]
    M200_measure=[]
    redshift=[]
    lambda_ngals=[]
    R_lambda=[]
    cidsave=[]
    rasave=[]
    decsave=[]
    N_back=[]
    X=200 #desired excess over critical density, ex. if X=200, calculates R/M200
    dX=1 #acceptance window around X
    interpradii=np.append([0],radii)
    j=0
    for i in range(0,test.size):
        cluster=ngals[i]
        dense=density[i]
        clustermass=mass[i]
        massdense=mass_density[i]
        x=clusterid[i]
        cidsave.append(x)
        #if 'NGALS' in globals(): # JCB
        if dataset=='redmapper':
            c_ngals=NGALS[np.where(cid==x)]
            if c_ngals.size == 0:
                c_ngals=np.array([0])
            c_ngals=float(c_ngals)
            lambda_ngals.append(c_ngals)
        if 'lambda_r' in globals():
            c_r=lambda_r[np.where(cid==x)]
            if c_r.size==0:
                c_r=np.array([0])
            c_r=float(c_r)
            R_lambda.append(c_r)
        c_z=truez[i]
        if c_z.size==0:
            c_z=o
        c_z=float(c_z)
        redshift.append(c_z)
        c_ra=rac[np.where(cid==x)]
        if c_ra.size==0:
            c_ra=np.array([0])
        c_ra=float(c_ra)
        rasave.append(c_ra)
        c_dec=decc[np.where(cid==x)]
        if c_dec.size==0:
            c_dec=np.array([0])
        c_dec=float(c_dec)
        decsave.append(c_dec)
    #
        backdense=background_dense[i]
        backdense1=float(backdense[0]) #background density for this cluster
    #
        critdense1=crit_density(rho_crit,c_z,0.23,0.77)
        critdense=critdense1*np.ones_like(radii)
        ratio=massdense/critdense
    #
        f=interp1d(radii,ratio)
        radii_new=np.linspace(rmin,rmax,10000)
        ratio_new=f(radii_new)
        r200m=radii_new[np.where((ratio_new>=X-dX)&(ratio_new<=X+dX))] #find possible r200s within acceptance range
        if r200m.size > 0:
            r200m=np.mean(r200m) #mean of all possible r200s is measured r200
        else:
            r200m=1. #bogus r200=0 if nothing within acceptance range
            print 'bad cluster:',x,'ratio min/max:',min(ratio_new),max(ratio_new)
            j=j+1
    #
        R200_measure.append(r200m)
    #
        vol=(4*np.pi/3.)*(r200m**3) #cluster volume inside R200
        nback=backdense1*vol #expected number of background galaxies within R200
        N_back.append(nback)
        interpmass=np.append([0],clustermass)
        interp=interp1d(interpradii,interpmass)
        m200m=interp(r200m) #measured M200 (mass within R200)
        M200_measure.append(m200m)


    print 'Total bad clusters:',j

    if r_aper_model=='hod':
        R200_measure=np.array(R200_measure)
    else:
        R200_measure=c1['R200_true']*0.7#np.array(R200_measure)
    #

    M200_measure=np.array(M200_measure)
    #if 'lambda_ngals' in globals(): # JCB
    if dataset=='redmapper':
        lambda_ngals=np.array(lambda_ngals)
    #if 'R_lambda' in globals():
    if dataset=='chandra':
        R_lambda=np.array(R_lambda)
    cidsave=np.array(cidsave)
    redshift=np.array(redshift)
    rasave=np.array(rasave)
    decsave=np.array(decsave)
    N_back=np.array(N_back)

    ######SELECTING MEMBER GALAXIES WITHIN R200######
    print 'calculating radial probabilities'

    #member file stuff
    true_members = [] 
    galaxyID=[]
    hostID=[]
    P_radial=[]
    angular_dist=[]
    P_redshift=[]
    hostR200=[]
    hostM200=[]
    hostN200=[]
    galaxyRA=[]
    galaxyDEC=[]
    galaxyZ=[]
    galZerr=[]
    galmagG=[]
    galmagR=[]
    galmagI=[]
    galmagZ=[]
    galmagGerr=[]
    galmagRerr=[]
    galmagIerr=[]
    galmagZerr=[]
    galamagR=[]
    #galmult=[]
    #galspread=[]
    #galkcorr=[]
    #galgr0=[]
    #cluster file stuff
    R200=[]
    M200=[]
    N200=[]
    N_background=[]
    cluster_ID=[]
    cluster_RA=[]
    cluster_DEC=[]
    cluster_Z=[]
    constants=[]
    n_total=[]
    flags=np.zeros_like(m1uniq)
    for i in range(0,len(m1uniq)):
        #first do all the cluster stuff
    #    ind=m1uniq[i]
        cind=clusterid[i]
    #    cind=cidsave[i]
        cluster_ID.append(cind)
        hostr=R200_measure[i]
        R200.append(hostr)
        hostm=M200_measure[i]
        M200.append(hostm)
        hostback=N_back[i]
        N_background.append(hostback)
        cluster_RA.append(central_ra[i])
        cluster_DEC.append(central_dec[i])
        cluster_Z.append(central_z[i])
        #now do all the galaxy stuff
        match0=np.where(m1==m1uniq[i])
        match=m2[match0]
        add=ang_diam_dist[i]
        angular=adist[match0] #angular distance to cluster center, degrees
        R=(2.*np.pi/360.)*angular*add #distance to cluster center in Mpc
        jx,=np.where(R<=hostr) #select only galaxies within R200 
    #Choosing only R<R200 prevents the values of the normalization constant
    #from taking on unphysical values. In rare cases where the constant
    #is still less than zero, the output file will flag it with flags=-1
        p_z=probz[i][jx]
        R=R[jx]
        ntot_z=p_z.sum()
        n_total.append(ntot_z)
        n_bg=bg_gal_density[i]*np.pi*hostr**2
        c=norm_constant(hostr,ntot_z,n_bg)
        if c<0:
            c = 0.
            if len(constants)>0: c=min(constants)
            flags[i]=-1
        constants.append(c)
        p_rad=radial_probability(R,hostr,ntot_z,bg_gal_density[i],c)
        #add in redshift probabilities
        pdist=p_rad*p_z
        n_p=np.sum(pdist)
        P_radial.append(p_rad)
        angular_dist.append(R) #save distance to cluster center in Mpc
        P_redshift.append(p_z)
        glxid=galid[match][jx]
        glxra=rag[match][jx]
        glxdec=decg[match][jx]
        glxz=zg[match][jx]
        glxmagg=magg[match][jx]
        glxmagr=magr[match][jx]
        glxmagi=magi[match][jx]
        glxmagz=magz[match][jx]
        glxmaggerr=magerr_g[match][jx]
        glxmagrerr=magerr_r[match][jx]
        glxmagierr=magerr_i[match][jx]
        glxmagzerr=magerr_z[match][jx]
        glxamagr=amagr[match][jx]
        trm=trueMembers[match][jx]
    #    glxspread=spread[match]
    #    glxmult=mult[match]
        glxzerr=zgerr[match][jx]
        #glxkcorr=kcorr[match][jx]
        #glxgr0=gr0[match][jx]
        galaxyID.append(glxid)
        tempID=cind*np.ones_like(glxid)
        hostID.append(tempID)
        tempR=hostr*np.ones_like(glxid)
        hostR200.append(tempR)
        tempM=hostm*np.ones_like(glxid)
        hostM200.append(tempM)
        tempn=n_p*np.ones_like(glxid)
        hostN200.append(tempn)
        galaxyRA.append(glxra)
        galaxyDEC.append(glxdec)
        galaxyZ.append(glxz)
        galZerr.append(glxzerr)
        galmagG.append(glxmagg)
        galmagR.append(glxmagr)
        galmagI.append(glxmagi)
        galmagZ.append(glxmagz)
        galmagGerr.append(glxmaggerr)
        galmagRerr.append(glxmagrerr)
        galmagIerr.append(glxmagierr)
        galmagZerr.append(glxmagzerr)
        galamagR.append(glxamagr)
        true_members.append(trm)
    #    galmult.append(glxmult)
    #    galspread.append(glxspread)
        #galkcorr.append(glxkcorr)
        #galgr0.append(glxgr0)

    galaxyID=np.array([item for sublist in galaxyID for item in sublist])
    galaxyRA=np.array([item for sublist in galaxyRA for item in sublist])
    galaxyDEC=np.array([item for sublist in galaxyDEC for item in sublist])
    galaxyZ=np.array([item for sublist in galaxyZ for item in sublist])
    hostID=np.array([item for sublist in hostID for item in sublist])
    P_radial=np.array([item for sublist in P_radial for item in sublist])
    P_redshift=np.array([item for sublist in P_redshift for item in sublist])
    angular_dist=np.array([item for sublist in angular_dist for item in sublist])
    hostR200=np.array([item for sublist in hostR200 for item in sublist])
    hostM200=np.array([item for sublist in hostM200 for item in sublist])
    hostN200=np.array([item for sublist in hostN200 for item in sublist])
    galZerr=np.array([item for sublist in galZerr for item in sublist])
    galmagG=np.array([item for sublist in galmagG for item in sublist])
    galmagR=np.array([item for sublist in galmagR for item in sublist])
    galmagI=np.array([item for sublist in galmagI for item in sublist])
    galmagZ=np.array([item for sublist in galmagZ for item in sublist])
    galmagGerr=np.array([item for sublist in galmagGerr for item in sublist])
    galmagRerr=np.array([item for sublist in galmagRerr for item in sublist])
    galmagIerr=np.array([item for sublist in galmagIerr for item in sublist])
    galmagZerr=np.array([item for sublist in galmagZerr for item in sublist])
    galamagR=np.array([item for sublist in galamagR for item in sublist])
    true_members=np.array([item for sublist in true_members for item in sublist])
    
    #galmult=np.array([item for sublist in galmult for item in sublist])
    #galspread=np.array([item for sublist in galspread for item in sublist])
    #galkcorr=np.array([item for sublist in galkcorr for item in sublist])
    #galgr0=np.array([item for sublist in galgr0 for item in sublist])
    constants=np.array(constants)
    n_total=np.array(n_total)

    mxp=1. #max(P_radial)
    P_radial=P_radial*(1./mxp)
    Pdist=P_radial*P_redshift
    Pmem = P_radial*P_redshift
    cluster_Z=np.array(cluster_Z)

    print
    print 'Masking Galaxies'

    ## Search for projected GC on a given radius(Mpc)
    ## Returns: column with the number of projected GC    

    Bg_FLAG,m1_bg,m2_bg = Bg_overlap(rac,decc,cluster_Z,rac,decc,cluster_Z,ang_diam_dist,r_in=4,r_out=6,dz=0.2)
    nOverlap, m1,m2 = overlap(rac,decc,cluster_Z,rac,decc,cluster_Z,ang_diam_dist,radius=6,dz=0.5)

    # Pmem = ProjectionEffects_Pmem(Pmem0,m1,m2,hostID,galaxyID,cluster_ID)
    # bg_probz = ProjectionEffects_Pmem(bg_probz,m1_bg,m2_bg,host_id,bgID,cluster_ID)
    ## There is no galaxyID for background galaxies. It should change the function

    print 
    print 'There are',np.count_nonzero(Bg_FLAG>0),' clusters with non proper local background'
    print 
    print 'calculating local background color distributions'

    gmag=data['gmag']
    rmag=data['rmag']
    imag=data['imag']
    zmag=data['zmag']
    z_gal=data['zg']
    z_gal_err=data['zgerr']
    host_id=data['host_id']

    gr_bg = [host_id,(gmag-rmag), rmag, bg_probz]
    ri_bg = [host_id,(rmag-imag), imag, bg_probz]
    iz_bg = [host_id,(imag-zmag), zmag, bg_probz]

    gr_hists,ri_hists,iz_hists=local_bg_histograms(ang_diam_dist,R200_measure,4.,6.,host_id,gmag,rmag,imag,zmag,bg_probz,n_total,bg_total,constants)

    scale_bg = doScale_bg(R200_measure,4.,6.,n_total,bg_total,constants)

    print 'max prad',max(P_radial)
    print 'min dist',min(angular_dist)

    bgtable=Table([gr_hists,ri_hists,iz_hists],names=['gr','ri','iz'])
    # bgtable.write(bghistfile,format='fits',overwrite=True)

    print 
    print 'calculating GMM probabilities'
    #treedata=zip(jimz,np.zeros_like(jimz))
    #tree=spatial.KDTree(treedata)

    zbin_edges,step=np.linspace(0.1,1.0,21,retstep=True)
    zbin_centers=zbin_edges[:-1]+(step/2.)

    z_treedata=zip(zbin_centers,np.zeros_like(zbin_centers))
    zhist_tree=spatial.KDTree(z_treedata)
    # cluster_Z=np.array(cluster_Z)

    gr=galmagG-galmagR
    ri=galmagR-galmagI
    iz=galmagI-galmagZ

    #pull mag_lim (0.4L*) vs redshift data
    # maglim_r, maglim_i, maglim_z = pullMagLim(magLim_file,cluster_Z,dm=3)
    # maglim_r, maglim_i, maglim_z = 24.3*np.ones_like(galmagR),23.50*np.ones_like(galmagI),22.9*np.ones_like(galmagZ) ## Flat cut
    # maglim_r, maglim_i, maglim_z = getMagLimVec(interp_magi,interp_ri,interp_iz,cluster_Z,dm=2)

    #Red Sequence evolution: (z,color_mean,slope,inter) 
    #The param is good up to z~0.6
    # gr_RSparam, ri_RSparam, iz_RSparam = pullRSparam(rsfile,cluster_Z)

    ## Mock Catalogs option: Follow the mean color evolution from the colorfile table
    #if dataset=='mock':
        # gr_RSparam, ri_RSparam, iz_RSparam = pullRSparam_mock(colorfile,rsfile,cluster_Z)

    #colorLabel=[['$r$','$g-r$'],['$i$','$r-i$'],['$z$','$i-z$']]

    #label=''
    ## GMM FIT
    # grinfo=GMM.fitFull(gr,galmagR,gr_hists,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_r,rs_mean=rs_mean_gr,Pmem_lim=0.0,color_label=colorLabel[0],tol=0.0000001)
    # grinfo=GMM.fitFullRS(gr,galmagR,gr_bg,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_r,param_rs=gr_RSparam,scale_vec=scale_bg,Pmem_lim=0.0,Nlabel=0,LabelSave=label,tol=0.0000001,plot=0)

    # riinfo=GMM.fitFull(ri,galmagI,ri_hists,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_i,rs_mean=rs_mean_ri,Pmem_lim=0.0,color_label=colorLabel[2],tol=0.0000001)
    # riinfo=GMM.fitFullRS(ri,galmagI,ri_bg,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_i,param_rs=ri_RSparam,scale_vec=scale_bg,Pmem_lim=0.0,Nlabel=1,LabelSave=label,tol=0.0000001,plot=0)

    # izinfo=GMM.fitFull(iz,galmagZ,iz_hists,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_z,rs_mean=rs_mean_iz,Pmem_lim=0.0,color_label=colorLabel[1],tol=0.0000001)
    # izinfo=GMM.fitFullRS(iz,galmagZ,iz_bg,Pmem,cluster_ID,cluster_Z,galaxyID,hostID,R200_measure,maglim_z,param_rs=iz_RSparam,scale_vec=scale_bg,Pmem_lim=0.0,Nlabel=2,LabelSave=label,tol=0.0000001)    


    # grmu_r=grinfo[0]
    # grmu_b=grinfo[1]
    # grsigma_r=grinfo[2]
    # grsigma_b=grinfo[3]
    # gralpha_r=grinfo[4]
    # gralpha_b=grinfo[5]
    # grconverged=grinfo[6]
    # grPred=grinfo[7]
    # grPblue=grinfo[8]
    # grPcolor=grinfo[9]
    # grprobgalid=grinfo[10]
    # grslope=grinfo[11]
    # gryint=grinfo[12]
    # gr_rsOffset=grinfo[-1]

    # rimu_r=riinfo[0]
    # rimu_b=riinfo[1]
    # risigma_r=riinfo[2]
    # risigma_b=riinfo[3]
    # rialpha_r=riinfo[4]
    # rialpha_b=riinfo[5]
    # riconverged=riinfo[6]
    # riPred=riinfo[7]
    # riPblue=riinfo[8]
    # riPcolor=riinfo[9]
    # riprobgalid=riinfo[10]
    # rislope=grinfo[11]
    # riyint=grinfo[12]
    # ri_rsOffset=riinfo[-1]

    # izmu_r=izinfo[0]
    # izmu_b=izinfo[1]
    # izsigma_r=izinfo[2]
    # izsigma_b=izinfo[3]
    # izalpha_r=izinfo[4]
    # izalpha_b=izinfo[5]
    # izconverged=izinfo[6]
    # izPred=izinfo[7]
    # izPblue=izinfo[8]
    # izPcolor=izinfo[9]
    # izprobgalid=izinfo[10]
    # izslope=grinfo[11]
    # izyint=grinfo[12]
    # iz_rsOffset=izinfo[-1]

    #Make cuts for galaxy membership probabilities 
    w,=np.where((Pmem>=p_lim))
    galid2=galaxyID[w]
    hostid2=hostID[w]
    galra2=galaxyRA[w]
    galdec2=galaxyDEC[w]
    galz2=galaxyZ[w]
    galzerr2=galZerr[w]
    galmagG2=galmagG[w]
    galmagR2=galmagR[w]
    galmagI2=galmagI[w]
    galmagZ2=galmagZ[w]
    galmagGerr2=galmagGerr[w]
    galmagRerr2=galmagRerr[w]
    galmagIerr2=galmagIerr[w]
    galmagZerr2=galmagZerr[w]
    galamagR2=galamagR[w]
    angular_dist2=angular_dist[w]
    true_members2=true_members[w]
    prad2=P_radial[w]
    pz2=P_redshift[w]
    #galgr02=galgr0[w]

    if dataset=='mock':
        Ngals_output = findMockGals(galid2,cluster_ID)
        missing = Ngals_input-Ngals_output
        Bg_FLAG = missing
        print missing,'were not deteceted as members'

    # gralpha_r.shape=(gralpha_r.size,)
    # gralpha_b.shape=(gralpha_b.size,)

    # grPbg=np.zeros_like(grPblue)

    # rialpha_r.shape=(rialpha_r.size,)
    # rialpha_b.shape=(rialpha_b.size,)    
    # izalpha_r.shape=(izalpha_r.size,)
    # izalpha_b.shape=(izalpha_b.size,)
    #assign host cluster redshift to each member galaxy
    zclust=zg[w]
    zclust_err=zgerr[w]
    for i in range(np.array(cluster_ID).size):
        ix,=np.where(hostid2==cluster_ID[i])
        zclust[ix]=z[i]
        if dataset=='redmapper':
            zclust_err[ix]=zcl_err[i] #JCB
    if dataset=='chandra':
        zclust_err=np.zeros_like(zclust)
    # ncounts_conv = (len(grconverged[grconverged=='False'])+len(riconverged[grconverged=='False'])+len(izconverged[grconverged=='False']))/3
    # print 'GMM convergence means is', ncounts_conv

    ######WRITING OUTPUT FILES######
    # grPmemb=grPcolor[w]*Pmem[w]
    # riPmemb=riPcolor[w]*Pmem[w]
    # izPmemb=izPcolor[w]*Pmem[w]
    #restPmemb=restpc2*prad2*pz2
    Pmemb=Pmem[w]
    # print 'sum(pr*pz*pc)',Pmemb.sum()

    Ngals = doNgals(cluster_ID,hostid2,Pmemb)

    # if dataset=='mock':
    #     dz = (g1['redshift']-g1['z_true'])/(1+g1['z_true'])
    #     trueMembers = g1['True']

    print 'writing member file'

    col1=pyfits.Column(name='COADD_OBJECTS_ID',format='K',array=galid2)
    # col2=pyfits.Column(name='HOST_HALOID',format='J',array=hostid2)
    col2=pyfits.Column(name='HOST_HALOID',format='K',array=hostid2)
    col3=pyfits.Column(name='RA',format='D',array=galra2)
    col4=pyfits.Column(name='DEC',format='D',array=galdec2)
    col5=pyfits.Column(name='ZP',format='E',array=galz2)
    col6=pyfits.Column(name='ZPE',format='E',array=galzerr2)
    col7=pyfits.Column(name='MAG_AUTO_G',format='E',array=galmagG2)
    col8=pyfits.Column(name='MAG_AUTO_R',format='E',array=galmagR2)
    col9=pyfits.Column(name='MAG_AUTO_I',format='E',array=galmagI2)
    col10=pyfits.Column(name='MAG_AUTO_Z',format='E',array=galmagZ2)
    col11=pyfits.Column(name='P_RADIAL',format='E',array=prad2)
    col12=pyfits.Column(name='P_REDSHIFT',format='E',array=pz2)
    # col13=pyfits.Column(name='GR_P_COLOR',format='E',array=grPcolor)
    # col14=pyfits.Column(name='RI_P_COLOR',format='E',array=riPcolor)
    # col15=pyfits.Column(name='IZ_P_COLOR',format='E',array=izPcolor)
    col16=pyfits.Column(name='P_MEMBER',format='E',array=Pmemb)
    # col17=pyfits.Column(name='RI_P_MEMBER',format='E',array=riPmemb)
    # col18=pyfits.Column(name='IZ_P_MEMBER',format='E',array=izPmemb)
    # col16=pyfits.Column(name='P_MEMBER',format='E',array=Pmemb)
    col19=pyfits.Column(name='AMAG_R',format='E',array=galamagR2)
    col20=pyfits.Column(name='DIST_TO_CENTER',format='E',array=angular_dist2)
    # col21=pyfits.Column(name='GRP_RED',format='E',array=grPred)
    # col22=pyfits.Column(name='GRP_BLUE',format='E',array=grPblue)
    # col23=pyfits.Column(name='GRP_BG',format='E',array=grPbg)
    # col24=pyfits.Column(name='RIP_RED',format='E',array=riPred)
    # col25=pyfits.Column(name='RIP_BLUE',format='E',array=riPblue)
    #col26=pyfits.Column(name='RIP_BG',format='E',array=riPbg)
    # col27=pyfits.Column(name='IZP_RED',format='E',array=izPred)
    # col28=pyfits.Column(name='IZP_BLUE',format='E',array=izPblue)
    # col29=pyfits.Column(name='GR_RS_OFFSET',format='E',array=gr_rsOffset)
    # col30=pyfits.Column(name='RI_RS_OFFSET',format='E',array=ri_rsOffset)
    # col31=pyfits.Column(name='IZ_RS_OFFSET',format='E',array=iz_rsOffset)
    #col29=pyfits.Column(name='IZP_BG',format='E',array=izPbg)
    #col30=pyfits.Column(name='RESTP_RED',format='E',array=restPred)
    #col31=pyfits.Column(name='RESTP_BLUE',format='E',array=restPblue)
    #col32=pyfits.Column(name='RESTP_BG',format='E',array=restPbg)
    #col33=pyfits.Column(name='REST_P_COLOR',format='E',array=restpc2)
    #col34=pyfits.Column(name='REST_P_MEMBER',format='E',array=restPmemb)
    #col35=pyfits.Column(name='GR0',format='E',array=galgr02)
    col36=pyfits.Column(name='HOST_REDSHIFT',format='E',array=zclust)
    #if 'lambda_ngals' in globals(): # JCB
    col37=pyfits.Column(name='HOST_REDSHIFT_ERR',format='E',array=zclust_err)
    col38=pyfits.Column(name='MAGERR_AUTO_G',format='E',array=galmagGerr2)
    col39=pyfits.Column(name='MAGERR_AUTO_R',format='E',array=galmagRerr2)
    col40=pyfits.Column(name='MAGERR_AUTO_I',format='E',array=galmagIerr2)
    col41=pyfits.Column(name='MAGERR_AUTO_Z',format='E',array=galmagZerr2)
    if dataset=='mock':
        col42=pyfits.Column(name='True',format='L',array=true_members2)
        cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col16,col19,col20,col36,col37,col38,col39,col40,col41,col42])
    else:
        cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col16,col19,col20,col36,col37,col38,col39,col40,col41])

    #cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40])
    # cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col24,col25,col27,col28,col36,col37,col38,col39,col40,col41])
    # cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col19,col20,col21,col22,col24,col25,col27,col28,col29,col30,col31,col36,col37,col38,col39,col40,col41])
    tbhdu=pyfits.BinTableHDU.from_columns(cols)
    mdata = Table(tbhdu.data)
    if member_outfile is not None:
        tbhdu.writeto(member_outfile,overwrite=True)

    print 'writing cluster file'

    R200=np.array(R200)
    M200=np.array(M200)
    N200=np.array(N200)
    N_background=np.array(N_background)
    cluster_ID=np.array(cluster_ID)
    cluster_RA=np.array(cluster_RA)
    cluster_DEC=np.array(cluster_DEC)
    cluster_Z=np.array(cluster_Z)

    # grsep=(grmu_r-grmu_b)/grsigma_r
    # risep=(rimu_r-rimu_b)/risigma_r
    # izsep=(izmu_r-izmu_b)/izsigma_r
    #restsep=(restmu_r-restmu_b)/restsigma_r

    # grw=np.where(grsep>=1.)
    # riw=np.where(risep>=1.)
    # izw=np.where(izsep>=1.)
    #restw=np.where(restsep>=1.)

    # grflag=np.zeros_like(grmu_r)
    # grflag[grw]=1
    # riflag=np.zeros_like(rimu_r)
    # riflag[riw]=1
    # izflag=np.zeros_like(izmu_r)
    # izflag[izw]=1
    #restflag=np.zeros_like(restmu_r)
    #restflag[restw]=1

    zeros=np.zeros_like(cluster_ID)

    # col1=pyfits.Column(name='MEM_MATCH_ID',format='J',array=cluster_ID)
    col1=pyfits.Column(name='MEM_MATCH_ID',format='K',array=cluster_ID)
    col2=pyfits.Column(name='RA',format='D',array=cluster_RA)
    col3=pyfits.Column(name='DEC',format='D',array=cluster_DEC)
    col4=pyfits.Column(name='Z',format='E',array=cluster_Z)
    col5=pyfits.Column(name='R200',format='E',array=R200)
    col6=pyfits.Column(name='M200',format='E',array=M200)
    col7=pyfits.Column(name='N200',format='E',array=Ngals)
    # col7=pyfits.Column(name='Nmem',format='E',array=Ngals)
    #if 'lambda_ngals' in globals(): # JCB
    if dataset=='redmapper':
        col8=pyfits.Column(name="LAMBDA_CHISQ",format='E',array=lambda_ngals)
    else:
        col8=pyfits.Column(name="LAMBDA_CHISQ",format='E',array=zeros)
    # col9=pyfits.Column(name='GR_SLOPE',format='E',array=grslope)
    # col10=pyfits.Column(name='GR_INTERCEPT',format='E',array=gryint)
    # col11=pyfits.Column(name='GRMU_R',format='E',array=grmu_r)
    # col12=pyfits.Column(name='GRMU_B',format='E',array=grmu_b)
    # col13=pyfits.Column(name='GRSIGMA_R',format='E',array=grsigma_r)
    # col14=pyfits.Column(name='GRSIGMA_B',format='E',array=grsigma_b)
    # col15=pyfits.Column(name='GRW_R',format='E',array=gralpha_r)
    # col16=pyfits.Column(name='GRW_B',format='E',array=gralpha_b)
    # col17=pyfits.Column(name='RI_SLOPE',format='E',array=rislope)
    # col18=pyfits.Column(name='RI_INTERCEPT',format='E',array=riyint)
    # col19=pyfits.Column(name='RIMU_R',format='E',array=rimu_r)
    # col20=pyfits.Column(name='RIMU_B',format='E',array=rimu_b)
    # col21=pyfits.Column(name='RISIGMA_R',format='E',array=risigma_r)
    # col22=pyfits.Column(name='RISIGMA_B',format='E',array=risigma_b)
    # col23=pyfits.Column(name='RIW_R',format='E',array=rialpha_r)
    # col24=pyfits.Column(name='RIW_B',format='E',array=rialpha_b)
    # # col25=pyfits.Column(name='GRMU_BG',format='E',array=grmu_bg)
    # # col26=pyfits.Column(name='GRSIGMA_BG',format='E',array=grsigma_bg)
    # # col27=pyfits.Column(name='GRW_BG',format='E',array=gralpha_bg)
    # #col28=pyfits.Column(name='RIMU_BG',format='E',array=rimu_bg)
    # #col29=pyfits.Column(name='RISIGMA_BG',format='E',array=risigma_bg)
    # #col30=pyfits.Column(name='RIW_BG',format='E',array=rialpha_bg)
    # col31=pyfits.Column(name='IZ_SLOPE',format='E',array=izslope)
    # col32=pyfits.Column(name='IZ_INTERCEPT',format='E',array=izyint)
    # col33=pyfits.Column(name='IZMU_R',format='E',array=izmu_r)
    # col34=pyfits.Column(name='IZMU_B',format='E',array=izmu_b)
    # col35=pyfits.Column(name='IZSIGMA_R',format='E',array=izsigma_r)
    # col36=pyfits.Column(name='IZSIGMA_B',format='E',array=izsigma_b)
    # col37=pyfits.Column(name='IZW_R',format='E',array=izalpha_r)
    # col38=pyfits.Column(name='IZW_B',format='E',array=izalpha_b)

    # col39=pyfits.Column(name='IZMU_BG',format='E',array=izmu_bg)
    #col40=pyfits.Column(name='IZSIGMA_BG',format='E',array=izsigma_bg)
    col41=pyfits.Column(name='DA',format='E',array=ang_diam_dist)
    col42=pyfits.Column(name='FLAGS',format='E',array=flags)
    #col43=pyfits.Column(name='RI_SEP_FLAG',format='L',array=riflag)
    #col44=pyfits.Column(name='IZ_SEP_FLAG',format='L',array=izflag)
    #col45=pyfits.Column(name='REST_SLOPE',format='E',array=restslope)
    #col46=pyfits.Column(name='REST_INTERCEPT',format='E',array=restyint)
    #col47=pyfits.Column(name='RESTMU_R',format='E',array=restmu_r)
    #col48=pyfits.Column(name='RESTMU_B',format='E',array=restmu_b)
    #col49=pyfits.Column(name='RESTMU_BG',format='E',array=restmu_bg)
    #col50=pyfits.Column(name='RESTSIGMA_R',format='E',array=restsigma_r)
    #col51=pyfits.Column(name='RESTSIGMA_B',format='E',array=restsigma_b)
    #col52=pyfits.Column(name='RESTSIGMA_BG',format='E',array=restsigma_bg)
    #col53=pyfits.Column(name='RESTW_R',format='E',array=restalpha_r)
    #col54=pyfits.Column(name='RESTW_B',format='E',array=restalpha_b)
    #col55=pyfits.Column(name='RESTW_BG',format='E',array=restalpha_bg)
    #col56=pyfits.Column(name='REST_SEP_FLAG',format='L',array=restflag)
    if dataset=='redmapper':
        col57=pyfits.Column(name='Z_ERR',format='E',array=zcl_err) #JCBB
    else:
        col57=pyfits.Column(name='Z_ERR',format='E',array=zeros) #JCBB
    col58=pyfits.Column(name='MOCK_FLAG',format='K',array=Bg_FLAG) #JCBB
    #col58=pyfits.Column(name='BCKGD',format='E',array=bg_gal_density)


    cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col41,col42])
    #cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col31,col32,col33,col34,col35,col36,col37,col38,col42]) #JCBB
    #cols=pyfits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57])
    tbhdu=pyfits.BinTableHDU.from_columns(cols)
    cdata = Table(tbhdu.data)
    if cluster_outfile is not None:
        tbhdu.writeto(cluster_outfile,overwrite=True)

    if dataset=='copa':
        mdata, cdata = old_memb_to_copa_columns(mdata,cdata)

    print 'success!! and there was great rejoicing!'
    return mdata, cdata


def getConfig(section, item, boolean=False,
		userConfigFile="/home/s1/jesteves/backup/Memb-assign/membassign_config_dc2.ini"):

        # userConfigFile="/home/s1/jesteves/Memb-assign/MembAssign_Config.ini"):

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	# # if config item not found, raise log warning
	# if (not configFile.has_option(section, item)):
    #     print('{item} from [{section}] NOT found in config file: {userConfigFile}!'.format(item=item, section=section,userConfigFile=userConfigFile))

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))

	if (not boolean):
		return configFile.get(section, item)

	else:
		return configFile.getboolean(section, item)

def getPath(type="inputPath"):
	path = getConfig("Paths",type)
	return path

def getInterval(variable,section='Variables'):
    string = getConfig(section,variable)
    lis_string = string.split(',')
    lis = [float(li) for li in lis_string ]
    return lis

def doMatchIDs(id1,id2):
    """Get the indices in common
    """
    idx = np.empty(0,dtype=int)
    for i in id1:
        w, = np.where(id2==i)
        idx = np.append(idx,w)
    return idx

def getAngDist(ra1, dec1, ra2, dec2):
  """
    Calculate the angular distance between two coordinates.
    
    Parameters
    ----------
    ra1 : float, array
        Right ascension of the first object in degrees.
    dec1 : float, array
        Declination of the first object in degrees.
    ra2 : float, array
        Right ascension of the second object in degrees.
    dec2 : float, array
        Declination of the second object in degrees.

    Returns
    -------
    Angle : float, array
        The angular distance in DEGREES between the first
        and second coordinate in the sky.
         
  """  
  
  delt_lon = (ra1 - ra2)*np.pi/180.
  delt_lat = (dec1 - dec2)*np.pi/180.
  # Haversine formula
  dist = 2.0*np.arcsin( np.sqrt( np.sin(delt_lat/2.0)**2 + \
         np.cos(dec1*np.pi/180.)*np.cos(dec2*np.pi/180.)*np.sin(delt_lon/2.0)**2 ) )  

  return dist/np.pi*180.

# Compute the Number of members --Sum(Pmem). 
def doNgals(idx_cls,idx_gal,Pmem):
    Ngals = []
    for ci in idx_cls:
        w,=np.where(idx_gal==ci)
        if len(w)>0:
            Pmem_cls_i = Pmem[w]
            Pmem_non_zero = Pmem_cls_i[Pmem_cls_i>=0]
            ngals_i = sum(Pmem_non_zero)
        else:
            ngals_i = 0
        Ngals.append(ngals_i)
    return np.array(Ngals)

# the second return vector is used to make the host_id vector
# for the galaxies in the annulus. The first return is the
# indicies  in the galaxy vector for thos galxies in the annulus
def annulus_match(ra_cluster,dec_cluster,ang_diam_dist,ra_galaxy,dec_galaxy,r_in,r_out):
    depth=10
    h=esutil.htm.HTM(depth)
    #Inner match
    degrees_i=(360/(2*np.pi))*(r_in/ang_diam_dist)
    m1i,m2i,disti=h.match(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=degrees_i,maxmatch=0)
    #outer match
    degrees_o=(360/(2*np.pi))*(r_out/ang_diam_dist)
    m1o,m2o,disto=h.match(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=degrees_o,maxmatch=0)
    indicies_into_galaxies_in_annulus=[]
    indicies_into_clusters=[]
    for i in range(len(ra_cluster)):
        w_i=np.where(m1i==i)
        w_o=np.where(m1o==i)
        indicies_into_m2_in_annulus = np.in1d(m2o[w_o],m2i[w_i],invert=True)
        indicies_into_galaxies_in_annulus_i=m2o[w_o][indicies_into_m2_in_annulus]
        indicies_into_galaxies_in_annulus.append(indicies_into_galaxies_in_annulus_i)
        indicies_into_clusters_i = m1o[w_o][indicies_into_m2_in_annulus]
        indicies_into_clusters.append(indicies_into_clusters_i)
    indicies_into_galaxies_in_annulus=np.concatenate(indicies_into_galaxies_in_annulus)
    indicies_into_clusters=np.concatenate(indicies_into_clusters)
    return indicies_into_galaxies_in_annulus, indicies_into_clusters 

#use host_id=indicies_into_clusters from previous function
def make_annuli_quantities(indices_into_galaxies, host_id, gal_id, ra_galaxy, dec_galaxy,zg, zgerr, gmag, rmag, imag, zmag):#, gmagerr, rmagerr, imagerr) :

    ra_galaxy = ra_galaxy[indices_into_galaxies]
    dec_galaxy = dec_galaxy[indices_into_galaxies]
    host_id =  host_id
    galID = gal_id[indices_into_galaxies]
    zg=  zg[indices_into_galaxies]
    zgerr=  zgerr[indices_into_galaxies]
    gmag=  gmag[indices_into_galaxies]
    rmag=  rmag[indices_into_galaxies]
    imag=  imag[indices_into_galaxies]
    zmag=  zmag[indices_into_galaxies]
#    gmagerr=  gmagerr[indices_into_galaxies]
#    rmagerr=  rmagerr[indices_into_galaxies]
#    imagerr=  imagerr[indices_into_galaxies]

    data = dict()
    data["ra"] = ra_galaxy
    data["dec"] = dec_galaxy
    data["host_id"] = host_id
    data["galaxyID"] = galID
    data["zg"] = zg
    data["zgerr"] = zgerr
    data["gmag"] = gmag
    data["rmag"] = rmag
    data["imag"] = imag
    data["zmag"] = zmag
#    data["gmagerr"] = gmagerr
#    data["rmagerr"] = rmagerr
#    data["imagerr"] = imagerr
    return data

def gaussian(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def PhotozProbabilities(intmin,intmax,galz,galzerr):
    zpts,zstep=np.linspace(intmin,intmax,20,retstep=True) #split redshift window for approximation
    trapezoid_area=TrapArea(zpts,zstep,galz,galzerr)
    arflip=trapezoid_area.swapaxes(0,1)
    prob=np.sum(arflip,axis=1)
    return prob

def TrapArea(pts,step,galz,galzerr):
    area=[]
    for i in range(len(pts)-1): #approximate integral using trapezoidal riemann sum
        zpt1=pts[i] #zpt1,zpt2 are left,right points for trapezoid, respectively
        zpt2=pts[i+1]
        gauss1=gaussian(zpt1,galz,galzerr) #gauss1/2 are gaussian values for left,right points, respectively
        gauss2=gaussian(zpt2,galz,galzerr)
        area1=((gauss1+gauss2)/2.)*step
        area.append(area1)
    area=np.array(area)
    return area

def background_photoz_probs(z_cluster,host_id,z_galaxy,z_galaxy_err,zfunc):
    m1uniq=np.arange(z_cluster.size).astype(int)
    center_z=z_cluster[m1uniq]
    window=zfunc(center_z)
    zmin=center_z-1.5*window#0.5
    zmax=center_z+1.5*window#0.5
    total=[]
    probz=[]
    for x in range(len(m1uniq)):
        w=np.where(host_id==x)
        membz=z_galaxy[w]
        membzerr=z_galaxy_err[w]
        prob=PhotozProbabilities(zmin[x],zmax[x],membz,membzerr)
        probz.append(prob)
        total.append(np.sum(prob))
    probz=np.array(probz)
    total=np.array(total)
    return probz, total

def Aeff_integrand(R,R200,n_total,n_bg,const):
    Rcore=R200/100.
    p=np.where( R>Rcore,2*np.pi*R*radial_probability(R,R200,n_total,n_bg,const),2*np.pi*Rcore*radial_probability(Rcore,R200,n_total,n_bg,const) )
    return p

def doScale_bg(R200,r_in,r_out,n_total,n_bg,const):
    area_annulus=(np.pi*r_out**2.)-(np.pi*r_in**2.)
    m1uniq=np.arange(R200.size).astype(int)
    scale_vec = []
    for x in m1uniq:
        # area_effective,err=integrate.quad(Aeff_integrand,0,3.,args=(R200[x],n_total[x],n_bg[x],const[x]))
        area_effective = (np.pi*(R200[x])**2.)
        scale=area_effective/area_annulus
        scale_vec.append(scale)
        # print 'aeff',area_effective
        # print 'aann',area_annulus
        # print 'scale',scale
    return scale_vec

def local_bg_histograms(ang_diam_dist,R200,r_in,r_out,host_id,gmag,rmag,imag,zmag,pz,n_total,n_bg,const):
    area_annulus=(np.pi*r_out**2.)-(np.pi*r_in**2.)
    hist_bins=np.arange(-1,4.1,0.01)
    m1uniq=np.arange(ang_diam_dist.size).astype(int)
    gr_hists=[]
    ri_hists=[]
    iz_hists=[]
    for x in m1uniq:
        area_effective,err=integrate.quad(Aeff_integrand,0,3.,args=(R200[x],n_total[x],n_bg[x],const[x]))
        scale=area_effective/area_annulus
        #print 'aeff',area_effective
        #print 'aann',area_annulus
        #print 'scale',scale
        w=np.where(host_id==m1uniq[x])
        gmag1=gmag[w]
        rmag1=rmag[w]
        imag1=imag[w]
        zmag1=zmag[w]
        pz1=pz[x]
        gr=gmag1-rmag1
        ri=rmag1-imag1
        iz=imag1-zmag1
        gr_h,gr_e=np.histogram(gr,bins=hist_bins,weights=pz1,density=True)
        ri_h,ri_e=np.histogram(ri,bins=hist_bins,weights=pz1,density=True)
        iz_h,iz_e=np.histogram(iz,bins=hist_bins,weights=pz1,density=True)
        gr_hists.append(gr_h)#*scale)
        ri_hists.append(ri_h)#*scale)
        iz_hists.append(iz_h)#*scale)
    gr_hists=np.array(gr_hists)
    ri_hists=np.array(ri_hists)
    iz_hists=np.array(iz_hists)
    return gr_hists, ri_hists, iz_hists

def local_bg_histograms_RS_off(mag1,mag2,paramRS,cluster_ID,host_id,pz):
    a, b = paramRS
    hist_bins=np.arange(-1,4.1,0.01)
    m1uniq=np.arange(cluster_ID.size).astype(int)
    color_hists=[]
    for x in m1uniq:
        cind=cluster_ID[i]
        idx_cls = np.where(cluster_ID==cind)
        slope, intercept = float(a[cls_id]),float(b[cls_id])
        w=np.where(host_id==m1uniq[x])
        mag1i, mag2i = mag1[w], mag2[w]
        pz1=pz[x]
        rs_color, color = (mag2i*slope+intercept), (mag1i-mag2i)
        offset_color = color-rs_color
        hist,edge = np.histogram(offset_color,bins=hist_bins,weights=pz1,density=True)
        color_hists.append(hist)
    return np.array(color_hists)

def background_galaxy_density(total,r_in,r_out):
#    vol_in=(4./3.)*np.pi*(r_in**3)
#    vol_out=(4./3.)*np.pi*(r_out**3)
#    vol=vol_out-vol_in
    area_in=np.pi*(r_in**2)
    area_out=np.pi*(r_out**2)
    area=area_out-area_in
    density=total/area
    return density


#Mass-richness relation functions (see Tinker et al 2011)
def ncen(M,log_Mmin,sigma):
    #takes logM_min and logSigma_logM from paper. returns Ncentral from paper
    sigma=10**sigma
    return (1./2.)*(1+erf((np.log10(M)-log_Mmin)/sigma))

def ntot(M,Msat,log_Mmin,sigma,alpha_sat,Mcut):
    #takes logMmin, logSigma_logM, logMsat, logMcut from paper. Returns Ntotal=Ncentral+Nsatellite from paper
    Msat=10**Msat
    Mcut=10**Mcut
    return ncen(M,log_Mmin,sigma)*(1+(((M/Msat)**alpha_sat)*np.exp(-Mcut/M)))

#Non-redshift varying HOD 
def hod_mass(N,params):
    #params: logMmin,logMsat,alphasat,logMcut,logsigmalogM directly from table 4 of paper
    Mmin=params[0]
    Msat=params[1]
    alpha=params[2]
    Mcut=params[3]
    sigma=params[4]
    mass=np.linspace(1*10**7,2*10**17,1*10**6)
    m=interp1d(ntot(mass,Msat,Mmin,sigma,alpha,Mcut),mass)
    n=m(N)
    return n

###Redshift varying HOD parameters

#Msat redshift dependence 
def logMsat(z,M0=12.33,a=-0.27):
    return M0 + a*z

#alpha_sat redshift dependence - currently used for redshift varying HOD model
def alpha_sat(z,alpha0=1.0,a=0.0,z0=0.5):
    if z> z0: return alpha0 + a*(z-z0) #+ (alpha0+a*z)
    else: return alpha0


def hod_mass_z(N,z,params):
    #params: logMmin,logMsat,alphasat,logMcut,logsigmalogM directly from table 4 of paper
    # Tinker et al. 2012 (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1104.1635)
    Mmin=params[0]
    Msat=params[1]
    alpha=params[2]
    Mcut=params[3]
    sigma=params[4]
    #mass=np.linspace(1*10**7,2*10**17,1*10**6)
    mass=np.logspace(10,16,num=60,dtype=float)
    m200c=np.zeros_like(N)
    for i in range(len(z)):
        Msat=logMsat(z[i],params[1],0.) # make msat to be a function of z
        #alpha=alpha_sat(z[i],params[2],-0.5,0.4) # make alpha a functin of z
        alpha=alpha_sat(z[i],params[2]) # fixed alpha
        m=interp1d(ntot(mass,Msat,Mmin,sigma,alpha,Mcut),mass,
                   bounds_error=False,fill_value='extrapolate')
        for j in range(len(N[i])): N[i][j]=max(N[i][j],0.1) #set minimum value for mass conversions to prevent code from failing
        m200c[i]=m(N[i])
    return m200c

#change critical density with redshift
def crit_density(p_c,z,Omega_m,Omega_lambda):
    return p_c*(Omega_m*(1+z)**3 + Omega_lambda)#/(Omega_m*(1+z)**3)*Omega_m


## Magnitude limite
def getMagLimVec(interp_magi,interp_ri,interp_iz,zvec,dm=0):
    mag_i,color_ri,color_iz = interp_magi(zvec),interp_ri(zvec),interp_iz(zvec)
    mag_r, mag_z = (color_ri+mag_i),(mag_i-color_iz)
    maglim_r, maglim_i, maglim_z = (mag_r+dm),(mag_i+dm),(mag_z+dm)

    return maglim_r, maglim_i, maglim_z
    
def pullMagLim(auxfile,zvec,dm=0):
    '''Get the magnitude limit for 0.4L*
    file columns: z, mag_lim_i, (g-r), (r-i), (i-z)
    mag_lim = mag_lim + dm
    input: Galaxy clusters redshift
    return: mag. limit for each galaxy cluster and for the r,i,z bands
    output = [maglim_r(array),maglim_i(array),maglim_z(array)]
    '''
    annis=np.loadtxt(auxfile)
    jimz=[i[0] for i in annis]  ## redshift
    jimgi=[i[1] for i in annis] ## mag(i-band)
    jimgr=[i[2] for  i in annis]## (g-r) color
    jimri=[i[3] for i in annis] ## (r-i) color
    jimiz=[i[4] for i in annis] ## (i-z) color
    
    interp_magi=interp1d(jimz,jimgi,fill_value='extrapolate')
    interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
    interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
    interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

    mag_i,color_ri,color_iz = interp_magi(zvec),interp_ri(zvec),interp_iz(zvec)
    mag_r, mag_z = (color_ri+mag_i),(mag_i-color_iz)
    maglim_r, maglim_i, maglim_z = (mag_r+dm),(mag_i+dm),(mag_z+dm)

    return maglim_r, maglim_i, maglim_z


def pullRSparam(auxfile,zvec):
    '''Get the red sequence fit parameters
    file columns: z, gr_mean, gr_slope, gr_intercept, ri_mean, ... , iz_intercept
    
    input: Galaxy clusters redshift
    return: slope and intercept for each galaxy cluster for the three colors
    output = [gr_mean(array),gr_slope(array), gr_intercept(array)],...
    '''
    annis=np.loadtxt(auxfile)
    jimz=[i[0] for i in annis]  ## redshift
    jimgr=[i[1] for i in annis] ## (g-r) color mean
    jimri=[i[4] for i in annis] ## (r-i) color mean
    jimiz=[i[7] for i in annis] ## (i-z) color mean
    
    param_gr=np.array([[i[2],i[3]] for i in annis]) ## RS Fit: [slope, intercept] (g-r) color 
    param_ri=np.array([[i[5],i[6]] for i in annis]) ## RS Fit: [slope, intercept] (r-i) color 
    param_iz=np.array([[i[8],i[9]] for i in annis]) ## RS Fit: [slope, intercept] (i-z) color 
    
    interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
    interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
    interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

    interp_gr_slope, interp_gr_intercept = interp1d(jimz,(param_gr[:,0]),fill_value='extrapolate'),interp1d(jimz,np.array(param_gr[:,1]),fill_value='extrapolate')
    interp_ri_slope, interp_ri_intercept = interp1d(jimz,param_ri[:,0],fill_value='extrapolate'),interp1d(jimz,param_ri[:,1],fill_value='extrapolate')
    interp_iz_slope, interp_iz_intercept = interp1d(jimz,param_iz[:,0],fill_value='extrapolate'),interp1d(jimz,param_iz[:,1],fill_value='extrapolate')

    gr_RSparam = [interp_gr(zvec),interp_gr_slope(zvec),interp_gr_intercept(zvec)]
    ri_RSparam = [interp_ri(zvec),interp_ri_slope(zvec),interp_ri_intercept(zvec)]
    iz_RSparam = [interp_iz(zvec),interp_iz_slope(zvec),interp_iz_intercept(zvec)]

    return gr_RSparam, ri_RSparam, iz_RSparam

def pullRSparam_mock(auxfile1,auxfile2,zvec):
    '''Get the red sequence fit parameters
    file columns: z, gr_mean, gr_slope, gr_intercept, ri_mean, ... , iz_intercept
    
    input: Galaxy clusters redshift
    return: slope and intercept for each galaxy cluster for the three colors
    output = [gr_mean(array),gr_slope(array), gr_intercept(array)],...
    '''
    annis=np.loadtxt(auxfile1)
    jimz=[i[0] for i in annis]  ## redshift
    jimgr=[i[1] for i in annis] ## (g-r) color mean
    jimri=[i[2] for i in annis] ## (r-i) color mean
    jimiz=[i[3] for i in annis] ## (i-z) color mean
    
    interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
    interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
    interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

    annis=np.loadtxt(auxfile2)
    jimz=[i[0] for i in annis]  ## redshift
    param_gr=np.array([[i[2],i[3]] for i in annis]) ## RS Fit: [slope, intercept] (g-r) color 
    param_ri=np.array([[i[5],i[6]] for i in annis]) ## RS Fit: [slope, intercept] (r-i) color 
    param_iz=np.array([[i[8],i[9]] for i in annis]) ## RS Fit: [slope, intercept] (i-z) color 

    interp_gr_slope, interp_gr_intercept = interp1d(jimz,(param_gr[:,0]),fill_value='extrapolate'),interp1d(jimz,np.array(param_gr[:,1]),fill_value='extrapolate')
    interp_ri_slope, interp_ri_intercept = interp1d(jimz,param_ri[:,0],fill_value='extrapolate'),interp1d(jimz,param_ri[:,1],fill_value='extrapolate')
    interp_iz_slope, interp_iz_intercept = interp1d(jimz,param_iz[:,0],fill_value='extrapolate'),interp1d(jimz,param_iz[:,1],fill_value='extrapolate')

    gr_RSparam = [interp_gr(zvec),interp_gr_slope(zvec),interp_gr_intercept(zvec)]
    ri_RSparam = [interp_ri(zvec),interp_ri_slope(zvec),interp_ri_intercept(zvec)]
    iz_RSparam = [interp_iz(zvec),interp_iz_slope(zvec),interp_iz_intercept(zvec)]

    return gr_RSparam, ri_RSparam, iz_RSparam


def RS_color(mag,param):
    slope, intercept = param
    rsColor = mag*slope+intercept
    return rsColor

def auxRSparam(param,zbin,zvec):
    ## Interpolates the data and returns an array (same size of the zvec)
    interp_slope=interp1d(zbin,param[0],fill_value='extrapolate')
    interp_int=interp1d(zbin,param[1],fill_value='extrapolate')
    return [interp_slope(zvec),interp_int(zvec)]

def getRSparam(file,zvec):
    RSfit=np.loadtxt(file)      ## z, slope_gr, int_gr, ...
    zbin = [i[0] for i in RSfit]
    param_gr0=[[i[2],i[3]] for i in RSfit]
    param_ri0=[[i[4],i[6]] for i in RSfit]
    param_zi0=[[i[8],i[9]] for i in RSfit]
    
    param_gr = auxRSparam(param_gr0,zbin,zvec)
    param_ri = auxRSparam(param_ri0,zbin,zvec)
    param_iz = auxRSparam(param_iz0,zbin,zvec)

    return param_gr, param_ri, param_iz

def getRSout(info,zvec):
    RSfit=np.loadtxt(file)      ## z, slope_gr, int_gr, ...
    zbin = [i[0] for i in RSfit]
    param_gr0=[[i[2],i[3]] for i in RSfit]
    param_ri0=[[i[4],i[6]] for i in RSfit]
    param_zi0=[[i[8],i[9]] for i in RSfit]
    
    param_gr = auxRSparam(param_gr0,zbin,zvec)
    param_ri = auxRSparam(param_ri0,zbin,zvec)
    param_iz = auxRSparam(param_iz0,zbin,zvec)

    return param_gr, param_ri, param_iz

def overlap(ra1,dec1,z1,ra2,dec2,z2,ang_diam_dist,radius=2,dz=0.1):
    '''Count the number of overlaped objectes in a cylindrius(RA+radius,DEC+radius,z+/-dm)
    '''
    depth=10
    h=esutil.htm.HTM(depth)
    degrees_i=(360/(2*np.pi))*(radius/ang_diam_dist)
    m1,m2,dist=h.match(ra1,dec1,ra2,dec2,radius=degrees_i,maxmatch=0)
    w, = np.where((z2[m2] > (z1[m1] - dz)) &
                     (z2[m2] < (z1[m1] + dz)) )
    m1,m2,dist=m1[w],m2[w],dist[w]
    # m1,adist,zdist = h.cylmatch(ra1, dec1, z1, ra2, dec2, z2, radius, dz, maxmatch = 0, unique=False, nkeep=-1)
    overlap_clusters=[]
    for i in range(len(ra1)):
        w,=np.where(m1==i)
        n_match=int(len(m1[w])-1)      ## Number of overlaped object in the radius=X
        overlap_clusters.append(n_match)
    return np.array(overlap_clusters), m1, m2

def Bg_overlap(ra1,dec1,z1,ra2,dec2,z2,ang_diam_dist,r_in=4,r_out=6,dz=0.1):
    _,m1_in,m2_in = overlap(ra1,dec1,z1,ra2,dec2,z2,ang_diam_dist,radius=r_in,dz=dz)
    _,m1_out,m2_out = overlap(ra1,dec1,z1,ra2,dec2,z2,ang_diam_dist,radius=r_out,dz=dz)
    BG_FLAG = []
    indicies_into_m2=[]
    indicies_into_m1=[]
    for i in range(len(ra1)):
        w_i,=np.where(m1_in==i)
        w_o,=np.where(m1_out==i)
        indicies_into_annulus = np.in1d(m2_out[w_o],m2_in[w_i],invert=True)
        ncls = int(np.count_nonzero(indicies_into_annulus))
        m2_i = m2_out[w_o][indicies_into_annulus]
        m1_i = m1_out[w_o][indicies_into_annulus]
        BG_FLAG.append(ncls)
        indicies_into_m2.append(m2_i)
        indicies_into_m1.append(m1_i)
    return np.array(BG_FLAG),np.concatenate(indicies_into_m1),np.concatenate(indicies_into_m2)

def ProjectionEffects_Pmem(Pmem,m1,m2,hostID,glxID,clusterID):
    ## Find common galaxies between GCs.
    ## Pmem = P_i x (1-Pj1) x (1 - Pj2) x ... x (1 - Pjn)
    for i in range(len(Pmem)):
        Pmem_B = np.ones_like(Pmem)
        ## For a given cluster i, it takes the projected clusters j.
        w, = np.where(m1==i)
        mi,mj = m1[w], m2[w]
        size_i = np.count_nonzero(w)
        lis = []; idxfinal = np.array([0])
        if size_i>1:
            cls_i = clusterID[mi[0]]
            gals_i, = np.where(hostID == cls_i)
            for jn in range(1,size_i):
                cls_j = clusterID[mj[jn]]
                gals_j, = np.where(hostID == cls_j)
                idxi, idxj = np.in1d(glxID[gals_i],glxID[gals_j]), np.in1d(glxID[gals_j],glxID[gals_i])
                count_commom_gals = np.count_nonzero(idxi)
                if count_commom_gals>0:
                    print 'The RM-', cls_i, 'cluster shares', count_commom_gals, 'galaxies with RM-', cls_j, 'cluster.'
                    idxij, idxji = gals_i[idxi], gals_j[idxj]           ## common galaxies indices on galaxy cluster i and galaxy cluster j
                    Pmem_B[idxij]=(1-Pmem[idxji])   ## Pmem = P_i x (1-Pj1) x (1 - Pj2) x ... x (1 - Pjn)
        Pmem = Pmem*Pmem_B
    ## end For
    return Pmem

def findMockGals(galID,clsID):
    """Find the true galaxies on the Mock Catalogs
    """
    res = np.empty((0,),int)
    for idx in clsID:
        w,=np.where((galID==idx)|(galID==-idx))
        ngals = len(w)
        res = np.append(res,int(ngals))
        # print "\n True Member Galaxies at RM-%i is"%(idx),ngals
    return res

# param_gr, param_ri, param_iz = getRSparam(RSfile,cluster_Z)
# rsColor_gr = RS_color(galmagR,param_gr)
# rsColor_ri = RS_color(galmagI,param_ri)
# rsColor_iz = RS_color(galmagZ,param_iz)

def sigma(R,R200,c=3):
    #Radial NFW profile implementation. Takes array of radii, value of R200,
    #and NFW concentration parameter (set to 3 by default)
    if R200>0:
        Rs=float(R200)/float(c)
        r=R/Rs
        r=np.where(np.logical_or(r<=1e-5,r==1.),r+0.001,r)
        pre=1./((r**2)-1)
        arctan_coeff=2./(np.sqrt(np.abs(r**2-1)))
        arctan_arg=np.sqrt(np.abs((r-1)/(r+1)))
        sigma=np.where(r>1,pre*(1-arctan_coeff*np.arctan(arctan_arg)),pre*(1-arctan_coeff*np.arctanh(arctan_arg)))
        return sigma*2*Rs #2Rs for new p(r)
    else:
        bogusval=-99.*np.ones_like(R)
        return bogusval

# def _radial_probability(R,R200,z_cluster,sigma_background):
#    sigma_cluster=sigma(R,R200)
#    radial_prob=sigma_cluster/(sigma_cluster+sigma_background)
#    return radial_prob

def norm_const_integrand(R,R200):
    return sigma(R,R200)*2*np.pi*R

def norm_constant(R200,n_total,n_background):
    integral,err=integrate.quad(norm_const_integrand,0.,R200,args=R200)
    const=(n_total-n_background)/integral
    return const

def radial_probability(R,R200,n_total,sigma_background,const):
#    sigma_total=n_total#/(np.pi*3.**2)
    p_r=const*sigma(R,R200)/(const*sigma(R,R200)+sigma_background)
    p_r=np.where(p_r>1.,1.,p_r)
    return p_r

def linear(p,x):
    return p[0]+p[1]*x

def clip(data,data2,mean=None,sigma=None,n=3.,side='high'):
    #does sigma clipping step
    if mean==None:
        mean=data.mean()
    if sigma==None:
        sigma=data.std()
    if side=='high':
        ix=(data<(mean+n*sigma))
    elif side=='low':
        ix=(data>(mean-n*sigma))
    elif side=='both':
        ix=((data>(mean-n*sigma)) & (data<(mean+n*sigma)))
    dat1=data[ix]
    dat1.shape=(len(dat1),1)
    dat2=data2[ix]
    dat2.shape=(len(dat2),1)
    return dat1,dat2

def pick_colors(mu,sigma,alpha,expred):
    #choose which gaussian fit corresponds to red/blue/background
    mudif=np.abs(mu-expred)
    if min(mudif)!=max(mudif):
        red=np.where(mudif==min(mudif))
        blue=np.where(mudif==max(mudif))
    elif min(mudif)==max(mudif):
        blue=1
        red=0
    mur=mu[red]
    mub=mu[blue]
    sigr=sigma[red]
    sigb=sigma[blue]
    if mub>mur and sigb<sigr:
        temp=red
        red=blue
        blue=temp
    mur=mu[red]
    mub=mu[blue]
    sigr=sigma[red]
    sigb=sigma[blue]
    alphar=alpha[red]
    alphab=alpha[blue]
    return mur,mub,sigr,sigb,alphar,alphab

def sigma_clip(col,distprob,expred,mu=None,sig=None,n=3.,side='high'):
    #compute GMM fit for data after sigma clipping
    gmm=mixture.GMM(n_components=2,tol=0.0000001,n_iter=500)
    gr3,distprob3=clip(col,distprob,mean=mu,sigma=sig,n=n,side=side)
    if len(gr3)>1:
        try:
            gmm.fit(gr3,data_weights=distprob3)
	    #gmm.fit(gr3,distprob3)
            mu=gmm.means_
            mu.shape=(len(mu),)
            alpha=gmm.weights_
            alpha.shape=(len(alpha),)
            covars=gmm.covars_
            sigma=np.sqrt(covars)
            sigma.shape=(len(sigma),)
            mur,mub,sigr,sigb,alphar,alphab=pick_colors(mu,sigma,alpha,expred)
        except RuntimeError:
            mur=-99
            mub=-99
            sigr=-99
            sigb=-99
            alphar=-99
            alphab=-99            
    else:
        mur=99
        mub=99
        sigr=99
        sigb=99
        alphar=99
        alphab=99
    return gr3,distprob3,mur,mub,sigr,sigb,alphar,alphab

def background_subtract(col,pdist,bg_hist):
    #takes color,pradial,predshift, and a background histogram to subtract off
    #returns data points and weights to be fed to GMM for fitting
    hist_bins=np.arange(-1,4.1,0.01)
    pz_hist,pz_binedges=np.histogram(col,bins=hist_bins,weights=pdist,density=True)
    #BW
    #Include radial weights in subtraction
    #Below code applies radial weights after subtraction
    #inds=np.digitize(col,bins=hist_bins)
    #binsum_pr=[sum(pr[inds==i]) for i in range(1,len(hist_bins))]
    #bin_pr_denom=[len(pr[inds==i]) for i in range(1,len(hist_bins))]
    #binsum_pr=np.array(binsum_pr)
    #bin_pr_denom=np.array(bin_pr_denom)
    #bin_pr_weights=binsum_pr/bin_pr_denom
    #bin_pr_weights=np.nan_to_num(bin_pr_weights)
    bgsub_hist=pz_hist-bg_hist
    bgsub_weighted=bgsub_hist#*bin_pr_weights
    center=(hist_bins[:-1]+hist_bins[1:])/2.
    width=np.diff(hist_bins)
    bgsub_weighted.shape=(len(bgsub_weighted[0]),)
    for i in range(len(bgsub_weighted)):
        if bgsub_weighted[i]<0:
            bgsub_weighted[i]=0
    #maxval=max(bgsub_weighted)
    n_obs=sum(bgsub_weighted)
    norm_constant=n_obs*width[0]
    h_weights=[]
    for i in bgsub_weighted:
        h_weights.append(i/norm_constant)
    center_fit=center
    center_fit.shape=(len(center_fit),1)
    h_weights=np.array(h_weights)
    h_weights.shape=(len(h_weights),1)
    bgsub_weighted.shape=(len(bgsub_weighted),1)
    return center_fit,bgsub_weighted,n_obs#bgsub_weighted,n_obs


### added function at 15th Apr, 2020
def readFile(filename,columns=None):
	'''
	Read a filename with fitsio.read and return an astropy table
	'''
    #	import fitsio
	hdu = fitsio.read(filename, columns=columns)
	return Table(hdu)


def getMembAssignmentFileList(nbatches):
    cluster_indir = getPath(type="cluster_indir")
    gal_indir = getPath(type="gal_indir")

    cluster_file = os.path.join(cluster_indir, getConfig("Files","clusterfile")) 
    gal_file = os.path.join(gal_indir, getConfig("Files","galfile"))

    galaxyPrefix = os.path.splitext(gal_file)[0]
    clusterPrefix = os.path.splitext(cluster_file)[0]

    m_list = [galaxyPrefix+'_%04i.fits'%(i+1) for i in range(nbatches)]
    c_list = [clusterPrefix+'_%04i.fits'%(i+1) for i in range(nbatches)]

    return m_list, c_list

def old_memb_to_copa_columns(gal,cat):
    old_columns = ['HOST_HALOID','COADD_OBJECTS_ID',
                   'ZP','ZPE','AMAG_R',
                   'P_MEMBER','P_RADIAL','P_REDSHIFT',
                   'DIST_TO_CENTER','HOST_REDSHIFT']
    new_columns = ['CID','mid','z','zerr','Mr','Pmem','Pr','pz0','R','redshift']

    data = gal#[old_columns].copy()
    for oc,nc in zip(old_columns,new_columns): data.rename_column(oc,nc)

    ## cdata
    old_columns = ['MEM_MATCH_ID','Z','R200','N200']
    new_columns = ['CID','redshift','R200','Ngals']

    cdata = cat#[old_columns].copy()
    for oc,nc in zip(old_columns,new_columns): cdata.rename_column(oc,nc)

    return data,cdata

def copa_to_old_memb_columns(gal,cat):
    old_columns = ['CID','mid','z','zerr']
    new_columns = ['MEM_MATCH_ID','COADD_OBJECTS_ID','MEDIAN_Z','Z_SIGMA']
    for oc,nc in zip(old_columns,new_columns): gal.rename_column(oc,nc)

    gal["FLAGS_GOLD"] = np.zeros_like(gal["COADD_OBJECTS_ID"])
    gal["MULT_NITER_MODEL"] = np.ones_like(gal["mag"][:,1])

    ## change mag 
    filters = ['G','R','I','Z']
    mag     = gal["mag"]
    magerr  = gal["magerr"]

    for i,fi in enumerate(filters):
        mag_cols  = 'MAG_AUTO_%s'%fi
        emag_cols = 'MAGERR_AUTO_%s'%fi

        gal[mag_cols] = mag[:,i]
        gal[emag_cols] = magerr[:,i]

    ## cluster data
    cat.rename_column("CID","MEM_MATCH_ID")
    cat.rename_column("redshift","z_lambda")
    return gal, cat

def readBuzzard(gal_file,cat_file,add_noise=False,zsigma=0.03):
    gal = readFile(gal_file)
    cat = readFile(cat_file)

    print 'column names:',gal.colnames
    ## gal data
    old_columns = ['CID','GID','z','zerr']
    new_columns = ['MEM_MATCH_ID','COADD_OBJECTS_ID','MEDIAN_Z','Z_SIGMA']
    for oc,nc in zip(old_columns,new_columns): gal.rename_column(oc,nc)

    filters = ['G','R','I','Z']
    mag = gal["mag"]
    magerr = gal["magerr"]

    if add_noise:
        z = gal['z_true']
        z_err = np.random.normal(scale=zsigma*(1+z),size=len(z))
        z_noise = z- z_err
        z_noise = np.where(z_noise<0.,0.,z_noise)

        gal['MEDIAN_Z'] = z_noise
        gal['Z_SIGMA'] = zsigma*(1+z)

    for i,fi in enumerate(filters):
        mag_cols = 'MAG_AUTO_%s'%fi
        emag_cols = 'MAGERR_AUTO_%s'%fi

        gal[mag_cols] = mag[:,i]
        gal[emag_cols] = magerr[:,i]

    ## new variables
    # gal["Mr"] = gal["amag"][:,1] ## r-abs mag
    gal['dmag'] = gal['dmag'][:]

    gal["FLAGS_GOLD"] = np.zeros_like(gal["COADD_OBJECTS_ID"])
    gal["MULT_NITER_MODEL"] = np.ones_like(gal["mag"][:,1])

    ## cluster data
    cat["MEM_MATCH_ID"] = cat["CID"]
    cat["z_lambda"] = cat["redshift"]

    return gal,cat
