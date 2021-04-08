# !/usr/bin/env python
# radial stuff algorithm

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.special import erf
from astropy.io.fits import getdata
from astropy.table import Table, vstack

from scipy.ndimage import gaussian_filter
from scipy import integrate
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

### local libraries
import healpy as hp
from projection_radec_to_xy import xy_to_radec,radec_to_xy

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

def crit_density(p_c,z,Omega_m,Omega_lambda):
    return p_c*(Omega_m*(1+z)**3 + Omega_lambda)#/(Omega_m*(1+z)**3)*Omega_m


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

#Msat redshift dependence 
def logMsat(z,M0=12.33,a=-0.27):
    return M0 + a*z

#alpha_sat redshift dependence - currently used for redshift varying HOD model
def alpha_sat(z,alpha0=1.0,a=0.0,z0=0.5):
    if z> z0: 
        return alpha0 + a*(z-z0) #+ (alpha0+a*z)
    else: 
        return alpha0

def hod_mass_z(Ngals, z_cls, params):
    ''' description: it computes the m200,c for the Tinker et al. 2012 Model
        input: Number of member galaxies along the cluster radius, cluster redshift and model paramenters.
        output: m200c
    '''
    #params: logMmin,logMsat,alphasat,logMcut,logsigmalogM directly from table 4 of paper
    # Tinker et al. 2012 (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1104.1635)
    
    Mmin=params[0]
    Msat=params[1]
    alpha=params[2]
    Mcut=params[3]
    sigma=params[4]

    mass = np.logspace(10,16,num=60,dtype=float)
    m200c = np.zeros_like(Ngals)
    
    Msat = logMsat(z_cls, params[1], 0.) # make msat to be a function of z
    #alpha=alpha_sat(z[i],params[2],-0.5,0.4) # make alpha a functin of z
    alpha=alpha_sat( z_cls, params[2]) # fixed alpha
    m=interp1d( ntot(mass,Msat,Mmin,sigma,alpha,Mcut), mass, bounds_error=False,fill_value='extrapolate')

    for i in range(len(Ngals)): Ngals[i]=max(Ngals[i],0.1) #set minimum value for mass conversions to prevent code from failing
    
    m200c=m(Ngals)

    return m200c

def binsCounts(x,n):
    """get n points per bin"""
    xs = np.sort(x,kind='mergesort')
    xedges = [ xs[0] ]
    for i in np.arange(0,len(xs),n):
        if (i+n)<len(xs)-1:
            xi = (xs[i+n+1]+xs[i+n])/2.
        else:
            xi = xs[-1]
        xedges.append(xi)

    xmean = [(xedges[i+1]+xedges[i])/2 for i in range(len(xedges)-1)]
    return np.array(xedges), np.array(xmean)

def binsCountsWeighted(x,weights,n):
    """get n points per bin"""
    idx = np.argsort(x,kind='mergesort')
    xs,ws = x[idx], weights[idx]

    xedges = []

    ni = 0
    for i in range(len(xs)):
        if ni>=n:
            xedges.append(xs[i])
            ni = 0
        else:
            ni += ws[i]
    # xedges = xs[w+1]
    xmean = [(xedges[i+1]+xedges[i])/2 for i in range(len(xedges)-1)]
    return np.array(xedges), np.array(xmean)

def doRadialBin(radii,pz,rvec,testPz=False):
    ngals,_ = np.histogram(radii,weights=pz,bins=rvec)[0]
    ng_density = ngals/(np.pi*(rvec[1:]**2-rvec[:-1]**2))
    r_mean = 0.5*(rvec[1:]+rvec[:-1])
    return ng_density, r_mean

def calcR200(radii,pz,cls_id,z_cls,nbkg,ra_c,dec_c,DA,rmax=3,pixelmap=None):
    w, = np.where(radii<=rmax)
    radii, pz = radii[w], pz[w]

    rmin,step=0.1,0.1
    rbin=np.r_[rmin:rmax:step,rmax]

    area = np.pi*rbin**2
    ngals_cls = np.array([np.sum(pz[radii<=ri]) for ri in rbin]) ## number of galaxies (<R) 
    
    if pixelmap is not None:
        theta_list = (rbin/DA)*(180/np.pi)
        maskfrac_profile = get_maskfrac_radial_profile(ra_c,dec_c,theta_list,pixelmap)
    else:
        maskfrac_profile = 1.#cat['MASKFRAC']

    ngals = ngals_cls-nbkg*area
    # ngals = ngals_cls-(nbkg*area)
    ngals = np.where(ngals<0.,0.,ngals)

    ####params=[11.6,12.45,1.0,12.25,-0.69]#parameters for mass conversion - see table 4 in Tinker paper
    params = [11.59,12.94,1.01,12.48,-0.69]#parameters for mass conversion - see table 4 in Tinker paper
    # params = [12.87,13.87,1.08,10.29,-0.12]

    mass = hod_mass_z(ngals,z_cls,params) #calculate mass given ngals (see above functions)

    volume = maskfrac_profile*(4./3)*np.pi*rbin**3
    mass_density=(mass)/volume

    # mass_density=np.where(mass_density<0.1,1e11,mass_density)
    pc=200*np.ones_like(radii)
    rho_crit = rhoc(0)
    critdense1 = crit_density(1.35984671381e+11,z_cls,0.23,0.77)
    critdense = critdense1*np.ones_like(rbin)

    X=200 #desired excess over critical density, ex. if X=200, calculates R/M200
    dX=1  #acceptance window around X
    ratio=mass_density/critdense

    f=interp1d(rbin,ratio,fill_value='extrapolate')
    radii_new=np.linspace(0.1,rmax,10000)
    ratio_new=f(radii_new)
    
    # if new_way:
    #     ratio_new=gaussian_filter(ratio_new,sigma=1) ## avoid noise

    w, = np.where((ratio_new>=X-dX)&(ratio_new<=X+dX))
    r200m=radii_new[w] #find possible r200s within acceptance range

    # w = np.nanargmin(np.abs(ratio_new-X))
    # r200m = radii_new[w]

    if r200m.size > 0:
        r200m=np.mean(r200m) #mean of all possible r200s is measured r200
    else:
        r200m=0.1 #bogus r200=0 if nothing within acceptance range
        #print 'bad cluster:',cls_id,'ratio min/max:',min(ratio_new),max(ratio_new)

    return r200m

def checkR200(r200,z_cls,M200=5e13):
    """ check if r200 is below a cluster with mass 5e13 Msun
        in the case of an error it computes R200 from the default Mass (M200)
    """
    # if r200<0.5:
    #     print('bad cluster')
        # r200 = 1.
    minv = convertR200toM200(r200,z_cls)
    if minv<M200:   
        r200 = convertM200toR200(M200,z_cls)
        
    return r200

## PDF radial
def profileNFW(R,R200,c=3):
    #Radial NFW profile implementation. Takes array of radii, value of R200,
    #and NFW concentration parameter (set to 3 by default)
    if R200>0:
        Rs=float(R200)/c
        r=R/Rs
        r=np.where(np.logical_or(r<=1e-5,r==1.),r+0.001,r)
        pre=1./((r**2)-1)
        arctan_coeff=2./(np.sqrt(np.abs(r**2-1)))
        arctan_arg=np.sqrt(np.abs((r-1)/(r+1)))
        sigma=np.where(r>1,pre*(1-arctan_coeff*np.arctan(arctan_arg)),pre*(1-arctan_coeff*np.arctanh(arctan_arg)))
        
        return sigma*2*Rs
    
    else:
        bogusval=-99.*np.ones_like(R)
        return bogusval

def doPDF(radii,R200,c=3,rc=0.2):
    density = profileNFW(radii,R200,c=c) ## without norm
    # density = np.where(radii<rc,np.mean(density[radii<rc]), density)
    
    return density

def norm_const_integrand(R,R200,c):
    return profileNFW(R,R200,c=c)*2*np.pi*R

def norm_constant(R200,c=3):
    integral,err=integrate.quad(norm_const_integrand,0.,R200,args=(R200,c))
    const=1/integral
    return const
    
def convertM200toR200(M200,z,nc=200):
    ## M200 in solar masses
    ## R200 in Mpc
    rho = rhoc(z)
    R200 = ( M200/(nc*4*np.pi*rho/3) )**(1/3.)
    return R200/h

def convertR200toM200(R200,z, nc=200):
    ## M200 in solar masses
    ## R200 in Mpc
    rho = rhoc(z)
    M200 = nc*4*np.pi*rho*R200**3/3
    return M200

def get_pdfs_function(radii,zgal,pdfr,pdfz):
    idx1 = np.argsort(radii)
    idx2 = np.argsort(zgal)
    
    fr = interp1d(radii[idx1], pdfr[idx1], kind='linear', fill_value='extrapolate')
    fz = interp1d(zgal[idx2], pdfz[idx2], kind='linear', fill_value='extrapolate')
    
    func = lambda r,z: fr(r)*fz(z)
    return func
###
def pix_radec(pixeis,nside=4096,nest=False):
    theta, phi = np.degrees(hp.pix2ang(nside=nside, ipix=pixeis, nest=nest))    
    dec,ra = (90-theta), phi
    return ra,dec

def radec_pix(ra,dec,nside=4096, nest=False):
    return np.array(hp.ang2pix(nside,np.radians(90-dec),np.radians(ra),nest=nest),dtype=np.int64)

def get_healpix_radec(center_ra,center_dec,radius=1.,cushion=(10./3600.),nside=4096,nest=False):
    center_ra_rad = np.radians(center_ra)
    center_dec_rad = np.radians(center_dec)

    center_vec = np.array([np.cos(center_dec_rad)*np.cos(center_ra_rad),
                        np.cos(center_dec_rad)*np.sin(center_ra_rad),
                        np.sin(center_dec_rad)])

    healpix_list = hp.query_disc(nside, center_vec, np.radians(radius+cushion), nest=nest, inclusive=True, fact=8)
    healpix_list = np.append(healpix_list,radec_pix(center_ra,center_dec, nside=nside, nest=nest))
    
    return np.unique(healpix_list)

#### Monte Carlo Mask Fraction
def generate_random_points(rad,Npoints,zmax):
    ## It makes a lattice
    x_rnd = np.random.uniform(-rad,rad,Npoints)
    y_rnd = np.random.uniform(-rad,rad,Npoints)
    z_rnd = np.random.uniform(0.,zmax,Npoints)

    ## It computes z and r
    radius_rnd = doDistAngle(x_rnd,y_rnd)
    
    w,=np.where(radius_rnd<rad)
    x_rnd,y_rnd,z_rnd = x_rnd[w],y_rnd[w],z_rnd[w]
    radius_rnd = radius_rnd[w]
    
    return x_rnd,y_rnd,radius_rnd,z_rnd

def generate_galaxy_distribution(pdfs,r200,ngals,zmax=1.5):
    """It generates a list of member galaxies with Pmem distribution
    input: ra,dec,radius[arcmin]
    """
    rad=r200          ##Mpc
    R200=r200
    Npoints=2000     ##initial points
    
    x_rnd,y_rnd,radius_rnd,z_rnd = generate_random_points(rad,Npoints,zmax)
    
    pmem_rnd = pdfs(radius_rnd,z_rnd)
    pmem_rnd = np.where(pmem_rnd<0.,0.,pmem_rnd)
    pmem_rnd_norm = pmem_rnd/sum(pmem_rnd)
    
    ## Draw the NFW model distribution
    idx_rnd = np.arange(len(radius_rnd))
    idx = np.random.choice(idx_rnd, size=ngals, p=pmem_rnd_norm)

    x, y, pmem = x_rnd[idx], y_rnd[idx], pmem_rnd[idx]
    return x,y,pmem

def get_maskfrac_montecarlo(pixelmap,pdfs,r200,thetaMpc,albers,ngals=200):    
    x,y,pmem = generate_galaxy_distribution(pdfs,r200,ngals,zmax=1.2)
    ra,dec = xy_to_radec(x,y,albers,thetaMpc)
    pixel_random_galaxies = radec_pix(ra,dec)
    
    maskfrac = 1.0*np.sum(pixelmap[pixel_random_galaxies])/pixel_random_galaxies.size
    return maskfrac

# def get_maskfrac(pixels, pixelmap):
#     mathced_signal = pixelmap[pixels]
#     maskfrac = 1.*np.sum(mathced_signal)/mathced_signal.size
#     return maskfrac

# def get_maskfrac_radial_aperture(ra_c,dec_c,theta,pixelmap):
#     pixel_circle = get_healpix_radec(ra_c,dec_c,radius=theta)
#     maskfrac = get_maskfrac(pixel_circle,pixelmap)
#     return maskfrac

# def get_maskfrac_radial_profile(ra_c,dec_c,theta_list,pixelmap):
#     pixel_circle_list = [get_healpix_radec(ra_c,dec_c,radius=ra) for ra in theta_list]
#     pixel_ring_list = [pixel_b[np.in1d(pixel_b,pixel_a,inverse=True)] for pixel_a, pixel_b in zip(pixel_circle_list[:-1],pixel_circle_list[1:])]
#     mask_frac_radii = [get_maskfrac(pixel,pixelmap) for pixel in pixel_ring_list]
    
#     return np.array(mask_frac_radii)
##

######################
def computeMaskFraction(pixelmap, gal, cat, r200, pdfr, pdfz, rvec, zvec):
    if pixelmap is None:
        if ('MASKFRAC' in cat.colnames):
	        return np.array(cat['MASKFRAC'])
        else:
            return np.zeros_like(cat['RA'])
    else:
        maskfrac=[]
        
        for i in range(len(cat)):
            cls_idx = cat['CID'][i]
            ra_c,dec_c = cat['RA'][i], cat['DEC'][i]

            R200 = r200[i]
            DA = cat['DA'][i]
            Mpc2theta= DA/rad2deg
            
            gidx, = np.where(gal['CID']==cls_idx)
            if gidx.size>0:
                rag,decg = gal['RA'][gidx],gal['DEC'][gidx]    
                _, _, albers = radec_to_xy(rag,decg,ra_c,dec_c,Mpc2theta)
            else:
                _, _, albers = radec_to_xy(ra_c,ra_c,ra_c,dec_c,Mpc2theta)

            fpdfs = get_pdfs_function(rvec,zvec,pdfr[i],pdfz[i])
            maskfraci = get_maskfrac_montecarlo(pixelmap,fpdfs,R200,Mpc2theta,albers,ngals=2000)
            
            maskfrac.append(maskfraci)
        return np.array(maskfrac)

def computeR200(gals, cat, nbkg, rmax=3, defaultMass=1e14,pixelmap=None,compute=True, h=0.7):
    ## estimate R200
    ncls = len(cat)
    r200m = []
    raper = []

    for idx in range(ncls):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        # magLim_i = cat['magLim'][idx,0] ## r-band cut

        gal = gals[(gals['CID']==cls_id)&(gals['dmag']<=0.)]
        # gal = gals[(gals['CID']==cls_id)&(gals['mag'][:,1]<=magLim_i)] ## r-band cut
        # gal = gals[(gals['CID']==cls_id)&(gals['amag'][:,1]<=-20.5)]

        rac,dec = cat['RA'][idx], cat['DEC'][idx]
        da = cat['DA'][idx]

        # if compute:
        r200i = calcR200(gal['R'],gal['pz0'],cls_id,z_cls,nbkg[idx],rac,dec,da,rmax=3,pixelmap=pixelmap)
        #r200i = checkR200(r200i,z_cls,M200=defaultMass)

        raperi = 1.*r200i

        # print('r200:',r200i)
        raper.append(raperi)
        r200m.append(r200i)
    
    return np.array(r200m), np.array(raper)

    # else: ## in the case that M200 is provided
    #     r200m = convertM200toR200(M200,cat['redshift'])
    #     return r200m

def computeN200(gals, cat, r200, nbkg, testPz=False):
    galsFlag = np.full(len(gals['Bkg']), False, dtype=bool)
    N200 = []
    keys = []
    count0 = 0
    good_indices, = np.where(nbkg>=0)
    for idx in good_indices:
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        # magLim_i = cat['magLim'][idx,1] ## magnitude cut in the i band

        mask = (gals['CID']==cls_id)&(gals['R']<=(r200[idx]))&(gals['dmag']<=0.)
        # mask = (gals['CID']==cls_id)&(gals['R']<=(r200[idx]))&(gals['mag'][:,2]<magLim_i)
        gal = gals[mask]

        N_field = float(nbkg[idx]*np.pi*r200[idx]**2)
        N_cls_field = len(gal['pz0'])

        if testPz:
            N_cls_field = np.sum(gal['pz0'])
        
        N200i = N_cls_field - N_field
        # N200i = N_cls_field
        if N200i<0:
            # print('There is something wrong: N200 is negative')
            N200i = N_cls_field
        N200.append(N200i)
        
        indices, = np.where((gals['CID']==cls_id)&(gals['R']<=r200[idx])&(gals['dmag']<=0.))
        if len(indices)>0:
            galsFlag[indices] = True
            new_idx = np.arange(count0,count0+len(indices),1,dtype=int)
            keys.append(new_idx)
            count0 += len(indices)

    return np.array(N200), galsFlag, keys

def computeRadialPDF(gals,cat,r200,raper,nbkg,keys,rvec,c=3.53):
    '''it's missing the normalization factor
    '''
    # pdf = np.empty(0,dtype=float)
    # pdf_bkg = np.empty(0,dtype=float)

    pdf_cls, pdf_cf, pdf_field = [], [], []
    
    rmed = (rvec[1:]+rvec[:-1])/2
    rvec2=rvec+4
    for idx in range(len(cat)):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2,ra = r200[idx], raper[idx]

        galIndices, = np.where((gals['CID']==cls_id)&(gals['Gal']==True))
        bkgGalaxies, = np.where((gals['Bkg']==True)&(gals['CID']==cls_id))

        g2, = np.where( (gals['CID']==cls_id)&(gals['dmag']<=0.))
        # g2, = np.where( (gals['CID']==cls_id)&(gals['mag'][:,1]<=cat['magLim'][idx,0]) )
        # g2, = np.where( (gals['CID']==cls_id)&(gals['amag'][:,1]<=-20.5) ) ## Mr<=-19.5

        radii = gals['R'][galIndices]
        radii2 = gals['R'][g2]
        radii_bkg = gals['R'][bkgGalaxies]
        
        probz = gals['pz0'][g2]
        probz_bkg = gals['pz0'][bkgGalaxies]

        # pdfi = (N2/area200)*doPDF(radii,r2,c=c)
        ni = nbkg[idx]
        area200 = np.pi*(r2/c)**2

        area_aper = np.pi*(ra)**2
        N2 = (len(radii)/area200)-ni
        
        normConstant = norm_constant(ra,c=c)

        # idxs = np.trunc(rmed/0.15)
        # idxs = np.where(idxs<=round(r2/0.15), idxs, round(r2/0.15))
        # rapers = np.unique(0.15*(idxs+1))
        # rapers.sort()
        
        # normConstant = np.array([norm_constant(rx,c=c) for rx in rapers])
        # normConstant = normConstant[idxs.astype(np.int)]

        # pdfi = normConstant*doPDF(radii,r2,c=c)
        # pdf = np.append(pdf,pdfi)

        pdf_cls_i = normConstant*doPDF(rmed,r2,c=c)
        pdf_cls.append(pdf_cls_i)

        area = np.pi*(rmed**2)
        pdf_cf_i = np.array([np.sum(probz[radii2<=ri]) for ri in rmed])#doRadialBin(radii2,probz,rvec,testPz=True)
        # pdf_cf_i = pdf_cf_i/area#gaussian_filter(pdf_cf_i,sigma=2)

        area2 = np.pi*(rvec2[1:]**2)
        pdf_f_i = np.array([np.sum(probz_bkg[radii_bkg<=ri]) for ri in rvec2[1:]])
        # pdf_f_i/= area2
        # pdf_f_i = gaussian_filter(pdf_f_i,sigma=2)
        
        pdf_cf.append(pdf_cf_i)
        pdf_field.append(pdf_f_i)
        
    # pdf_bkg = np.ones_like(pdf)
    # pdf_field = [np.ones_like(pdf_i) for pdf_i in pdf_cls]

    pdf_list = [pdf_cls, pdf_cf, pdf_field]
    return pdf_list


def interpData(x,y,xmin,xmax):
    bins = np.linspace(xmin,xmax,100)
    yint = interp1d(x,y,fill_value='extrapolate')

    return bins, yint(bins)

def plotPradial(radii,sigma,galaxies,nbkg,cls_id='none'):
    bin1 = np.trunc(galaxies['R']/0.1)
    g_radial_bin = galaxies.group_by(bin1)
    radii_bin = (g_radial_bin.groups.keys+1/2)*0.1

    area = [radii_bin[0]**2]
    for i in range(len(radii_bin)-1):
        area.append(np.pi*(radii_bin[i+1]**2-radii_bin[i]**2))

    ngals = g_radial_bin['GID'].groups.aggregate(np.count_nonzero)/area
    # print(ngals)

    r, ng = interpData(radii,sigma,0.05,8.)
    # ng = (ng/ng.max())*np.max(ngals)
    # print(ng)
    
    plt.clf()
    plt.figure(figsize=(12,8))
    plt.scatter(radii_bin,ngals,color='r', label='Data')
    plt.plot(r,ng+nbkg,color='k',label='Model NFW')
    plt.axhline(nbkg,linestyle='--',color='k')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\Sigma \; [gals/Mpc^{2}]$')
    plt.xlabel(r'$R \; [Mpc]$')
    
    ymax = np.max([ng.max(),ngals.max()])
    try:
        plt.ylim(nbkg/2 ,2*ymax)
    except:
        print('something weird')
    # plt.xlim(0.01,3)
    plt.title(cls_id)
    # plt.legend()
    plt.legend()
    plt.savefig('./check/probRadial/%s_radial_density.png'%(cls_id))

def doPr(R,R200,ntot,sBkg,c=3.53):
    kappa = (ntot-sBkg)*norm_constant(R200,c=c)
    Pr = (kappa)*doPDF(R,R200,c=c)/((kappa)*doPDF(R,R200,c=c) + sBkg)
    return Pr

def Aeff_integrand(R,R200,ntot,sBkg):
    Rcore=R200/100.
    p=np.where( R>Rcore,2*np.pi*R*doPr(R,R200,ntot,sBkg),2*np.pi*Rcore*doPr(Rcore,R200,ntot,sBkg) )
    return p

def scaleBkg(R200,ntot,sBkg,r_in=4.,r_out=6):
    area_effective,err=integrate.quad(Aeff_integrand,0,R200,args=(R200,ntot,sBkg))
    scale=area_effective
    return scale