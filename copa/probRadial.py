# !/usr/bin/env python
# radial stuff algorithm

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from astropy import units as u


from scipy.special import erf
from astropy.io.fits import getdata
from astropy.table import Table, vstack

from scipy import integrate
from scipy.interpolate import interp1d

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.283)
Msol = 1.98847e33
Mpc2cm = 3.086e+24

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

def doRadialBin(radii,pz,width=0.1,testPz=False):
    rvec = np.arange(0.1,3+width,width)
    
    ngals_profile = []
    for ri in rvec:
        # ni = np.sum( pz[ (radii>ri-width) & (radii<ri) ] )
        ni = len(pz)
        if testPz:
            ni = np.sum( pz[ (radii<ri) ] )
        ngals_profile.append(ni)

    return np.array(ngals_profile), rvec

def calcR200(radii,pz,cls_id,z_cls,nbkg,rmax=3,testPz=False):
    ngals_cls, rbin = doRadialBin(radii,pz,width=0.1,testPz=testPz)

    ngals = ngals_cls-nbkg*np.pi*rbin**2
    ####params=[11.6,12.45,1.0,12.25,-0.69]#parameters for mass conversion - see table 4 in Tinker paper
    params = [11.59,12.94,1.01,12.48,-0.69]#parameters for mass conversion - see table 4 in Tinker paper
    mass = hod_mass_z(ngals,z_cls,params) #calculate mass given ngals (see above functions)

    volume = (4./3)*np.pi*rbin**3
    mass_density=(mass)/volume

    mass_density=np.where(mass_density<0.1,1e11,mass_density)
    pc=200*np.ones_like(radii)
    rho_crit = rhoc(0)
    critdense1 = crit_density(rho_crit,z_cls,0.23,0.77)
    critdense = critdense1*np.ones_like(rbin)

    X=2000 #desired excess over critical density, ex. if X=200, calculates R/M200
    dX=10  #acceptance window around X
    ratio=mass_density/critdense

    f=interp1d(rbin,ratio,fill_value='extrapolate')
    radii_new=np.linspace(0.1,rmax,10000)
    ratio_new=f(radii_new)
    r200m=radii_new[np.where( (ratio_new>=X-dX)&(ratio_new<=X+dX) )] #find possible r200s within acceptance range
    
    if r200m.size > 0:
        r200m=np.median(r200m) #mean of all possible r200s is measured r200

    else:
        ## Try r500
        X=500
        r200m=radii_new[np.where( (ratio_new>=X-dX)&(ratio_new<=X+dX) )] #find possible r200s within acceptance range
        
        if r200m.size > 0:
            r200m=np.median(r200m)/0.65 #mean of all possible r200s is measured r200
        else:
            r200m = 0
            print('bad cluster:',cls_id,'ratio min/max:',min(ratio_new),max(ratio_new))

    return r200m

def checkR200(r200,z_cls,M200=5e15):
    """ check if r200 is less than 0.5Mpc
        in the case of an error it computes R200 from the default Mass (M200)
    """
    if r200<0.5:
        print('bad cluster')
        r200 = convertM200toR200(M200,z_cls)
        # r200 = 1.
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

def doPDF(radii,R200,c=3):
    density = profileNFW(radii,R200,c=c) ## without norm
    return density

def norm_const_integrand(R,R200,c):
    return profileNFW(R,R200,c=c)*2*np.pi*R

def norm_constant(R200,c=3):
    integral,err=integrate.quad(norm_const_integrand,0.,R200,args=(R200,c))
    const=1/integral
    return const
    
def convertM200toR200(M200,z):
    ## M200 in solar masses
    ## R200 in Mpc
    rho = rhoc(z)
    R200 = ( M200/(200*4*np.pi*rho/3) )**(1/3.)
    return R200

######################
def computeR200(gals, cat, nbkg, rmax=3, defaultMass=1e14,testPz=False,compute=True):
    ## estimate R200
    ncls = len(cat)
    r200m = []

    for idx in range(ncls):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        magLim_i = cat['magLim'][idx,1]

        gal = gals[(gals['CID']==cls_id)&(gals['mag'][:,2]<=magLim_i)]

        if compute:
            r200i = calcR200(gal['R'],np.ones_like(gal['PDFz']),cls_id,z_cls,nbkg[idx],rmax=rmax,testPz=testPz)
        else:
            r200i = 0.1

        r200i = checkR200(r200i,z_cls,M200=defaultMass)
        print('r200:',r200i)
        r200m.append(r200i)
    
    return np.array(r200m)

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
        magLim_i = cat['magLim'][idx,1] ## magnitude cut in the i band

        mask = (gals['CID']==cls_id)&(gals['R']<=(r200[idx]))&(gals['mag'][:,2]<magLim_i)
        gal = gals[mask]

        N_field = float(nbkg[idx]*np.pi*r200[idx]**2)
        N_cls_field = len(gal['PDFz'])

        if testPz:
            N_cls_field = np.sum(gal['PDFz'])
        
        N200i = N_cls_field - N_field
        # N200i = N_cls_field
        if N200i<0:
            # print('There is something wrong: N200 is negative')
            N200i = N_cls_field
        N200.append(N200i)
        
        indices, = np.where((gals['CID']==cls_id)&(gals['R']<=r200[idx])&(gals['mag'][:,2]<magLim_i))
        if len(indices)>0:
            galsFlag[indices] = True
            new_idx = np.arange(count0,count0+len(indices),1,dtype=int)
            keys.append(new_idx)
            count0 += len(indices)

    return np.array(N200), galsFlag, keys

def computeRadialPDF(gals,cat,r200,nbkg,keys,c=3.53,plot=False):
    '''it's missing the normalization factor
    '''
    pdf = np.empty(0,dtype=float)
    pdf_bkg = np.empty(0,dtype=float)

    for idx,_ in enumerate(keys):
        cls_id, z_cls = cat['CID'][idx], cat['redshift'][idx]
        r2 = r200[idx]

        galIndices = np.where((gals['CID']==cls_id)&(gals['Gal']==True))
        radii = gals['R'][galIndices]

        # pdfi = (N2/area200)*doPDF(radii,r2,c=c)
        ni = nbkg[idx]
        area200 = np.pi*(r2/c)**2
        N2 = (len(radii)/area200)-ni
        normConstant = 1*norm_constant(r2,c=c)
        pdfi = normConstant*doPDF(radii,r2,c=c)
        pdf = np.append(pdf,pdfi)

        if plot:
            ni = nbkg[idx]
            area200 = np.pi*(r2/c)**2
            N2 = (len(pdfi)/area200)-ni

            galaxies = np.where((gals['CID']==cls_id))
            radx = gals[galaxies]['R']
            pdfx = N2*normConstant*doPDF(radx,r2,c=c)
            plotPradial(radx,pdfx,gals[galaxies],ni,cls_id='Buzzard_%i'%(cls_id))

    pdf_bkg = np.ones_like(pdf)

    return pdf, pdf_bkg


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


# def Aeff_integrand(R,R200,kappa,sBkg):
#     Rcore=R200/100.
#     p=np.where( R>Rcore,2*np.pi*R*doPr(R,R200,kappa,sBkg),2*np.pi*Rcore*doPr(Rcore,R200,kappa,sBkg) )
#     return p

# def scaleBkg(R200,kappa,sBkg,r_in=4.,r_out=6):
#     area_annulus=(np.pi*r_out**2.)-(np.pi*r_in**2.)
#     area_effective,err=integrate.quad(Aeff_integrand,0,R200,args=(R200,kappa,sBkg))
#     scale=area_effective/area_annulus
#     return scale
