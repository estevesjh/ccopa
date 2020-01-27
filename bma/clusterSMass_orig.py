import numpy as np
#import pyfits as pf
from astropy.io import fits as pf
import fitsio

from astropy.table import Table, join

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
    return (weightedmass.sum())/10.**10.


def haloStellarMass (filename="vt-stellar-masses.txt", outfile="vt-halos-stellar-masses.txt") :

    import logging

    logging.info('Starting clusterSMass_orig.haloStellarMass()')

    #ra,dec,z,zerr,id,central,hostid, gr,gre, gi,gie, kri,krie, kii,kiie, \
    #    i,distmod,rabs,iabs, mcMass, taMass, maiss, std = np.genfromtxt(filename, unpack=True)
    #id, gr_o,  grostd,  gi_o, giostd, kri, kristd, kii, kiistd, i,  distmod, r_abs, i_abs, mcMass, taMass, mass, std, bestsp,COADD_OBJECTS_ID_1,hostid,ra,dec,MAG_AUTO_G,MAG_AUTO_R,MAG_AUTO_I,MAG_AUTO_Z,P_RADIAL,P_REDSHIFT,GR_P_COLOR,RI_P_COLOR,IZ_P_COLOR,GR_P_MEMBER,RI_P_MEMBER,IZ_P_MEMBER,DIST_TO_CENTER,GRP_ED,GRP_BLUE,GRP_BACKGROUND,RIP_RED,RIP_BLUE,RIP_BACKGROUND,IZP_RED,IZP_BLUE,IZP_BG,MAGERR_AUTO_G,MAGERR_AUTO_R, MAGERR_AUTO_I,MAGERR_AUTO_Z,z,R200,M200,N200,LAMBDA_CHISQ,GR_SEP_FLAG,RI_SEP_FLAG,IZ_SEP_FLAG = np.genfromtxt(filename, unpack=True)

    #id,mass,std,COADD_OBJECTS_ID_1,hostid,ZP,ZPE,MAG_AUTO_G,MAG_AUTO_R,MAG_AUTO_I,MAG_AUTO_Z,P_RADIAL,P_REDSHIFT,GR_P_COLOR,RI_P_COLOR,IZ_P_COLOR,GR_P_MEMBER,RI_P_MEMBER, IZ_P_MEMBER, AMAG_R,DIST_TO_CENTER,GRP_RED,GRP_BLUE,GRP_BG,RIP_RED,RIP_BLUE, RIP_BG,IZP_RED,IZP_BLUE,IZP_BG,RESTP_RED,RESTP_BLUE,RESTP_BG,REST_P_COLOR,REST_P_MEMBER,MAGERR_AUTO_G,MAGERR_AUTO_R,MAGERR_AUTO_I,MAGERR_AUTO_Z,z,R200,M200,N200,GR_SEP_FLAG,RI_SEP_FLAG,IZ_SEP_FLAG = np.genfromtxt(filename, unpack=True)

    #id, hostid, gr_o,  std,  gi_o, std, kri, std, kii, std, i,  distmod, r_abs, i_abs, mcMass, taMass, mass, std,z, sfr, sfrstd, age,agestd,bestsp,zmet_best, zmet_mean, GR_P_COLOR, RI_P_COLOR, IZ_P_COLOR, GR_P_MEMBER, RI_P_MEMBER, IZ_P_MEMBER, DIST_TO_CENTER, GRP_RED, GRP_BLUE, RIP_RED, RIP_BLUE, IZP_RED, IZP_BLUE = np.genfromtxt(filename, unpack=True, delimiter=",")

    d = fitsio.FITS(filename)[1]

    id = d['ID'][:]
    hostid = d['MEM_MATCH_ID'][:]
    mass = d['mass'][:]
    std = d['mass_err'][:]
    z= d['Z'][:]
    sfr= d['ssfr'][:]
    sfrstd= d['ssfr_std'][:]
    age= d['mass_weight_age'][:]
    agestd= d['mass_weight_age_err'][:]
    bestsp= d['best_model'][:]
    zmet_best= d['best_zmet'][:]
    zmet_mean= d['zmet'][:]
    GR_P_COLOR= d['GR_P_COLOR'][:]
    RI_P_COLOR= d['RI_P_COLOR'][:]
    IZ_P_COLOR= d['IZ_P_COLOR'][:]
    GR_P_MEMBER= d['P_RADIAL'][:] # NOTE the difference since p_tot=p_z*p_r
    RI_P_MEMBER= d['P_REDSHIFT'][:] # NOTE the difference since p_tot=p_z*p_r
    IZ_P_MEMBER= d['P_MEMBER'][:] # NOTE the difference since p_tot=p_z*p_r so 
    DIST_TO_CENTER= d['DIST_TO_CENTER'][:]
    GRP_RED= d['GRP_RED'][:]
    GRP_BLUE= d['GRP_BLUE'][:]
    RIP_RED= d['RIP_RED'][:]
    RIP_BLUE= d['RIP_BLUE'][:]
    ## Johnny Comments - I switch (i-z) to (r-i) for the Pblue, and Pred
    IZP_RED= RIP_RED
    IZP_BLUE= RIP_BLUE

#     IZP_RED= d['IZP_RED'][:]
#     IZP_BLUE= d['IZP_BLUE'][:]

    best_chisq = d['best_chisq'][:]

    halo_out =[]
    zout_out =[]
    ngals_out =[]
    sum_mass_out =[]
    sum_mass_std_out =[]
    lambda_gr_out =[]
    lambda_gr_err_jk_out =[]
    lambda_ri_out =[]
    lambda_ri_err_jk_out =[]
    lambda_iz_out =[]
    lambda_iz_err_jk_out =[]
    lambda_iz_red_out =[]
    lambda_iz_blue_out =[]
    lambda_iz_red_err_jk_out =[]
    lambda_iz_blue_err_jk_out =[]
    ssfr_weight_iz_out =[]

# FOR X-ray cat
#    id,gr_o, std,gi_o, std,kri,std,kii,std,i,  distmod,r_abs,  i_abs,  mcMass,taMass,mass,std,sfr, sfrstd,age,agestd,bestsp,COADD_OBJECTS_ID,hostid,RA,  DEC,ZP,ZPE,  DERED_G_1,DERED_R_1,DERED_I_1,DERED_Z_1,P_RADIAL,  P_REDSHIFT, GR_P_COLOR,  RI_P_COLOR,IZ_P_COLOR,  GR_P_MEMBER, RI_P_MEMBER, IZ_P_MEMBER, AMAG_R, DIST_TO_CENTER,GRP_RED,GRP_BLUE, GRP_BG, RIP_RED,RIP_BLUE, RIP_BG, IZP_RED,IZP_BLUE, IZP_BG, RESTP_RED, RESTP_BLUE,RESTP_BG,REST_P_COLOR, REST_P_MEMBER,OBJID,  RA_1,DEC_1, ZPHOT,  ZPHOTERR,DERED_U,  DERED_G_2,DERED_R_2,DERED_I_2,DERED_Z_2,ERR_U,ERR_G, ERR_R, ERR_I, ERR_Z, ID,z,R200,  M200,N200,LAMBDA,GR_SLOPE,GR_INTERCEPT,GRMU_R,GRMU_B, GRSIGMA_R, GRSIGMA_B, GRW_R,GRW_B,  RI_SLOPE, RI_INTERCEPT,RIMU_R, RIMU_B, RISIGMA_R, RISIGMA_B,  RIW_R,  RIW_B,  GRMU_BG,  GRSIGMA_BG,GRW_BG,  RIMU_BG,  RISIGMA_BG,RIW_BG, IZ_SLOPE, IZ_INTERCEPT,IZMU_R, IZMU_B,IZSIGMA_R,  IZSIGMA_B, IZW_R,  IZW_B,  IZMU_BG,IZSIGMA_BG,IZW_BG, GR_SEP_FLAG,RI_SEP_FLAG,IZ_SEP_FLAG,REST_SLOPE,  REST_INTERCEPT,RESTMU_R,  RESTMU_B,  RESTMU_BG,RESTSIGMA_R,RESTSIGMA_B,RESTSIGMA_BG,RESTW_R,  RESTW_B,  RESTW_BG,REST_SEP_FLAG = np.genfromtxt(filename, unpack=True)

    #id,mass,std,COADD_OBJECTS_ID_1,hostid,ra,dec,ZP,ZPE,MAG_AUTO_G,MAG_AUTO_R,MAG_AUTO_I,MAG_AUTO_Z,P_RADIAL,P_REDSHIFT,GR_P_COLOR,RI_P_COLOR,IZ_P_COLOR,GR_P_MEMBER,RI_P_MEMBER,IZ_P_MEMBER,AMAG_R,DIST_TO_CENTER,GRP_RED,GRP_BLUE,GRP_BG,RIP_RED,RIP_BLUE,RIP_BG,IZP_RED,IZP_BLUE,IZP_BG,RESTP_RED,RESTP_BLUE,RESTP_BG,REST_P_COLOR,REST_P_MEMBER,MAGERR_AUTO_G,MAGERR_AUTO_R,MAGERR_AUTO_I,MAGERR_AUTO_Z,z,R200,M200,N200,GR_SEP_FLAG,RI_SEP_FLAG,IZ_SEP_FLAG,RESTW_R,RESTW_B,RESTW_BG,REST_SEP_FLAG = np.genfromtxt(filename, unpack=True)


    id = np.array(id).astype(int)
    #central = np.array(central).astype(int)
    hostid = np.array(hostid).astype(int)

    #Control on probabilities ==0 - add other colors

    idx_bad = (IZ_P_COLOR <0.0001)
    IZ_P_COLOR[idx_bad] = 0.0001
    idx_bad = (IZP_RED <0.0001)
    IZP_RED[idx_bad] = 0.0001
    idx_bad = (IZP_BLUE <0.0001)
    IZP_BLUE[idx_bad] = 0.0001

    halos = np.unique(hostid)
    halos = np.sort(halos)

    # logging.debug('# hostid z median(ra) median(dec) median(z) ngals log(stellar_mass)  (h=0.7, Om=0.3, flat)')
    logging.debug('hostid, ngals, sum_mass, sum_mass_std, lambda_iz, mass_std_iz, lambda_gr_err_jk, ssfr_weight_iz')
   
    for halo in halos:
        ix = np.nonzero(hostid == halo)
        #zmedian = np.median(z[ix])
        #ramedian = np.median(ra[ix])
        #decmedian = np.median(dec[ix])
        ngals = id[ix].size
        linear_mass = 10**mass[ix]
        linear_mass_weight_gr = 10**mass[ix]*GR_P_MEMBER[ix]
        linear_mass_weight_ri = 10**mass[ix]*RI_P_MEMBER[ix]
        linear_mass_weight_iz = 10**mass[ix]*IZ_P_MEMBER[ix]
        mass_errors = np.log(10.)*linear_mass*std[ix]
        mass_std = np.sqrt((mass_errors**2).sum())

        mass_err_gr = np.log(10.)*linear_mass*std[ix]*GR_P_MEMBER[ix]
        mass_err_ri = np.log(10.)*linear_mass*std[ix]*RI_P_MEMBER[ix]
        mass_err_iz = np.log(10.)*linear_mass*std[ix]*IZ_P_MEMBER[ix]
        mass_std_gr = 10**(-10)*np.sqrt((mass_err_gr**2).sum())
        mass_std_ri = 10**(-10)*np.sqrt((mass_err_ri**2).sum())
        mass_std_iz = 10**(-10)*np.sqrt((mass_err_iz**2).sum())

        linear_mass_weight_gr_red = 10**mass[ix]*GR_P_MEMBER[ix]/GR_P_COLOR[ix]*GRP_RED[ix]
        linear_mass_weight_ri_red = 10**mass[ix]*RI_P_MEMBER[ix]/RI_P_COLOR[ix]*RIP_RED[ix]
        linear_mass_weight_iz_red = 10**mass[ix]*IZ_P_MEMBER[ix]*IZP_RED[ix] #/IZ_P_COLOR[ix]*IZP_RED[ix] change all colors!!

        linear_mass_weight_gr_blue = 10**mass[ix]*GR_P_MEMBER[ix]/GR_P_COLOR[ix]*GRP_BLUE[ix]
        linear_mass_weight_ri_blue = 10**mass[ix]*RI_P_MEMBER[ix]/RI_P_COLOR[ix]*RIP_BLUE[ix]
        linear_mass_weight_iz_blue = 10**mass[ix]*IZ_P_MEMBER[ix]*IZP_BLUE[ix] #/IZ_P_COLOR[ix]*IZP_BLUE[ix]

        #jacknife test
        weightmass_gr = 10**mass[ix]*GR_P_MEMBER[ix]
        lambda_gr_err_jk = (jackknife_var(weightmass_gr, lambda_star_jk))**0.5
        weightmass_ri = 10**mass[ix]*RI_P_MEMBER[ix]
        lambda_ri_err_jk = (jackknife_var(weightmass_ri, lambda_star_jk))**0.5
        weightmass_iz = 10**mass[ix]*IZ_P_MEMBER[ix]
        lambda_iz_err_jk = (jackknife_var(weightmass_iz, lambda_star_jk))**0.5
        weightmass_iz_red = 10**mass[ix]*IZ_P_MEMBER[ix]*IZP_RED[ix] #/IZ_P_COLOR[ix]*IZP_RED[ix]
        lambda_iz_red_err_jk = (jackknife_var(weightmass_iz_red, lambda_star_jk))**0.5
        weightmass_iz_blue = 10**mass[ix]*IZ_P_MEMBER[ix]*IZP_BLUE[ix] #/IZ_P_COLOR[ix]*IZP_BLUE[ix]
        lambda_iz_blue_err_jk = (jackknife_var(weightmass_iz_blue, lambda_star_jk))**0.5

        sum_mass = linear_mass.sum()
        lambda_gr = (linear_mass_weight_gr.sum())/10.**10.
        lambda_ri = (linear_mass_weight_ri.sum())/10.**10.
        lambda_iz = (linear_mass_weight_iz.sum())/10.**10.
        sum_mass_std = mass_std/sum_mass/np.log(10.)
        sum_mass = np.log10(sum_mass)

        lambda_gr_red = (linear_mass_weight_gr_red.sum())/10.**10.
        lambda_ri_red = (linear_mass_weight_ri_red.sum())/10.**10.
        lambda_iz_red = (linear_mass_weight_iz_red.sum())/10.**10.

        lambda_gr_blue = (linear_mass_weight_gr_blue.sum())/10.**10.
        lambda_ri_blue = (linear_mass_weight_ri_blue.sum())/10.**10.
        lambda_iz_blue = (linear_mass_weight_iz_blue.sum())/10.**10.

        #SFR computation
        this_sfr = sfr[ix]
        this_mass = mass[ix]
        this_pmem = IZ_P_MEMBER[ix]  
        this_chisq = best_chisq[ix]
        ix_sfr = (this_mass>10.)# & (this_chisq<10.))#((this_sfr <-8)&(this_mass>10.)&(this_pmem>0.2))
        ssfr_weight_iz = np.log10(np.sum(this_sfr[ix_sfr]*10**this_mass[ix_sfr]))#*this_pmem[ix_sfr]))
        #ssfr_weight_iz = np.log10(np.sum(10**this_sfr[ix_sfr]*10**this_mass[ix_sfr]))      
        #ssfr_weight_iz = np.log10(np.sum(this_pmem[ix_sfr])*np.median(10**this_sfr[ix_sfr]*10**this_mass[ix_sfr]*this_pmem[ix_sfr]))

        #sum_mass_gr = np.log10(sum_mass_gr)
        #sum_mass_ri = np.log10(sum_mass_ri)
        #sum_mass_iz = np.log10(sum_mass_iz)
        #lambda_rm = LAMBDA_CHISQ[ix[0][0]]

        #TO UNCOMMENT!!!!!!!
        #M200_GMM = M200[ix[0][0]]
        zout = z[ix[0][0]]

        halo_out.append(halo)
        zout_out.append(zout)
        ngals_out.append(ngals)
        sum_mass_out.append(sum_mass)
        sum_mass_std_out.append(sum_mass_std)
        lambda_gr_out.append(lambda_gr)
        lambda_gr_err_jk_out.append(lambda_gr_err_jk)
        lambda_ri_out.append(lambda_ri)
        lambda_ri_err_jk_out.append(lambda_ri_err_jk)
        lambda_iz_out.append(lambda_iz)
        lambda_iz_err_jk_out.append(lambda_iz_err_jk)
        lambda_iz_red_out.append(lambda_iz_red)
        lambda_iz_blue_out.append(lambda_iz_blue)
        lambda_iz_red_err_jk_out.append(lambda_iz_red_err_jk)
        lambda_iz_blue_err_jk_out.append(lambda_iz_blue_err_jk)
        ssfr_weight_iz_out.append(ssfr_weight_iz)

        logging.debug('{:10d}  {:4d}   {:6.3f}   {:6.3f} {:6.4f} {:6.3f} {:6.4f} {:6.4f}'.format(
            halo, ngals, sum_mass, sum_mass_std,lambda_iz,mass_std_iz,lambda_gr_err_jk,ssfr_weight_iz))
        
    halo_out = np.array(halo_out)
    zout_out = np.array(zout_out)
    ngals_out = np.array(ngals_out)
    sum_mass_out = np.array(sum_mass_out)
    sum_mass_std_out = np.array(sum_mass_std_out)
    lambda_gr_out = np.array(lambda_gr_out)
    lambda_gr_err_jk_out = np.array(lambda_gr_err_jk_out)
    lambda_ri_out = np.array(lambda_ri_out)
    lambda_ri_err_jk_out = np.array(lambda_ri_err_jk_out)
    lambda_iz_out = np.array(lambda_iz_out)
    lambda_iz_err_jk_out = np.array(lambda_iz_err_jk_out)
    lambda_iz_red_out = np.array(lambda_iz_red_out)
    lambda_iz_blue_out = np.array(lambda_iz_blue_out)
    lambda_iz_red_err_jk_out = np.array(lambda_iz_red_err_jk_out)
    lambda_iz_blue_err_jk_out = np.array(lambda_iz_blue_err_jk_out)
    ssfr_weight_iz_out = np.array(ssfr_weight_iz_out)


    col1=pf.Column(name='MEM_MATCH_ID',format='J',array=halo_out)
    col2=pf.Column(name='Z',format='E',array=zout_out)
    col3=pf.Column(name='NGALS',format='E',array=ngals_out)
    col4=pf.Column(name='SUM_MASS',format='E',array=sum_mass_out)
    col5=pf.Column(name='SUM_MASS_STD',format='E',array=sum_mass_std_out)
    col6=pf.Column(name='MU_PR',format='E',array=lambda_gr_out) # Mostly meaningless but represents input of p_radial
    col7=pf.Column(name='MU_PR_ERR_JK',format='E',array=lambda_gr_err_jk_out) # Mostly meaningless but represents input of p_radial
    col8=pf.Column(name='MU_PZ',format='E',array=lambda_ri_out) # Mostly meaningless but represents input of p_redshift
    col9=pf.Column(name='MU_PZ_ERR_JK',format='E',array=lambda_ri_err_jk_out) # Mostly meaningless but represents input of p_redshift
    col10=pf.Column(name='MU',format='E',array=lambda_iz_out) # The true Mu* value as computed with p_tot=p_z*p_r
    col11=pf.Column(name='MU_ERR_JK',format='E',array=lambda_iz_err_jk_out) # The true Mu* value as computed with p_tot=p_z*p_r
    col12=pf.Column(name='MU_RED',format='E',array=lambda_iz_red_out)
    col13=pf.Column(name='MU_BLUE',format='E',array=lambda_iz_blue_out)
    col14=pf.Column(name='MU_RED_ERR_JK',format='E',array=lambda_iz_red_err_jk_out)
    col15=pf.Column(name='MU_BLUE_ERR_JK',format='E',array=lambda_iz_blue_err_jk_out)
    col16=pf.Column(name='SSFR',format='E',array=ssfr_weight_iz_out)

    cols=pf.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16])
    tbhdu=pf.BinTableHDU.from_columns(cols)
    tbhdu.writeto(outfile,overwrite=True)

    logging.info('Returning from clusterSMass_orig.haloStellarMass()')

def computeMuStar(d,colorFit=False,trueMembers=False):
    import logging
    
    if trueMembers:
        d = d[d['True']==True]
        print('True members catalog size:', len(d))

    id = d['GID'][:]
    hostid = d['CID'][:]
    mass = d['mass'][:]
    std = d['mass_err'][:]
    z= d['redshift'][:]
    sfr= d['ssfr'][:]
    sfrstd= d['ssfr_std'][:]
    age= d['mass_weight_age'][:]
    agestd= d['mass_weight_age_err'][:]
    bestsp= d['best_model'][:]
    zmet_best= d['best_zmet'][:]
    zmet_mean= d['zmet'][:]
    best_chisq = d['best_chisq'][:]
    
    P_MEMBER= d['Pmem'][:]
    P_COLOR= d['Pc'][:]
    DIST_TO_CENTER= d['R'][:]
    
	# RI_P_COLOR= d['RI_P_COLOR'][:]
	# IZ_P_COLOR= d['IZ_P_COLOR'][:]
    
    if colorFit:
    	P_RED= d['Pred'][:]
    	P_BLUE= d['Pblue'][:]

    if trueMembers:
        P_MEMBER = np.ones_like(P_MEMBER,dtype=float)
        P_COLOR = P_MEMBER
        
    halo_out =[]
    zout_out =[]
    # ngals_out =[]
    
    sum_mass_out =[]
    sum_mass_std_out =[]

    lambda_out =[]
    lambda_err_jk_out =[]
    
    if colorFit:
        lambda_red_out =[]
        lambda_blue_out =[]
        
        lambda_red_err_jk_out =[]
        lambda_blue_err_jk_out =[]

    ssfr_weight_out =[] 

    id = np.array(id)#.astype(int)
    #central = np.array(central).astype(int)
    hostid = np.array(hostid).astype(int)

    #Control on probabilities ==0 - add other colors

    idx_bad = (P_COLOR <0.0001)
    P_COLOR[idx_bad] = 0.0001

    if colorFit:
        idx_bad = (P_RED <0.0001)
        P_RED[idx_bad] = 0.0001
        idx_bad = (P_BLUE <0.0001)
        P_BLUE[idx_bad] = 0.0001

    halos = np.unique(hostid)
    halos = np.sort(halos)

    # logging.debug('# hostid z median(ra) median(dec) median(z) ngals log(stellar_mass)  (h=0.7, Om=0.3, flat)')
    logging.debug('hostid, sum_mass, sum_mass_std, lambda, mass_std, lambda_err_jk, ssfr_weight')

    for halo in halos:
        ix = np.nonzero(hostid == halo)
        
        # ngals = id[ix].size
        linear_mass = 10**mass[ix]
        linear_mass_weight = 10**mass[ix]*P_MEMBER[ix]
        
        mass_errors = np.log(10.)*linear_mass*std[ix]
        mass_std = np.sqrt((mass_errors**2).sum())

        mass_err_weight = np.log(10.)*linear_mass*std[ix]*P_MEMBER[ix]
        mass_std_weight = 10**(-10)*np.sqrt((mass_err_weight**2).sum())

        # #jacknife test
        weightmass = 10**mass[ix]*P_MEMBER[ix]
        lambda_err_jk = (jackknife_var(weightmass, lambda_star_jk))**0.5
            
        sum_mass = linear_mass.sum()
        lambda_weight = (linear_mass_weight.sum())/10.**10.
        sum_mass_std = mass_std/sum_mass/np.log(10.)
        sum_mass = np.log10(sum_mass)
            
        #SFR computation
        this_sfr = sfr[ix]
        this_mass = mass[ix]
        this_pmem = P_MEMBER[ix]  
        this_chisq = best_chisq[ix]
        ix_sfr = (this_mass>10.)# & (this_chisq<10.))#((this_sfr <-8)&(this_mass>10.)&(this_pmem>0.2))
        ssfr_weight = np.log10(np.sum(this_sfr[ix_sfr]*10**this_mass[ix_sfr]))#*this_pmem[ix_sfr]))
        
        #TO UNCOMMENT!!!!!!!
        #M200_GMM = M200[ix[0][0]]
        zout = z[ix[0][0]]

        halo_out.append(halo)
        zout_out.append(zout)
        # ngals_out.append(ngals)
        sum_mass_out.append(sum_mass)
        sum_mass_std_out.append(sum_mass_std)
        
        lambda_out.append(lambda_weight)
        lambda_err_jk_out.append(lambda_err_jk)    

        ssfr_weight_out.append(ssfr_weight)
            
        if colorFit:
            linear_mass_weight_red = 10**mass[ix]*P_MEMBER[ix]*P_RED[ix]
            linear_mass_weight_blue = 10**mass[ix]*P_MEMBER[ix]*P_BLUE[ix]

            weightmass_red = 10**mass[ix]*P_MEMBER[ix]*P_RED[ix]
            lambda_red_err_jk = (jackknife_var(weightmass_red, lambda_star_jk))**0.5
            weightmass_blue = 10**mass[ix]*P_MEMBER[ix]*P_BLUE[ix] 
            lambda_blue_err_jk = (jackknife_var(weightmass_blue, lambda_star_jk))**0.5

            lambda_weight_red = (linear_mass_weight_red.sum())/10.**10.
            lambda_weight_blue = (linear_mass_weight_blue.sum())/10.**10.

            lambda_red_out.append(lambda_weight_red)
            lambda_blue_out.append(lambda_weight_blue)
            
            lambda_red_err_jk_out.append(lambda_red_err_jk)
            lambda_blue_err_jk_out.append(lambda_blue_err_jk)

        logging.debug('{:10d}  {:6.3f}   {:6.3f} {:6.4f} {:6.3f} {:6.4f} {:6.4f}'.format(
            halo, sum_mass, sum_mass_std,lambda_weight,mass_std,lambda_err_jk,ssfr_weight))
        
    halo_out = np.array(halo_out)
    zout_out = np.array(zout_out)
    # ngals_out = np.array(ngals_out)
    sum_mass_out = np.array(sum_mass_out)
    sum_mass_std_out = np.array(sum_mass_std_out)
    lambda_out = np.array(lambda_out)
    lambda_err_jk_out = np.array(lambda_err_jk_out)
    ssfr_weight_out = np.array(ssfr_weight_out)

    if colorFit:
        lambda_red_out = np.array(lambda_red_out)
        lambda_blue_out = np.array(lambda_blue_out)
        lambda_red_err_jk_out = np.array(lambda_red_err_jk_out)
        lambda_blue_err_jk_out = np.array(lambda_blue_err_jk_out)

    col1=pf.Column(name='CID',format='J',array=halo_out)
    # col2=pf.Column(name='redshift',format='E',array=zout_out)
    # col3=pf.Column(name='NGALS',format='E',array=ngals_out)
    col4=pf.Column(name='SUM_MASS',format='E',array=sum_mass_out)
    col5=pf.Column(name='SUM_MASS_STD',format='E',array=sum_mass_std_out)
    col10=pf.Column(name='MU',format='E',array=lambda_out) # The true Mu* value as computed with p_tot=p_z*p_r
    col11=pf.Column(name='MU_ERR_JK',format='E',array=lambda_err_jk_out) # The true Mu* value as computed with p_tot=p_z*p_r
    col16=pf.Column(name='SSFR',format='E',array=ssfr_weight_out)
    cols=pf.ColDefs([col1,col4,col5,col10,col11,col16])

    if colorFit:
        col12=pf.Column(name='MU_RED',format='E',array=lambda_red_out)
        col13=pf.Column(name='MU_BLUE',format='E',array=lambda_blue_out)
        col14=pf.Column(name='MU_RED_ERR_JK',format='E',array=lambda_red_err_jk_out)
        col15=pf.Column(name='MU_BLUE_ERR_JK',format='E',array=lambda_blue_err_jk_out)

        cols=pf.ColDefs([col1,col4,col5,col10,col11,col12,col13,col14,col15,col16])

    if trueMembers:
        col1=pf.Column(name='CID',format='J',array=halo_out)
        # col2=pf.Column(name='redshift',format='E',array=zout_out)
        # col3=pf.Column(name='NGALS',format='E',array=ngals_out)
        col4=pf.Column(name='SUM_MASS_TRUE',format='E',array=sum_mass_out)
        col5=pf.Column(name='SUM_MASS_STD_TRUE',format='E',array=sum_mass_std_out)
        col10=pf.Column(name='MU_TRUE',format='E',array=lambda_out) # The true Mu* value as computed with p_tot=p_z*p_r
        col11=pf.Column(name='MU_ERR_JK_TRUE',format='E',array=lambda_err_jk_out) # The true Mu* value as computed with p_tot=p_z*p_r
        col16=pf.Column(name='SSFR_TRUE',format='E',array=ssfr_weight_out)
        cols=pf.ColDefs([col1,col4,col5,col10,col11,col16])

    tbhdu=pf.BinTableHDU.from_columns(cols)
    # tbhdu.writeto(outfile,overwrite=True)
    data = Table(tbhdu.data)

    return data

def haloStellarMassCOPA(filename="vt-stellar-masses.txt", outfile="vt-halos-stellar-masses.txt",cluster_infile='vt-clusters.txt',colorFit=False,simulation=False):

    import logging

    logging.info('Starting clusterSMass_orig.haloStellarMass()')

    d = Table(fitsio.read(filename))
    oldCat = Table(fitsio.read(cluster_infile))

    ## compute MuStar
    data = computeMuStar(d,colorFit=colorFit)
    ## append the input catalog file
    print('append the input catalog file')
    outCat = join(oldCat,data,keys='CID', join_type='outer')

    if simulation:
        data_true = computeMuStar(d,colorFit=False,trueMembers=True)
        outCat = join(outCat,data_true,keys='CID', join_type='outer')
    
    outCat.write(outfile,format='fits',overwrite=True)

    logging.info('Returning from clusterSMass_orig.haloStellarMass()')