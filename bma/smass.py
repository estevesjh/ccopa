import numpy as np
import loadPopColors
#import helperfunctions
#import weightedstats as ws
#import pyfits as pf
from astropy.io import fits as pf

#
# inputDataDict is what holds the input photometry
# it expects id, i, ierr, gr,ri,iz, grerr, rierr, izerr
#
#

#### Johnny at 27 october 2019.
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

def luminosityDistance(z):
    DL = float( cosmo.luminosity_distance(z) / u.Mpc ) # em Mpc
    return DL

# luminosityDistance = np.vectorize(luminosityDistance)

#### Johnny at 29 Feb 2020
from numba import jit
@jit(nopython=True)
def doCalcCOPA(ldistDict,splineDict,splines,zmet,index,ids,haloid,mi,ierr,ri,iz,gr,grerr,rierr,izerr,allzed):
    # prepping for output
    nvariables = 27
    out_index, out_id, out_haloid, out_gr, out_stdgr, out_gi, out_stdgi, \
    out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
    out_bestzmet,out_stdzmet, out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, out_sfr, out_stdsfr, out_age, out_stdage, \
    out_zmet, out_zed = [[] for i in range(nvariables)]
    
    out_bestsp, out_bestchisq = [], []

    size = haloid.size

    #size = 10
    for galaxy in range(0,size) :
        zed = allzed[galaxy]

        logging.debug('{i} of {len}, z = {z}'.format(
            i = galaxy+1, len = size, z = zed))

        rest_gr,rest_gi,weight,chisqs = [], [],[],[]
        masslight, sfrs,ages, zmets,kii, kri = [],[],[],[],[],[]
        minChiSq = 999; spIndex = -1
        for sp in range(0,len(splines)) :
            # for speed
            skey = str(sp) + "-" + str(zed)
            sgr = splines[sp][0](zed)
            sri = splines[sp][1](zed)
            siz = splines[sp][2](zed)
            sgrr = splines[sp][4](zed) ;# restframe g-r
            sgir = splines[sp][5](zed) ;# restframe g-i
            skii = splines[sp][6](zed) ;# kcorrection: i_o - i_obs
            skri = splines[sp][7](zed) ;# kcorrection: r_o - i_obs
            sml = splines[sp][8](zed) ;# log(mass/light)  (M_sun/L_sun)
            ssfr = splines[sp][9](zed)
            if (ssfr<-20.): ssfr=-20.
            sage_cosmic = splines[sp][10](zed)
            sage = splines[sp][11](zed)
            szmet = zmet[sp]
                #To be changed if SFH changes

            #splineDict[skey] = sgr,sri,siz,sgrr,sgir,skii,skri,sml

            gre = grerr[galaxy]
            rie = rierr[galaxy]
            ize = izerr[galaxy]
            gr_chisq = pow((gr[galaxy] - sgr)/gre,2)
            ri_chisq = pow((ri[galaxy] - sri)/rie,2)
            iz_chisq = pow((iz[galaxy] - siz)/ize,2)
            rest_gr.append(sgrr)
            rest_gi.append(sgir)
            kii.append(skii)
            kri.append(skri)
            masslight.append(sml)
            sfrs.append(ssfr)
            ages.append(sage)
            zmets.append(szmet)
            chisq = gr_chisq + ri_chisq + iz_chisq
            probability = 1-chi2.cdf(chisq, 3-1) ;# probability of chisq greater than this
            weight.append(probability)
            chisqs.append(chisq)
        spIndex = np.argmax(weight)
        rest_gr = np.array(rest_gr)
        rest_gi = np.array(rest_gi)
        kii = np.array(kii)
        kri = np.array(kri)
        masslight = np.array(masslight)
        sfrs = np.array(sfrs)
        idx_sfr = (sfrs<-8.)
        ages = np.array(ages)
        weight = np.array(weight)
        gr_weighted = rest_gr * weight
        gi_weighted = rest_gi * weight
        kii_weighted = kii * weight
        kri_weighted = kri * weight
        #weight_norm = weight/np.sum(weight)
        masslight_weighted = masslight * weight
        sfr_weighted = 10**sfrs[idx_sfr] * weight[idx_sfr]
        age_weighted = ages * weight
        zmet_weighted = zmets * weight
        w1 = weight.sum()
        w2 = (weight**2).sum()
        if w1 == 0 : w1 = 1e-10
        if w2 == 0 : w2 = 1e-10
        mean_gr  = gr_weighted.sum()/w1
        mean_gi  = gi_weighted.sum()/w1
        mean_kii = kii_weighted.sum()/w1
        mean_kri = kri_weighted.sum()/w1
        mean_masslight = masslight_weighted.sum()/w1
        #try :
        #    if weight.shape[0]>1.:
        #        mean_sfr = float(ws.numpy_weighted_median(sfrs, weights=weight)) #np.median(sfr_weighted) #sfr_weighted.sum()/w1
        #    else:
        #        mean_sfr = -70.
        #except :
        #    mean_sfr = -70.
        #print mean_sfr
        mean_age = age_weighted.sum()/w1
        mean_zmet = zmet_weighted.sum()/w1
        mean_sfr = sfr_weighted.sum()/w1
        # unbiased weighted estimator of the sample variance
        w3 = w1**2 - w2
        if w3 == 0 : w3 = 1e-10
        var_gr = ( w1/w3 ) * (weight*(rest_gr - mean_gr)**2).sum()
        var_gi = ( w1/w3 ) * (weight*(rest_gi - mean_gi)**2).sum()
        var_kii = ( w1/w3 ) * (weight*(kii - mean_kii)**2).sum()
        var_kri = ( w1/w3 ) * (weight*(kri - mean_kii)**2).sum()
        var_masslight = ( w1/w3 ) * (weight*(masslight - mean_masslight)**2).sum()
        var_sfr = ( w1/w3 ) * (weight*(sfrs - mean_sfr)**2).sum()
        var_age = ( w1/w3 ) * (weight*(ages - mean_age)**2).sum()
        var_zmet = ( w1/w3 ) * (weight*(zmets - mean_zmet)**2).sum()
        std_gr = var_gr**0.5
        std_gi = var_gi**0.5
        std_kii = var_kii**0.5
        std_kri = var_kri**0.5
        std_masslight = var_masslight**0.5
        std_sfr = var_sfr**0.5
        std_age = var_age**0.5
        std_zmet = var_zmet**0.5 
        if std_gr > 99.99 : std_gr = 99.99
        if std_gi > 99.99 : std_gi = 99.99
        if std_kii > 99.99 : std_kii = 99.99
        if std_kri > 99.99 : std_kri = 99.99
        if std_sfr > 99.99 : std_sfr = 99.99
        if std_age > 99.99 : std_age = 99.99
        if std_masslight > 99.99 : std_masslight = 99.99
        if std_zmet > 99.99 : std_zmet = 99.99
	# Comment -distanceModulus out for fsps versions <2.5, as their mags don't include distance modulus
        if zed in ldistDict :
            lumdist = ldistDict[zed]
        else :
            # lumdist = cd.luminosity_distance(zed) ;# in Mpc
            lumdist = luminosityDistance(zed)

            ldistDict[zed] = lumdist
        distanceModulus = 5*np.log10(lumdist*1e6/10.)
        iabs = mi[galaxy] + mean_kii - distanceModulus
        rabs = mi[galaxy] + mean_kri - distanceModulus
        taMass = taylorMass(mean_gi, iabs)
        mcMass = mcintoshMass(mean_gr, rabs)
        fsMass = fspsMass( mean_masslight, iabs )
        # JTA: to make purely distance modulus
        #iabs = i[galaxy] - distanceModulus 
        #fsMass = gstarMass( iabs )

        # saving for output
        out_index.append( index[galaxy])
        out_id.append( ids[galaxy] )
        out_haloid.append( haloid[galaxy] )
        out_gr.append( mean_gr )
        out_stdgr.append(std_gr )
        out_gi.append( mean_gi )
        out_stdgi.append( std_gi)
        out_kii.append( mean_kii )
        out_stdkii.append( std_kii )
        out_kri.append( mean_kri )
        out_stdkri .append( std_kri )
        out_iobs.append( mi[galaxy] )
        out_distmod.append( distanceModulus )
        out_iabs.append( iabs )
        out_rabs.append( rabs )
        out_mass_gr.append( mcMass )
        out_mass_gi.append( taMass )
        out_mass.append( fsMass )
        out_stdmass.append( std_masslight )
        out_bestsp.append( spIndex )
        out_bestzmet.append( zmets[spIndex] )
        out_bestchisq.append(chisqs[spIndex])
        out_sfr.append( mean_sfr )
        out_stdsfr.append( std_sfr  )
        out_age.append( mean_age )
        out_stdage.append( std_age  )
        out_zmet.append( mean_zmet )
        out_stdzmet.append( std_zmet  )
        out_zed.append( allzed[galaxy] )
    
    out_index = np.array(out_index).astype(int)
    out_id = np.array(out_id)#.astype(int)
    out_haloid = np.array(out_haloid).astype(int)
    out_gr = np.array(out_gr)
    out_stdgr = np.array(out_stdgr)
    out_gi = np.array(out_gi)
    out_stdgi = np.array(out_stdgi)
    out_kii = np.array(out_kii)
    out_stdkii = np.array(out_stdkii)
    out_kri = np.array(out_kri)
    out_stdkri  = np.array(out_stdkri )
    out_iobs = np.array(out_iobs)
    out_distmod = np.array(out_distmod)
    out_iabs = np.array(out_iabs)
    out_rabs = np.array(out_rabs)
    out_mass_gr = np.array(out_mass_gr)
    out_mass_gi = np.array(out_mass_gi)
    out_mass = np.array(out_mass)
    out_stdmass = np.array(out_stdmass)
    out_sfr = np.array(out_sfr)
    out_stdsfr = np.array(out_stdsfr)
    out_age = np.array(out_age)
    out_stdage = np.array(out_stdage)
    out_bestsp = np.array(out_bestsp)
    out_zmet = np.array(out_zmet)
    out_bestzmet = np.array(out_bestzmet)
    out_zed =np.array(out_zed)
    out_bestchisq = np.array(out_bestchisq)

    return [out_index,out_id,out_haloid,out_gr,out_stdgr,out_gi,out_stdgi,out_kii,out_stdkii,out_kri,out_stdkri,out_iobs,out_distmod,out_iabs,out_rabs,out_mass_gr,
            out_mass_gi,out_mass,out_stdmass,out_sfr,out_stdsfr,out_age,out_stdage,out_bestsp,out_zmet,out_bestzmet,out_zed,out_bestchisq]

def calcCOPA(inputDataDict, outfile, indir="simha/", lib="miles"):
    # import CosmologicalDistance
    from scipy.stats import chi2
    import os
    import logging
         
    logging.debug('Starting smass.calc()')

    # cd = CosmologicalDistance.CosmologicalDistance(omega_m=0.286,omega_l=0.714)
    ldistDict = dict()
    splineDict = dict()
    splines, zmet = loadPopColors.doAll(indir, lib=lib)
    index = inputDataDict["indices"]
    ids  = inputDataDict["id"]
    haloid  = inputDataDict["haloid"]
    mi  = inputDataDict["i"]
    ierr  = inputDataDict["ierr"]
    gr  = inputDataDict["gr"]
    ri  = inputDataDict["ri"]
    iz  = inputDataDict["iz"]
    grerr  = inputDataDict["grerr"]
    rierr  = inputDataDict["rierr"]
    izerr  = inputDataDict["izerr"]
    allzed = inputDataDict["zed"]

    #print zmet
    # protect against too small of errors => values = 0
    ix = np.nonzero(grerr < 0.02)
    grerr[ix] = 0.02
    ix = np.nonzero(rierr < 0.02)
    rierr[ix] = 0.02
    ix = np.nonzero(izerr < 0.02)
    izerr[ix] = 0.02

    # out_index,out_id,out_haloid,out_gr,out_stdgr,out_gi,out_stdgi,out_kii,out_stdkii,out_kri,out_stdkri,\
    # out_iobs,out_distmod,out_iabs,out_rabs,out_mass_gr,out_mass_gi,out_mass,out_stdmass,out_sfr,out_stdsfr,\
    # out_age,out_stdage,out_bestsp,out_zmet,out_bestzmet,out_zed,out_bestchisq = doCalcCOPA(ldistDict,splineDict,splines,zmet,index,ids,haloid,mi,ierr,gr,ri,iz,gr,grerr,rierr,izerr,allzed)

    # prepping for output
    nvariables = 27
    out_index, out_id, out_haloid, out_gr, out_stdgr, out_gi, out_stdgi, \
    out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
    out_bestzmet,out_stdzmet, out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, out_sfr, out_stdsfr, out_age, out_stdage, \
    out_zmet, out_zed = [[] for i in range(nvariables)]
    
    out_bestsp, out_bestchisq = [], []

    size = haloid.size

    #size = 10
    for galaxy in range(0,size) :
        zed = allzed[galaxy]

        logging.debug('{i} of {len}, z = {z}'.format(
            i = galaxy+1, len = size, z = zed))

        rest_gr,rest_gi,weight,chisqs = [], [],[],[]
        masslight, sfrs,ages, zmets,kii, kri = [],[],[],[],[],[]
        minChiSq = 999; spIndex = -1
        for sp in range(0,len(splines)) :
            # for speed
            skey = str(sp) + "-" + str(zed)
            sgr = splines[sp][0](zed)
            sri = splines[sp][1](zed)
            siz = splines[sp][2](zed)
            sgrr = splines[sp][4](zed) ;# restframe g-r
            sgir = splines[sp][5](zed) ;# restframe g-i
            skii = splines[sp][6](zed) ;# kcorrection: i_o - i_obs
            skri = splines[sp][7](zed) ;# kcorrection: r_o - i_obs
            sml = splines[sp][8](zed) ;# log(mass/light)  (M_sun/L_sun)
            ssfr = splines[sp][9](zed)
            if (ssfr<-20.): ssfr=-20.
            sage_cosmic = splines[sp][10](zed)
            sage = splines[sp][11](zed)
            szmet = zmet[sp]
                #To be changed if SFH changes

            #splineDict[skey] = sgr,sri,siz,sgrr,sgir,skii,skri,sml

            gre = grerr[galaxy]
            rie = rierr[galaxy]
            ize = izerr[galaxy]
            gr_chisq = pow((gr[galaxy] - sgr)/gre,2)
            ri_chisq = pow((ri[galaxy] - sri)/rie,2)
            iz_chisq = pow((iz[galaxy] - siz)/ize,2)
            rest_gr.append(sgrr)
            rest_gi.append(sgir)
            kii.append(skii)
            kri.append(skri)
            masslight.append(sml)
            sfrs.append(ssfr)
            ages.append(sage)
            zmets.append(szmet)
            chisq = gr_chisq + ri_chisq + iz_chisq
            probability = 1-chi2.cdf(chisq, 3-1) ;# probability of chisq greater than this
            weight.append(probability)
            chisqs.append(chisq)
        spIndex = np.argmax(weight)
        rest_gr = np.array(rest_gr)
        rest_gi = np.array(rest_gi)
        kii = np.array(kii)
        kri = np.array(kri)
        masslight = np.array(masslight)
        sfrs = np.array(sfrs)
        idx_sfr = (sfrs<-8.)
        ages = np.array(ages)
        weight = np.array(weight)
        gr_weighted = rest_gr * weight
        gi_weighted = rest_gi * weight
        kii_weighted = kii * weight
        kri_weighted = kri * weight
        #weight_norm = weight/np.sum(weight)
        masslight_weighted = masslight * weight
        sfr_weighted = 10**sfrs[idx_sfr] * weight[idx_sfr]
        age_weighted = ages * weight
        zmet_weighted = zmets * weight
        w1 = weight.sum()
        w2 = (weight**2).sum()
        if w1 == 0 : w1 = 1e-10
        if w2 == 0 : w2 = 1e-10
        mean_gr  = gr_weighted.sum()/w1
        mean_gi  = gi_weighted.sum()/w1
        mean_kii = kii_weighted.sum()/w1
        mean_kri = kri_weighted.sum()/w1
        mean_masslight = masslight_weighted.sum()/w1
        #try :
        #    if weight.shape[0]>1.:
        #        mean_sfr = float(ws.numpy_weighted_median(sfrs, weights=weight)) #np.median(sfr_weighted) #sfr_weighted.sum()/w1
        #    else:
        #        mean_sfr = -70.
        #except :
        #    mean_sfr = -70.
        #print mean_sfr
        mean_age = age_weighted.sum()/w1
        mean_zmet = zmet_weighted.sum()/w1
        mean_sfr = sfr_weighted.sum()/w1
        # unbiased weighted estimator of the sample variance
        w3 = w1**2 - w2
        if w3 == 0 : w3 = 1e-10
        var_gr = ( w1/w3 ) * (weight*(rest_gr - mean_gr)**2).sum()
        var_gi = ( w1/w3 ) * (weight*(rest_gi - mean_gi)**2).sum()
        var_kii = ( w1/w3 ) * (weight*(kii - mean_kii)**2).sum()
        var_kri = ( w1/w3 ) * (weight*(kri - mean_kii)**2).sum()
        var_masslight = ( w1/w3 ) * (weight*(masslight - mean_masslight)**2).sum()
        var_sfr = ( w1/w3 ) * (weight*(sfrs - mean_sfr)**2).sum()
        var_age = ( w1/w3 ) * (weight*(ages - mean_age)**2).sum()
        var_zmet = ( w1/w3 ) * (weight*(zmets - mean_zmet)**2).sum()
        std_gr = var_gr**0.5
        std_gi = var_gi**0.5
        std_kii = var_kii**0.5
        std_kri = var_kri**0.5
        std_masslight = var_masslight**0.5
        std_sfr = var_sfr**0.5
        std_age = var_age**0.5
        std_zmet = var_zmet**0.5 
        if std_gr > 99.99 : std_gr = 99.99
        if std_gi > 99.99 : std_gi = 99.99
        if std_kii > 99.99 : std_kii = 99.99
        if std_kri > 99.99 : std_kri = 99.99
        if std_sfr > 99.99 : std_sfr = 99.99
        if std_age > 99.99 : std_age = 99.99
        if std_masslight > 99.99 : std_masslight = 99.99
        if std_zmet > 99.99 : std_zmet = 99.99
	# Comment -distanceModulus out for fsps versions <2.5, as their mags don't include distance modulus
        if zed in ldistDict :
            lumdist = ldistDict[zed]
        else :
            # lumdist = cd.luminosity_distance(zed) ;# in Mpc
            lumdist = luminosityDistance(zed)

            ldistDict[zed] = lumdist
        distanceModulus = 5*np.log10(lumdist*1e6/10.)
        iabs = mi[galaxy] + mean_kii - distanceModulus
        rabs = mi[galaxy] + mean_kri - distanceModulus
        taMass = taylorMass(mean_gi, iabs)
        mcMass = mcintoshMass(mean_gr, rabs)
        fsMass = fspsMass( mean_masslight, iabs )
        # JTA: to make purely distance modulus
        #iabs = i[galaxy] - distanceModulus 
        #fsMass = gstarMass( iabs )

        # saving for output
        out_index.append( index[galaxy])
        out_id.append( ids[galaxy] )
        out_haloid.append( haloid[galaxy] )
        out_gr.append( mean_gr )
        out_stdgr.append(std_gr )
        out_gi.append( mean_gi )
        out_stdgi.append( std_gi)
        out_kii.append( mean_kii )
        out_stdkii.append( std_kii )
        out_kri.append( mean_kri )
        out_stdkri .append( std_kri )
        out_iobs.append( mi[galaxy] )
        out_distmod.append( distanceModulus )
        out_iabs.append( iabs )
        out_rabs.append( rabs )
        out_mass_gr.append( mcMass )
        out_mass_gi.append( taMass )
        out_mass.append( fsMass )
        out_stdmass.append( std_masslight )
        out_bestsp.append( spIndex )
        out_bestzmet.append( zmets[spIndex] )
        out_bestchisq.append(chisqs[spIndex])
        out_sfr.append( mean_sfr )
        out_stdsfr.append( std_sfr  )
        out_age.append( mean_age )
        out_stdage.append( std_age  )
        out_zmet.append( mean_zmet )
        out_stdzmet.append( std_zmet  )
        out_zed.append( allzed[galaxy] )
    
    out_index = np.array(out_index).astype(int)
    out_id = np.array(out_id)#.astype(int)
    out_haloid = np.array(out_haloid).astype(int)
    out_gr = np.array(out_gr)
    out_stdgr = np.array(out_stdgr)
    out_gi = np.array(out_gi)
    out_stdgi = np.array(out_stdgi)
    out_kii = np.array(out_kii)
    out_stdkii = np.array(out_stdkii)
    out_kri = np.array(out_kri)
    out_stdkri  = np.array(out_stdkri )
    out_iobs = np.array(out_iobs)
    out_distmod = np.array(out_distmod)
    out_iabs = np.array(out_iabs)
    out_rabs = np.array(out_rabs)
    out_mass_gr = np.array(out_mass_gr)
    out_mass_gi = np.array(out_mass_gi)
    out_mass = np.array(out_mass)
    out_stdmass = np.array(out_stdmass)
    out_sfr = np.array(out_sfr)
    out_stdsfr = np.array(out_stdsfr)
    out_age = np.array(out_age)
    out_stdage = np.array(out_stdage)
    out_bestsp = np.array(out_bestsp)
    out_zmet = np.array(out_zmet)
    out_bestzmet = np.array(out_bestzmet)
    out_zed =np.array(out_zed)
    out_bestchisq = np.array(out_bestchisq)

    col0=pf.Column(name='index',format='J',array=out_index)
    col1=pf.Column(name='CID',format='J',array=out_haloid)
    col2=pf.Column(name='redshift',format='E',array=out_zed)
    try:
        col3=pf.Column(name='ID',format='K',array=out_id)
    except:
        col3=pf.Column(name='ID',format='30A',array=out_id)
    col4=pf.Column(name='gr_o',format='E',array=out_gr)
    col5=pf.Column(name='gr_o_err',format='E',array=out_stdgr)
    col6=pf.Column(name='gi_o',format='E',array=out_gi)
    col7=pf.Column(name='gi_o_err',format='E',array=out_stdgi)
    col8=pf.Column(name='kri',format='E',array=out_kri)
    col9=pf.Column(name='kri_err',format='E',array=out_stdkri)
    col10=pf.Column(name='kii',format='E',array=out_kii)
    col11=pf.Column(name='kii_err',format='E',array=out_stdkii)
    col12=pf.Column(name='iobs',format='E',array=out_iobs)
    col13=pf.Column(name='distmod',format='E',array=out_distmod)
    col14=pf.Column(name='rabs',format='E',array=out_rabs)
    col15=pf.Column(name='iabs',format='E',array=out_iabs)
    col16=pf.Column(name='mcMass',format='E',array=out_mass_gr)
    col17=pf.Column(name='taMass',format='E',array=out_mass_gi)
    col18=pf.Column(name='mass',format='E',array=out_mass)
    col19=pf.Column(name='mass_err',format='E',array=out_stdmass)
    col20=pf.Column(name='ssfr',format='E',array=out_sfr)
    col21=pf.Column(name='ssfr_std',format='E',array=out_stdsfr)
    col22=pf.Column(name='mass_weight_age',format='E',array=out_age)
    col23=pf.Column(name='mass_weight_age_err',format='E',array=out_stdage)
    col24=pf.Column(name='best_model',format='E',array=out_bestsp)
    col25=pf.Column(name='best_zmet',format='E',array=out_bestzmet)
    col26=pf.Column(name='zmet',format='E',array=out_zmet)
    col27=pf.Column(name='best_chisq',format='E',array=out_bestchisq)

    cols=pf.ColDefs([col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27])
    tbhdu=pf.BinTableHDU.from_columns(cols)
    tbhdu.writeto(outfile,overwrite=True)
    logging.debug('Returning from smass.calc()')

#### end Johnny

def calc (inputDataDict, outfile, indir="simha/", lib="miles") :
    from scipy.stats import chi2
    import CosmologicalDistance
    import os
    import logging

    logging.debug('Starting smass.calc()')

    cd = CosmologicalDistance.CosmologicalDistance(omega_m=0.286,omega_l=0.714)
    ldistDict = dict()
    splineDict = dict()
    splines, zmet = loadPopColors.doAll(indir, lib=lib)
    ids  = inputDataDict["id"]
    haloid  = inputDataDict["haloid"]
    i  = inputDataDict["i"]
    ierr  = inputDataDict["ierr"]
    gr  = inputDataDict["gr"]
    ri  = inputDataDict["ri"]
    iz  = inputDataDict["iz"]
    grerr  = inputDataDict["grerr"]
    rierr  = inputDataDict["rierr"]
    izerr  = inputDataDict["izerr"]
    allzed = inputDataDict["zed"]
    GR_P_COLOR=inputDataDict["GR_P_COLOR"]
    RI_P_COLOR=inputDataDict["RI_P_COLOR"]
    IZ_P_COLOR=inputDataDict["IZ_P_COLOR"]
    P_RADIAL=inputDataDict["P_RADIAL"]
    P_REDSHIFT=inputDataDict["P_REDSHIFT"]
    P_MEMBER=inputDataDict["P_MEMBER"]
    DIST_TO_CENTER=inputDataDict["DIST_TO_CENTER"]
    GRP_RED=inputDataDict["GRP_RED"]
    GRP_BLUE=inputDataDict["GRP_BLUE"]
    RIP_RED=inputDataDict["RIP_RED"]
    RIP_BLUE=inputDataDict["RIP_BLUE"]
    IZP_RED=inputDataDict["IZP_RED"]
    IZP_BLUE=inputDataDict["IZP_BLUE"]

    #print zmet
    # protect against too small of errors => values = 0
    ix = np.nonzero(grerr < 0.02)
    grerr[ix] = 0.02
    ix = np.nonzero(rierr < 0.02)
    rierr[ix] = 0.02
    ix = np.nonzero(izerr < 0.02)
    izerr[ix] = 0.02

    # prepping for output
    out_id,out_haloid, out_gr, out_stdgr, out_gi, out_stdgi, \
        out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
        out_bestzmet,out_stdzmet, out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, out_sfr, out_stdsfr, out_age, out_stdage, \
        out_GR_P_COLOR,out_RI_P_COLOR,out_IZ_P_COLOR,out_P_RADIAL,out_P_REDSHIFT,out_P_MEMBER,out_DIST_TO_CENTER, \
        out_GRP_RED,out_GRP_BLUE,out_RIP_RED,out_RIP_BLUE,out_IZP_RED,out_IZP_BLUE,out_zmet, out_zed =\
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    out_bestsp, out_bestchisq = [], []

    size = haloid.size
    #size = 10
    for galaxy in range(0,size) :
        zed = allzed[galaxy]

        logging.debug('{i} of {len}, z = {z}'.format(
            i = galaxy+1, len = size, z = zed))

        rest_gr,rest_gi,weight,chisqs = [], [],[],[]
        masslight, sfrs,ages, zmets,kii, kri = [],[],[],[],[],[]
        minChiSq = 999; spIndex = -1
        for sp in range(0,len(splines)) :
            # for speed
            skey = str(sp) + "-" + str(zed)
            #if skey in splineDict :
            #    sgr,sri,siz,sgrr,sgir,skii,skri,sml = splineDict[skey]
            #else :
            #    sgr = splines[sp][0](zed)
            #    sri = splines[sp][1](zed)
            #    siz = splines[sp][2](zed)
            #    sgrr = splines[sp][4](zed) ;# restframe g-r
            #    sgir = splines[sp][5](zed) ;# restframe g-i
            #    skii = splines[sp][6](zed) ;# kcorrection: i_o - i_obs
            #    skri = splines[sp][7](zed) ;# kcorrection: r_o - i_obs
            #    sml = splines[sp][8](zed) ;# log(mass/light)  (M_sun/L_sun)
            #    ssfr = splines[sp][9](zed)
            #    sage_cosmic = splines[sp][10](zed)
            #    sage = splines[sp][11](zed)
            #    zmet = splines[12]
            sgr = splines[sp][0](zed)
            sri = splines[sp][1](zed)
            siz = splines[sp][2](zed)
            sgrr = splines[sp][4](zed) ;# restframe g-r
            sgir = splines[sp][5](zed) ;# restframe g-i
            skii = splines[sp][6](zed) ;# kcorrection: i_o - i_obs
            skri = splines[sp][7](zed) ;# kcorrection: r_o - i_obs
            sml = splines[sp][8](zed) ;# log(mass/light)  (M_sun/L_sun)
            ssfr = splines[sp][9](zed)
            if (ssfr<-20.): ssfr=-20.
            sage_cosmic = splines[sp][10](zed)
            sage = splines[sp][11](zed)
            szmet = zmet[sp]
                #To be changed if SFH changes

            #splineDict[skey] = sgr,sri,siz,sgrr,sgir,skii,skri,sml

            gre = grerr[galaxy]
            rie = rierr[galaxy]
            ize = izerr[galaxy]
            gr_chisq = pow((gr[galaxy] - sgr)/gre,2)
            ri_chisq = pow((ri[galaxy] - sri)/rie,2)
            iz_chisq = pow((iz[galaxy] - siz)/ize,2)
            rest_gr.append(sgrr)
            rest_gi.append(sgir)
            kii.append(skii)
            kri.append(skri)
            masslight.append(sml)
            sfrs.append(ssfr)
            ages.append(sage)
            zmets.append(szmet)
            chisq = gr_chisq + ri_chisq + iz_chisq
            probability = 1-chi2.cdf(chisq, 3-1) ;# probability of chisq greater than this
            weight.append(probability)
            chisqs.append(chisq)
        spIndex = np.argmax(weight)
        rest_gr = np.array(rest_gr)
        rest_gi = np.array(rest_gi)
        kii = np.array(kii)
        kri = np.array(kri)
        masslight = np.array(masslight)
        sfrs = np.array(sfrs)
        idx_sfr = (sfrs<-8.)
        ages = np.array(ages)
        weight = np.array(weight)
        gr_weighted = rest_gr * weight
        gi_weighted = rest_gi * weight
        kii_weighted = kii * weight
        kri_weighted = kri * weight
        #weight_norm = weight/np.sum(weight)
        masslight_weighted = masslight * weight
        sfr_weighted = 10**sfrs[idx_sfr] * weight[idx_sfr]
        age_weighted = ages * weight
        zmet_weighted = zmets * weight
        w1 = weight.sum()
        w2 = (weight**2).sum()
        if w1 == 0 : w1 = 1e-10
        if w2 == 0 : w2 = 1e-10
        mean_gr  = gr_weighted.sum()/w1
        mean_gi  = gi_weighted.sum()/w1
        mean_kii = kii_weighted.sum()/w1
        mean_kri = kri_weighted.sum()/w1
        mean_masslight = masslight_weighted.sum()/w1
        #try :
        #    if weight.shape[0]>1.:
        #        mean_sfr = float(ws.numpy_weighted_median(sfrs, weights=weight)) #np.median(sfr_weighted) #sfr_weighted.sum()/w1
        #    else:
        #        mean_sfr = -70.
        #except :
        #    mean_sfr = -70.
        #print mean_sfr
        mean_age = age_weighted.sum()/w1
        mean_zmet = zmet_weighted.sum()/w1
        mean_sfr = sfr_weighted.sum()/w1
        # unbiased weighted estimator of the sample variance
        w3 = w1**2 - w2
        if w3 == 0 : w3 = 1e-10
        var_gr = ( w1/w3 ) * (weight*(rest_gr - mean_gr)**2).sum()
        var_gi = ( w1/w3 ) * (weight*(rest_gi - mean_gi)**2).sum()
        var_kii = ( w1/w3 ) * (weight*(kii - mean_kii)**2).sum()
        var_kri = ( w1/w3 ) * (weight*(kri - mean_kii)**2).sum()
        var_masslight = ( w1/w3 ) * (weight*(masslight - mean_masslight)**2).sum()
        var_sfr = ( w1/w3 ) * (weight*(sfrs - mean_sfr)**2).sum()
        var_age = ( w1/w3 ) * (weight*(ages - mean_age)**2).sum()
        var_zmet = ( w1/w3 ) * (weight*(zmets - mean_zmet)**2).sum()
        std_gr = var_gr**0.5
        std_gi = var_gi**0.5
        std_kii = var_kii**0.5
        std_kri = var_kri**0.5
        std_masslight = var_masslight**0.5
        std_sfr = var_sfr**0.5
        std_age = var_age**0.5
        std_zmet = var_zmet**0.5 
        if std_gr > 99.99 : std_gr = 99.99
        if std_gi > 99.99 : std_gi = 99.99
        if std_kii > 99.99 : std_kii = 99.99
        if std_kri > 99.99 : std_kri = 99.99
        if std_sfr > 99.99 : std_sfr = 99.99
        if std_age > 99.99 : std_age = 99.99
        if std_masslight > 99.99 : std_masslight = 99.99
        if std_zmet > 99.99 : std_zmet = 99.99
	# Comment -distanceModulus out for fsps versions <2.5, as their mags don't include distance modulus
        if zed in ldistDict :
            lumdist = ldistDict[zed]
        else :
            lumdist = cd.luminosity_distance(zed) ;# in Mpc
            ldistDict[zed] = lumdist
        distanceModulus = 5*np.log10(lumdist*1e6/10.)
        iabs = i[galaxy] + mean_kii - distanceModulus
        rabs = i[galaxy] + mean_kri - distanceModulus
        taMass = taylorMass(mean_gi, iabs)
        mcMass = mcintoshMass(mean_gr, rabs)
        fsMass = fspsMass( mean_masslight, iabs )
        # JTA: to make purely distance modulus
        #iabs = i[galaxy] - distanceModulus 
        #fsMass = gstarMass( iabs )

        # saving for output
        out_id.append( ids[galaxy] )
        out_haloid.append( haloid[galaxy] )
        out_gr.append( mean_gr )
        out_stdgr.append(std_gr )
        out_gi.append( mean_gi )
        out_stdgi.append( std_gi)
        out_kii.append( mean_kii )
        out_stdkii.append( std_kii )
        out_kri.append( mean_kri )
        out_stdkri .append( std_kri )
        out_iobs.append( i[galaxy] )
        out_distmod.append( distanceModulus )
        out_iabs.append( iabs )
        out_rabs.append( rabs )
        out_mass_gr.append( mcMass )
        out_mass_gi.append( taMass )
        out_mass.append( fsMass )
        out_stdmass.append( std_masslight )
        out_bestsp.append( spIndex )
        out_bestzmet.append( zmets[spIndex] )
        out_bestchisq.append(chisqs[spIndex])
        out_sfr.append( mean_sfr )
        out_stdsfr.append( std_sfr  )
        out_age.append( mean_age )
        out_stdage.append( std_age  )
        out_zmet.append( mean_zmet )
        out_stdzmet.append( std_zmet  )
        out_zed.append( allzed[galaxy] )
        out_GR_P_COLOR.append( GR_P_COLOR[galaxy])
        out_RI_P_COLOR.append( RI_P_COLOR[galaxy])
        out_IZ_P_COLOR.append( IZ_P_COLOR[galaxy])
        out_P_RADIAL.append( P_RADIAL[galaxy])
        out_P_REDSHIFT.append( P_REDSHIFT[galaxy])
        out_P_MEMBER.append( P_MEMBER[galaxy])
        out_DIST_TO_CENTER.append( DIST_TO_CENTER[galaxy])
        out_GRP_RED.append( GRP_RED[galaxy])
        out_GRP_BLUE.append( GRP_BLUE[galaxy])
        out_RIP_RED.append( RIP_RED[galaxy])
        out_RIP_BLUE.append( RIP_BLUE[galaxy])
        out_IZP_RED.append( IZP_RED[galaxy])
        out_IZP_BLUE.append( IZP_BLUE[galaxy])

    out_id = np.array(out_id).astype(int)
    out_haloid = np.array(out_haloid).astype(int)
    out_gr = np.array(out_gr)
    out_stdgr = np.array(out_stdgr)
    out_gi = np.array(out_gi)
    out_stdgi = np.array(out_stdgi)
    out_kii = np.array(out_kii)
    out_stdkii = np.array(out_stdkii)
    out_kri = np.array(out_kri)
    out_stdkri  = np.array(out_stdkri )
    out_iobs = np.array(out_iobs)
    out_distmod = np.array(out_distmod)
    out_iabs = np.array(out_iabs)
    out_rabs = np.array(out_rabs)
    out_mass_gr = np.array(out_mass_gr)
    out_mass_gi = np.array(out_mass_gi)
    out_mass = np.array(out_mass)
    out_stdmass = np.array(out_stdmass)
    out_sfr = np.array(out_sfr)
    out_stdsfr = np.array(out_stdsfr)
    out_age = np.array(out_age)
    out_stdage = np.array(out_stdage)
    out_bestsp = np.array(out_bestsp)
    out_GR_P_COLOR = np.array(out_GR_P_COLOR)
    out_RI_P_COLOR = np.array(out_RI_P_COLOR)
    out_IZ_P_COLOR = np.array(out_IZ_P_COLOR)
    out_P_RADIAL = np.array(out_P_RADIAL)
    out_P_REDSHIFT = np.array(out_P_REDSHIFT)
    out_P_MEMBER = np.array(out_P_MEMBER)
    out_DIST_TO_CENTER = np.array(out_DIST_TO_CENTER)
    out_GRP_RED = np.array(out_GRP_RED)
    out_GRP_BLUE = np.array(out_GRP_BLUE)
    out_RIP_RED = np.array(out_RIP_RED)
    out_RIP_BLUE = np.array(out_RIP_BLUE)
    out_IZP_RED = np.array(out_IZP_RED)
    out_IZP_BLUE = np.array(out_IZP_BLUE)
    out_zmet = np.array(out_zmet)
    out_bestzmet = np.array(out_bestzmet)
    out_zed =np.array(out_zed)
    out_bestchisq = np.array(out_bestchisq)

    col1=pf.Column(name='MEM_MATCH_ID',format='J',array=out_haloid)
    col2=pf.Column(name='Z',format='E',array=out_zed)
    try:
        col3=pf.Column(name='ID',format='K',array=out_id)
    except:
        col3=pf.Column(name='ID',format='30A',array=out_id)
    col4=pf.Column(name='gr_o',format='E',array=out_gr)
    col5=pf.Column(name='gr_o_err',format='E',array=out_stdgr)
    col6=pf.Column(name='gi_o',format='E',array=out_gi)
    col7=pf.Column(name='gi_o_err',format='E',array=out_stdgi)
    col8=pf.Column(name='kri',format='E',array=out_kri)
    col9=pf.Column(name='kri_err',format='E',array=out_stdkri)
    col10=pf.Column(name='kii',format='E',array=out_kii)
    col11=pf.Column(name='kii_err',format='E',array=out_stdkii)
    col12=pf.Column(name='iobs',format='E',array=out_iobs)
    col13=pf.Column(name='distmod',format='E',array=out_distmod)
    col14=pf.Column(name='rabs',format='E',array=out_rabs)
    col15=pf.Column(name='iabs',format='E',array=out_iabs)
    col16=pf.Column(name='mcMass',format='E',array=out_mass_gr)
    col17=pf.Column(name='taMass',format='E',array=out_mass_gi)
    col18=pf.Column(name='mass',format='E',array=out_mass)
    col19=pf.Column(name='mass_err',format='E',array=out_stdmass)
    col20=pf.Column(name='ssfr',format='E',array=out_sfr)
    col21=pf.Column(name='ssfr_std',format='E',array=out_stdsfr)
    col22=pf.Column(name='mass_weight_age',format='E',array=out_age)
    col23=pf.Column(name='mass_weight_age_err',format='E',array=out_stdage)
    col24=pf.Column(name='best_model',format='E',array=out_bestsp)
    col25=pf.Column(name='best_zmet',format='E',array=out_bestzmet)
    col26=pf.Column(name='zmet',format='E',array=out_zmet)
    col27=pf.Column(name='best_chisq',format='E',array=out_bestchisq)
    col28=pf.Column(name='GR_P_COLOR',format='E',array=out_GR_P_COLOR)
    col29=pf.Column(name='RI_P_COLOR',format='E',array=out_RI_P_COLOR)
    col30=pf.Column(name='IZ_P_COLOR',format='E',array=out_IZ_P_COLOR)
    col31=pf.Column(name='P_RADIAL',format='E',array=out_P_RADIAL)
    col32=pf.Column(name='P_REDSHIFT',format='E',array=out_P_REDSHIFT)
    col33=pf.Column(name='P_MEMBER',format='E',array=out_P_MEMBER)
    col34=pf.Column(name='DIST_TO_CENTER',format='E',array=out_DIST_TO_CENTER)
    col35=pf.Column(name='GRP_RED',format='E',array=out_GRP_RED)
    col36=pf.Column(name='GRP_BLUE',format='E',array=out_GRP_BLUE)
    col37=pf.Column(name='RIP_RED',format='E',array=out_RIP_RED)
    col38=pf.Column(name='RIP_BLUE',format='E',array=out_RIP_BLUE)
    col39=pf.Column(name='IZP_RED',format='E',array=out_IZP_RED)
    col40=pf.Column(name='IZP_BLUE',format='E',array=out_IZP_BLUE)

    cols=pf.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40])
    tbhdu=pf.BinTableHDU.from_columns(cols)
    tbhdu.writeto(outfile,clobber=True)
    
    data = np.array([out_id, out_haloid, out_gr, out_stdgr, out_gi, out_stdgi, \
        out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
        out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, \
        out_zed,out_sfr, out_stdsfr, out_age, out_stdage, out_bestsp,out_bestzmet,out_zmet,out_GR_P_COLOR,out_RI_P_COLOR,out_IZ_P_COLOR,out_P_RADIAL,out_P_REDSHIFT,out_P_MEMBER,out_DIST_TO_CENTER, \
        out_GRP_RED,out_GRP_BLUE,out_RIP_RED,out_RIP_BLUE,out_IZP_RED,out_IZP_BLUE])

    #header = "# id, haloid, gr_o,  std,  gi_o, std, kri, std, kii, std, i,  distmod, "
    #header = header + "r_abs, i_abs, mcMass, taMass, mass, std, zed, sfr, sfrstd, age, agestd, bestsp,best_zmet,mean_zmet \
    #     out_GR_P_COLOR,out_RI_P_COLOR,out_IZ_P_COLOR,out_GR_P_MEMBER,out_RI_P_MEMBER,out_IZ_P_MEMBER,out_DIST_TO_CENTER, \
    #    out_GRP_RED,out_GRP_BLUE,out_RIP_RED,out_RIP_BLUE,out_IZP_RED,out_IZP_BLUE\n"
    #fd = open(outfile,"w")
    #fd.write(header)
    #fd.close()
    #np.savetxt(outfile+".dat", data.T, "%d,%d,%6.3f,%6.4f,%6.3f,%6.4f,%6.3f,%6.4f,%6.3f,%6.4f,\
    #     %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.4f,%6.3f,%6.3f,%6.3f,%d,%6.3f, %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.4f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f, %6.3f")
    #os.system("cat {} >> {}; rm {}".format(outfile+".dat", outfile, outfile+".dat"))

    logging.debug('Returning from smass.calc()')


def taylorMass (gi, iabs) :
    # equation 8, taylor et al 2011 MNRAS, V418, Issue 3, pp. 1587-1620
    mass = -0.68 + 0.70*gi - 0.4*(iabs - 4.58)
    # assumes h=0.7
    out_stdgi = np.array(out_stdgi)
    out_kii = np.array(out_kii)
    out_stdkii = np.array(out_stdkii)
    out_kri = np.array(out_kri)
    out_stdkri  = np.array(out_stdkri )
    out_iobs = np.array(out_iobs)
    out_distmod = np.array(out_distmod)
    out_iabs = np.array(out_iabs)
    out_rabs = np.array(out_rabs)
    out_mass_gr = np.array(out_mass_gr)
    out_mass_gi = np.array(out_mass_gi)
    out_mass = np.array(out_mass)
    out_stdmass = np.array(out_stdmass)
    out_sfr = np.array(out_sfr)
    out_stdsfr = np.array(out_stdsfr)
    out_age = np.array(out_age)
    out_stdage = np.array(out_stdage)
    out_bestsp = np.array(out_bestsp)
    data = np.array([out_id, out_gr, out_stdgr, out_gi, out_stdgi, \
        out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
        out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, \
        out_sfr, out_stdsfr, out_age, out_stdage, out_bestsp])
    header = "# id, gr_o,  std,  gi_o, std, kri, std, kii, std, i,  distmod, "
    header = header + "r_abs, i_abs, mcMass, taMass, mass, std, sfr, sfrstd, age, agestd, bestsp\n"

    data = np.array([out_id, out_gr, out_stdgr, out_gi, out_stdgi, \
        out_kri, out_stdkri, out_kii, out_stdkii, out_iobs, out_distmod, \
        out_rabs, out_iabs, out_mass_gr, out_mass_gi, out_mass, out_stdmass, out_bestsp])
    header = "# id, gr_o,  std,  gi_o, std, kri, std, kii, std, i,  distmod, "
    header = header + "r_abs, i_abs, mcMass, taMass, mass, std, bestsp\n"
    fd = open(outfile,"w")
    fd.write(header)
    fd.close()
    np.savetxt(outfile+".dat", data.T, "%d,%6.3f,%6.4f,%6.3f,%6.4f,%6.3f,%6.4f,%6.3f,%6.4f,\
         %6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.4f,%6.3f,%6.3f,%6.3f,%6.3f,%d")
    os.system("cat {} >> {}; rm {}".format(outfile+".dat", outfile, outfile+".dat"))


def mcintoshMass (gr, rabs, h=0.7) :
    # equation 1, McIntosh et al 2014, arXiv:1308.0054v2
    mass = -0.406 + 1.097*gr - 0.4*(rabs - 5*np.log10(h) - 4.64)
    # log(mass/h^2) 
    return mass

def taylorMass (gi, iabs) :
    mass = -0.68 + 0.70*gi - 0.4*(iabs - 4.58)
    # assumes h=0.7
    return mass

def fspsMass( masstolight , iabs) :
    # following the above, which assumes h=0.7
    mass =  masstolight - 0.4*(iabs - 4.58)
    return mass
# how far off is just distance modulus?
def gstarMass(iabs) :
    mass =  - 0.4*(iabs - 4.58)
    return mass
