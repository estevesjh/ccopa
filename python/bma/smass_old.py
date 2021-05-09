import numpy as np
import h5py

#import helperfunctions
#import weightedstats as ws
#import pyfits as pf
from astropy.io import fits as pf
#
# inputDataDict is what holds the input photometry
# it expects ids, i, ierr, gr,ri,iz, grerr, rierr, izerr
#
#

#### Johnny at 27 october 2019.
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

def luminosityDistance(z):
    DL = float( cosmo.luminosity_distance(z) / u.Mpc ) # em Mpc
    return DL

def calc_copa_v2(infile, outfile, indir="simha/", lib="miles"):
    # import CosmologicalDistance
    from scipy.stats import chi2
    import os
    import logging
    import loadPopColors

    inputDataDict = load_input(infile)

    logging.debug('Starting smass.calc()')

    # cd = CosmologicalDistance.CosmologicalDistance(omega_m=0.286,omega_l=0.714)
    ldistDict = dict()
    splineDict = dict()
    splines, zmet = loadPopColors.doAll(indir, lib=lib)
    index = inputDataDict["indices"]
    ids  = inputDataDict["mid"]
    haloid  = inputDataDict["sid"]
    mi  = inputDataDict["i"]
    ierr  = inputDataDict["ierr"]
    gr  = inputDataDict["gr"]
    ri  = inputDataDict["ri"]
    iz  = inputDataDict["iz"]
    grerr  = inputDataDict["grerr"]
    rierr  = inputDataDict["rierr"]
    izerr  = inputDataDict["izerr"]
    allzed = inputDataDict["redshift"]

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

    columns = ['index','mid','sid','redshift','gr_o','gr_o_err','gi_o','gi_o_err','kii','kii_err','kri','kri_err','iobs',
    'distmod','iabs','rabs','mcMass','taMass','mass','mass_err','ssfr','ssfr_std','mass_weight_age','mass_weight_age_err''best_model','best_zmet',
    'zmet','best_chisq']

    vectors = [out_index,out_id,out_haloid,out_gr,out_stdgr,out_gi,out_stdgi,out_kii,out_stdkii,out_kri,out_stdkri,out_iobs,out_distmod,out_iabs,out_rabs,out_mass_gr,
            out_mass_gi,out_mass,out_stdmass,out_sfr,out_stdsfr,out_age,out_stdage,out_bestsp,out_zmet,out_bestzmet,out_zed,out_bestchisq]

    mydict = dict().fromkeys(columns)
    for col,arr in zip(columns,vectors):
        mydict[col] = arr
    
    write_hdf5_file(outfile,mydict,columns)

    logging.debug('Returning from smass.calc()')


def load_input(fname):
    fmaster = h5py.File(fname, 'r')
    columns = list(fmaster['bma'].keys())

    mydict = dict().fromkeys(columns)
    for col in columns:
        mydict[col] = fmaster['bma/%s/'%col][:]
    return mydict

def write_hdf5_file(fname,table,columns):
    fmaster = h5py.File(fname, 'a')
    fmaster.create_group('bma_old')
    for col in columns:
        fmaster.create_dataset('bma_old/%s/'%col, data= table[col][:])
    fmaster.close()

#### end Johnny

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
