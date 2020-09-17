# Credit for most of this code is due to Alex Drlica-Wagner
import numpy as np
import fitsio
import glob
import os

from astropy.table import Table, join

def readfile(filename,columns=None):
    '''
    Read a filename with fitsio.read and return an astropy table
    '''
    hdu = fitsio.read(filename, columns=columns)
    return Table(hdu)

def loadfiles(filenames, columns=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    out = []
    i = 1
    print
    for f in filenames:
        print 'File {i}: {f}'.format(i=i, f=f)
        out.append(fitsio.read(f, columns=columns))
        i += 1

    return np.concatenate(out)

def printCompleteMsg(outFile):
    print '\n--> Status: File Complete <{outFile}>\n'.format(outFile=outFile)


# For combining afterburner outputs on mof healpixed data
def combineAfterBurnerOutputs():
    print
    option = raw_input('Enter clusters or members: ')

    if (option == 'cluster' or option == 'clusters'):

        filenames = sorted(glob.glob('/data/des40.a/data/jburgad/clusters/outputs_brian/lgt5_mof_clusters_*'))
        data = loadfiles(filenames) # Call function and load all data sets
        out_dir = '/data/des40.a/data/jburgad/clusters/outputs_brian/'
        outfile = out_dir + 'lgt5_mof_clusters_full.fit'

        # Define afterburner column names for printing (disregard variable names)
        hostid = data['MEM_MATCH_ID'][:]
        zg = data['RA'][:]
        galid = data['DEC'][:]
        gro = data['Z'][:]
        gro_err = data['R200'][:]
        gio = data['M200'][:]
        gio_err = data['N200'][:]
        kri = data['LAMBDA_CHISQ'][:]
        kri_err = data['GR_SLOPE'][:]
        kii = data['GR_INTERCEPT'][:]
        kii_err = data['GRMU_R'][:]
        iobs = data['GRMU_B'][:]
        distmod = data['GRSIGMA_R'][:]
        rabs = data['GRSIGMA_B'][:]
        iabs = data['GRW_R'][:]
        mcMass = data['GRW_B'][:]
        taMass = data['RI_SLOPE'][:]
        mass = data['RI_INTERCEPT'][:]
        mass_err = data['RIMU_R'][:]
        ssfr = data['RIMU_B'][:]
        ssfr_std = data['RISIGMA_R'][:]
        mass_weight_age = data['RISIGMA_B'][:]
        mass_weight_age_err = data['RIW_R'][:]
        best_model = data['RIW_B'][:]
        best_zmet = data['GRMU_BG'][:]
        zmet = data['GRSIGMA_BG'][:]
        best_chisq = data['GRW_BG'][:]
        grpc = data['IZ_SLOPE'][:]
        ripc = data['IZ_INTERCEPT'][:]
        izpc = data['IZMU_R'][:]
        grpm = data['IZMU_B'][:]
        ripm = data['IZSIGMA_R'][:]
        izpm = data['IZSIGMA_B'][:]
        dist2c = data['IZW_R'][:]
        grpr = data['IZW_B'][:]
        grpb = data['FLAGS'][:]
        ripr = data['Z_ERR'][:]

        # Now print out the combined catalog
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        names = ['MEM_MATCH_ID','RA','DEC','Z','R200','M200','N200','LAMBDA_CHISQ','GR_SLOPE','GR_INTERCEPT','GRMU_R','GRMU_B','GRSIGMA_R','GRSIGMA_B','GRW_R','GRW_B','RI_SLOPE','RI_INTERCEPT','RIMU_R','RIMU_B','RISIGMA_R','RISIGMA_B','RIW_R','RIW_B','GRMU_BG','GRSIGMA_BG','GRW_BG','IZ_SLOPE','IZ_INTERCEPT','IZMU_R','IZMU_B','IZSIGMA_R','IZSIGMA_B','IZW_R','IZW_B','FLAGS','Z_ERR']
        array_list = [hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,zmet,best_chisq,grpc,ripc,izpc,grpm,ripm,izpm,dist2c,grpr,grpb,ripr]
        fits.write(array_list, names=names)

    elif (option == 'member' or option == 'members'):

        filenames = sorted(glob.glob('/data/des40.a/data/jburgad/clusters/outputs_brian/lgt5_mof_members_*'))
        data = loadfiles(filenames) # Call function and load all data sets
        out_dir = '/data/des40.a/data/jburgad/clusters/outputs_brian/'
        outfile = out_dir + 'lgt5_mof_members_full.fit'

        # Define afterburner column names for printing (disregard variable names)
        hostid = data['COADD_OBJECTS_ID'][:]
        zg = data['HOST_HALOID'][:]
        galid = data['RA'][:]
        gro = data['DEC'][:]
        gro_err = data['ZP'][:]
        gio = data['ZPE'][:]
        gio_err = data['MAG_AUTO_G'][:]
        kri = data['MAG_AUTO_R'][:]
        kri_err = data['MAG_AUTO_I'][:]
        kii = data['MAG_AUTO_Z'][:]
        kii_err = data['P_RADIAL'][:]
        iobs = data['P_REDSHIFT'][:]
        distmod = data['GR_P_COLOR'][:]
        rabs = data['RI_P_COLOR'][:]
        iabs = data['IZ_P_COLOR'][:]
        mcMass = data['P_MEMBER'][:]
        taMass = data['AMAG_R'][:]
        mass = data['DIST_TO_CENTER'][:]
        mass_err = data['GRP_RED'][:]
        ssfr = data['GRP_BLUE'][:]
        ssfr_std = data['GRP_BG'][:]
        mass_weight_age = data['RIP_RED'][:]
        mass_weight_age_err = data['RIP_BLUE'][:]
        best_model = data['IZP_RED'][:]
        best_zmet = data['IZP_BLUE'][:]
        zmet = data['GR0'][:]
        best_chisq = data['HOST_REDSHIFT'][:]
        grpc = data['HOST_REDSHIFT_ERR'][:]
        ripc = data['MAGERR_AUTO_G'][:]
        izpc = data['MAGERR_AUTO_R'][:]
        grpm = data['MAGERR_AUTO_I'][:]
        ripm = data['MAGERR_AUTO_Z'][:]

        # Now print out the combined catalog
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        names = ['COADD_OBJECTS_ID','HOST_HALOID','RA','DEC','ZP','ZPE','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z','P_RADIAL','P_REDSHIFT','GR_P_COLOR','RI_P_COLOR','IZ_P_COLOR','P_MEMBER','AMAG_R','DIST_TO_CENTER','GRP_RED','GRP_BLUE','GRP_BG','RIP_RED','RIP_BLUE','IZP_RED','IZP_BLUE','GR0','HOST_REDSHIFT','HOST_REDSHIFT_ERR','MAGERR_AUTO_G','MAGERR_AUTO_R','MAGERR_AUTO_I','MAGERR_AUTO_Z']
        array_list = [hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,zmet,best_chisq,grpc,ripc,izpc,grpm,ripm]
        fits.write(array_list, names=names)
    printCompleteMsg(outfile)


# For combining the standard outputs of the BMA computeStellarMass 
def combineBMAStellarMassOutput(stellarMassOutPrefix):
    '''
    Note: to read a subset of data, columns = ['column1','column2']; data = loadfiles(filenames,columns=columns)
    '''
    import logging
    logging.info('Starting combineCat.combineBMAStellarMassOutput()')

    fileNames = sorted(glob.glob(stellarMassOutPrefix+'*'))
    # Call function and load all data sets
    data = loadfiles(fileNames) 
    outFile = stellarMassOutPrefix + 'full.fits'

    # Define BMA stellar mass column names for printing
    hostid = data['MEM_MATCH_ID'][:]
    zg = data['Z'][:]
    galid = data['ID'][:]
    gro = data['gr_o'][:]
    gro_err = data['gr_o_err'][:]
    gio = data['gi_o'][:]
    gio_err = data['gi_o_err'][:]
    kri = data['kri'][:]
    kri_err = data['kri_err'][:]
    kii = data['kii'][:]
    kii_err = data['kii_err'][:]
    iobs = data['iobs'][:]
    distmod = data['distmod'][:]
    rabs = data['rabs'][:]
    iabs = data['iabs'][:]
    mcMass = data['mcMass'][:]
    taMass = data['taMass'][:]
    mass = data['mass'][:]
    mass_err = data['mass_err'][:]
    ssfr = data['ssfr'][:]
    ssfr_std = data['ssfr_std'][:]
    mass_weight_age = data['mass_weight_age'][:]
    mass_weight_age_err = data['mass_weight_age_err'][:]
    best_model = data['best_model'][:]
    best_zmet = data['best_zmet'][:]
    zmet = data['zmet'][:]
    best_chisq = data['best_chisq'][:]
    grpc = data['GR_P_COLOR'][:]
    ripc = data['RI_P_COLOR'][:]
    izpc = data['IZ_P_COLOR'][:]
    grpm = data['P_RADIAL'][:]
    ripm = data['P_REDSHIFT'][:]
    izpm = data['P_MEMBER'][:]
    dist2c = data['DIST_TO_CENTER'][:]
    grpr = data['GRP_RED'][:]
    grpb = data['GRP_BLUE'][:]
    ripr = data['RIP_RED'][:]
    ripb = data['RIP_BLUE'][:]
    izpr = data['IZP_RED'][:]
    izpb = data['IZP_BLUE'][:]

    # Now print out the combined catalog
    fits = fitsio.FITS(outFile, 'rw', clobber=True)
    names = ['MEM_MATCH_ID','Z','ID','gr_o','gr_o_err','gi_o','gi_o_err','kri','kri_err','kii','kii_err','iobs','distmod','rabs','iabs','mcMass','taMass','mass','mass_err','ssfr','ssfr_std','mass_weight_age','mass_weight_age_err','best_model','best_zmet','zmet','best_chisq','GR_P_COLOR','RI_P_COLOR','IZ_P_COLOR','P_RADIAL','P_REDSHIFT','P_MEMBER','DIST_TO_CENTER','GRP_RED','GRP_BLUE','RIP_RED','RIP_BLUE','IZP_RED','IZP_BLUE']
    array_list = [hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,zmet,best_chisq,grpc,ripc,izpc,grpm,ripm,izpm,dist2c,grpr,grpb,ripr,ripb,izpr,izpb]
    fits.write(array_list, names=names)
    
    logging.info('Returning from combineCat.combineBMAStellarMassOutput()')

## Johnny at 27th October 2019
# For combining the standard outputs of the BMA computeStellarMass COPA
def combineBMAStellarMassOutputCOPA(stellarMassOutPrefix,save=False):
    '''
    Note: to read a subset of data, columns = ['column1','column2']; data = loadfiles(filenames,columns=columns)
    '''
    import logging
    logging.info('Starting combineCat.combineBMAStellarMassOutput()')

    fileNames = sorted(glob.glob(stellarMassOutPrefix+'_'+('[0-9]' * 5)+'.fits'))
    # Call function and load all data sets
    data = loadfiles(fileNames) 
    outFile = stellarMassOutPrefix + 'full.fits'

    # Define BMA stellar mass column names for printing
    index = data['index'][:]
    hostid = data['CID'][:]
    zg = data['redshift'][:]
    galid = data['ID'][:]
    gro = data['gr_o'][:]
    gro_err = data['gr_o_err'][:]
    gio = data['gi_o'][:]
    gio_err = data['gi_o_err'][:]
    kri = data['kri'][:]
    kri_err = data['kri_err'][:]
    kii = data['kii'][:]
    kii_err = data['kii_err'][:]
    iobs = data['iobs'][:]
    distmod = data['distmod'][:]
    rabs = data['rabs'][:]
    iabs = data['iabs'][:]
    mcMass = data['mcMass'][:]
    taMass = data['taMass'][:]
    mass = data['mass'][:]
    mass_err = data['mass_err'][:]
    ssfr = data['ssfr'][:]
    ssfr_std = data['ssfr_std'][:]
    mass_weight_age = data['mass_weight_age'][:]
    mass_weight_age_err = data['mass_weight_age_err'][:]
    best_model = data['best_model'][:]
    best_zmet = data['best_zmet'][:]
    zmet = data['zmet'][:]
    best_chisq = data['best_chisq'][:]
    
    if save:
        # Now print out the combined catalog
        fits = fitsio.FITS(outFile, 'rw', clobber=True)
        names = ['index','CID','redshift','ID','gr_o','gr_o_err','gi_o','gi_o_err','kri','kri_err','kii','kii_err','iobs','distmod','rabs','iabs','mcMass','taMass','mass','mass_err','ssfr','ssfr_std','mass_weight_age','mass_weight_age_err','best_model','best_zmet','zmet','best_chisq']
        array_list = [index,hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,zmet,best_chisq]
        fits.write(array_list, names=names)
        
    logging.info('Returning from combineCat.combineBMAStellarMassOutput()')
    return Table(data)

def joinTable(data0,data1):
    data0['index'] = np.arange(0,len(data0),1,dtype=int)
    
    new_data = join(data0, data1, keys='index', join_type='outer')
    new_data.remove_columns(['index','ID'])
    return new_data
    
def matchBMAStellarMassOutputCOPA(stellarMassInfile,stellarMassOutPrefix,save=False):
    import logging
    logging.info('Starting combineCat.matchBMAStellarMassOutput()')

    outfilename = stellarMassOutPrefix+'.fits'
    print('out member files:',outfilename)

    BMAdata = combineBMAStellarMassOutputCOPA(stellarMassOutPrefix,save=save)    
    olddata = readfile(stellarMassInfile)

    BMAdata.remove_columns(['CID','redshift'])
    
    new_data = joinTable(olddata,BMAdata)

    new_data.write(outfilename,format='fits',overwrite=True)
    logging.info('Returning from combineCat.matchBMAStellarMassOutput()')

## end Johnny 