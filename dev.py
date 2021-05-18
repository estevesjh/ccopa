import sys
import numpy as np
import h5py
from time import time
# sys.path.append('/home/johnny/Documents/copa_2.1')
sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana

t0 = time()
cfg = '/home/s1/jesteves/git/ccopa/config_files/config_buzzard_v2.yaml'
copa = copacabana(cfg,dataset='buzzard_v2')

# Make infile
# copa.make_input_file()

## make fake-photoz
# sys.path.append('/home/s1/jesteves/git/ccopa/scripts')
# import make_gaussianPz
# make_gaussianPz.main()

## Run Copa & BMA
#copa.run_bma_healpix(nCores=60,overwrite=False)

# copa.kwargs['z_window'] = -1
# copa.run_copa_healpix('testEmu', pz_file='emuBPZ', nCores=60)
copa.compute_muStar('testEmu', overwrite=True)

# total_time = (time()-t0)/60
# print('\n')
# print('Total time: %.2f min'%total_time)

# copa.run_copa(run, pz_file='emuBPZ_zww', nCores=60, old_code=False)


# from scipy.interpolate import interp1d
# def getMagLimModel_04L(auxfile,zvec,dm=0):
#     '''
#     Get the magnitude limit for 0.4L*

#     file columns: z, mag_lim_i, (g-r), (r-i), (i-z)
#     mag_lim = mag_lim + dm

#     input: Galaxy clusters redshift
#     return: mag. limit for each galaxy cluster and for the r,i,z bands
#     output = [maglim_r(array),maglim_i(array),maglim_z(array)]
#     '''
#     annis=np.loadtxt(auxfile)
#     jimz=[i[0] for i in annis]  ## redshift
#     jimgi=[i[1] for i in annis] ## mag(i-band)
#     jimgr=[i[2] for  i in annis]## (g-r) color
#     jimri=[i[3] for i in annis] ## (r-i) color
#     jimiz=[i[4] for i in annis] ## (i-z) color

#     interp_magi=interp1d(jimz,jimgi,fill_value='extrapolate')
#     interp_gr=interp1d(jimz,jimgr,fill_value='extrapolate')
#     interp_ri=interp1d(jimz,jimri,fill_value='extrapolate')
#     interp_iz=interp1d(jimz,jimiz,fill_value='extrapolate')

#     mag_i,color_ri,color_iz = interp_magi(zvec),interp_ri(zvec),interp_iz(zvec)
#     mag_r, mag_z = (color_ri+mag_i),(mag_i-color_iz)

#     maglim_r, maglim_i, maglim_z = (mag_r+dm),(mag_i+dm),(mag_z+dm)

#     magLim = np.vstack([maglim_r, maglim_i, maglim_z])

#     return magLim#.transpose()



# import glob
# mydir = '/data/des61.a/data/johnny/Buzzard/Buzzard_v2.0.0/output/tiles/'
# basename = 'buzzard_v2.0.0_copa_golden_{:05d}.hdf5'
# files = glob.glob(mydir+'*.hdf5')

# for mfile in files:
#     print('infile: %s \n'%mfile)
#     master = h5py.File(mfile,'a')
#     cluster = master['clusters/main/']

#     zcls = cluster['redshift']
#     magLim = getMagLimModel_04L(auxfile,zcls,dm=0).T

#     cluster.create_dataset('magLim',data=magLim)
#     master.close()