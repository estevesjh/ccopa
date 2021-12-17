import sys
import numpy as np
import h5py
from time import time
# sys.path.append('/home/johnny/Documents/copa_2.1')
sys.path.append('/home/s1/jesteves/git/ccopa/python')
from main import copacabana
from make_input_files.make_photoz_pz import generate_photoz_models

t0 = time()
# cfg = '/home/s1/jesteves/git/ccopa/config_files/config_buzzard_v2.yaml'
# copa = copacabana(cfg,dataset='buzzard_v2')

# cfg = '/home/s1/jesteves/git/ccopa/config_files/config_copa_dc2.yaml'
# copa = copacabana(cfg)


root = '/home/s1/jesteves/git/buzzardAnalysis/mainAnalysis/'
cfg  = root+'config_buzzard_rm_v2.yaml'
copa = copacabana(cfg,dataset='buzzard_v2')
copa.run_bma_healpix(nCores=65,overwrite=True)

# copa.kwargs['dmag_lim']     = 0.
# copa.kwargs['r_aper_model'] = 'r200'
# copa.kwargs['z_window']     = 0.03

# # generate_photoz_models('gaussian',[copa.master_fname],0.03,
# #                        group_name='guass003-corr2',nCores=60)

# copa.run_copa('%s-r200-ztest-bkg'%('gauss003-v2'),pz_file=u'guass003-corr2', nCores=60)
# copa.compute_muStar('%s-r200-ztest-bkg'%('gauss003-v2'), overwrite=True)

# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.03,
#                        group_name='guass003-corr',nCores=60)

# copa.run_copa_healpix('%s-r200-ztest'%('gauss003-v2'),pz_file=u'guass003-corr', nCores=60)
# copa.compute_muStar('%s-r200-ztest'%('gauss003-v2'), overwrite=True)

# generate_photoz_models('gaussian',[copa.master_fname],0.03,
#                        group_name='guass003-counts',method='counts',nCores=60)
# copa.run_copa('%s-r200-ztest'%('gauss003-bkg'),pz_file=u'guass003', nCores=60)
# copa.run_copa('%s-r200-ztest'%('gauss003-count-bkg'), pz_file=u'guass003-count', nCores=60)
# copa.compute_muStar('%s-r200-ztest'%('gauss003-count-bkg'), overwrite=True)
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.03,
#                        group_name='guass003-upd',nCores=60)

# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.01,
#                        group_name='guass001-corr',nCores=60)

# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.05,
#                        group_name='guass005-corr',nCores=60)

# copa.run_copa_healpix('%s-r200-ztest'%('gauss003'),pz_file=u'guass003-upd', nCores=60)
# copa.run_copa_healpix('%s-r200-ztest'%('gauss003-corr'),pz_file=u'guass003-corr', nCores=60)
# copa.run_copa_healpix('%s-r200-ztest'%('gauss003-count'), pz_file=u'guass003-count', nCores=60)

# copa.compute_muStar('%s-r200-ztest'%('gauss003'), overwrite=True)
# copa.compute_muStar('%s-r200-ztest'%('gauss003-corr'), overwrite=True)
# copa.compute_muStar('%s-r200-ztest'%('gauss003-count'), overwrite=True)

## Make infile
# copa.make_input_file()
# copa.run_bma_healpix(nCores=60,overwrite=False)

# zw = './aux_files/zwindow_model_buzzard_dnf.txt'
# generate_photoz_models('bias'    ,copa.master_fname_tile_list,0.03,zmodel_file=zw,group_name='dnf_model',nCores=60)
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.01,zmodel_file=None,group_name='pz',nCores=60)
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.03,zmodel_file=None,group_name='pz',nCores=60)
# generate_photoz_models('gaussian',copa.master_fname_tile_list,0.05,zmodel_file=None,group_name='pz',nCores=60)

# copa.kwargs['r_aper_model'] = 'hod'
# copa.kwargs['dmag_lim']     = 1.

# copa.kwargs['z_window']     = 0.01
# copa.run_copa_healpix('%s-rhod-zerr'%('gauss001'), pz_file='gauss001',  nCores=60)
# copa.compute_muStar('%s-rhod-zerr'%('gauss001'), overwrite=True)

# copa.kwargs['z_window']     = 0.03
# copa.run_copa_healpix('%s-rhod'%('gauss003'),   pz_file='gauss003', nCores=60)

# # copa.kwargs['z_window']     = 0.05
# # copa.run_copa_healpix('%s-rhod-zerr'%('gauss005'), pz_file='gauss005',  nCores=60)
# # copa.compute_muStar('%s-rhod-zerr'%('gauss005'), overwrite=True)

# copa.kwargs['dmag_lim']     = 1.
# copa.kwargs['r_aper_model'] = 'r200'

# copa.kwargs['z_window']     = 0.01
# copa.run_copa_healpix('%s-r200-zerr'%('gauss001'), pz_file='gauss001',  nCores=60)
# copa.compute_muStar('%s-r200-zerr'%('gauss001'), overwrite=True)

# copa.kwargs['z_window']     = 0.03
# copa.run_copa_healpix('%s-r200-zerr'%('gauss003'),   pz_file='gauss003', nCores=60)
# copa.compute_muStar('%s-r200-zerr'%('gauss003'), overwrite=True)

# copa.kwargs['z_window']     = 0.05
# copa.run_copa_healpix('%s-r200-zerr'%('gauss005'), pz_file='gauss005',  nCores=60)
# copa.compute_muStar('%s-r200-zerr'%('gauss005'), overwrite=True)


# copa.compute_muStar('%s-rhod'%('dnf003')     , overwrite=True)
# copa.compute_muStar('%s-rhod'%('dnf')     , overwrite=True)
# copa.compute_muStar('%s-rhod'%('gauss001'), overwrite=True)
# copa.compute_muStar('%s-rhod'%('gauss003'), overwrite=True)
# copa.compute_muStar('%s-rhod'%('gauss005'), overwrite=True)


# total_time = (time()-t0)/60/60
# print('\n')
# print('Total time: %.2f hours'%total_time)