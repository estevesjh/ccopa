import sys
import numpy as np
import h5py
sys.path.append('/home/johnny/Documents/copa_2.1')

from libs.main import copacabana

cfg = 'libs/config_copa_dc2.yaml'
copa = copacabana(cfg)

# copa.make_input_file()
# copa.run_bma(nCores=50)
## run to all galaxies inside 3Mpc
## redshift of the cluster

runs    = ['bpz-rhod','emuBPZ-rhod','gauss003-rhod','gauss005-rhod']
pzfiles = ['bpz'     ,'emuBPZ'     ,'gauss003'     ,'gauss005'     ]
z_window= [0.03, 0.05, 0.03, 0.05]

# run copa
for run,zw,pzfile in zip(runs[1:2],z_window[1:2],pzfiles[1:2]):
    print('\nRun: %s'%(run))
    print('z window: %.2f'%(zw))
    copa.kwargs['z_window'] = zw
    copa.run_copa(run, pz_file=pzfile, nCores=40, old_code=False)
    copa.compute_muStar(run, overwrite=True, true_members=False)
    copa.compute_muStar(run, overwrite=True, true_members=True)

    # copa.run_copa(run+'_old', pz_file=pzfile, nCores=40, old_code=True)
    # copa.compute_muStar(run+'_old', overwrite=True, true_members=True, nCores=20)
    # copa.compute_muStar(run+'_old', overwrite=True, true_members=False,nCores=20)
    # copa.run_copa(run+'_zfile_old',pz_file=pzfile, nCores=60, old_code=True)
    # copa.compute_muStar(run+'_zfile_old',overwrite=True, true_members=False)
    # copa.compute_muStar(run+'_zfile_old',overwrite=True, true_members=True ,nCores=40)

runs2   = ['bpz-r200','emuBPZ-r200','gauss003-r200','gauss005-r200']
pzfiles = ['bpz'     ,'emuBPZ'     ,'gauss003'     ,'gauss005'     ]
z_window= [0.03, 0.05, 0.03, 0.05]
copa.kwargs['r_aper_model'] = 'r200'

# run old membership assignment
for run,zw,pzfile in zip(runs2[1:],z_window[1:],pzfiles[1:]):
    print('\nRun: %s'%(run))
    print('z window: %.2f'%(zw))
    copa.kwargs['z_window'] = zw
    copa.run_copa(run,       pz_file=pzfile, nCores=40, old_code=False)
    copa.compute_muStar(run, overwrite=True, true_members=False)
    copa.compute_muStar(run, overwrite=True, true_members=True)

    # copa.run_copa(run+'_zfile_old',pz_file=pzfile, nCores=60, old_code=True)
    # copa.compute_muStar(run+'_zfile_old',overwrite=True, true_members=False)
    # copa.compute_muStar(run+'_zfile_old',overwrite=True, true_members=True ,nCores=40)
print('Done')
print('\n')

# copa.run_copa('gauss005-r200',pz_file='gauss005', nCores=40)
#copa.compute_muStar('gauss005-r200')

# run = 'gauss005-r200'
# gal = copa.load_copa_out('members',run=run)
# cat = copa.load_copa_out('cluster',run=run)

# file_out = '/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/cosmoDC2_v1.1.4_copa_{run_name}_{dtype}.fits'
# gal.write(file_out.format(run_name=run,dtype='members'),format='fits',overwrite=True)
# cat.write(file_out.format(run_name=run,dtype='cluster'),format='fits',overwrite=True)


# copa.run_copa('bpz-rhod_old',pz_file=None, nCores=40, old_code=True)
# copa.compute_muStar('bpz-rhod_old')

# copa.run_copa('bpz-rhod',pz_file=None, nCores=40)
# copa.compute_muStar('bpz-rhod')

# copa.run_copa('emuBPZ-rhod',pz_file='emuBPZ', nCores=40)
# copa.compute_muStar('emuBPZ-rhod')

# copa.run_copa('gauss005-rhod',pz_file='gauss005', nCores=40)
# copa.compute_muStar('gauss005-rhod')

# copa.kwargs['z_window'] = 0.03
# copa.run_copa('gauss003-rhod',pz_file='gauss003', nCores=40)
# copa.compute_muStar('gauss003-rhod')


# def delete_group(fname,path):
#     fmaster = h5py.File(fname,'a')
#     group   = fmaster[path]

#     cols    = group.keys()
#     for col in cols:
#         del group[col]

# fname = '/data/des61.a/data/johnny/CosmoDC2/sample2021/outputs/cosmoDC2_v1.1.4_copa.hdf5'
# group1= 'members/copa/bpz-rhod_old/'
# group2= 'clusters/copa/bpz-rhod_old/'
# delete_group(fname,group1)
# delete_group(fname,group2)

### test new smass.py implementation

# fname = copa.master_fname
# print('fname:',fname)

# import h5py
# import numpy as np
# import matplotlib.pyplot as plt

# fname = '/home/s1/jesteves/copa_v2.1/datasets/cosmoDC2_v1.1.4_copa-v2.1_dev_small.hdf5'
# hf = h5py.File(fname,'r')

# columns = ['RA','DEC','z']

# secondary = hf['secondary/main/']
# main      = hf['master/main']
# mid       = np.array(secondary['mid'][:])

# mydict = dict()
# for col in columns:
#     arr         = np.array(main[col][:])
#     mydict[col] = arr[mid]

# columns2 = list(secondary.keys())
# for col in columns2:
#     arr         = np.array(secondary[col][:])
#     mydict[col] = arr[mid]

# hf.close()

# ra = mydict['RA'][:]
# dec= mydict['DEC'][:]
# radii = mydict['R'][:]

# idx = np.random.randint(low=0,high=ra.size,size=50000)
# plt.scatter(ra[idx],dec[idx],c=radii[idx],alpha=0.1)
# plt.savefig('./radec_copa.png')

# import h5py
# import numpy as np

# hf = h5py.File(fname,'r')

# cols = list(hf['secondary/bma/'].keys())

# sid = hf['secondary/bma/sid'][:]
# mid = hf['secondary/bma/mid'][:]
# zcls= hf['secondary/bma/zmet'][:]

# midm= np.array(hf['secondary/main/mid'][:])
# zclm= np.array(hf['secondary/main/redshift'][:])

#diff= zcls-zclm[sid]
## diff= mid-midm[sid]

# print 'mean:',np.mean(diff)
# print 'std:' ,np.std(diff)
# print '\n'

# for col in cols:
    # a_new=hf['bma/%s'%col][:]
#     a_old=hf['bma_old/%s'%col][:]
#     diff = a_new-a_old

#     print 'column %s'%col
#     print 'mean:',np.mean(diff)
#     print 'std:' ,np.std(diff)
#     print '\n'