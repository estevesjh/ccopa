import sys
import numpy as np
import h5py
from time import time
#sys.path.append('/home/johnny/Documents/copa_2.1')
sys.path.append('/home/s1/jesteves/git/ccopa/python')

from main import copacabana

t0 = time()
cfg = '/home/s1/jesteves/git/ccopa/config_files/config_buzzard_v2.yaml'
copa = copacabana(cfg,dataset='buzzard_v2')
copa.make_input_file()

total_time = (time()-t0)/60
#print('\n')
print('Total time: %.2f min'%total_time)
#copa.run_bma(nCores=50)

# copa.run_copa(run, pz_file='emuBPZ_zww', nCores=60, old_code=False)
# copa.compute_muStar(run, overwrite=True, true_members=False)
# copa.compute_muStar(run, overwrite=True, true_members=True)
