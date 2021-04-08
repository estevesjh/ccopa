import sys
import numpy as np
import h5py
sys.path.append('/home/johnny/Documents/copa_2.1')

from libs.main import copacabana

cfg = 'libs/config_copa_dc2.yaml'
copa = copacabana(cfg)

copa.make_input_file()
copa.run_bma(nCores=50)

copa.kwargs['z_window'] = 0.05

run = 'gauss005'

copa.run_copa(run, pz_file=pzfile, nCores=40, old_code=False)
copa.compute_muStar(run, overwrite=True, true_members=False)
copa.compute_muStar(run, overwrite=True, true_members=True)
