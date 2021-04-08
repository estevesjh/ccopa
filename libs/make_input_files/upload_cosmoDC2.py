#!/usr/bin/env python
import numpy as np
import h5py
import yaml
import os

def upload_cosmoDC2_hf5_files(name_file0,healpix_list,columns_copa=None):
    out = []
    for hpx in healpix_list:
        if os.path.isfile(name_file0.format(hpx)):
            data = cosmoDC2_hf5_file_to_dictionary(name_file0,hpx,columns_copa=columns_copa)
            out.append(data)
        else:
            print('file not found %s'%(name_file0.format(hpx)))
    return stack_dict(out)
    
def cosmoDC2_hf5_file_to_dictionary(name_file0,hpx,columns_copa=None):
    hf = h5py.File(name_file0.format(hpx), 'r')
    print(name_file0.format(hpx))
    cidx           = hf['cluster_ids'][:]
    if columns_copa is None:
        columns_copa   = list(hf['copa/%i/'%cidx[0]].keys())
    columns_photoz = list(hf['photoz/'].keys())
    columns        = columns_copa+columns_photoz

    data = dict().fromkeys(columns)

    for col in columns_copa:
        res = [hf['copa/%i/%s'%(cid,col)][:] for cid in cidx]
        data[col] = np.concatenate(res)
    
    indices = data['indices']
    for col in columns_photoz:
        res = hf['photoz/%s'%(col)][:]
        data[col] = res[indices]
    hf.close()
    return data

def stack_dict(in_list):
    columns = list(in_list[0].keys())
    new_dict= dict().fromkeys(columns)
    for col in columns:
        res = [mydict[col] for mydict in in_list]
        new_dict[col] = np.concatenate(res)
    return new_dict
