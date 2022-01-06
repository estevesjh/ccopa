#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pandas as pd
import h5py
import yaml
import os

from time import time
import esutil

from astropy.table import Table, vstack, join
from astropy.io.fits import getdata

from upload_cosmoDC2 import upload_cosmoDC2_hf5_files, stack_dict
from projection_radec_to_xy import radec_to_theta,radec_to_xy

rad2deg= 180/np.pi
h=0.7

cls_columns = ['CID','redshift','RA','DEC','M200_true','R200_true','magLim']
# cls_columns = ['CID','redshift','RA','DEC','M200_true','R200_true','DA','magLim','MASKFRAC']

## master file
def make_master_file(cdata,data,file_out,yaml_file,header):
    with open(yaml_file) as file:
        cd,gd = yaml.load_all(file.read())

    ## load members with copacabana colnames 
    data['mid'] = np.arange(0,data['CID'].size,1,dtype=np.int64)

    ## load clusters with copacabana colnames
    cdata = rename_columns_table(Table(cdata),cd)#rename_dict(cdata,cd)
    cdata = cdata[cls_columns]

    ## Writing master file
    write_master_file(file_out,header,cdata,data)

def write_master_file(name_file,header,table0,table1,columns1=None):
    """write the clusters and members directories
    params:
    :name_file str: file's name of the dataset
    :table0: cluster table
    :table1: members table quantities
    :columns1 list: members columns names
    """
    fmaster = h5py.File(name_file, "w")

    ## match cluster table with galaxy table
    #cidx, = np.where( np.in1d( table0['CID'],np.unique(table1['CID']) ) )
    match  = esutil.numpy_util.match(np.unique(table1['CID']),table0['CID'])
    cidx   = match[1]

    columns0 = list(table0.keys())
    for col in columns0:
        fmaster.create_dataset('clusters/main/%s/'%(col), data = table0[col][cidx] )

    if not columns1: columns1 = table1.keys()
    for col in columns1:
        fmaster.create_dataset('members/main/%s/'%(col), data = table1[col][:])
    
    fmaster.close()

# def make_members_data(data,col_dict):
#     columns = list(col_dict.keys())
#     mdata = dict().fromkeys(columns)
#     mdata['mid'] = np.arange(0,data[col_dict['CID']].size,1,dtype=int)

#     for col in columns:
#         mdata[col] = data[col_dict[col]]
    
#     return mdata

### BMA 
def make_bma_catalog_cut(fname,kwargs,rmax,overwrite=False):
    fmaster = h5py.File(fname, "a")

    ## check if bma already exists
    #if check_not_hf5(fmaster['members'],'bma'):
    print('making bma cutout')
    columns = fmaster['members/main']
    radii   = fmaster['members/main/R'][:]*h
    dmag    = fmaster['members/main/dmag'][:]
    mag     = fmaster['members/main/mag'][:]
    mid     = fmaster['members/main/mid'][:]
    z       = fmaster['members/main/redshift'][:]
    
    indices = apply_magnitude_selection(fname,kwargs)
    mask    = radii[indices]<=rmax

    if 'FLAG' in fmaster['members/main'].keys():
        flag = fmaster['members/main/FLAG']
        mask &= flag < 8
    
    for i in range(3):
        color = mag[indices,i] - mag[indices,i+1]
        mask &= (color<=4.)&(color>=-1.)
    
    cut     = indices[mask]
    try:
        fmaster.create_group('members/bma/')
        fmaster.create_dataset('members/bma/mid_cut/', data=mid[cut])
    except:
        del fmaster['members/bma/mid_cut/']
        fmaster.create_dataset('members/bma/mid_cut/', data=mid[cut])

    fmaster['members/bma/'].attrs['rmax'] = rmax
    fmaster['members/bma/'].attrs['dmag'] = kwargs['mag_selection']
    fmaster['members/bma/'].attrs['nsize'] = int(cut.size)

    bma_indices = mid[cut]
    # else:
    #     bma_indices = fmaster['members/bma/mid_cut'][:]
    #     if overwrite:
    #         print('in construction')
    #         # exclude_group(fmaster,'members/bma')
    #         # fmaster.close()
    #         # make_bma_catalog_cut(fname,rmax,dmag_lim=dmag_lim)
    #     else:
    #         print('BMA cut already exists; overwrite=False')
    fmaster.close()
    return bma_indices


def make_bma_input_temp_file(fname,files,indices,nsize,nchunks):
    mcols       = ['mid','CID','redshift','mag','magerr']                ## master/main/ columns
    out_columns = ['mid','CID','redshift','i','ierr','gr','ri','iz','grerr','rierr','izerr']

    if not os.path.isfile(files[0]):
        mydict = read_hdf5_file_to_dict(fname,indices=indices,cols=mcols,path='members/main/')

        out= dict().fromkeys(out_columns)
        for col in out_columns[:3]:
            out[col] = mydict[col][:]
        
        out['indices'] = np.arange(0,out['mid'].size,1,dtype=np.int64)

        out['i']    = mydict['mag'][:,2]
        out['ierr'] = mydict['magerr'][:,2]

        out['gr']   = mydict['mag'][:,0] - mydict['mag'][:,1]
        out['ri']   = mydict['mag'][:,1] - mydict['mag'][:,2]
        out['iz']   = mydict['mag'][:,2] - mydict['mag'][:,3]

        out['grerr']= np.sqrt(mydict['magerr'][:,0]**2 + mydict['magerr'][:,1]**2)
        out['rierr']= np.sqrt(mydict['magerr'][:,1]**2 + mydict['magerr'][:,2]**2)
        out['izerr']= np.sqrt(mydict['magerr'][:,2]**2 + mydict['magerr'][:,3]**2)

        write_bma_dict_temp_files(files,out,nsize,nchunks)

def write_bma_dict_temp_files(files,table,nsize,nchunks):
    columns = table.keys()
    idxs    = np.linspace(0,nsize,nchunks+1,dtype=np.int64)
    
    for i,file in enumerate(files):
        if not os.path.isfile(file):
            hf = h5py.File(file,'w')
            hf.create_group('bma')
            for col in columns:
                ilo,iup = idxs[i],idxs[i+1]
                hf.create_dataset('bma/%s/'%col,data=table[col][ilo:iup])
            hf.close()
    return files
      
def read_hdf5_file_to_dict(file,cols=None,indices=None,path='/'):
    hf = h5py.File(file, 'r')
    #hf.visititems(show_h5_group)

    try:
        mygroup = hf[path]
    except:
        hf.visititems(show_h5_group)
        print('Error group not found: %s'%path)
        exit()

    if cols is None: cols  = list(mygroup.keys())
    if indices is None: indices = np.arange(0,len(mygroup[cols[0]]),1,dtype=np.int64)

    mydict= dict().fromkeys(cols)
    for col in cols:
        try:
            mydict[col] = np.array(mygroup[col][:][indices])
        except:
            print('Error: %s'%(file))
    
    hf.close()

    return mydict

def combine_hdf5_files(files,path='/'):
    """ Combine hdf5 files with same data structure
    """
    mylist = []
    count  = 0
    for file in files:
        if os.path.isfile(file):
            mydict = read_hdf5_file_to_dict(file,path=path)
            mylist.append(mydict)
        else:
            print('missing the temp file: %s'%(file))
            count+=1

    all_dict=stack_dict(mylist)
    return all_dict, count

def wrap_up_temp_files(fname,files,path='members/bma/',overwrite=False):
    ## load temp files into a dict
    table, nmissing = combine_hdf5_files(files,path='bma/')
    nsize = table['mid'].size

    ## write bma output on the master file
    columns = list(table.keys())
    #columns.remove('mid')

    fmaster   = h5py.File(fname,'a')
    nsize_old = fmaster[path+'mid_cut/'][:].size
    if nsize_old!= len(table):
        print('Error: output table doesnt match input table')
        print(fname,'\n')

    checkColumns= np.sum([not check_not_hf5(fmaster[path],col) for col in columns])

    if (checkColumns==0)or(overwrite):
        for col in columns:
            fmaster[path].create_dataset(col,data=table[col][:])
    else:
        print('Not writing BMA temp out files. Master file has already a BMA output. Overwrite is set to false.')
    fmaster.close()

    print('all files combined with success')
    return nmissing


### copa
def load_copa_input_catalog(fname,kwargs,pz_file=None,simulation=True):
    print('loading clusters')
    cdata = load_cluster_main_catalog(fname)

    print('apply magnitude selection')
    indices = apply_magnitude_selection(fname,kwargs)

    print('loading full members catalog')
    mydict = read_hdf5_file_to_dict(fname,cols=None,indices=indices,path='members/main/') ## None load all columns

    print('making color columns')
    mydict = set_color_columns(mydict)

    if simulation:
        print('selecting fake photo-z catalog')
        mydict= select_photoz_catalog(mydict,fname,group=pz_file)

    print('making galaxies cut')
    mydict1 = make_galaxies_cut(mydict,kwargs)

    print('assigning background galaxies')
    radii          = mydict1['R']*h
    mydict1['Bkg'] = (radii>=kwargs['r_in'])&(radii<=kwargs['r_out'])
    mydict1['Gal'] = (radii<=kwargs['rmax'])

    print('computing physical quantities')
    mydict1 = compute_physical_coordinates(mydict1,cdata)

    # print('computing the number of galaxies in each cluster')
    # cdata  = count_input_gals(cdata,mydict1)
    table   = dict_to_table(mydict1)
    
    return table, cdata

def apply_magnitude_selection(fname,kwargs):
    if kwargs['mag_selection'] is None:
        out = read_hdf5_file_to_dict(fname,cols=['dmag'],path='members/main/')
        indices = np.where(out['dmag'][:]<=kwargs['dmag_lim'])[0]
    else:
        out = read_hdf5_file_to_dict(fname,cols=[kwargs['mag_selection']],path='members/indices/')
        indices = out[kwargs['mag_selection']][:]
    return indices.astype(int)

def select_photoz_catalog(data,fname,group=None):
    pz_cols = ['z','zerr','pz0','zoffset','zwindow']
    if group is None:
        return data
    else:
        pzcat   = read_hdf5_file_to_dict(fname,cols=pz_cols,indices=data['mid'][:],path=u'members/%s'%(group))
        #idx,    = np.where(np.in1d(pzcat['mid'],data['mid'],assume_unique=True))
        for col in pz_cols:
            data[col] = np.array(pzcat[col][:])
            #data[col] = arr[idx]
        return data

def count_input_gals(cdata,data):
    cid = cdata['CID'][:]
    cgid= data['CID'][:]

    keys = list(chunks(cgid,cid))

    gal_mask  =  data['Gal'][:]
    bkg_mask  =  data['Bkg'][:]
    tru_mask  =  data['True'][:]

    ngals_gal =  [np.count_nonzero(gal_mask[idx]) for idx in keys]
    ngals_bkg =  [np.count_nonzero(bkg_mask[idx]) for idx in keys]
    ngals_tru =  [np.count_nonzero(tru_mask[idx]) for idx in keys]
    cdata['ngal_input'] = np.array(ngals_gal)
    cdata['nbkg_input'] = np.array(ngals_bkg)
    cdata['ntru_input'] = np.array(ngals_bkg)

    return cdata

def compute_physical_coordinates(galaxies,clusters):
    ra  = galaxies['RA'][:]
    dec = galaxies['DEC'][:]
    rac = clusters['RA'][:]
    decc= clusters['DEC'][:]

    Mpc2theta= clusters['DA'][:]/rad2deg
    cid = clusters['CID']
    cgid= galaxies['CID']

    keys = list(chunks(cgid,cid))
    galaxies = ini_new_dict_columns(galaxies,['dx','dy','theta'],values=[-99,-99,-99],size=ra.size)

    for i,idx in enumerate(keys):
        dx,dy,_ = radec_to_xy(ra[idx],dec[idx],rac[i],decc[i],Mpc2theta[i]) ## Mpc
        theta   = radec_to_theta(ra[idx],dec[idx],rac[i],decc[i])           ## degrees
        galaxies['dx'][idx]    = dx
        galaxies['dy'][idx]    = dy
        galaxies['theta'][idx] = theta
    return galaxies

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def ini_new_dict_columns(mydict,cols,values,size=1):
    for col,val in zip(cols,values):
        mydict[col] = val*np.ones((size,))
    return mydict


def set_color_columns(table):
    #################  g-r , g-i , r-i , r-z , i-z
    colors_indices = [[0,1],[0,2],[1,2],[1,3],[2,3]]
    mag = table['mag'][:]

    color = np.empty_like(table['z'][:])
    for choose_color in colors_indices:
        i,i2 = choose_color
        colori = (mag[:,i]-mag[:,i2])
        color = np.c_[color,colori]

    table['color'] = color[:,1:]
    return table

def make_galaxies_cut(mydict,kwargs):
    rmax    = 3.
    rin     = kwargs['r_in']
    rout    = kwargs['r_out']
    dz_max  = kwargs['dz_max']
    zmax    = kwargs['zmax_gal']
    zmin    = kwargs['zmin_gal']

    radii   = mydict['R']*h
    zgal    = mydict['z']
    zoff    = mydict['zoffset']
    color   = mydict['color'][:]

    mask  = (radii<=rout)
    mask &= (zgal>=zmin)&(zgal<=zmax)
    # mask &= (np.abs(zoff)<=dz_max)
    mask &= (color[:,0]<=3.5)&(color[:,3]<=3.5)&(color[:,4]<=3.5)
    mask &= (color[:,0]>=-1.0)&(color[:,3]>=-1.0)&(color[:,4]>=-1.0)

    cut, = np.where(mask)
    
    print('galaxy catalog')
    print('all: ',radii.size)
    print('cut: ',cut.size)

    cols     = list(mydict.keys())
    new_dict = dict().fromkeys(cols)
    for col in cols:
        new_dict[col] = mydict[col][cut]
        
    return new_dict

def load_cluster_main_catalog(fname):
    fmaster = h5py.File(fname,'r')
    cols    = list(fmaster['clusters/main/'].keys())
    table   = Table() 
    for col in cols:
        table[col] = fmaster['clusters/main/%s'%col][:]
    return table

def make_chunks(galaxies,clusters,nchunks):
    c_id    = clusters['CID'][:]
    cid      = np.unique(c_id)
    gal_cid  = galaxies['CID'][:]

    gal_list = []
    cls_list = []
    indices = np.linspace(0,len(cid),nchunks+1,dtype=np.int64)
    keys = list(chunks(gal_cid,cid))

    for i in range(nchunks):
        ki = cid[indices[i]:indices[i+1]]
        keys = list(chunks(gal_cid,ki))
        
        gidxs = np.concatenate(keys)
        cidxs, = np.where(np.in1d(c_id,ki))

        cls_list.append(clusters[cidxs] )
        gal_list.append(galaxies[gidxs])

        #header_chucks(i,gidxs.size,cidxs.size)
    return gal_list, cls_list

def header_chucks(i,gsize,csize):
    print('Chunk %i'%i)
    print('Gal size:',gsize)
    print('Clu size:',csize)
    print('\n')

def write_copa_output_pandas(fname,gal,cat,run_name,overwrite=True):
    ## pandas
    df1 = gal.to_pandas()
    df2 = cat.to_pandas()
    
    hdf = pd.HDFStore(fname)
    hdf.append('members/copa/%s'%run_name , df1, format='table', data_columns=True)
    hdf.append('clusters/copa/%s'%run_name, df2, format='table', data_columns=True)
    hdf.close()
    pass

def check_group(fname,path):
    hf = h5py.File(fname,'r')

    all_items = list(allkeys(hf))
    hf.close()
    return path in all_items

def write_copa_output(fname,gal,cat,run_name,overwrite=False):
    if check_group(fname,'clusters/copa/%s'%(run_name)):
        print('overwriting groups: members and clusters/copa/%s'%(run_name))
        delete_group(fname,u'clusters/copa/%s'%(run_name))
        delete_group(fname,u'members/copa/%s'%(run_name))

    fmaster = h5py.File(fname,'a')
    if 'copa' not in fmaster['members'].keys():
        fmaster.create_group('members/copa/')

    ## save galaxy output
    path = 'members/copa/%s'%run_name
    if run_name not in fmaster['members/copa/'].keys():
        fmaster.create_group(path)
    
    gout = fmaster[path]
    cols = gal.colnames
    check= cols[0] not in gout.keys()
    
    if check :
        for col in cols: gout.create_dataset(col,data=gal[col])
            
    ## save cluster output
    if 'copa' not in fmaster['clusters'].keys():
        fmaster.create_group('clusters/copa/')

    path = 'clusters/copa/%s'%run_name
    if run_name not in fmaster['clusters/copa/'].keys():
        fmaster.create_group(path)
    
    cout = fmaster[path]
    cols = cat.colnames
    check= cols[0] not in cout.keys()

    if check:
        for col in cols: cout.create_dataset(col,data=cat[col])

    fmaster.close()
    pass

def delete_group(fname,path):
    try:
        fmaster = h5py.File(fname,'a')
        group   = fmaster[path]
        cols    = group.keys()
        if len(cols)>0:
            for col in cols: del group[col]
        fmaster.close()
    except:
        fmaster.close()
        print('Error: failed to delete group %s'%path)
        return
    
def load_copa_output(fname,dtype,run,old_code=False):
    if dtype=='cluster':
        ## load copa and bma
        copa  = read_hdf5_file_to_dict(fname,path='clusters/copa/%s/'%(run))
        cat   = Table(copa)
        return cat

    if dtype=='members':
        ## load copa, bma and members
        copa_dict = read_hdf5_file_to_dict(fname,path='members/copa/%s'%(run))
        # bma_dict  = read_hdf5_file_to_dict(fname,path='members/bma/')

        # midxs   = np.sort(copa_dict['mid'][:])
        # members = read_hdf5_file_to_dict(fname,indices=midxs,cols=None,path='members/main/') 
        
        # print('Matching Copa output with main and BMA')
        # main = Table(members)
        copa = Table(copa_dict)
        # bma  = Table(bma_dict)
        
        # ## repeated cols
        # main.remove_columns(['GID','z','zerr','zoffset','pz0']) ## taking only the photoz info from the output

        # if not old_code:
        #     toto = join(main,copa,keys=['mid','CID']) ## matching with the mid and the cluster ID
        # else:
        #     toto = copa
        
        # gal  = join(toto, bma, keys=['mid','CID'])
        return copa

### auxialry functions

def table_to_dict(table):
    columns = table.colnames
    mydict = dict().fromkeys(columns)
    for col in columns:
        mydict[col] = table[col]
    return mydict

def dict_to_table(mydict):
    cols = list(mydict.keys())
    table = Table()
    for col in cols:
        table[col] = mydict[col][:]
    return table

def rename_columns_table(data,column_dict):
    cols_table = list(data.columns)

    for key in column_dict.keys():
        if key not in cols_table:
            data.rename_column(column_dict[key],key)

    return data

def rename_dict(data,columns_dict):
    columns = [col for col in columns_dict]
    newdict = dict().fromkeys(columns)

    columns = add_additional_columns(columns_dict,list(data.keys())) ## columns not specified on the yaml file

    for col in columns:
        newdict[col] = data[columns_dict[col]]
    return newdict

def add_additional_columns(mydict,cols_dataset):
    cols_default   = [mydict[col] for col in list(mydict.keys())]
    cols_additional= [col for col in cols_dataset if col not in cols_default]
    if len(cols_additional)>0:
        for col in cols_additional:
            mydict[col] = col
    return mydict

def check_not_hf5(hf,path):
    return path not in hf.keys()

### useful functions
def show_h5_group(name,node):
    ''' show all the h5 groups
    Example:
    fmaster = h5py.File(copa.master_fname,'r')
    fmaster.visititems(show_h5_group)

    ## https://stackoverflow.com/questions/45562169/traverse-hdf5-file-tree-and-continue-after-return
    '''
    if isinstance(node, h5py.Group):
        print(name,node)
    return None

def show_h5_dataset(name,node):
    if isinstance(node, h5py.Dataset):
        print(name,node)
    return None

def show_h5_all(name,node):
    print(name,node)
    return None

def rename_group(fname,path, group_new, group_old):
    fmaster   = h5py.File(fname,'a')
    files = fmaster[path]
    group = files[group_old]

    columns = group.keys()

    mydict = dict()
    for col in columns:
        mydict[col] = group[col][:]
        del group[col]
    
    del group

    if group_new not in files.keys():
        files.create_group(group_new)
        group = files[group_new]
    else:
        group = files[group_new]
        for col in group.keys():
            del group[col]
    
    for col in columns:
        group.create_dataset(col, data=mydict[col][:])     
    fmaster.close()

def upload_dataFrame(infile,keys='members'):
    hdf = pd.HDFStore(infile, mode='r')
    df1 = hdf.get(keys)
    hdf.close()
    data = Table.from_pandas(df1)    
    return data


def store_structure(name,node):
    if isinstance(node, h5py.Group):
        groups.append(node.name)
        #print(name,node)
    
    if isinstance(node, h5py.Dataset):
        datasets.append(node.name)
        #print(name,node)
        
    return None

def allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys

def copy_h5_file(infile,outfile):
    hf = h5py.File(infile,'r')

    groups = []
    datasets = []
    hf.visititems(store_structure)
    hf2 = h5py.File(outfile,'w')

    for g in groups:
        hf2.create_group(g)

    for d in datasets:
        hf2.create_dataset(d,data=hf[d])

    hf.close()
    hf2.close()

    os.rename(outfile,infile)