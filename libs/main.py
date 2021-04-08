#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import h5py
import yaml
import os
import sys

from time import time
from joblib import Parallel, delayed

# local libraries
from make_input_files.make_input_datasets import *
from make_input_files.upload_cosmoDC2 import upload_cosmoDC2_hf5_files

import bma.smass as smass
from bma.cluster_smass import compute_mu_star, compute_mu_star_true

from copac.membAssignment import clusterCalc, compute_ptaken, compute_ngals

from scripts.afterburner import old_memb

h=0.7

class copacabana:
    """Copacabana"""
    def __init__(self,config_file,dataset='cosmoDC2',simulation=True):
        
        self.kwargs       = get_files(config_file)
        self.gfile        = self.kwargs['members_infile']
        self.cfile        = self.kwargs['cluster_infile']
        self.master_fname = self.kwargs['master_outfile']
        self.yaml_file    = self.kwargs['columns_yaml_file']
        self.dataset   = dataset
        
        self.simulation = simulation
        self.header     = None#get_header(dataset)#'./data/annis_mags_04_Lcut.txt'

        self.out_dir       = os.path.dirname(self.master_fname)
        self.temp_file_dir = check_dir(os.path.join(self.out_dir,'temp_file'))
        self.pdf_file_dir  = check_dir(os.path.join(self.out_dir, 'pdfs'))
        
    def make_input_file(self,overwrite=False):
        t0 = time()
        if (not os.path.isfile(self.master_fname))or(overwrite):
            if self.dataset=='cosmoDC2':
                print('load cosmoDC2 infile: %s'%self.gfile)
                cdata= table_to_dict(Table(getdata(self.cfile)))
                keys = np.unique(cdata['healpix_pixel'])
                data = upload_cosmoDC2_hf5_files(self.gfile,keys)
                print('loading process complete: %.2f s \n'%(time()-t0))

            if self.dataset=='buzzard_v1.98':
                ## to do
                print('Work in progress')
                exit()
            
            make_master_file(cdata,data,self.master_fname,self.yaml_file,self.header)
            print('write master file: %s'%self.master_fname)
        else:
            print('master file already exists')

    def run_bma(self,nCores=4,batchStart=0,batchEnd=None,rmax=3,combine_files=True,remove_temp_files=False,overwrite=False):
        print('Starting BMA')
        indices = make_bma_catalog_cut(self.master_fname,rmax,dmag_lim=2,overwrite=overwrite)

        self.bma_indices = indices
        self.bma_nsize   = indices.size
        self.bma_nchunks = self.kwargs['bma_number_of_chunks']

        print('sample size: %i'%(self.bma_nsize))
        print('divided in nchunks: %i \n'%(self.bma_nchunks))
        
        self.bma_temp_input_files = [self.temp_file_dir+'/input_{:05d}.hdf5'.format(i) for i in range(self.bma_nchunks)]
        self.bma_temp_output_files= [self.temp_file_dir+'/output_{:05d}.hdf5'.format(i) for i in range(self.bma_nchunks)]

        make_bma_input_temp_file(self.master_fname,self.bma_temp_input_files,
                                 self.bma_indices,self.bma_nsize,self.bma_nchunks)
        
        t0 = time()
        self.bma_trigger(self.bma_temp_input_files, self.bma_temp_output_files,
                        nCores=nCores,batchStart=batchStart,batchEnd=batchEnd, overwrite=overwrite)
        tt = (time()-t0)/60
        print('stellarMass total time: %.2f min \n'%(tt))

        ## the files are combined only if all the temp files exists
        if combine_files:
            print('wrapping up temp files')
            nmissing = wrap_up_temp_files(self.master_fname,self.bma_temp_output_files,path='members/bma/',overwrite=overwrite)           
            if nmissing>0:
                print('there are some missing files, plese check the batch numbers and rerun it.\n')

        if remove_temp_files:
            remove_files(self.bma_temp_output_files)
            remove_files(self.bma_temp_input_files)

    def bma_trigger(self,infiles,outfiles,
                   nCores=2,batchStart=0,batchEnd=None,overwrite=False):
        if batchEnd is None: batchEnd = self.bma_nchunks

        batches = np.arange(batchStart,batchEnd,1,dtype=np.int64)        
        inPath = self.kwargs["lib_inPath"]
        
        ## check outfiles
        checkFiles = np.sum([os.path.isfile(myfile) for myfile in np.array(outfiles)[batches]])
        if (checkFiles>0)&(not overwrite):
            # print('output files already exists; overwrite=False')
            print('Not running BMA again. If you want to overwrite the current files, please set overwrite to True')        
            print('exiting the function')
            return

        print('starting parallel process')
        print('runing on the following batches:',batches)
        Parallel(n_jobs=nCores)(
            delayed(smass.calc_copa_v2)(infiles[i], outfiles[i], inPath) for i in batches)
        print('ended smass calc')
        
    def run_copa(self,run_name,pz_file=None,nCores=20,old_code=False):
        print('\nStarting Copa')
        print('run %s'%run_name)
        blockPrint()
        # galaxies, clusters= load_copa_input_catalog(self.master_fname,self.kwargs,pz_file=pz_file,simulation=self.simulation)
        # galaxies.write(self.temp_file_dir+'/%s_copa_test_gal.fits'%run_name,format='fits',overwrite=True)
        # clusters.write(self.temp_file_dir+'/%s_copa_test_cls.fits'%run_name,format='fits',overwrite=True)

        galaxies = Table(getdata(self.temp_file_dir+'/%s_copa_test_gal.fits'%run_name))
        clusters = Table(getdata(self.temp_file_dir+'/%s_copa_test_cls.fits'%run_name))

        self.nclusters = len(clusters)
        self.ngalaxies = len(galaxies)
        self.copa_nchunks = self.kwargs['copa_number_of_chunks']
        
        gal_list, cluster_list = make_chunks(galaxies,clusters,self.copa_nchunks)
        # gal_files = [self.temp_file_dir+'/{name}_{type}_input_{id:05d}.fits'.format(name=run_name,type='members',id=i) for i in range(self.copa_nchunks)]

        # for gal,fname in zip(gal_list,gal_files):
        #     gal.write(fname,format='fits',overwrite=True)

        t0 = time()
        if not old_code:
            cat, g0 = self.copa_trigger(run_name,gal_list,cluster_list,nCores=nCores)        
            ## compute Ptaken
            galOut = compute_ptaken(g0)

            ### update Ngals
            catOut = computeNgals(galOut,cat)
        else:
            catOut, galOut = self.old_memb_trigger(run_name,gal_list,cluster_list,nCores=nCores)
        enablePrint()

        ### saving output
        write_copa_output(self.master_fname,galOut,catOut,run_name,overwrite=True)

        # save total computing time
        totalTime = time() - t0
        totalTimeMsg = "Total time: {}s".format(totalTime)
        print(totalTimeMsg)

    
    def copa_trigger(self,run_name,gal_list,cluster_list,nCores=2):
        self.copa_temp_cluster_output_files = [self.temp_file_dir+'/{name}_{type}_output_{id:05d}.fits'.format(name=run_name,type='cluster',id=i) for i in range(self.copa_nchunks)]
        self.copa_temp_members_output_files = [self.temp_file_dir+'/{name}_{type}_output_{id:05d}.fits'.format(name=run_name,type='members',id=i) for i in range(self.copa_nchunks)]
        self.copa_pdf_output_files          = [self.pdf_file_dir+'/{name}_{type}_output_{id:05d}.hdf5'.format(name=run_name,type='pdf',id=i) for i in range(self.copa_nchunks)]

        # ckwargs = [{'outfile_pdfs':pdfi,'member_outfile':membi,'cluster_outfile':clsi,'r_in':self.kwargs['r_in'],
        #             'r_out':self.kwargs['r_out'], 'sigma_z':self.kwargs['z_window'], 'simulation': self.simulation,
        #             'r_aper_model':self.kwargs['r_aper_model'],'pixelmap':self.kwargs['pixelmap_file']}
        #            for clsi,membi,pdfi in zip(self.copa_temp_cluster_output_files,self.copa_temp_members_output_files,self.copa_pdf_output_files)]

        ckwargs = [{'outfile_pdfs':pdfi,'member_outfile':None,'cluster_outfile':None,'r_in':self.kwargs['r_in']/h,
                    'r_out':self.kwargs['r_out']/h, 'sigma_z':self.kwargs['z_window'], 'zfile':self.kwargs['z_model_file'], 
                    'simulation': self.simulation, 'r_aper_model':self.kwargs['r_aper_model'],'pixelmap':self.kwargs['pixelmap_file']}
                    for pdfi in self.copa_pdf_output_files]
            
        print('copa parallel process')
        out = Parallel(n_jobs=nCores)(delayed(clusterCalc)(gal_list[i], cluster_list[i], **ckwargs[i]) 
                                      for i in range(self.copa_nchunks))
        g0, cat = getOutFile(out)

        return cat, g0
    
    def old_memb_trigger(self,run_name,gal_list,cluster_list,nCores=4):
        ckwargs = {'dataset':'copa','sigma_z':self.kwargs['z_window'], 'zfile':self.kwargs['z_model_file'], 
                   'r_aper_model':self.kwargs['r_aper_model'],'zmin_gal':self.kwargs['zmin_gal'],'zmax_gal':self.kwargs['zmax_gal']}
        out = Parallel(n_jobs=nCores)(delayed(old_memb)(cluster_list[i],gal_list[i], **ckwargs) for i in range(self.copa_nchunks))
        g0,cat = getOutFile(out)
        return cat, g0

    def load_copa_out(self,dtype,run):
        return load_copa_output(self.master_fname,dtype,run)

    def compute_muStar(self,run,true_members=False,overwrite=True,nCores=20):
        fmaster = h5py.File(self.master_fname,'r')
        check   = 'mass' in fmaster['members/bma/'].keys()
        check2  = 'MU' not in fmaster['clusters/copa/%s'%run].keys()
        fmaster.close()

        if not check:
            print('please run BMA before running compute_muStar()')

        if check2 or overwrite: 
            if true_members:
                compute_mu_star_true(self.master_fname,run,ngals=True,nCores=nCores)
            else:
                compute_mu_star(self.master_fname,run,nCores=nCores)
            


################
def get_files(config_file):
    with open(config_file) as file:
        kwargs = yaml.safe_load(file)
    return kwargs

def init_out_bma_files(fname,files):
    fmaster = h5py.File(fname,'a')
    for file in files:
        if check_not_hf5(fmaster,file): fmaster.create_group(file)
    fmaster.close()


def check_dir(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
    return path_dir

def remove_files(files):
    for file in files:
        os.remove(file)

def getOutFile(out):
    gal = [toto[0] for toto in out if not isinstance(toto[0],float)]
    cat = [toto[1] for toto in out if not isinstance(toto[1],float)]

    if (len(gal)>0)&(len(cat)>0):
        galAll = vstack(gal)
        catAll = vstack(cat)
    else:
        galAll = np.nan
        catAll = np.nan
    
    return galAll, catAll

def computeNgals(g,cat):
    good_indices, = np.where(cat['Nbkg']>0)
    Ngals = compute_ngals(g,cat['CID'][good_indices],cat['R200'][good_indices],true_gals=False,col='Pmem')
    cat['Ngals'] = -1.
    cat['Ngals'][good_indices] = Ngals
    return cat

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == 'main':
    ## run example
    ## cosmoDC2 dataset

    cfg = 'config_copa_dc2.yaml'
    copa = copacabana(cfg)

    copa.make_input_file()
    copa.run_bma(nCores=4)
    copa.run_copa()