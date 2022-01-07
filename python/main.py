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
from make_input_files.pre_processing_copa import preProcessing

import bma.smass as smass
from bma.cluster_smass import compute_mu_star, compute_mu_star_true

from copac.membAssignment import clusterCalc, compute_ptaken, compute_ngals

from old_memb.afterburner import old_memb

h = 0.7


class copacabana:
    """Copacabana"""
    def __init__(self, config_file, dataset='cosmoDC2', simulation=True):
        
        self.kwargs       = get_files(config_file)
        self.gfile        = self.kwargs['members_infile']
        self.cfile        = self.kwargs['cluster_infile']
        self.master_fname = self.kwargs['master_outfile']
        self.yaml_file    = self.kwargs['columns_yaml_file']
        self.dataset      = dataset
        
        self.simulation = simulation
        self.header     = None

        self.out_dir       = os.path.dirname(self.master_fname)+'/'
        self.temp_file_dir = check_dir(os.path.join(self.out_dir,'temp_file'))
        self.pdf_file_dir  = check_dir(os.path.join(self.out_dir, 'pdfs'))

        self.healpix = self.kwargs['healpixel_setup']
        self.tiles   = self.kwargs['healpixel_list']
        if self.healpix:
            self.tiles = np.array(self.tiles)#[:3]
            self.tile_path   = check_dir(os.path.join(self.out_dir,'tiles'))
            basename = os.path.basename(self.master_fname)
            self.master_fname_tile       = os.path.join(self.tile_path,basename)
            self.master_fname_tile_list  = [self.master_fname_tile.format(hpx) for hpx in self.tiles]
            self.temp_file_dir_list      = [check_dir(os.path.join(self.temp_file_dir,'{:05d}'.format(hpx))) for hpx in self.tiles]
            print('master file: \n','\n'.join(self.master_fname_tile_list))
            print('outdir:', self.out_dir)
            print('tile path:', self.tile_path)

    def pre_processing_healpix(self, healpix_list=None):
        if healpix_list is None:
            healpix_list = self.tiles
        cdata = Table(getdata(self.cfile))

        with open(self.yaml_file) as file:
            cd, gd = yaml.load_all(file.read())
        
        print('cluster columns')
        print(cd)

        for tile in healpix_list:
            tile        = int(tile)
            infile      = self.gfile.format(tile)
            mfile       = self.master_fname_tile.format(tile)

            mask   = cdata[cd['tile']]==tile
            ctile = table_to_dict(cdata[mask])

            print('tile : %i'%tile)
            print('counts: %i'%(np.count_nonzero(mask)))
            
            if np.count_nonzero(mask)>0:
                print('Loading Data')
                print('infile: %s'%infile)
                t0     = time()
                data   = table_to_dict(upload_dataFrame(infile,keys='members'))
                
                print('ngals : %.2e'%(len(data[gd['RA']][:])))
                pp = preProcessing(ctile,data,
                                   dataset=self.dataset,auxfile=self.kwargs['mag_model_file'],
                                   columns_cls=cd, columns_gal=gd)

                pp.make_cutouts(rmax=8)
                pp.make_relative_variables(z_window=0.03,nCores=60)
                pp.assign_true_members()
                pp.apply_mag_cut(dmag_cut=3)
                
                print('Writing Master File')
                make_master_file(pp.cdata,pp.out,mfile,self.yaml_file,self.header)

                partial_time = time()-t0
                print('Partial time: %.2f s \n'%(partial_time))
            else:
                print('Error: the tile %i is empty\n'%tile)

    def make_input_file(self,healpix_list=None,overwrite=False):
        t0 = time()
        if (not os.path.isfile(self.master_fname))or(overwrite):
            if self.dataset=='cosmoDC2':
                print('load cosmoDC2 infile: %s'%self.gfile)
                cdata= table_to_dict(Table(getdata(self.cfile)))
                keys = np.unique(cdata['healpix_pixel'])[:10]
                data = upload_cosmoDC2_hf5_files(self.gfile,keys)
                print('loading process complete: %.2f s \n'%(time()-t0))
            
                make_master_file(cdata,data,self.master_fname,self.yaml_file,self.header)
                print('write master file: %s'%self.master_fname)
            if self.dataset=='buzzard_v2':
                print('Running pre_processing_healpix() instead \n')
                self.pre_processing_healpix(healpix_list=healpix_list)

            if self.dataset=='des_y3':
                print('Running pre_processing_healpix() instead \n')
                self.pre_processing_healpix(healpix_list=healpix_list)

        else:
            print('master file already exists')

    def run_bma_healpix(self,run_name,nCores=4,batchStart=0,batchEnd=None,rmax=3,combine_files=True,remove_temp_files=False,overwrite=False):
        print('\n')
        print(5*'-----')
        print('Starting BMA')
        self.tiles_size  = len(self.tiles)
        self.bma_nchunks_per_tile = split_chuncks(self.kwargs['bma_number_of_chunks'],self.tiles_size)
        self.bma_nchunks = self.bma_nchunks_per_tile*self.tiles_size
        print('divided in nchunks: %i'%(self.bma_nchunks))
        print('each tile divided in %i chunks \n'%(self.bma_nchunks_per_tile))

        nsize = 0
        self.bma_temp_input_files = []
        self.bma_temp_output_files= []
        for hpx,mfile in zip(self.tiles,self.master_fname_tile_list):
            print('Healpixel: {:05d}'.format(hpx))
            temp_infile = [self.temp_file_dir+'/{:05d}/{}_input_{:03d}.hdf5'.format(hpx,run_name,i) for i in range(self.bma_nchunks_per_tile)]
            temp_outfile= [self.temp_file_dir+'/{:05d}/{}_output_{:03d}.hdf5'.format(hpx,run_name,i) for i in range(self.bma_nchunks_per_tile)]

            #idx = make_bma_catalog_cut(mfile,self.kwargs,rmax,overwrite=overwrite)
            idx = query_indices_catalog(mfile, run_name, self.kwargs, pmem_th=0.01, rmax=3, overwrite=overwrite)
            make_bma_input_temp_file(mfile,temp_infile,idx,len(idx),self.bma_nchunks_per_tile)

            self.bma_temp_input_files.append(temp_infile)
            self.bma_temp_output_files.append(temp_outfile)
            print('sample size: %i \n'%(len(idx)))
            nsize += len(idx)
        print('temp files')
        print('Total Sample Size: %i'%nsize)
        print('input :',flatten_list(self.bma_temp_input_files)[-3:])
        print('output:',flatten_list(self.bma_temp_output_files)[-3:])

        ## run BMA
        print('bma parallel')
        t0 = time()
        self.bma_trigger(flatten_list(self.bma_temp_input_files), flatten_list(self.bma_temp_output_files),
                        nCores=nCores,batchStart=batchStart,batchEnd=batchEnd, overwrite=overwrite)
        tt = (time()-t0)/60.
        print('stellarMass total time: %.2f min \n'%(tt))

        ## the files are combined only if all the temp files exists
        if combine_files:
            print('wrapping up temp files')
            for i,mfile in enumerate(self.master_fname_tile_list):
                bma_temp_output_files = self.bma_temp_output_files[i]
                print('bma out files')
                print('\n'.join(bma_temp_output_files[-3:]))
                nmissing = wrap_up_temp_files(mfile,bma_temp_output_files,run_name,nsize,path='members/bma/',overwrite=overwrite)           

            if nmissing>0:
                print('there are some missing files, plese check the batch numbers and rerun it.\n')

        if remove_temp_files:
            print('removing temp files')
            remove_files(flatten_list(self.bma_temp_output_files))
            remove_files(flatten_list(self.bma_temp_input_files))


    def run_bma(self,run_name,nCores=4,batchStart=0,batchEnd=None,rmax=3,combine_files=True,remove_temp_files=False,overwrite=False):
        print('Starting BMA')
        #indices = make_bma_catalog_cut(self.master_fname,rmax,dmag_lim=2,overwrite=overwrite)
        indices = query_indices_catalog(self.master_fname, run_name, self.kwargs, pmem_th=0.01, rmax=3, overwrite=overwrite)

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
            nmissing = wrap_up_temp_files(self.master_fname,self.bma_temp_output_files,run_name,self.bma_nsize,path='members/bma/',overwrite=overwrite)
            if nmissing>0:
                print('there are some missing files, plese check the batch numbers and rerun it.\n')

        if remove_temp_files:
            remove_files(self.bma_temp_output_files)
            remove_files(self.bma_temp_input_files)

    def bma_trigger(self,infiles,outfiles,
                   nCores=2,batchStart=0,batchEnd=None,overwrite=False):
        if batchEnd is None: batchEnd = len(outfiles)

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

    def run_copa_healpix(self,run_name,pz_file=None,nCores=20,old_code=False):
        print('\nStarting Copa')
        print('run %s'%run_name)
        #blockPrint()
        gal_list, cluster_list = [],[]
        for hpx,mfile in zip(self.tiles,self.master_fname_tile_list):
            # print('here')
            print('master_file:',mfile)
            galaxies, clusters= load_copa_input_catalog(mfile,self.kwargs,pz_file=pz_file,simulation=self.simulation)

            # galaxies = Table(getdata(self.temp_file_dir+'/{:05d}/{}_copa_test_gal.fits'.format(hpx,run_name[:-5])))
            # clusters = Table(getdata(self.temp_file_dir+'/{:05d}/{}_copa_test_gal.fits'.format(hpx,run_name[:-5])))
            
            galaxies['tile'] = hpx
            clusters['tile'] = hpx
            print('\n')
            print('tile file:',self.temp_file_dir+'/{:05d}/{}_copa_test_gal.fits'.format(hpx,run_name))
            # print('\n')
            galaxies.write(self.temp_file_dir+'/{:05d}/{}_copa_test_gal.fits'.format(hpx,run_name),format='fits',overwrite=True)
            clusters.write(self.temp_file_dir+'/{:05d}/{}_copa_test_cls.fits'.format(hpx,run_name),format='fits',overwrite=True)

            gal_list.append(galaxies)
            cluster_list.append(clusters)
        
        # self.nclusters = len(clusters)
        # self.ngalaxies = len(galaxies)
        # self.copa_nchunks = self.kwargs['copa_number_of_chunks']
        
        # gal_list, cluster_list = make_chunks(galaxies,clusters,self.copa_nchunks)
        # # gal_files = [self.temp_file_dir+'/{name}_{type}_input_{id:05d}.fits'.format(name=run_name,type='members',id=i) for i in range(self.copa_nchunks)]

        # for gal,fname in zip(gal_list,gal_files):
        #     gal.write(fname,format='fits',overwrite=True)

        t0 = time()
        if not old_code:
            cat, g0 = self.copa_trigger(run_name,gal_list,cluster_list,nCores=nCores)

            ## compute Ptaken
            galOut = compute_ptaken(g0)

            ### update Ngals
            catOut = computeNgals(galOut,cat)
            
            # write_copa_output(self.master_fname,galOut,catOut,run_name,overwrite=True)
        else:
            catOut, galOut = self.old_memb_trigger(run_name,gal_list,cluster_list,nCores=nCores)
        
        for hpx,mfile in zip(self.tiles,self.master_fname_tile_list):
            gali = galOut[galOut["tile"]==hpx]
            cati = catOut[catOut["tile"]==hpx]
            write_copa_output(mfile,gali,cati,run_name,overwrite=True)

        #enablePrint()
        # save total computing time
        totalTime = time() - t0
        totalTimeMsg = "Total time: {}s".format(totalTime)
        print(totalTimeMsg)

    def run_copa(self,run_name,pz_file=None,nCores=20,old_code=False):
        print('\nStarting Copa')
        print('run %s'%run_name)
        #blockPrint()
        galaxies, clusters= load_copa_input_catalog(self.master_fname,self.kwargs,pz_file=pz_file,simulation=self.simulation)
        galaxies['tile'] = 0
        #galaxies.write(self.temp_file_dir+'/%s_copa_test_gal.fits'%run_name,format='fits',overwrite=True)
        #clusters.write(self.temp_file_dir+'/%s_copa_test_cls.fits'%run_name,format='fits',overwrite=True)

        #galaxies = Table(getdata(self.temp_file_dir+'/%s_copa_test_gal.fits'%run_name))
        #clusters = Table(getdata(self.temp_file_dir+'/%s_copa_test_cls.fits'%run_name))

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
        #enablePrint()

        ### saving output
        # try:
        write_copa_output(self.master_fname,galOut,catOut,run_name,overwrite=True)
        
        # except:
        #     galOut.write(self.temp_file_dir+'/%s_copa_output_gal.fits'%run_name,format='fits',overwrite=True)
        #     catOut.write(self.temp_file_dir+'/%s_copa_output_cls.fits'%run_name,format='fits',overwrite=True)

        # save total computing time
        totalTime = time() - t0
        totalTimeMsg = "Total time: {}s".format(totalTime)
        print(totalTimeMsg)

    def make_copa_tmp_files(self,run_name,gal_list,cluster_list):
        self.copa_temp_cluster_input_files = []
        self.copa_temp_members_input_files = []

        for i in range(len(gal_list)):
            cfile = self.temp_file_dir+'/{name}_{type}_input_{id:05d}.fits'.format(name=run_name,type='cluster',id=i)
            gfile = self.temp_file_dir+'/{name}_{type}_input_{id:05d}.fits'.format(name=run_name,type='members',id=i)
            gal_list[i].write(gfile,format='fits',overwrite=True)
            cluster_list[i].write(cfile,format='fits',overwrite=True)
            self.copa_temp_cluster_input_files.append(cfile)
            self.copa_temp_members_input_files.append(gfile)
        pass

    def copa_trigger(self,run_name,gal_list,cluster_list,nCores=2,):
        self.copa_nchunks = len(gal_list)
        self.copa_pdf_output_files          = [self.pdf_file_dir+'/{name}_{type}_output_{id:05d}.hdf5'.format(name=run_name,type='pdf',id=i) for i in range(self.copa_nchunks)]
        
        if self.healpix:
            self.copa_pdf_output_files = [self.pdf_file_dir+'/{name}_{type}_output_{id:05d}.hdf5'.format(name=run_name,type='pdf',id=hpx) for hpx in self.tiles]

        ckwargs = [{'outfile_pdfs':pdfi,'member_outfile':None,'cluster_outfile':None,'r_in':self.kwargs['r_in']/h, 'pz_factor':self.kwargs['pz_factor'],
                    'r_out':self.kwargs['r_out']/h, 'sigma_z':self.kwargs['z_window'], 'zfile':self.kwargs['z_model_file'], 
                    'simulation': self.simulation, 'r_aper_model':self.kwargs['r_aper_model'],'pixelmap':self.kwargs['pixelmap_file']}
                    for pdfi in self.copa_pdf_output_files]
        
        self.make_copa_tmp_files(run_name,gal_list,cluster_list)
        gal_list,cluster_list=0.,0.

        print('copa parallel process')
        out = Parallel(n_jobs=nCores)(delayed(clusterCalc)(self.copa_temp_members_input_files[i], self.copa_temp_cluster_input_files[i], **ckwargs[i]) 
                                      for i in range(self.copa_nchunks))
        g0, cat = getOutFile(out)

        return cat, g0
    
    def old_memb_trigger(self,run_name,gal_list,cluster_list,nCores=4):
        self.copa_nchunks = len(gal_list)
        if self.kwargs['z_window']<0: zfile = self.kwargs['z_model_file']
        else: zfile = None
        ckwargs = {'dataset':'copa','sigma_z':self.kwargs['z_window'], 'zfile':zfile, 
                   'r_aper_model':self.kwargs['r_aper_model'],'zmin_gal':self.kwargs['zmin_gal'],'zmax_gal':self.kwargs['zmax_gal']}
        out = Parallel(n_jobs=nCores)(delayed(old_memb)(cluster_list[i],gal_list[i], **ckwargs) for i in range(self.copa_nchunks))
        g0,cat = getOutFile(out)
        return cat, g0

    def load_copa_out(self,dtype,run):
        if self.healpix:
            data = []
            for mfile in self.master_fname_tile_list:
                data.append(load_copa_output(mfile,dtype,run))
            return vstack(data)
        else:
            return load_copa_output(self.master_fname,dtype,run)

    def compute_muStar(self,run,true_members=False,overwrite=True,nCores=20):
        if not self.healpix:
            fmaster = h5py.File(self.master_fname,'r')
            check   = 'mass' in fmaster['members/bma/'].keys()
            check2  = 'MU' not in fmaster['clusters/copa/%s'%run].keys()
            fmaster.close()

            if not check:
                print('please run BMA before running compute_muStar()')

            if check2 or overwrite: 
                # if true_members:
                compute_mu_star(self.master_fname,run,nCores=nCores)
                if self.simulation:
                    compute_mu_star_true(self.master_fname,run,ngals=True,nCores=nCores)
        else:
            for mfile in self.master_fname_tile_list:
                compute_mu_star(mfile,run,nCores=nCores)
                if self.simulation:
                    compute_mu_star_true(mfile,run,ngals=True,nCores=nCores)

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
    Ngalsf= compute_ngals(g,cat['CID'][good_indices],cat['R200'][good_indices],true_gals=False,col='Pmem_flat')
    Ngalso= compute_ngals(g,cat['CID'][good_indices],cat['R200'][good_indices],true_gals=False,col='Pmem_old')

    cat['Ngals'] = -1.
    cat['Ngals'][good_indices] = Ngals

    cat['Ngals_flat'] = -1.
    cat['Ngals_flat'][good_indices] = Ngalsf

    cat['Ngals_old'] = -1.
    cat['Ngals_old'][good_indices] = Ngalso
    return cat

def split_chuncks(a,b):
    q = a//b
    if (a%b) != 0: q=q+1
    return q

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def flatten_list(regular_list):
    return sum(regular_list,[])

if __name__ == 'main':
    ## run example
    ## cosmoDC2 dataset

    cfg = 'config_copa_dc2.yaml'
    #copa = copacabana(cfg)

    #copa.make_input_file()
    #copa.run_bma(nCores=4)
    #copa.run_copa()
