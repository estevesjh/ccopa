# libraries
import configparser as ConfigParser
import logging
from time import time, sleep
from os import path
import os
from astropy.table import Table, vstack
import astropy.io.fits as pyfits

import glob
import fitsio

# local libraries
import membAssignment as membAssign
import helper as helper

####
import numpy as np
from dask import compute, delayed
from joblib import Parallel, delayed
####

file_path_script = __file__

def getConfig(section, item, boolean=False, getAllVariables=False, userConfigFile="./copa/CCOPA_Config.ini"):

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	# if config item not found, raise log warning
	if (not configFile.has_option(section, item)):
		msg = '{item} from [{section}] NOT found in config file: {userConfigFile}!'.format(
			item=item, section=section,
			userConfigFile=userConfigFile)
		if (section != 'Log'):
			logging.warning(msg)
		else:
			print(msg)
		return ""

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))
	if (section != 'Log'):
		logging.debug(msg)
	else:
		print(msg)

	if (not boolean):
		return configFile.get(section, item)
		
	else:
		return configFile.getboolean(section, item)

	if getAllVariables:
		out = dict()
		for key in configFile.items():
			for val in key[1]:
				out[val] = key[1][val]
		return out

def isOperationSet(operation,section="Operations"):
	return getConfig(boolean=True, section=section,
		item=operation)

def createLog():
	logLevel = getConfig("Log","level")
	logFileName = getConfig("Log","logFile")
	myFormat = '[%(asctime)s] [%(levelname)s]\t%(module)s - %(message)s'
	if logLevel == 'DEBUG':
		logging.basicConfig(
			filename=logFileName,
			level=logging.DEBUG,
			format=myFormat)
	else:
		logging.basicConfig(
			filename=logFileName,
			level=logging.INFO,
			format=myFormat)

def commonValues(values):
	idx_sort = np.argsort(values)
	sorted_values = values[idx_sort]
	vals, idx_start, count = np.unique(sorted_values, return_counts=True,
                                return_index=True)

	# sets of indices
	res = np.split(idx_sort, idx_start[1:])
	#filter them with respect to their size, keeping only items occurring more than once

	vals = vals[count > 1]
	commonValuesIndicies = [ri for ri in res if ri.size>1]
	
	return commonValuesIndicies

def computePmem(g0,plim=0.01):
	"""
	it computes p_taken
	"""
	print('compute ptaken')
	# g0.sort('Pmem')
	pmem = g0['Pmem']
	ptaken = np.ones_like(pmem,dtype=float)

	## find common galaxies
	gid = np.array(g0['GID'])#.astype(np.int)
	commonGroups = commonValues(gid)

	count = 0
	for indices in commonGroups:
		pm_group = np.array(pmem[indices])
		pt_group = np.array(ptaken[indices])

		idx_sort = np.argsort(-1*pm_group) ## -1* to reverse order

		pm_group_s = pm_group[idx_sort]
		pt_group_s = pt_group[idx_sort]

		new_pm = 0
		toto = 1
		pm = []
		for i in range(indices.size):
			toto *= (1-new_pm)
			new_pm = toto*pm_group_s[i]
			pt_group_s[i] = toto
			pm.append(new_pm)

		pmem[indices[idx_sort]] = np.array(pm)
		ptaken[indices[idx_sort]] = pt_group_s

	g0['Pmem'] = pmem
	g0['Ptaken'] = ptaken

	if 'True' in g0.colnames:
		pmem = np.where(g0['True']==True, 1., pmem) ## don't trow the true members away !
	mask = (pmem>=plim)

	return g0[mask]

def computeNgals(g,cat):
	good_indices = np.where(cat['Nbkg']>0)
	Ngals = membAssign.computeNgals(g,cat['CID'][good_indices],true_gals=False,col='Pmem')
	cat['Ngals'] = -1.
	cat['Ngals'][good_indices] = Ngals
	return cat

def getClusterColsName(massProxy=False):
	section = 'ClsColumns'
	
	ID = getConfig(section,"ID")
	RA = getConfig(section,"ra")
	DEC = getConfig(section,"dec")
	REDSHIFT = getConfig(section,"redshift")
	
	colNames = [ID,RA,DEC,REDSHIFT]

	if massProxy:
		MASS_PROXY = getConfig(section,"massProxy")
		colNames.append(MASS_PROXY)
	
	return colNames

def getGalaxyColsName():
	section = 'GalColumns'
	
	ID = getConfig(section,"ID")
	RA = getConfig(section,"ra")
	DEC = getConfig(section,"dec")
	REDSHIFT = getConfig(section,"z")

	MAG1 = getConfig(section,"m1")
	MAG2 = getConfig(section,"m2")
	MAG3 = getConfig(section,"m3")
	MAG4 = getConfig(section,"m4")
	
	FLAG = getConfig(section,"flagQ")
	errREDSHIFT = getConfig(section,"zErr")

	MAGERR1 = getConfig(section,"mErr1")
	MAGERR2 = getConfig(section,"mErr2")
	MAGERR3 = getConfig(section,"mErr3")
	MAGERR4 = getConfig(section,"mErr4")

	colNames = [ID,RA,DEC,REDSHIFT,MAG1,MAG2,MAG3,MAG4,FLAG,
			    errREDSHIFT,MAGERR1,MAGERR2,MAGERR3,MAGERR4]

	return colNames

def group_table_by(table,key,idx):
    new_table = table[:0].copy()
    for i in idx:
        ti = Table(table[np.where(table[key]==i)])
        if len(ti)>0:
            new_table = vstack([new_table,ti])
    return new_table

# def readFile(file_gal):
# 	galaxy_cat = pyfits.open(file_gal)
# 	g0 = Table(galaxy_cat[1].data)
# 	return g0
	
# def readFiles(files):
# 	allData = []
# 	for file_gal in files:
# 		if path.isfile(file_gal):
# 			g0 = readFile(file_gal)
# 			allData.append(g0)
		
# 		else:
# 			print('file not found %s'%(file_gal))
	
# 	g = vstack(allData)
# 	return g

def readFile(filename,columns=None):
    '''
    Read a filename with fitsio.read and return an astropy table
    '''
    hdu = fitsio.read(filename, columns=columns)
    return Table(hdu)

def readFiles(filenames, columns=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    out = []
    i = 1
    print
    for f in filenames:
        print('File {i}: {f}'.format(i=i, f=f))
        out.append(fitsio.read(f, columns=columns))
        i += 1

    return Table(np.concatenate(out))

# def writeFiles(member_outfile,cluster_outfile):
# 	logging.info('Writing output')
# 	clusterNames = sorted(glob.glob(cluster_outfile+'*'))
# 	memberNames = sorted(glob.glob(member_outfile+'*'))

#     # Call function and load all data sets
# 	# out0 = readFiles(clusterNames)
# 	# out1 = readFiles(memberNames)

# 	out0.write(cluster_outfile + ".fits")
# 	out1.write(member_outfile + ".fits")

def getOutFile(out):
	gal = [toto[0] for toto in out]
	cat = [toto[1] for toto in out]

	galAll = vstack(gal)
	catAll = vstack(cat)

	return galAll, catAll

def writeFilesParallel(galAll,catAll,getTrueMembers=False):
	
	memberPrefix = getConfig("Files","galaxyOutFilePrefix")
	clusterPrefix = getConfig("Files","clusterOutFilePrefix")

	fits = '.fits'
	if getTrueMembers:
		fits = 'true.fits'

	galAll.write(memberPrefix+fits, format='fits', overwrite=True)
	catAll.write(clusterPrefix+fits, format='fits', overwrite=True)

def getMembAssignmentFileList(nbatches):
	
	if not os.path.isdir('./ccopa_tmp_cat'):
		os.makedirs('./ccopa_tmp_cat')

	cluster_infile = getConfig("Files","clusterInputFile")
	member_infile = getConfig("Files","galaxyInputFile")

	# galaxyPrefix = './ccopa_tmp_cat/'+os.path.basename(member_infile)
	# clusterPrefix = './ccopa_tmp_cat/'+os.path.basename(cluster_infile)

	galaxyPrefix = os.path.splitext(member_infile)[0]
	clusterPrefix = os.path.splitext(cluster_infile)[0]

	m_list = [galaxyPrefix+'_%04i.fits'%(i+1) for i in range(nbatches)]
	c_list = [clusterPrefix+'_%04i.fits'%(i+1) for i in range(nbatches)]

	return m_list, c_list

def writeSmallFiles(cluster_cat,galaxy_cat,nbatches):
	m_out, c_out = getMembAssignmentFileList(nbatches)

	ncls = len(cluster_cat)
	islice = np.linspace(0,ncls,nbatches+1,dtype=int)

	cluster_list = [cluster_cat[islice[idx]:islice[idx+1]] for idx in range(nbatches)]
	galaxy_list = [group_table_by(galaxy_cat,'CID',cid['CID']) for cid in cluster_list]

	
	for i in range(nbatches):
		gi,ci = galaxy_list[i], cluster_list[i]
		## writing small files
		gi.write(m_out[i],format='fits',overwrite=True)
		ci.write(c_out[i],format='fits',overwrite=True)

	return galaxy_list, cluster_list

def triggerMembAssignment(idx,cInfile,gInfile,**kwargs):
	ci = readFile(cInfile)
	gi = readFile(gInfile)

	gal, cat = membAssign.clusterCalc(gi,ci,**kwargs)

	return gal, cat

def checkFiles(file_list):
	count=0
	for file_i in file_list:
		## if doesn't exits
		if not os.path.isfile(file_i):
			count+=1
	return count

def getKwargs(m_out=None,c_out=None):
	simulation = isOperationSet(operation="simulationTest")
	computeR200 = isOperationSet(operation="computeR200")

	r_in = round( float(getConfig('Cuts', "radius_bkg_in")), 1)
	r_out = round( float(getConfig('Cuts', "radius_bkg_out")), 1)

	plim = float(getConfig('Cuts', "p_low_lim"))
	M200 = float(getConfig('Cuts', "M200"))

	kwargs = {"member_outfile":m_out,"cluster_outfile":c_out,"r_in":r_in,"r_out":r_out,'M200':M200,"p_low_lim":plim,'simulation':simulation,'computeR200':computeR200}
	return kwargs

def loadTable(kwargs):
	galaxyInFile = getConfig("Files","galaxyInputFile")
	clusterInFile = getConfig("Files","clusterInputFile")

	simulation = kwargs['simulation']
	rcut = round( float(getConfig('Cuts', "radius_cutouts")), 1)
	window = round(float(getConfig('Cuts', "redshiftWindow")),2) ## just valid for simulations

	zmin = round(float(getConfig('Cuts', "zmin_gal")),3)
	zmax = round(float(getConfig('Cuts', "zmax_gal")),3)

	columnsLabelsCluster = getClusterColsName(massProxy=True)
	clusters = helper.readClusterCat(clusterInFile, colNames=columnsLabelsCluster, massProxy=True, simulation=simulation)

	columnsLabelsGalaxies = getGalaxyColsName()
	galaxies = helper.readGalaxyCat(galaxyInFile, clusters, colNames=columnsLabelsGalaxies, r_in=kwargs['r_in'], r_out=kwargs['r_out'], radius=rcut, window=window, zrange=(zmin,zmax), simulation=simulation)

	return galaxies, clusters

def loadTables(kwargs,parallel=False,nbatches=20):
	if parallel:
		m_list, c_list = getMembAssignmentFileList(nbatches)
		
		n_members_missing = checkFiles(m_list)
		n_cluster_missing = checkFiles(c_list)

		if (n_members_missing>0) or (n_cluster_missing>0):
			print('creating temporary files')

			galaxies, clusters = loadTable(kwargs)
			_, _ = writeSmallFiles(clusters,galaxies,nbatches)

		return m_list, c_list

	else:
		galaxies, clusters = loadTable(kwargs)
		return galaxies, clusters

def get_date_time():
	from datetime import datetime
	now = datetime.now()
	dt_string = now.strftime("%H:%M:%S_%m%d%Y")

	return now,dt_string

def save_run_info(totalTime, date, run_info_file = 'run_info.out'):
	import json
	
	out_dict = getConfig("Files","bla",getAllVariables=True)
	out_dict['scriptPath'] = file_path_script
	out_dict['dateTime'] = date.strftime("%H:%M:%S - %m/%d/%Y")
	out_dict['TotalTime'] = str(round(totalTime,3))+' seconds'
	
	out_j = json.dumps(out_dict)
	f=open(run_info_file,"w")
	f.write(out_j)
	f.close()

def computeMembAssignment():
	logging.info('Starting COPACABANA - COlor Probabilistic Assignment for Clusters and Bayesian ANAlysis')

	# get initial time
	total_t0 = time()

	# get initial date
	date0, date0_str = get_date_time()

	parallel = isOperationSet(operation="parallel")

	kwargs = getKwargs()

	if parallel:
		nbatches = int(getConfig("Parallel", "batches"))
		nprocess = int(getConfig("Parallel", "process"))

		m_list, c_list = loadTables(kwargs,parallel=True,nbatches=nbatches)

		sleep(2)
		out = Parallel(n_jobs=nprocess)(delayed(triggerMembAssignment)(idx,c_list[idx],m_list[idx],**kwargs) for idx in range(nbatches) )
		g0, cat = getOutFile(out)

	else:
		galaxies, clusters = loadTables(kwargs,parallel=False)
		g0, cat = membAssign.clusterCalc(galaxies,clusters,**kwargs)

	## update Pmem and compute Ptaken
	galOut = computePmem(g0,plim=kwargs['p_low_lim'])

	### update Ngals
	catOut = computeNgals(galOut,cat)

	### saving output
	writeFilesParallel(galOut,catOut)

	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')

	logging.debug('Returning from ComputeMemberAsignment()')

	### saving run info
	run_info_file = './auxTable/run_info/copa_info_%s.out'%(date0_str)
	save_run_info(totalTime,date0,run_info_file=run_info_file)

if __name__ == "__main__":
	computeMembAssignment()
