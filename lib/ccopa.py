# libraries
import configparser as ConfigParser
import logging
from time import time
from os import path
from astropy.table import Table, vstack

# local libraries
import membAssignment as membAssign
import helper as helper

####
import numpy as np
from dask import compute, delayed


def getConfig(section, item, boolean=False,
		userConfigFile="CCOPA_Config.ini"):

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


def extractTarGz(tarFileName, path):
	import tarfile
	tar = tarfile.open(tarFileName, "r:gz")
	tar.extractall(path=inputPath)
	tar.close()


def getInputPath():
	inputPath = getConfig("Paths","inputPath")
	# if a inputPath is not set, go after the .tar.gz file
	if (not inputPath):

		# if tarFile doesn't exist, abort
		tarName = getConfig("Files","tarFile")

		if (not tarName or not path.isfile(tarName) or
				not tarName.endswith("tar.gz")):

			return ""

		# defining inputPath to uncompress file
		inputPath = "./simha_miles_Nov2016/"
		extractTarGz(tarFileName=tarName, path=inputPath)

	return inputPath


def getStellarMassOutPrefix():
	stellarMassOutPrefix = getConfig("Files","stellarMassOutPrefix")

	if not stellarMassOutPrefix:
		logging.critical("Can't continue without stellarMassOutPrefix defined! Exiting.")
		exit()

	return stellarMassOutPrefix


# def combineFits():
# 	from combineCat import combineBMAStellarMassOutput
# 	stellarMassOutPrefix = getStellarMassOutPrefix()
# 	combineBMAStellarMassOutput(stellarMassOutPrefix)

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

def parallelComputeMemberAsignment(galaxy_cat, cluster_cat, member_outfile, cluster_outfile, nbatches=5,nprocess=10,r_in=6,r_out=8,bandwidth=0.01,p_low_lim=0.01):
	print('parallels')
	logging.info('Calling parallelism inside parallelComputeMemberAsignment().')
	
	ncls = len(cluster_cat)
	islice = np.linspace(0,ncls,nbatches,dtype=int)
	delayed_results = []
    
	for i in range(nbatches-1):
		ci = cluster_cat[islice[i]:islice[i+1]]
		gi = group_table_by(galaxy_cat,'CID',ci['CID'])

		m_out = member_outfile+'%04i.fits'%(islice[i])
		c_out = cluster_outfile+'%04i.fits'%(islice[i])

		out = delayed(membAssign.clusterCalc)(gi,ci,m_out,c_out,r_in=r_in,r_out=r_out,bandwidth=0.01,p_low_lim=p_low_lim)
		delayed_results.append(out)

	#total = delayed_results.compute()
	results = compute(*delayed_results)
	#results = compute(*delayed_results, scheduler='processes', num_workers=nprocess)

	logging.debug('Returning from ComputeMemberAsignment()')

def ComputeMemberAsignment():
	parallel = False
	# start logging
	createLog()

	logging.info('Starting CCOPA - Cluster COlor Probabilistic Assignment.')

	# get initial time
	total_t0 = time()

	nbatches = int(getConfig("Parallel", "batches"))
	nprocess = int(getConfig("Parallel", "process"))
	
	computeR200 = isOperationSet(operation="computeR200")
	query = isOperationSet(operation="query")

	galaxyOutFile = getConfig("Files","galaxyOutFilePrefix") 
	clusterOutFile = getConfig("Files","clusterOutFilePrefix")

	galaxyInFile = getConfig("Files","galaxyInputFile")
	clusterInFile = getConfig("Files","clusterInputFile")

	rMax = round( float(getConfig('Cuts', "radius_cutouts")), 1)
	r_in = round( float(getConfig('Cuts', "radius_bkg_in")),1)
	r_out = round(float(getConfig('Cuts', "radius_bkg_out")),1)

	zmin = round(float(getConfig('Cuts', "zmin_gal")),3)
	zmax = round(float(getConfig('Cuts', "zmax_gal")),3)
	plim = float(getConfig('Cuts', "p_low_lim"))	
	Nflag= int(getConfig('Cuts', "flag_low"))
	window = round(float(getConfig('Cuts', "redshiftWindow")),2)

	if (not clusterInFile):
		logging.critical("Can't continue without either inputPath or clusterInputFile defined! Exiting.")
		exit()

	## Prepare Cluster Catalog to MembAssign
	columnsLabelsCluster = getClusterColsName(massProxy=True)
	clusters = helper.readClusterCat(clusterInFile, colNames=columnsLabelsCluster)
	clusters = clusters[0:3]
	## Prepare Galaxy Catalog to MembAssign # cut rmax [Mpc] around each cluster
	if query:
		print('to do')
		exit()
		# heleperQuery.queryGalaxyCat(galaxyOutFile, clusters, radius=12, zrange=(zmin,zmax), Nflag=Nflag)
	else:
		columnsLabelsGalaxies = getGalaxyColsName()
		galaxies = helper.readGalaxyCat(galaxyInFile, clusters, zrange=(zmin,zmax), window=0.1,
										r_in=r_in, r_out=r_out, radius=rMax,
										Nflag=Nflag,colNames=columnsLabelsGalaxies)
	print('finish galaxy cat uploads')
	if parallel:
		# Compute MembAssign: write galaxy and cluster catalog in galaxyOutFile
		parallelComputeMemberAsignment(galaxies,clusters,galaxyOutFile,clusterOutFile,
										r_in=r_in,r_out=r_out,bandwidth=0.01,p_low_lim=plim,
										nbatches=nbatches,nprocess=nprocess)
		
	else:
		membAssign.clusterCalc(galaxies,clusters,galaxyOutFile,clusterOutFile,
										r_in=r_in,r_out=r_out,bandwidth=0.01,p_low_lim=plim)

	# # generate concatenated fits file
	# logging.info('Combining fits.')
	# combineFits()

	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')


if __name__ == "__main__":
    ComputeMemberAsignment()
