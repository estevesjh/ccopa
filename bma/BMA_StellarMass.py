# new libraries
import ConfigParser
import logging

from time import time
from os import path
import numpy as np
from joblib import Parallel, delayed
from astropy.io.fits import getdata
import os

# imports from nike.py below
import smass
import helperfunctions
import clusterSMass_orig
import colorModeling

def getConfig(section, item, boolean=False, getAllVariables=False,userConfigFile="./bma/BMA_StellarMass_Config.ini"):

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
			print msg
		return ""

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))
	if (section != 'Log'):
		logging.debug(msg)
	else:
		print msg

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

def stellarMassOutFile():
	stellarMassInfile = getConfig("Files","membersInputFile")
	stellarMassOutFile = os.path.splitext(stellarMassInfile)[0]+'_stellarMass.fits'

	if not stellarMassOutFile:
		logging.critical("Can't continue without stellarMassOutPrefix defined! Exiting.")
		exit()

	return stellarMassOutFile

def combineFits():
	from combineCat import combineBMAStellarMassOutput
	stellarMassOutPrefix = getStellarMassOutPrefix()
	combineBMAStellarMassOutput(stellarMassOutPrefix)

def matchFits(combineFiles=False):
	from combineCat import matchBMAStellarMassOutputCOPA
	stellarMassOutPrefix = getStellarMassOutPrefix()
	membersInFile = getConfig("Files","membersInputFile")
	
	matchBMAStellarMassOutputCOPA(membersInFile,stellarMassOutPrefix,save=combineFiles)

def getSize():
	inPath = getInputPath()
	membersInFile = getConfig("Files","membersInputFile")

	if (not inPath or not membersInFile):
		logging.critical("Can't continue without either inputPath or membersInputFile defined! Exiting.")
		exit()

	nsize = getdata(membersInFile).size

	return nsize

def get_date_time():
	from datetime import datetime
	now = datetime.now()
	dt_string = now.strftime("%H:%M:%S_%m%d%Y")

	return now,dt_string

def save_run_info(totalTime, date, run_info_file = 'run_info.out'):
	import json

	out_dict = getConfig(getAllVariables=True)
	out_dict['dateTime'] = date.strftime("%H:%M:%S - %m/%d/%Y")
	out_dict['TotalTime'] = str(round(totalTime,3)+' seconds'
	out_dict['scriptPath'] = file_path_script

	out_j = json.dumps(out_dict)
	f=open(run_info_file,"w")
	f.write(out_j)
	f.close()

def computeStellarMass(batch, memPerJob):
	# For running the stellar masses (takes the longest)
	batchIndex = batch + memPerJob
	job = int(batchIndex / memPerJob)

	logging.debug('Starting computeStellarMass() with batch = {b}; job = {j}.'.format(
		b = batch, j = job))

	stellarMassOutFile = getConfig("Files","stellarMassOutPrefix") + "{:0>5d}.fits".format(job)

	inPath = getInputPath()
	membersInFile = getConfig("Files","membersInputFile")

	if (not inPath or not membersInFile):
		logging.critical("Can't continue without either inputPath or membersInputFile defined! Exiting.")
		exit()

	inputDataDict = helperfunctions.readCCOPA(membersInFile, batch, batchIndex)

	smass.calcCOPA(inputDataDict, outfile=stellarMassOutFile, indir=inPath, lib="miles")

	logging.debug('Returning from computeStellarMass() with batch = {b}; job = {j}.'.format(
		b = batch, j = job))

def computeColorModel():
	stellarMassFile = stellarMassOutFile()
	clusterOutFile  = getConfig("Files","clusterStellarMassOutFile")

	ncls_per_bin = 20
	rs_outfile = getConfig("colorModel", "rs_outfile")
	
	colorModeling.colorModel(clusterOutFile,stellarMassFile,output_rs=rs_outfile)

def computeClusterStellarMass():
	stellarMassFile = stellarMassOutFile()
	
	clusterOutFile  = getConfig("Files","clusterStellarMassOutFile")
	clusterInfile  = getConfig("Files","clusterInputFile")
	
	simulation = isOperationSet(operation="simulationTest")
	colorFit = isOperationSet(operation="colorModel")

	logging.info('Computing cluster stellar mass.')
	# clusterSMass_orig.haloStellarMass(filename = stellarMassFile, outfile = clusterOutFile)
	clusterSMass_orig.haloStellarMassCOPA(filename = stellarMassFile, outfile = clusterOutFile, cluster_infile=clusterInfile, colorFit=colorFit, simulation=simulation)

def parallelComputeStellarMass(batchStart=0,
		batchMax=25936, nJobs=100, nCores=20):
		# nJobs is normally = 100
	batchesList = np.linspace(batchStart, batchMax, nJobs, endpoint=False, dtype=int)

	logging.info('Calling parallelism inside parallelComputeStellarMass().')
	Parallel(n_jobs=nCores)(delayed(computeStellarMass)
		(batch, (batchMax - batchStart) / nJobs) 
		for batch in batchesList)

	# generate concatenated fits file
	#logging.info('Combining fits.')
	#combineFits()
	
def main():
	# start logging
	createLog()

	logging.info('Starting BMA Stellar Masses program.')

	# get initial time
	total_t0 = time()

	# get initial date
	date0, date0_str = get_date_time()

	# check and parallel compute stellar mass,
	#	if it is the case
	if (isOperationSet(operation="stellarMass")): ## get in config>operation>stellarMass (True or False)
		logging.info('Starting parallel stellar masses operation.')
		section = "Parallel"

		stellarMass_t0 = time()
		# get parallel information
		batchStart = int(getConfig(section, "batchStart"))
		# batchMax   = int(getConfig(section, "batchMax"))
		nJobs 	   = int(getConfig(section, "nJobs"))
		nCores 	   = int(getConfig(section, "nCores"))

		batchMax = getSize()
		
		# call function to parallel compute
		parallelComputeStellarMass(batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)

		# save time to compute stellar mass
		stellarMassTime = time() - stellarMass_t0
		stellarMassMsg = "Stellar Mass (parallel) time: {}s".format(stellarMassTime)
		logging.info(stellarMassMsg)

	# append output table into input file
	logging.info('Matching fits.')
	matchFits()
	
	# check and compute cluster stellar mass,
	#	if it is the case
	if (isOperationSet(operation="clusterStellarMass")):
		logging.info('Starting cluster stellar mass operation.')
		clusterStellarMassTime_t0 = time()
		computeClusterStellarMass()

		# save time to compute cluster stellar mass
		clusterStellarMassTime = time() - clusterStellarMassTime_t0
		clusterStellarMassMsg = "Cluster Stellar Mass time: {}s".format(clusterStellarMassTime)
		logging.info(clusterStellarMassMsg)

	if (isOperationSet(operation="colorModeling")):
		logging.info('Starting color modeling operation.')
		colorModelTime_t0 = time()
		computeColorModel()

		# save time to compute cluster stellar mass
		colorModelTime = time() - colorModelTime_t0
		colorModelMsg = "Cluster Stellar Mass time: {}s".format(colorModelTime)
		logging.info(colorModelMsg)

	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')

	### saving run info
	run_info_file = './auxTable/run_info/bma_info_%s.out'%(date0_str)
	save_run_info(totalTime,date0,run_info_file=run_info_file)

if __name__ == "__main__":
    main()
