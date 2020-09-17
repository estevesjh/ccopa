import configparser as ConfigParser
import argparse
import sys
import os


def header():
	print(5*'--'+'Welcome to Copacabana'+5*'--'+'\n')

def create_arg_parser():
	""""Creates and returns the ArgumentParser object."""
	parser = argparse.ArgumentParser(description='Copcabana')

	parser.add_argument('galaxyInputFile',
					help='Filename to the input galaxy catalog.')
	parser.add_argument('clusterInputFile',
					help='Filename to the input cluster catalog.')
	parser.add_argument('outputPrefix',
					help='Prefix to the output catalog')
	parser.add_argument('--truthTable',  nargs='?', type=str2bool,
					const=True, default=False,
					help='Activate mode to run for true members. Only possible for simulations!')
	parser.add_argument('--pdfFile', default=None,
					help='Prefix to the pdf hdf5 files')
				
	return parser

def checkFiles(parsed_args):
	files = [parsed_args.galaxyInputFile,parsed_args.clusterInputFile]

	for filename in files:
		if not os.path.exists(filename):
			print("File %s does not exist"%(filename))
			print("Critical error: exiting the program")
			exit()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setInputFile(section,item,value,userConfigFile='CCOPA_Config.ini'):
	""" Set a new value for a item in a given config file"""

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	print('Seting %s : %s \n'%(item,value))
	configFile.set(section,item,value)

	with open(userConfigFile, 'w') as config:
		configFile.write(config)

header()

copaConfigFile="./copa/copa_config_bz.ini"
bmaConfigFile="./bma/BMA_StellarMass_Config.ini"

arg_parser = create_arg_parser()
args = arg_parser.parse_args(sys.argv[1:])

# galalaxy_file_name = input('Inuput the galaxy catalog filename:')
# cluster_file_name = input('Inuput the cluster catalog filename:')

checkFiles(args)

if args.truthTable:
	args.outputPrefix += '_truth_table'
	args.pdfFile += '_truth'
	setInputFile('Operations','truthTable','True',userConfigFile=copaConfigFile)
	setInputFile('Operations','truthTable','True',userConfigFile=bmaConfigFile)
else:
	setInputFile('Operations','truthTable','False',userConfigFile=copaConfigFile)
	setInputFile('Operations','truthTable','False',userConfigFile=bmaConfigFile)

## set files name in the copa config file
setInputFile('Files','clusterInputFile',args.clusterInputFile,userConfigFile=copaConfigFile)
setInputFile('Files','galaxyInputFile',args.galaxyInputFile,userConfigFile=copaConfigFile)

setInputFile('Files','clusterOutFilePrefix',args.outputPrefix,userConfigFile=copaConfigFile)
setInputFile('Files','galaxyOutFilePrefix',args.outputPrefix+'_members',userConfigFile=copaConfigFile)

if args.pdfFile is not None:
	setInputFile('Files','pdfOutfilePrefix',args.pdfFile,userConfigFile=copaConfigFile)
	
## set files name in the BMA config file
setInputFile('Files','membersinputfile',args.outputPrefix+'_members.fits',userConfigFile=bmaConfigFile)
setInputFile('Files','clusterinputfile',args.outputPrefix+'.fits',userConfigFile=bmaConfigFile)

setInputFile('Files','stellarmassoutprefix',args.outputPrefix+'_stellarMass_members_',userConfigFile=bmaConfigFile)
setInputFile('Files','clusterstellarmassoutfile',args.outputPrefix+'_stellarMass.fits',userConfigFile=bmaConfigFile)
