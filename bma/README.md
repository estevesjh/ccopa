# BMAStellarMasses

## BMA Stellar Masses (from vt-clustertools). 

**Computes stellar masses and cluster stellar mass.**

Updated and adjusted from previous version:
https://github.com/SSantosLab/vt-clustertools/tree/master/BMAStellarMasses

## Getting Started 

### File structure

Copy or clone the files in this repository to your directory (*/BMAStellarMass/*, for example).

File structure required:

	.
	|-- BMA_StellarMass_Config.ini
	|-- BMA_StellarMass.py
	|-- clusterSMass_orig.py
	|-- combineCat.py
	|-- CosmologicalDistance.py
	|-- helperFunctions.py
	|-- loadPopColors.py
	`-- smass.py

### Configuration File

A configuration file is required to run: 

    BMA_StellarMass_Config.ini.
    
This file is separated by the following sections: **[Paths]**, **[Files]**, **[Log]**, **[Operations]** and **[Parallel]**.
It is convenient to save a "backup version" before changing this config file.

```bash
cp  BMA_StellarMass_Config.ini BMA_StellarMass_Config_BCKP.ini
```

Specify your desired path, input and output filenames, operations to run, and parallel configuration.

Complete example of configuration file (without comments):

	[Paths]
	inputPath: /data/des61.a/data/pbarchi/galaxyClusters/simha_miles_Nov2016/
			
	[Files]
	tarFile: ./simha_miles_Nov2016.tar.gz
	membersInputFile: /data/des61.a/data/pbarchi/galaxyClusters/test/membersFiles/test_m1_1cluster.fits
	stellarMassOutPrefix: /des/des61.a/data/pbarchi/clusters/test/testStellarMasses_
	clusterStellarMassOutFile: /des/des61.a/data/pbarchi/clusters/test/testClusterStellarMasses_full.fits
		
	[Log]
	logFile: ./data/des61.a/data/pbarchi/galaxyClusters/test/BMA-StellarMass.log
	level: INFO
			
	[Operations]
	stellarMass: True
	clusterStellarMass: True
			
	[Parallel]
	batchStart: 0
	batchMax: 146737
	nJobs: 100
	nCores: 20

#### [Paths]

The *inputPath* item is the old *indir* variable used in the previous version of this code. 

Just to have on record, the following is the "default path" which was assigned to *indir* (now, *inputPath*):

	inputPath: /data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_Nov2016/

There's a backup of these files in the following path (current "default value" for *inputPath*):

	inputPath: /data/des61.a/data/pbarchi/galaxyClusters/simha_miles_Nov2016/

If a path to these files does not exist for some reason, you can leave *inputPath* blank and specify a compressed file (*.tar.gz*) in the *tarFile* item (see Subsection **[Files]**).

#### [Files]

..* *tarfile*: If *inputPath* is left blank, the system tries to uncompress the file specified in *tarFile*. The default tarFile is available in this repository:

	tarFile: ./simha_miles_Nov2016.tar.gz

..* *membersInputFile*: fits input file with cluster members.

..* *stellarMassOutPrefix*: prefix to stellar mass output files. If:

	stellarMassOutPrefix: ./sMassOut_

the stellar mass output file from the first job will be named *sMassOut_00001.fits*, and, after combining the outputs from all of the jobs, the final output from stellar mass operation will be *sMassOut_full.fits*.

..* *clusterStellarMassOutFile*: fits output file name for cluster stellar mass operation.

#### [Log]

The system logs everything that happens in the following format:

	<datetime> [<LEVEL>]	<file> - <msg>

Example of a line of the log file (first line):

	[2018-11-16 16:22:33,440] [INFO]	BMA_StellarMass - Starting BMA Stellar Masses program.

You can specify one of two log levels: *INFO* or *DEBUG*. *DEBUG* outputs all messages from the system, including values of the variables in each calculation step -- this should be used only on debug runs -- if something goes wrong or if you want to trace every detail. The default value for log level is *INFO*, with which you will get *INFO*, *WARNING* and *CRITICAL* messages.

Example of the log section:

	[Log]
	logFile: ./BMA-StellarMass.log
	level: INFO

#### [Operations]

In Operations section you can define the steps you want to run (with boolean values: True or False). For example, if you desire to run only the stellar mass code, your [Operations] section would look like:

    stellarMass: True
    clusterStellarMass: False

#### [Parallel]

Parallel computing instructions comes from previous version of the code.

There are two parameters that need to be entered for each batch submission:

1. *batchStart*: the starting member point for this batch out of all the members. The first batch should begin at 0.
2. *batchMax*: the ending member point for this batch out of all members. To avoid duplicates this number should increase by the same amount batchStart increases when doing multiple batches.

The two other parameters can be adjusted as necessary, but, generally, they should stay the same:

3. *nJobs*: the number of jobs the batch submission is split into. *100* is a good number.
4. *nCores*: the total number of cores used by the computing machine at any given time. *20* is suggested for the DES cluster.

Previous experience about adjusting these parameters:
*What I did was load up the full member output catalog from afterburner to see how many members there were, and since we can use 5 DES cluster machines (des30,40/41,50/51, though ask Marcelle you might be able to use more), I then divided the total member number by 5. Let’s say it was 1 million. Then each machine should compute stellar masses for 200k members. So, for your first batch of jobs on des30, let’s say, you would use:*

	batchStart: 0
	batchMax: 200000

*Then on the next machine, des40, you would use:*

	batchStart: 200000
	batchMax: 400000

*In this way you would continue until you had all 1 million members running on 500 jobs (100 for each machine) over 5 machines.*

## Running

With the config file set up, run it with:
```bash
bash runBMA_StellarMass_bckp.sh
```
    
This script activates the necessary environment and calls the main script BMA_StellarMass.py.

If you already have an activated astro environment you can just run the python script (using *screen* to avoid disconnection issues):
```bash
python BMA_StellarMass.py
```    

### Activating environment

In the bash script *runBMA_StellarMass_bckp.sh* there is a function to activate the environment:

```bash
activateEnvironment() {
	# setting up anaconda
	export CONDA_DIR=/cvmfs/des.opensciencegrid.org/fnal/anaconda2
	source $CONDA_DIR/etc/profile.d/conda.sh
	# activating astro environment
	conda activate des18a
}
```

## Authors

* DES Galaxy Clusters collaboration (https://github.com/SSantosLab/vt-clustertools/tree/master/BMAStellarMasses)
