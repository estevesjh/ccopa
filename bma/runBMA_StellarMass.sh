#!/bin/bash

# from: http://home.fnal.gov/~kadrlica/fnalstart.html
activateEnvironment() {
	# setting up anaconda
	export CONDA_DIR=/cvmfs/des.opensciencegrid.org/fnal/anaconda2
	source $CONDA_DIR/etc/profile.d/conda.sh
	# activating astro environment
	conda activate des18a
}

activateEnvironment
# Run python script
python BMA_StellarMass.py
