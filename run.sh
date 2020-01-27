### This code runs Copacabana that consists in copa and BMA stellar mass code

## Insert the input files
fileGal="/data/des61.a/data/johnny/Buzzard/ccopaTests/Chinchilla-0Y1a_v1.6_truth_1000_highMass_clusters_members.fits"
fileCls="/data/des61.a/data/johnny/Buzzard/ccopaTests/Chinchilla-0Y1a_v1.6_truth_1000_highMass_clusters.fits"
fileOut="./out/Chinchilla-0Y1a_v1.6_truth_1000_highMass_ccopa_pz_005_Rfixed"

#fileGal="/data/des61.a/data/johnny/SPLUS/SPLUS_STRIPE82_GAL_10sigma_master_catalog_dr_march2019.fits"
#fileCls="/data/des61.a/data/johnny/SPLUS/catalogueofclustersfinal_withoutbcgs.fits"
#fileOut="/data/des61.a/data/johnny/SPLUS/splus_STRIPE82_master_DR2_SN_10_galaxyClusters_pmem"

## set the input files
python set_input_files.py $fileGal $fileCls $fileOut

echo "running copa"
python ./copa/main.py

echo "running BMA stellar mass"
python bma/BMA_StellarMass.py

echo "it is done"
