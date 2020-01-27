#!/usr/bin/env python
"""
Generating Copacaban validation web pag
"""
__author__ = "Johnny Esteves"

DESCRIPTION = """
	<p> Welcome to Copacabana web verification page. The input galaxy catalog is a selction of the most massive GC
	in the Buzzard simulation v1.6. The sample has 1000 GC with mass greater than 1e14 M_sun ranging between redshift 0.1 and 0.9.
	The galaxy catalog input has all galaxies inside a circle of 12Mpc around each cluster center. 
	We selected as true member galaxies all the galaxies inside the projected radius r200 of a given HALOID.
	</p>
"""

DESCRIPTION_TABLE = open('./auxTable/description_table_html.txt','r').read()

def get_image_path():
	import glob
	# resul_fname_list = ['./img/mu_identity_4.png','./img/mu_residual_4_ntrue.png']
	fname_list0 = glob.glob("./img/*evolution.png")
	fname_list0.reverse()

	fname_list1 = glob.glob("./img/scale_*.png")
	fname_list1.reverse()

	fname_list2 = glob.glob("./img/redshift*")

	# color_fname_list = ['./img/animated_gi_o.gif','./img/color_redshift.png']

	return [fname_list0,fname_list1,fname_list2]

def filter_info(dicto):
	colnames = ['Date and Time:','Total time runing:',
				'Cluster Input File:','Galaxy Input File:',
				'Cluster Output File:','Galaxy Output File:']

	col_old = ['dateTime','TotalTime','galaxyinputfile','clusterinputfile',
 			  'clusteroutfileprefix','galaxyoutfileprefix']
			   
	new_d = dict()
	for keys,keys_new in zip(col_old,colnames):
		new_d[keys_new] = dicto[keys]

	return new_d

def main():
	import json
	from copa.makeHTML import sections, build_index_page

	filename = 'index.html'
	run_info_file = './auxTable/run_info/copa_17:18:13_01272020.out'

	f1, f2, f3 = get_image_path()
	info_dict = json.load(open(run_info_file,"r"))

	## make info table
	info_dict = filter_info(info_dict)

	## Creating the first section
	header = sections('Dataset',title='Dataset: Buzzard Simulation v1.6')
	header.add_to_section(DESCRIPTION)
	header.add_figure(["./img/sky_plot.png"],caption=['Distributions of the sources on the sky'])
	header.add_html_table(info_dict,title='1',buttom=None)
	header.add_to_section(DESCRIPTION_TABLE)

	## Creating the COPA section
	copa = sections('Copa',title='COPA - COlor Probabilistic Assignment')
	copa.add_slide_images(f3)

	## Creating the BMA section
	bma = sections('BMA',title='BMA Stellar Mass')
	bma.add_to_section('<p> Here you can find the galaxy catalog outputs. </p>')
	bma.add_slide_images(f1)

	bma.add_to_section('<p> Here you can find the cluster catalog outputs. </p>')
	bma.add_slide_images(f2)

	build_index_page([header,copa,bma],fname=filename)

main()
