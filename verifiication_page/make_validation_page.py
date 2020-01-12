#!/usr/bin/env python
"""
Generating html
"""
__author__ = "Johnny Esteves"

INDEX = """
<!DOCTYPE html>
<html>
<title>Copacabana: Validation Page</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Montserrat">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<style>
body, h1,h2,h3,h4,h5,h6 {font-family: "Montserrat", sans-serif}
.w3-row-padding img {margin-bottom: 16px}
/* Set the width of the sidebar to 120px */
.w3-sidebar {width: 32px;background: #222;}
/* Add a left margin to the "page content" that matches the width of the sidebar (120px) */
/* Remove margins from "page content" on small screens */
@media only screen and (max-width: 600px) {#main {margin-left: 0}}
</style>
<body class="w3-white">

<header class="w3-container w3-top w3-blue w3-xlarge">
  <span class="w3-right w3-padding">Copacabana</span>
</header>

<!-- Overlay effect when opening sidebar on small screens -->
<div class="w3-overlay w3-animate-opacity" onclick="w3_close()" style="cursor:pointer" title="close side menu" id="myOverlay"></div>

<!-- Navbar on small screens (Hidden on medium and large screens) -->
<div class="w3-top">
  <div class="w3-bar w3-theme w3-top w3-left-align w3-large">
  <!--div class="w3-bar w3-theme w3-top w3-left-align w3-large"-->
    <a href="#" class="w3-bar-item w3-btn w3-text-white">Dataset</a>
    <a href="#results" class="w3-bar-item w3-btn w3-text-white w3-hide-small">Results</a>
    <a href="#color" class="w3-bar-item w3-btn w3-text-white w3-hide-small">Color</a>
  </div>
</div>

<!-- PAGE CONTENT -->
<div class="w3-main w3-content" style="max-width:1600px;margin-top:60px">
<div class="w3-padding-large" style="max-width:1600px;margin-top:60px" id="main">
	<!-- Header/Home -->
	  
	<div class="w3-content w3-justify w3-text-grey w3-padding-64" id="dataset">
	<h2 class="w3-text-grey">Dataset</h2>
	<hr style="width:200px" class="w3-opacity">
	<p> Welcome to Copacabana web verification page. Here we have details of the input data and some analystics of the output catalog. </p>

	<p>
	<table class="w3-table w3-bordered">
	<tr>
		<th>Dataset Name</th>
		<th>Size</th>
		<th>Date</th>
	</tr>
	<tr>
		<td>Buzzard v1.6</td>
		<td>1000 GC</td>
		<td>01/01/20</td>
	</tr>
	</table>
	</p>
	<br> <b/br>

	<h4 class="w3-text-grey"> Sky Plot </h4>
	<figure>
        <td> <a id="./img/sky_plot_sample.png"></a> <img src="./img/sky_plot_sample.png" style="width:100%%"> </td>
        <figurecaption> <h4> Figure: Distribution of the sources on the sky </h4> </figurecaption>
    </figure>
	
  <!-- Results Section -->
%(results_section)s

  <!-- Color Section -->
%(color_section)s

  <!-- Footer -->
  <footer class="w3-content w3-padding-64 w3-text-grey w3-xlarge">
    <a href="https://github.com/estevesjh/ccopa"> <i class="fa fa-github w3-hover-opacity"></i> </a>
    <p class="w3-medium">Fermi wiki page <a href="https://www.w3schools.com/w3css/default.asp" target="_blank" class="w3-hover-text-green">w3.css</a></p>
  <!-- End footer -->
  </footer>
</div>
<!-- END PAGE CONTENT -->
</div>

</body>
</html>
"""

RESULTS_SECTION = """
  <div class="w3-content w3-justify w3-text-grey w3-padding-64" id="results">
    <h2 class="w3-padding-16 w3-text-grey">Results</h2>
    <!-- Grid for photos -->
    <div class="w3-section">
      <p> to write </p>

      %(figure)s

    </div>
"""

COLOR_SECTION = """
  <div class="w3-padding-64 w3-content w3-text-grey" id="color">
    <h2 class="w3-text-grey">Color</h2>
    <hr style="width:200px" class="w3-opacity w3-center">
    <div class="w3-section w3-center">

      %(figure)s

  </div>
"""

FIGURE = """
    <figure>
        <td> <a id="%(fname)s"></a> <img src="%(fname)s" style="width:100%%"> </td>
        <figurecaption> <h4> Figure: %(caption)s </h4> </figurecaption>
    </figure>
"""

def insert_figure_list(fname_list, caption_list, SECTION):
	# figure_lines: is the figure sytanx added to the section (SECTION)
	# new_section: is the section syntax added with the figure filenames and captions

	tmp = [ FIGURE%dict(fname=fi,caption=ci) for (fi,ci) in zip(fname_list,caption_list)]

	figure_lines = '\n <br> </br>'.join(tmp)
	new_section = SECTION%dict(figure=figure_lines)
	
	return new_section

def get_image_captions():
	resul_caption_1 = 'Identity'
	resul_caption_2 = 'Residual'

	color_caption_1 = 'Color distribution for different redshift intrevals.'
	color_caption_2 ='Colors as a function of the redshift.'

	color_caption_list = [color_caption_1,color_caption_2]
	resul_caption_list = [resul_caption_1,resul_caption_2]

	return resul_caption_list, color_caption_list

def get_image_path():
	resul_fname_list = ['./img/mu_identity_4.png','./img/mu_residual_4_ntrue.png']
	color_fname_list = ['./img/animated_gi_o.gif','./img/color_redshift.png']

	return resul_fname_list, color_fname_list

def main():
	filename = 'web_page_test_0.html'
	
	resul_caption_list, color_caption_list = get_image_captions()
	resul_fname_list, color_fname_list = get_image_path()

	color_section = insert_figure_list(color_fname_list, color_caption_list, COLOR_SECTION)
	resul_section = insert_figure_list(resul_fname_list, resul_caption_list, RESULTS_SECTION)

	# rname1 = ['./img/mu_identity_4.png','Identity']
	# rname2 = ['./img/mu_residual_4_ntrue.png','Residual']

	# cname1 = ['./img/animated_gi_o.gif','Color distribution for different redshift intrevals.']
	# cname2 = ['./img/color_redshift.png','Colors as a function of the redshift.']

	# figure_color_row1 = FIGURE%dict(fname=cname1[0],caption=cname1[1])
	# figure_color_row2 = FIGURE%dict(fname=cname2[0],caption=cname2[1])

	# figure_resul_row1 = FIGURE%dict(fname=rname1[0],caption=rname1[1])
	# figure_resul_row2 = FIGURE%dict(fname=rname2[0],caption=rname2[1])

	# color_section = COLOR_SECTION%dict(figure1=figure_color_row1, figure2=figure_color_row2)
	# resul_section = RESULTS_SECTION%dict(figure1=figure_resul_row1, figure2=figure_resul_row2)

	index = INDEX%dict(results_section=resul_section,color_section=color_section)

	with open(filename,'w') as out:
		out.write(index)

main()

# REDSHIFT_SECTION = """
# <div class="w3-padding-64 w3-content" id="redshift">
#   <h2 class="w3-text-grey">Redshift</h2>
#   <hr style="width:200px" class="w3-opacity">

#   <!-- Grid for photos -->
#   <div class="w3-row-padding" style="margin:0 -16px">
#     <div class="w3-half">
#       <img src="/w3images/wedding.jpg" style="width:100%">
#     </div>

#     <div class="w3-half">
#       <img src="/w3images/underwater.jpg" style="width:100%">
#     </div>
#   <!-- End photo grid -->
#   </div>
# <!-- End Portfolio Section -->
# </div>
# """
