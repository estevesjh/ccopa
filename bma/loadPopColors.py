import numpy as np
import logging
# reload(loadPopColors); simha=loadPopColors.doallnames( "/Users/annis/Code/fsps/simha/",lib="basel"); keys = np.fromiter(iter(simha.values()),dtype="a100");

# do them all
#   lib can be miles or basel

# Feb17 Antonella modification: mass weighted ages are computed in loadsplines, for now only simha SFH is allowed
def doAll ( dir = "simha/", lib="basel") :
    logging.debug('Starting loadPopColors.doAll()')
    data = []
    zmets = []
    metallicites_basel = [10,14,17, 20, 22]
    metallicites_miles = [12,17,20,21,22]        
    metallicites = metallicites_basel
    if lib == "miles" : metallicites = metallicites_miles
    start_list = [1.0,3.0,5.0,7.0]#,9.0,11.0] 
    trunc_list = [5, 7, 9, 11, 13] 
    tau_list = [0.3, 0.7, 1.0, 1.3, 2.0]#, 9.0] 
    theta_list = [-1.00,-1.73,-5.66]
    #metallicites = [20, ]
    #start_list = [1.0, ] 
    #trunc_list = [7,] 
    #tau_list = [1.0, ] 
    #theta_list = [-0.524, ] 
    counter = 0
    for metal in metallicites:
        if (metal == 10): zmet=0.002
        if (metal == 12): zmet=0.0031
        if (metal == 14): zmet=0.0049
        if (metal == 17): zmet=0.0096
        if (metal == 20): zmet=0.019
        if (metal == 21): zmet=0.024
        if (metal == 22): zmet=0.03 

        for start in start_list :
            for trunc in trunc_list :
                if (start < (trunc-2.1)) :
                    for tau in tau_list :
                        for theta in theta_list :
                            file = "s-" + str(metal) + "-" +str(start) + "-"
                            file = file + str(trunc) + "-" + str(tau) + str("%4.3f" % theta)
                            file = dir + file + ".mags"

                            #print counter, file; counter += 1

                            logging.debug('metal: {m}; start: {s}; trunc: {trunc}; tau: {tau}; theta: {theta}'.format(
                                m = metal, s = start, trunc = trunc, tau = tau, theta = theta))

                            gr,ri,iz,zY,grr,gir,kii,kri,ml,sfr,age,mass_weight_age = loadSplines(file,start,trunc,tau,theta)
                            data.append([gr,ri,iz,zY,grr,gir,kii,kri,ml,sfr,age,mass_weight_age])
                            zmets.append(zmet)

    logging.debug('Returning from loadPopColors.doAll()')
                        
    return data,zmets

#   lib can be miles or basel
def doallnames ( dir = "simha/", lib="basel") :
    data = []
    metallicites_basel = [10,14,17, 20, 22]
    metallicites_miles = [2,3,4,5]        
    metallicites = metallicites_basel
    if lib == "miles" : metallicites = metallicites_miles
    start_list = [0.7, 1.0, 1.5, 2.0] 
    trunc_list = [7, 9, 11, 13] 
    tau_list = [0.3, 1.0, 1.3, 2.0, 9.0, 13.0] 
    theta_list = [-0.175, -0.524, -0.785, -1.047, -1.396] 
    counter = 0
    names = dict()
    for metal in metallicites:
        for start in start_list :
            for trunc in trunc_list :
                for tau in tau_list :
                    for theta in theta_list :
                        file = "s-" + str(metal) + "-" +str(start) + "-"
                        file = file + str(trunc) + "-" + str(tau) + str(theta)
                        file = dir + file + ".mags"
                        names[counter] = file       
                        names[counter,"metals"] = metal
                        counter += 1
    return names


#   Log(Z/Zsol):  0.000
#   Fraction of blue HB stars:  0.200; Ratio of BS to HB stars:  0.000
#   Shift to TP-AGB [log(Teff),log(Lbol)]:  0.00  0.00
#   IMF: 1
#   Mag Zero Point: AB (not relevant for spec/indx files)
#   SFH: Tage= 14.96 Gyr, log(tau/Gyr)=  0.954, const=  0.000, fb=  0.000, tb=  11.00 Gyr, sf_start=  1.500, dust=(  0.00,  0.00)
#   SFH 5: tau=  9.000 Gyr, sf_start=  1.500 Gyr, sf_trunc=  9.000 Gyr, sf_theta= -0.785, dust=(  0.00,  0.00)
#
#   z log(age) log(mass) Log(lbol) log(SFR) mags (see FILTER_LIST)
#20.000  5.5000 -70.0000   0.0000 -70.0000  99.000  99.000  99.000  99.000  99.000
#20.000  5.5250 -70.0000   0.0000 -70.0000  99.000  99.000  99.000  99.000  99.000

def loadSplines ( filename,start,trunc,tau,theta ) :
    logging.debug('Starting loadPopColors.loadSplines()')

    from scipy.interpolate import interp1d
    from scipy.interpolate import UnivariateSpline

    dummyNum1 = -70.0000
    dummyNum2 = 99.000
    
    hdr, data = read(filename)
    zed, age, mass, lum, sfr, gr, ri, iz, zY, grr, gir, kii, kri = lineToNP(data) 

    mass_weight_age = np.zeros(age.shape[0])
    count=0
    for age_i in age:
            mass_weight_age[count] = mass_age_simha(age_i,start,trunc,tau,theta)
            count=count+1
    ix = np.nonzero((zed > 0) & (zed < 1.6)  & (gr < 90) & (ri < 90) & (iz <90) & (np.invert(np.isnan(mass)) ) & (np.invert(np.isinf(lum))))
    #ix = np.nonzero((zed > 0) & (zed < 1.6) & (sfr > -69. ) & (gr < 90) & (ri < 90) & (iz <90) & (np.invert(np.isnan(mass)) ) & (np.invert(np.isinf(lum))))
    #ug = UnivariateSpline(zed[ix], ug[ix], s=0)
    gr = UnivariateSpline(zed[ix], gr[ix], s=0)
    ri = UnivariateSpline(zed[ix], ri[ix], s=0)
    iz = UnivariateSpline(zed[ix], iz[ix], s=0)
    zY = UnivariateSpline(zed[ix], zY[ix], s=0)
    grr = UnivariateSpline(zed[ix], grr[ix], s=0)
    gir = UnivariateSpline(zed[ix], gir[ix], s=0)
    kii = UnivariateSpline(zed[ix], kii[ix], s=0)
    kri = UnivariateSpline(zed[ix], kri[ix], s=0)
    ml = UnivariateSpline(zed[ix], mass[ix]-lum[ix], s=0)
    sfr = UnivariateSpline(zed[ix], sfr[ix], s=0)
    age = UnivariateSpline(zed[ix], age[ix], s=0)
    mass_weight_age = UnivariateSpline(zed[ix], mass_weight_age[ix], s=0)

    logging.debug('Returning from loadPopColors.loadSplines()')
    return gr,ri,iz,zY,grr,gir,kii,kri,ml,sfr,age,mass_weight_age
    

def read (file) :
    dummyNum = -70.0000
    header = []
    data = []
    fd = open(file, "r") 
    for line in fd :
        if line[0] == "#" :
            header.append(line)
        else :
            words = line.split()
            zed = float(words[0])
            age = float(words[1])
            mass = float(words[2])
            lum = float(words[3])
            sfr = float(words[4])
            #ug = float(words[5])
            gr = float(words[5])
            ri = float(words[6])
            iz = float(words[7])
            zY = float(words[8])
            grr = float(words[9])
            gir = float(words[10])
            kii = float(words[11])
            kri = float(words[12])
            if age != dummyNum :
                newLine = [zed, age, mass, lum, sfr, gr, ri, iz, zY,grr,gir,kii,kri]
                data.append(newLine)
    return header, data

# convert to np.arrays
def lineToNP(data) :
    zed, age, mass, lum, sfr = [],[],[],[],[]
    gr, ri, iz, grr, gir, zY = [],[],[],[],[],[]
    kii, kri = [],[]
    for j in range(len(data)-1,-1,-1) :
        zed.append(data[j][0])
        age.append(data[j][1])
        mass.append(data[j][2])
        lum.append(data[j][3])
        sfr.append(data[j][4])
        #ug.append(data[j][5])
        gr.append(data[j][5])
        ri.append(data[j][6])
        iz.append(data[j][7])
        zY.append(data[j][8])
        grr.append(data[j][9])
        gir.append(data[j][10])
        kii.append(data[j][11])
        kri.append(data[j][12])
    zed = np.array(zed)
    age = np.array(age)
    mass = np.array(mass)
    lum = np.array(lum)
    sfr = np.array(sfr)
    #ug = np.array(ug)
    gr = np.array(gr)
    ri = np.array(ri)
    iz = np.array(iz)
    zY = np.array(zY)
    grr = np.array(grr)
    gir = np.array(gir)
    kii = np.array(kii) ;# k-correction taking obs i to restframe i: kii= i_o - i_obs
    kri = np.array(kri) ;# k-correction taking obs i to restframe r: kri= r_o - i_obs
    return zed, age, mass, lum, sfr, gr, ri, iz, zY, grr, gir, kii, kri
        

def mass_age_simha(age,start,trunc,tau,theta):
    dt = age-start
    if dt<=(trunc-start):
        mass_age = (-2.*tau**2+np.exp(dt/tau)*(dt**2-2.*dt*tau+2.*tau**2))/(np.exp(dt/tau)*(dt-tau)+tau)   

    else:
        dtt=trunc=start
        num = -2.*tau**3+np.exp(dtt/tau)*(tau*dtt**2-2.*dtt*tau**2+2*tau**3+dtt*dt**2/2.-dtt**3/2.)+theta*(dt**3/3.+dtt**3/6.-dtt*dt**2/2.)
        den = tau*(np.exp(dtt/tau)*(dtt-tau)+tau)+dtt*np.exp(dtt/tau)*(dt-dtt)+theta*(dt**2/2.+dtt**2/2.-dtt*dt)
        mass_age = num/den

    return mass_age


# end
