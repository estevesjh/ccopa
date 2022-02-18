import numpy as np

def convert_mag_to_fluxes(mag,f_o=10**12):
    return f_o*10.**(-mag/2.5)

def convert_mag_to_lupmag(mags, a=2.5*np.log10(np.exp(1)), f_o=10**12):
    # define quantities
    m_o = 2.5*np.log10(f_o)
    #mags = np.c_[g,r,i,z]
    fluxes = convert_mag_to_fluxes(mags)

    # sigma = 10**np.std(np.log10(fluxes),axis=0)
    sigma = np.array([28.79631884352346, 39.89162179800252, 67.37310780328083, 118.70759986796102]) # g,r,i,z
    b = np.sqrt(a)*sigma

    # define g,r,i,z fluxes as a numpy array
    # luptitudes and errors
    lups = (m_o - 2.5*np.log10(b)) - a*np.arcsinh(fluxes/(2*b))
    lup_errors = np.sqrt(((a**2) * (sigma**2)) / (4*(b**2) + (fluxes**2)))
    
    return lups, lup_errors

def get_input_galpro(mags, redshift, a=2.5*np.log10(np.exp(1)), f_o=10**12):
    lups, lup_errors = convert_mag_to_lupmag(mags, a=a, f_o=f_o)
    g_r = lups[:, 0] - lups[:, 1]
    r_i = lups[:, 1] - lups[:, 2]
    i_z = lups[:, 2] - lups[:, 3]

    # colour errors
    g_r_err = ((lup_errors[:, 0]**2) + (lup_errors[:, 1]**2))**0.5
    r_i_err = ((lup_errors[:, 1]**2) + (lup_errors[:, 2]**2))**0.5
    i_z_err = ((lup_errors[:, 2]**2) + (lup_errors[:, 3]**2))**0.5

    # combining everything
    x_target = np.c_[lups, g_r, r_i, i_z, lup_errors, g_r_err, r_i_err, i_z_err, redshift]
    return x_target

def transform_to_1d(x, y):
    ynew = y[:, 1]
    xnew = np.vstack([x.T, y[:, 0]]).T
    return xnew, ynew[:, np.newaxis]