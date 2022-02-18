import numpy as np
import model
from convert_mag_to_lupmag import get_input_galpro, transform_to_1d

gp_root = '/data/des61.a/data/johnny/DESY3/galpro_files/DES_WF/'


def load_galpro_model(path='./'):
    """ Compute Stellar Masses for DES wide field galaxies
    """
    y_target = np.zeros(1000, dtype=float)[:, np.newaxis]
    gpmodel = model.Model('model_1d_des', y_target, y_target, y_target, root=path)
    return gpmodel

def compute_stellar_mass(mags, zcls, path='./'):
    """ Compute Stellar Masses for DES wide field galaxies
    input:
    mags: np.array(nsize, 4): for bands g, r, i, z
    zcls: np.arrya(nsize): redshift of the cluster
    """
    print('mag shape: ', mags.shape)
    x_target = get_input_galpro(mags, zcls)
    y_target = np.zeros(x_target[:, 0].size, dtype=float)[:, np.newaxis]

    gpmodel = model.Model('model_1d_des', x_target, y_target, x_target, root=path)
    point_estimates = gpmodel.point_estimate(save_estimates=False, make_plots=False)
    return point_estimates

def train_galpro(model_name, path ='./'):
    x_train = np.load(gp_root+'data/x_train.npy')
    y_train = np.load(gp_root+'data/y_train.npy')
    x_test = np.load(gp_root+'data/x_test.npy')
    y_test = np.load(gp_root+'data/y_test.npy')    

    x_test1d, y_test1d = transform_to_1d(x_test, y_test)
    x_train1d, y_train1d = transform_to_1d(x_train, y_train)

    gpmodel = model.Model(model_name, x_train1d, y_train1d, x_test1d, y_test1d,\
                          root=path, save_model=True)
    print('Model Trained Succsefully !')
    print('Storage at %s' % (path+'/model/'))