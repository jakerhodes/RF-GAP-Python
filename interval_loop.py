from rfgap import RFGAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import dataprep, load_regression
from phate import PHATE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import random
import json
import multiprocessing
from joblib import Parallel, delayed
import os


sns.set_theme()

def is_in_interval(y, lwr, upr):
    return np.logical_and(y >= lwr, y <= upr)

def get_coverage(y, y_lwr, y_upr):
    return np.mean((y >= y_lwr) & (y <= y_upr))

def get_width_stats(y_lwr, y_upr):
    widths = y_upr - y_lwr
    mean = np.mean(widths)
    sd = np.std(widths)
    min = np.min(widths)
    q1 = np.quantile(widths, 0.25)
    q2 = np.quantile(widths, 0.50)
    q3 = np.quantile(widths, 0.75)
    max = np.max(widths)
    return mean, sd, min, q1, q2, q3, max


def interval_loop(data_name = None, n_neighbors = 'auto', level = 0.95, random_state = 42, test_size = 0.2):

    filename = (data_name + '_' + str(n_neighbors) + '_' + 
        str(level) + '_' + str(random_state) + '_' + 
        str(test_size) + '.json')

    if os.path.isfile('intervals/' + filename):
        print("File already exists: ", print(data_name, n_neighbors, level, random_state, test_size))
        return

    results = {}

    results['data_name'] = data_name
    results['n_neighbors'] = n_neighbors
    results['level'] = level
    results['random_state'] = random_state
    results['test_size'] = test_size

    try:
        x, y = load_regression('../datasets/regression/' + data_name)
        n, d   = x.shape
        results['n'] = n
        results['d'] = d
        print(data_name, n_neighbors, level, random_state, test_size)

    except:
        np.savetxt(data_name + '_error.txt', 'Error loading dataset')
        return

    if isinstance(n_neighbors, int) and n_neighbors > n:
        return

    x_train, x_test, y_train, y_test, inds_train, inds_test = train_test_split(x, y, np.arange(n),
                                                                               test_size = test_size, 
                                                                               random_state = random_state)


    rf = RFGAP(oob_score = True, random_state = random_state, y = y_test)
    rf.fit(x_train, y_train)

    results['oob_score'] = rf.oob_score_
    results['test_score'] = rf.score(x_test, y_test)

    y_pred_test, y_pred_lwr_test, y_pred_upr_test = rf.predict_interval(X_test = x_test,
                                                                    n_neighbors = n_neighbors,
                                                                    level = level)

    results['coverage'] = get_coverage(y_test, y_pred_lwr_test, y_pred_upr_test)

    width_stats = get_width_stats(y_pred_lwr_test, y_pred_upr_test)
    results['width_mean'] = width_stats[0]
    results['width_sd'] = width_stats[1]
    results['width_min'] = width_stats[2]
    results['width_q1'] = width_stats[3]
    results['width_q2'] = width_stats[4]
    results['width_q3'] = width_stats[5]
    results['width_max'] = width_stats[6]

    with open('intervals/' + filename, 'w') as f:
        json.dump(results, f)


    
if __name__ == '__main__':

    random.seed(42)
    random_states = [random.randint(0, 10000) for _ in range(10)] 

    n_jobs = 25

    # data_names = ['AirfoilSelfNoise', 'AirQuality', 'Automobile', 'AutoMPG',
    #             'BeijingPM25', 'CommunityCrime', 'ComputerHardware',
    #             'ConcreateCompressiveStrength', 'ConcreteSlumpTest', 
    #             'CyclePowerPlant', 'EnergyEfficiency', 'FacebookMetrics',
    #             'FiveCitiesPM25', 'Hydrodynamics', 'IstanbulStock', 
    #             'Naval Propulsion Plants', 'OpticalNetwork', 
    #             'Parkinsons', 'Protein', 'SML2010']
    
    data_names = ['BeijingPM25']


    # data_names = data_names[::-1]

    levels = [0.80, 0.90, 0.95, 0.99]
    levels = levels[::-1]

    test_sizes = [0.2]
    n_neighborss = ['auto', 'all', 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    arguments = [
                (data_name, n_neighbors, level, random_state, test_size)
                for data_name in data_names
                for level in levels
                for test_size in test_sizes
                for n_neighbors in n_neighborss
                for random_state in random_states
                ]


    # Run the interval_loop function in parallel
    Parallel(n_jobs = n_jobs)(delayed(interval_loop)(*args) for args in arguments)

    
