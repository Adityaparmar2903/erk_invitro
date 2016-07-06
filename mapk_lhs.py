import os
from mapk_model import model
from run_mapk_mcmc import *
import numpy as np
import pickle
from scipy.stats.distributions import norm
from matplotlib import pyplot as plt
import csv

model_list = [build_markevich_2step, build_erk_autophos_any,
          build_erk_autophos_uT, build_erk_autophos_phos,
          build_erk_activate_mkp]

num_ini = 200
num_top = 48

obj_func = np.ones((len(model_list), num_ini))
obj_func[:] = np.nan
arr_ind = np.empty((len(model_list), num_ini))
arr_ind_sort = np.empty((len(model_list), num_ini))
mod_par = np.empty((len(model_list)))

label = []
for r in range(num_top):
    label.append('R-%d' %(r+1))


for j , mdel in enumerate(model_list):
    model = mdel()
    p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
    mod_par[j] = len(p_to_fit)

i = 1
for j in range(len(model_list)):
    for k in range(num_ini):
        arr_ind[j][k] = k
        fname = "output_powell_subprob/%s_%s_%d.pkl" % (i,j,k)
        if not os.path.exists(fname):
            print fname + ' does not exist.'
            continue
        result = pickle.load(open(fname, "rb"))
        if np.isnan(result.fun) or np.isinf(result.fun):
                continue
        obj_func[j][k] = result.fun

for j in range(len(model_list)):
    inds = obj_func[j].argsort()
    arr_ind_sort[j] = arr_ind[j][inds]

arr_ind_sort = np.int_(arr_ind_sort)

"""
for j in range(len(model_list)):
    best_par = np.empty((num_top, mod_par[j]))
    stat = np.empty((2, mod_par[j]))
    k = arr_ind_sort[j,:num_top]
    for l in range(num_top):
        m = k[l]
        fname = "output_powell_penalty/.pkl" % (i,j,m)
        if not os.path.exists(fname):
            print fname + ' does not exist.'
            continue
        result = pickle.load(open(fname, "rb"))
        if np.isnan(result.fun) or np.isinf(result.fun):
            continue
        best_par[l] = result.x
    stat[0] = np.mean(best_par, axis=0)
    stat[1] = np.std(best_par, axis=0)
    print stat
    fname_new = 'stat-.pkl' % (j)
    with open(fname_new, 'wb') as fh:
        pickle.dump(stat, fh)
"""
for j in range(len(model_list)):
    best_par = np.empty((num_top, mod_par[j]))
    k = arr_ind_sort[j,:num_top]
    for l in range(num_top):
        m = k[l]
        fname = "output_powell_subprob/%s_%s_%d.pkl" % (i,j,m)
        if not os.path.exists(fname):
            print fname + ' does not exist.'
            continue
        result = pickle.load(open(fname, "rb"))
        if np.isnan(result.fun) or np.isinf(result.fun):
                continue
        best_par[l] = result.x
        best_par1 = np.transpose(best_par)
    print np.transpose(best_par)
    fl = open('%s.csv' % (model_list[j].__name__), 'w')

    writer = csv.writer(fl)
    writer.writerow(label)
    for values in best_par1:
        writer.writerow(values)
    fl.close()
