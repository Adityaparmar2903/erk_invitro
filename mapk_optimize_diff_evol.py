from mapk_model import model
from run_mapk_mcmc import *
import numpy as np
from matplotlib import pyplot as pyplot
import scipy.optimize
import pickle

model_list = [build_markevich_2step, build_erk_autophos_any,
          build_erk_autophos_uT, build_erk_autophos_phos,
          build_erk_activate_mkp]

def neg_likelihood(x, *args, **kwargs):
    nlh = -1.0*likelihood(x, *args, **kwargs)
    return nlh

data = read_data()

for i, mdl in enumerate(model_list):
    model = mdl()
    p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
    bounds = []
    bds = (-8,3)
    for j in range(len(p_to_fit)):
        bounds.append(bds)
    result = scipy.optimize.differential_evolution(neg_likelihood,
                                                   bounds, args=(model,data))
    fname = '%d.pkl' % (i)
    with open(fname, 'wb') as fh:
        pickle.dump(result, fh)
