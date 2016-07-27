from copy import copy
from run_mapk_mcmc import *
import numpy as np
import scipy.optimize
from pyDOE import *
from scipy.stats.distributions import norm
import sys
import pickle
import numdifftools as nd

model_funs = [build_markevich_2step, build_erk_autophos_any,
	      build_erk_autophos_uT, build_erk_autophos_phos,
	      build_erk_activate_mkp]

method_names  = ['Nelder-Mead', 'Powell','CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP', 'Newton-CG', 'trust-ncg', 'dogleg', 'differential_evolution']

def generate_init(model, jj, ns, output_file, lognorm=False, best_reg=True):
	#select the parameters
	if best_reg:
		p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
		num_ind = len(p_to_fit)*ns
                ini_val = lhs(len(p_to_fit), samples = ns)
		fname_new = 'stat-%d.pkl' % (jj)
		stat = pickle.load(open(fname_new, "rb"))
		means = stat[0]
		stdvs = stat[1]
		if lognorm:
                	 for ind in range(len(p_to_fit)):
                                ini_val[:,ind] = norm(loc=means[ind], scale=stdvs[ind]).ppf(ini_val[:,ind])
                else:
                        # First shift unit hypercube to be centered around origin
                        # Then scale hypercube along each dimension by 2 stdevs
                        # Finally shift hypercube to be centered around nominal values
                        ini_val = means + 2 * stdvs * (ini_val - 0.5)
	else:
		p_to_fit = [p for p in model.parameters if p.name[0] == 'k']
		nominal_values = np.array([q.value for q in p_to_fit])
		log_nominal_values = np.log10(nominal_values)

		#latin hypercube sampling for picking a starting point
		num_ind = len(p_to_fit)*ns
		ini_val = lhs(len(p_to_fit), samples = ns)
		means = log_nominal_values
		stdvs = np.ones(len(p_to_fit))
		if lognorm:
			for ind in range(len(p_to_fit)):
				ini_val[:,ind] = norm(loc=means[ind], scale=stdvs[ind]).ppf(ini_val[:,ind])
		else:
			# First shift unit hypercube to be centered around origin
			# Then scale hypercube along each dimension by 2 stdevs
			# Finally shift hypercube to be centered around nominal values
			ini_val = means + 2 * stdvs * (ini_val - 0.5)

	np.save(output_file, ini_val)

def generate_all_inits(ns):
	for jj, mf in enumerate(model_funs):
		model = mf()
		fname = mf.__name__ + '_init'
		generate_init(model, jj, ns, fname)

def neg_likelihood(x, *args, **kwargs):
	# Get actual negative likelihood
	nlh = -1.0*likelihood(x, *args, **kwargs)
	# Penalize violated constraints
	try:
		nlh += numpy.sum(x[numpy.where(x > 3)] - 3) * 1000
		nlh += numpy.sum(-8 - x[numpy.where(x < -8)]) * 1000
	except Exception as e:
		print "Couldn't apply constraints."
	return nlh

def jaco(x, *args, **kwargs):
	delta = 0.01
	fx = neg_likelihood(x, *args, **kwargs)
	fxd = []
	for i, xx in enumerate(x):
		y = copy(x)
		y[i] = xx + delta
		fxd.append(neg_likelihood(y, *args, **kwargs))
	#fxd = [neg_likelihood(xx + delta, *args, **kwargs) for xx in x]
	jacob = [(xx - fx) / delta for xx in fxd]
	#jacob = nd.Jacobian(neg_likelihood)(x, *args, **kwargs)
	print 'J: ', jacob
	return numpy.array(jacob)
	#return jacob[0]

def hessia(x, *args, **kwargs):
	hessi = nd.Hessian(neg_likelihood)(x, *args, **kwargs)
	print 'H: ', hessi
	return hessi

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print 'Not enough input arguments.'
		sys.exit()
	from_idx = int(sys.argv[1])
	to_idx = int(sys.argv[2])
	if to_idx < from_idx:
		print 'Invalid from-to pair.'
		sys.exit()
	model_id = int(sys.argv[3])
	if model_id >= len(model_funs):
		print 'Invalid model id.'
		sys.exit()
	method_id = int(sys.argv[4])
	if method_id >= len(method_names):
		print 'Invalid method id.'
		sys.exit()
	#read data from file
	data = read_data()

	model = model_funs[model_id]()
	method = method_names[method_id]

	ini_val = np.load(model_funs[model_id].__name__ + '_init.npy')

	for i in range(from_idx, to_idx):
		result = scipy.optimize.minimize(neg_likelihood, ini_val[i],
						jac=jaco,
						hess=hessia,
						args=(model, data),
					 	method=method)

		fname = '%s_%s_%d.pkl' % (method_id, model_id,i)
		with open(fname, 'wb') as fh:
			pickle.dump(result, fh)
