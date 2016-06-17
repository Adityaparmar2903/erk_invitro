import os
from mapk_model import model
from run_mapk_mcmc import *
import numpy as np
import pickle
from scipy.stats.distributions import norm
from matplotlib import pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


#list the algorithms
method_list = ['Nelder-Mead', 'Powell', 'COBYLA', 'TNC']

#list the models
model_list = [build_markevich_2step, build_erk_autophos_any,
	      build_erk_autophos_uT, build_erk_autophos_phos,
	      build_erk_activate_mkp]


#no. of initial values sampled using Latin Hypercube Sampling
num_ini = 1000
from_id = 0
to_id = 1000
num_top = 10

#read the data
data = read_data()


obj_func = np.ones((len(method_list),len(model_list),num_ini))
obj_func[:] = np.nan
func_eval = np.ones((len(method_list),len(model_list),num_ini))
func_eval[:] = np.nan
ave_obj_func = np.ones((len(method_list),len(model_list)))
ave_func_eval = np.ones((len(method_list),len(model_list)))
obj_func_top = np.ones((len(method_list),len(model_list),num_top))
obj_func_top[:] = np.nan
func_eval_top = np.ones((len(method_list),len(model_list),num_top))
func_eval_top[:] = np.nan
obj_func_sort = np.ones((len(method_list),len(model_list),num_ini))
obj_func_sort[:] = np.nan
ind_best = np.ones((len(method_list),len(model_list)))
ind_best[:] = np.nan
arr_ind = np.empty((len(method_list),len(model_list),num_ini))
arr_ind_sort = np.empty((len(method_list),len(model_list),num_ini))

run_ind = np.arange(num_ini)
top_ind = np.arange(num_top)

colors = ["red","blue","green","yellow","cyan","darkviolet","sienna","darkgray","black"]
mark = ["o","^","D","s","H","x","+","."]
plt.ion()

#read the pkl files and store the result
for i in range(len(method_list)):
	for j in range(len(model_list)):
		#model = model_list[j]()
		for k in range(0,1000):
			arr_ind[i][j][k] = k
			fname = "output_final/%s_%s_%d.pkl" % (i,j,k)
			if not os.path.exists(fname):
				print fname + ' does not exist.'
				continue
			results = pickle.load(open(fname, "rb"))
			#for l in range(5):
			l = k%5
			result = results[l]
			if np.isnan(result.fun) or np.isinf(result.fun):
    				continue
			obj_func[i][j][k] = result.fun
			func_eval[i][j][k] = result.nfev
			#para_vec[i][j][k] = result.x
		ave_obj_func[i][j] = np.nanmean(obj_func[i][j])
		ave_func_eval[i][j] = np.nanmean(func_eval[i][j])

for ii in range(len(method_list)):
	for jj in range(len(model_list)):
		for kk in range(num_top):
			obj_func_top[ii][jj][kk] = (np.sort(obj_func[ii][jj]))[kk]

for ii in range(len(method_list)):
	for jj in range(len(model_list)):
		inds = obj_func[ii][jj].argsort()
		arr_ind_sort[ii][jj] = arr_ind[ii][jj][inds]
		

for ii in range(len(method_list)):
	for jj in range(len(model_list)):
		ind_best[ii][jj] = arr_ind_sort[ii][jj][0]

ind_best = np.int_(ind_best)

for rr in range(len(method_list)):
	for ss in range(len(model_list)):
		tt = ind_best[rr][ss]
		fname1 = "output_final/%s_%s_%d.pkl" % (rr,ss,tt)
		if not os.path.exists(fname1):
			print fname1 + ' does not exist.'
			continue
		results1 = pickle.load(open(fname1, "rb"))
		ll = tt%5
		result1 = results1[ll]
		if np.isnan(result.fun) or np.isinf(result.fun):
    			continue
		best_p = result1.x
		model = model_list[jj]()
		pd = parameter_dict(model, best_p)
		plot_fit(model,data,pd)
		plt.suptitle("%s-%s" % (method_list[rr], model_list[ss].__name__), fontsize=14, 					fontweight='bold')
		plt.savefig("Plots/Plot_fit/%s-%s.png" % (method_list[rr], model_list[ss].__name__),
				format='png', dpi=600)
		plt.savefig("Plots/Plot_fit/%s-%s.pdf" % (method_list[rr], model_list[ss].__name__),
				format='pdf', dpi=600)

		





			




			
		
"""OBJECTIVE FUNCTION"""
"""	
		
#for each algorithm plot the results for different models
for a in range(len(method_list)):
	plt.figure()
	plt.suptitle(method_list[a], fontsize=14, fontweight='bold')
	for b in range(len(model_list)):
		plt.plot(run_ind+1, np.sort(obj_func[a][b]), linestyle='',
			marker='.', markeredgewidth=0.0, color = colors[b],
			label = model_list[b].__name__)
	plt.legend(loc='upper left')
	plt.xlabel("Run index")
	plt.ylabel("Objective function value")
	#plt.ylim(-1e5,0)
	plt.grid(True)
	plt.savefig("Plots/%s.pdf" % (method_list[a]),format='pdf')
	plt.savefig("Plots/%s.png" % (method_list[a]),format='png')
		
#for each model plot the results for different algorithms
for aa in range(len(model_list)):
	plt.figure()
	plt.suptitle(model_list[aa].__name__, fontsize=14, fontweight='bold')
	for bb in range(len(method_list)):
		plt.plot(run_ind+1, np.sort(obj_func[bb][aa]), linestyle='', 
			marker='.', markeredgewidth=0.0, color = colors[bb],
			label = method_list[bb])
	plt.legend(loc='upper left')
	plt.xlabel("Run index")
	plt.ylabel("Objective function value")
	#plt.ylim(-1e5,0)
	plt.grid(True)
	plt.savefig("Plots/%s.pdf" % (model_list[aa].__name__),format='pdf')
	plt.savefig("Plots/%s.png" % (model_list[aa].__name__),format='png')

#for each algorithm plot best results for different models  top 10
for w in range(len(method_list)):
	plt.figure()
	plt.suptitle(method_list[w], fontsize=14, fontweight='bold')
	for v in range(len(model_list)):
		plt.plot(top_ind+1, obj_func_top[w][v], linestyle='solid',
			marker='o', markeredgewidth=0.0, color = colors[v],
			label = model_list[v].__name__)
	plt.legend(loc='upper left')
	plt.xlabel("Run index")
	plt.ylabel("Objective function value")
	#plt.ylim(-1e5,0)
	plt.grid(True)
	plt.savefig("Plots/Top10-%s.pdf" % (method_list[w]),format='pdf')
	plt.savefig("Plots/Top10-%s.png" % (method_list[w]),format='png')


#for each model plot best results for different algorithms top 10
for y in range(len(model_list)):
	plt.figure()
	plt.suptitle(model_list[y].__name__, fontsize=14, fontweight='bold')
	for z in range(len(method_list)):
		plt.plot(top_ind+1, obj_func_top[z][y], linestyle='solid', 
			marker='o', markeredgewidth=0.0, color = colors[z],
			label = method_list[z])
	plt.legend(loc='upper left')
	plt.xlabel("Run index")
	plt.ylabel("Objective function value")
	#plt.ylim(-1e5,0)
	plt.grid(True)
	plt.savefig("Plots/Top10-%s.pdf" % (model_list[y].__name__),format='pdf')
	plt.savefig("Plots/Top10-%s.png" % (model_list[y].__name__),format='png')

"""

"""HEATMAP"""

"""

#plot heatmap for objective function values
column_labels = method_list
row_labels = [model_list[g].__name__ for g in range(len(model_list))]
data = ave_obj_func
fig, ax = plt.subplots()
plt.suptitle('Objective Function Value (Average)', fontsize=14, fontweight='bold')
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)
plt.show() 
#plt.savefig("Plots/heatmap_obj_func2.png",format='png')
plt.savefig("Plots/heatmap_obj_func2.pdf",format='pdf')

#plot heatmap for no. of function evaluations
column_labels = method_list
row_labels = [model_list[h].__name__ for h in range(len(model_list))]
data = ave_func_eval
fig, ax = plt.subplots()
plt.suptitle('Objective Function Evaluations (Average)', fontsize=14, fontweight='bold')
heatmap = ax.pcolor(data, cmap=plt.cm.Reds)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)
plt.show()
#plt.savefig("Plots/heatmap_func_eval2.png",format='png')
plt.savefig("Plots/heatmap_func_eval2.pdf",format='pdf')

"""

"""MODEL PERFORMANCE"""

"""

plt.figure()
plt.suptitle('Model performance', fontsize=14, fontweight='bold')
#objective function vs function evaluations
for aaa in range(len(model_list)):
	#for bbb in range(len(method_list)):
	plt.plot(ave_obj_func[:,aaa], ave_func_eval[:,aaa], linestyle='solid', 
			linewidth= 3.0, color=colors[aaa],
			label=model_list[aaa].__name__)
	plt.legend(loc='upper right')
	plt.xlabel("Objective function value (Average)")
	plt.ylabel("Calls to the objective function (Average)")
	plt.grid(True)
plt.savefig("Plots/Model_Average_Performance.png",format='png')
plt.savefig("Plots/Model_Average_Performance.pdf",format='pdf')

plt.figure()
plt.suptitle('Method performance', fontsize=14, fontweight='bold')
#objective function vs function evaluations
for aaa in range(len(method_list)):
	#for bbb in range(len(method_list)):
	plt.plot(ave_obj_func[aaa], ave_func_eval[aaa], linestyle='solid', 
			linewidth= 3.0, color=colors[aaa],
			label=method_list[aaa])
	plt.legend(loc='upper right')
	plt.xlabel("Objective function value (Average)")
	plt.ylabel("Calls to the objective function (Average)")
	plt.grid(True)
plt.savefig("Plots/Method_Average_Performance.png",format='png')
plt.savefig("Plots/Method_Average_Performance.pdf",format='pdf')

data_heat = [go.Heatmap(z=ave_obj_func,x=[(model_list[h].__name__) for h in range(len(model_list))],y=method_list)]
py.iplot(data_heat, filename='heatmap_obj_func')

data_heat = [go.Heatmap(z=ave_func_eval,x=[(model_list[h].__name__) for h in range(len(model_list))],y=method_list)]
py.iplot(data_heat, filename='heatmap_func_eval')
		

"""
	
				

				
