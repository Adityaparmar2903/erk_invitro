import os
#from mapk_model import model
#from run_mapk_mcmc import *
import numpy as np
import pickle
from scipy.stats.distributions import norm
from matplotlib import pyplot as plt
#import plotly.plotly as py
#import plotly.graph_objs as go

method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP', 'Newton-CG', 'trust-ncg', 'dogleg']
model_list = [build_markevich_2step, build_erk_autophos_any,
          build_erk_autophos_uT, build_erk_autophos_phos,
          build_erk_activate_mkp]
#model_list = ['robertson']
#model_list_name = ['robertson']

num_ini = 300
num_top = 10
obj_func = np.ones((len(method_list),len(model_list),num_ini))
obj_func[:] = np.nan
func_eval = np.ones((len(method_list),len(model_list),num_ini))
func_eval[:] = np.nan
obj_func_top = np.ones((len(method_list),len(model_list),num_top))
obj_func_top[:] = np.nan
ave_obj_func = np.ones((len(method_list),len(model_list)))
ave_func_eval = np.ones((len(method_list),len(model_list)))
arr_ind = np.empty((len(method_list),len(model_list),num_ini))
run_ind = np.arange(num_ini)
top_ind = np.arange(num_top)
colors = ['red', 'darkgreen', 'blue', 'cyan', 'magenta',
         'yellow', 'orange', 'sienna', 'black', 'lawngreen', 'slategrey']
mark = ["o","^","D","s","H","x","+","."]
plt.ion()

def read_pkl(list1, list2, num, data1, data2, data3, data4, data5):
    for i in range(len(list1)):
    	for j in range(len(list2)):
    		for k in range(num):
    			data5[i][j][k] = k
    			fname = "output/%s_%s_%d.pkl" % (i,j,k)
    			if not os.path.exists(fname):
    				print fname + ' does not exist.'
    				continue
    			result = pickle.load(open(fname, "rb"))
    			if np.isnan(result.fun) or np.isinf(result.fun):
        				continue
    			data1[i][j][k] = result.fun
    			data2[i][j][k] = result.nfev
    		data3[i][j] = np.nanmean(obj_func[i][j])
    		data4[i][j] = np.nanmean(func_eval[i][j])

def sort_run_ind(list1, list2, data):
    arr_ind_sort = np.empty((len(list1), len(list2), num_ini))
    ind_best = np.ones((len(list1),len(list2)))
    ind_best[:] = np.nan
    for i in range(len(list1)):
        for j in range(len(list2)):
            inds = data[i][j].argsort()
            arr_ind_sort[i][j] = arr_ind[i][j][inds]
    for i in range(len(list1)):
        for j in range(len(list2)):
            ind_best[i][j] = arr_ind_sort[i][j][0]
    ind_best = np.int_(ind_best)
    return ind_best

def plot_fit_imp(list1, list2):
    for r in range(len(list1)):
        for s in range(len(list2)):
            best_ind = sort_run_ind(list1, list2, obj_func)
            t = best_ind[r][s]
            fname1 = "output_robertson/%s_%s_%d.pkl" % (r,s,t)
            if not os.path.exists(fname1):
                print fname1 + ' does not exist.'
                continue
            result1 = pickle.load(open(fname1, "rb"))
            if np.isnan(result1.fun) or np.isinf(result1.fun):
                    continue
            best_p = result1.x
            model = model_list[s]()
            pd = parameter_dict(model, best_p)
            plot_fit(model, data, pd)
            plt.suptitle("%s-%s" % (method_list[r], model_list[s].__name__),
                        fontsize=14, fontweight='bold')
            plt.savefig("Plots/Plot_fit/%s-%s.png"
                        % (method_list[r], model_list[s].__name__),
                        format='png')

def plot_obj_func(t, func, list1, list2, **keyword):
    for a in range(len(list1)):
        plt.figure()
        plt.suptitle(list1[a], fontsize=14, fontweight='bold')
        for b in range(len(list2)):
            if ('Method' in keyword):
                plt.plot(t, np.sort(func[a][b]), linestyle='solid', marker='.',
                            markeredgewidth=0.0, color=colors[b],
                            label=list2[b])
            else:
                plt.plot(t, np.sort(func[b][a]), linestyle='solid', marker='.',
                            markeredgewidth=0.0, color=colors[b],
                            label=list2[b])

        plt.legend(loc='upper left')
        plt.ylim((0,8000))
        plt.xlabel('Run index')
        plt.ylabel('Objective function value')
        plt.ylim(0,3)
        plt.grid(True)
        if ('Top10' in keyword):
            plt.savefig("Plots/Top10-%s.png" % (list1[a]),
            format='png')
        else:
            plt.savefig("Plots/%s.png" % (list1[a]),
            format='png')

def plot_performance(name, xaxis, yaxis, **keyword):
    plt.figure()
    plt.suptitle(name, fontsize=14, fontweight='bold')
    if ('model' in keyword):
        for a in range(len(model_list)):
            plt.plot(xaxis[:,a], yaxis[:,a], linestyle='solid',
                    linewidth=3.0, color=colors[a],
                    label=model_list_name[a])
    else:
        for a in range(len(method_list)):
            plt.plot(xaxis[a], yaxis[a], linestyle='solid',
                    linewidth=3.0, color=colors[a],
                    label=method_list[a])
    plt.legend(loc='upper left')
    plt.xlabel("Objective function value (Average)")
    plt.ylabel("Calls to the objective function (Average)")
    plt.grid(True)
    if('model' in keyword):
            plt.savefig("Plots/Performance/Model_Average_Performance.png",
                        format='png')
    else:
            plt.savefig("Plots/Performance/Method_Average_Performance.png",
                        format='png')

def plot_heatmap(data, **keyword):
    row_labels = method_list
    column_labels = [model_list_name[g] for g in range(len(model_list))]
    if ('calls' in keyword):
        data = np.transpose(ave_func_eval)
        fig, ax = plt.subplots()
        plt.suptitle('Objective Function Evaluations (Average)', fontsize=14,
                      fontweight='bold')
        heatmap = ax.pcolor(data, cmap=plt.cm.Reds)
    else:
        data = np.transpose(ave_obj_func)
        fig, ax = plt.subplots()
        plt.suptitle('Objective Function Value (Average)', fontsize=14,
                      fontweight='bold')
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.show()
    if ('calls' in keyword):
        plt.savefig("Plots/Heatmap/heatmap_func_eval2.png",format='png')
    else:
        plt.savefig("Plots/Heatmap/heatmap_obj_func2.png",format='png')

def plot_obj_func_sort(index):
    plt.figure()
    inds = obj_func[index].argsort()
    for j in range(len(method_list)):
        for t in range(model_list):
            obj_func[j][t] = obj_func[j][t][inds]
    for i, meth in enumerate(method_list):
        plt.plot(ind+1, obj_func[i], linestyle = 'solid',
                 marker='.', markeredgewidth=0.0, color = colors[i] , label = meth)
        plt.legend()
        plt.xlabel("Run Index")
        plt.ylabel("Objective Function Value (log)")
        plt.grid(True)

#data = read_data()
read_pkl(method_list, model_list, num_ini, obj_func, func_eval, ave_obj_func, ave_func_eval, arr_ind)

for i in range(len(method_list)):
    for j in range(len(model_list)):
        for k in range(num_top):
            obj_func_top[i][j][k] = (np.sort(obj_func[i][j]))[k]

#plot_fit_imp(method_list, model_list)
plot_obj_func(run_ind+1, obj_func, method_list, model_list_name, Method=True)
plot_obj_func(top_ind+1, obj_func_top, method_list, model_list_name, Method=True, Top10=True)
plot_obj_func(run_ind+1, obj_func, model_list_name, method_list)
plot_obj_func(top_ind+1, obj_func_top, model_list_name, method_list, Top10=True)
plot_performance('Model_Performance', ave_obj_func, ave_func_eval ,model=True)
plot_performance('Method_Performance', ave_obj_func, ave_func_eval)
plot_heatmap(ave_obj_func)
plot_heatmap(ave_func_eval, calls=True)
plot_obj_func_sort(1)
"""
#HEATMAP plotly
data_heat = [go.Heatmap(z=ave_obj_func,x=[(model_list[h].__name__) for h in range(len(model_list))],y=method_list)]
py.iplot(data_heat, filename='heatmap_obj_func')

data_heat = [go.Heatmap(z=ave_func_eval,x=[(model_list[h].__name__) for h in range(len(model_list))],y=method_list)]
py.iplot(data_heat, filename='heatmap_func_eval')
"""
