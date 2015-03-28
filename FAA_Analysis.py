#!/usr/bin/env python 
"""
Functions to analyse and plot data produced by the: 
Fly Behavioral Expression of Ethanol Reward assay 
"""
import os
import operator
import math
import FAA_IO as FAA

from collections import defaultdict
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Example dataset that you might have produced with the fly-alcohol-assay

#All EtOH 45% Dur 7.5 minutes Data
#set1_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/Ethological Data/EtOH Dose Expts Part 7/EtOH 45 - 7.5 minutes set 1'
#set2_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/Ethological Data/EtOH Dose Expts Part 7/EtOH 45 - 7.5 minutes set 2'
#set3_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/Ethological Data/EtOH Dose Expts Part 7/EtOH 45 - 7.5 minutes set 3'
#set4_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/Ethological Data/EtOH Dose Expts Part 5/EtOH 45 - 7.5 minutes'
#set5_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/Ethological Data/EtOH Dose Expts Part 4/EtOH 45 - 7.5 minutes'
#
#EtOH45_data = [set1_path, set2_path, set3_path, set4_path, set5_path] <<-- this is the "grand dataset" to input into the processing and analysis functions

def transform_processed_data(processed_data, norm = None, diff = None):
    """
    Function that takes in data that was returned by process_data()
    and transforms it in various ways
    
    1) Can Normalize by taking the time to end chamber *for each lane* for the 
    first training trial and dividing both the training trial as well as 
    testing trial values
    
    2) Can take Differences between trials within a single experiment
    (i.e. t1-t2, t2-t3, t3-t4, etc...)
    positive values indicate the the subsequent trial was faster than the former
    negative values indicate the opposite.
    """
    
    def normalize_df(expt_set, cond_dict, numerator_label, denominator_label):
        
        df = expt_set[cond_dict[numerator_label]]  
        denominator = expt_set[cond_dict[denominator_label]]['t1']
        
        tunnel = df['tunnel']
        #df_new = df[['t1','t2','t3']]
        df_new =df.drop('tunnel', 1)
        df_new = df_new.divide(denominator, axis='rows')
        df_new.insert(0, 'tunnel', tunnel)
        df_new.condition_type = numerator_label
        return df_new
    
    trans_expt_data = []
    
    for expt_set in processed_data:       
        #Need to pre-allocate an empty list so that normalized data can be re-inserted in the correct condition order
        trans_set = [None]*len(expt_set)
        
        #following section performs normalization of processed data
        if norm is True:      
            cond_dict = {condition_expt.condition_type: indx for indx, condition_expt in enumerate(expt_set)}
        
            #Get test labels because they will always occur alphabetically before training labels
            test_labels = [condition_expt.condition_type for condition_expt in expt_set if "Test" in condition_expt.condition_type]
            
            for test_label in test_labels:
                train_label = 'Training_' + test_label.split("_")[1]
            
                norm_train = normalize_df(expt_set, cond_dict, train_label, train_label)
                norm_test = normalize_df(expt_set, cond_dict, test_label, train_label)
                
                trans_set[cond_dict[train_label]] = norm_train
                trans_set[cond_dict[test_label]] = norm_test
                                
            trans_expt_data.append(trans_set)
        
        #following section perform between trial difference subtractions of processed data
        if diff is True:
            trans_df_set = []
            for df in expt_set:                
                cond = df.condition_type                
                tunnel = df['tunnel']
                df = df.drop('tunnel', 1)                                
                #loops through number of columns in our dataframe (we dropped 1 column already) and takes difference scores
                #stores difference as {d1: t1-t2, d2: t2-t3, etc...}
                df_dict = {'d{}'.format(indx): df['t{}'.format(indx)]-df['t{}'.format(indx+1)] for indx in range(1, len(df.columns))}
                        
                trans_df = pd.DataFrame(df_dict)
                trans_df.insert(0, 'tunnel', tunnel)                
                trans_df.condition_type = cond                
                trans_df_set.append(trans_df)           
            trans_expt_data.append(trans_df_set)
    return trans_expt_data

def process_data(grand_dataset, expected_numTrials = None, norm = False, diff = False, dv = 'total_time'):
    """
    Function to process a list of paths to experiment set directories
    This particular function by default parses out time to end-chamber 
    information ("total_time") for the different conditions    

    Also accepts an expected number of trials (expected_numTrials) in case 
    some trials are not in the data (i.e. weren't collected, absent due to errors, etc.)
    
    Finally, user can specify which dependent variable they want to process.
    The choices are: "total_time", "total_distance", "mean_velocity", "mean_angular_velocity"
    
    The output of this function can be further processed with the:
    process_grand_data_summary() function
    """
    if dv is 'total_time':
        c_to_drop = [2,3,4]
        c_to_drop2 = [0,2,3,4]
    elif dv is 'total_distance':
        c_to_drop = [1,3,4]
        c_to_drop2 = [0,1,3,4]
    elif dv is 'mean_velocity':
        c_to_drop = [1,2,4]
        c_to_drop2 = [0,1,2,4]
    elif dv is 'mean_angular_velocity':
        c_to_drop = [1,2,3]
        c_to_drop2 = [0,1,2,3]
        
    expt_data = []
    
    for expt in grand_dataset:        
        expt_set = FAA.load_set(expt)        
        # Get the conditions from the loaded expt class
        # Will typically be something like:
        # ['Training_EtOH45', 'Training_Air', 'Test_EtOH45', 'Test_Air']
        conditions = [key for key in expt_set.__dict__.keys() if ("Train" in key) or ("Test" in key)]
        conditions.sort()
        set_data = []   
            
        for cond in conditions:            
           # Get the num trials from the loaded expt class
           # Will typically be something like:
           # ['Trial_1', 'Trial_2', 'Trial_3']           
           trials = [key for key in getattr(expt_set, cond).__dict__.keys() if "Trial" in key]
           trials.sort()          
           cond_dfs = []
           
           for indx, trial in enumerate(trials): 
               #Dataframe will contain in the following order:
               #tunnel (0), total_time (1), total_distance (2), mean_velocity (3), and mean_angular_velocity (4)               
               get_dataframe = operator.attrgetter('.'.join((cond, trial, "Summarized_Walk_Table")))
               df = get_dataframe(expt_set)           
               if indx == 0:
                   # Drop all columns except tunnel and our dependent variable of interest
                   # This is okay to hardcode!
                   df = df.drop(df.columns[c_to_drop], axis=1)
               else:
                   #drop all columns except  our dependent variable of interest
                   df = df.drop(df.columns[c_to_drop2], axis=1)                   
               #rename total_time so that it incorporates the trial info the data came from
               df = df.rename(columns = {dv:'t{}'.format(indx+1)})               
               cond_dfs.append(df)
           # If we are expecting a certain number of trials, this will fill in 
           # nonexisting trials with NaN placeholders
           if expected_numTrials:
               while len(cond_dfs) < expected_numTrials:
                   template_df = pd.DataFrame(columns=["t{}".format(len(cond_dfs)+1)], index=range(0,6))
                   cond_dfs.append(template_df)
           cond_times = pd.concat(cond_dfs, axis=1)
           #Add a "metatag" to each dataframe that describes the experiment condition
           cond_times.condition_type = cond
           cond_times.faa_instance = expt_set
           set_data.append(cond_times)          
        expt_data.append(set_data)
        
    #Do post processing transformations on data?
    if norm is True or diff is True:
        expt_data = transform_processed_data(expt_data, norm=norm, diff=diff)
    return expt_data

def process_grand_data_summary(grand_dataset, norm=False, dv = 'total_time'):   
    """
    Takes the processed data (from process_data() function) and concatenates
    replicates of the same experiment condition data together. This allows for 
    summary style calculations, plotting, and statistics for
    a specific experiment condition.
    """        
    expt_time_data = process_data(grand_dataset, 3, norm, dv = dv)    
    # To get grand data summary we want to concatenate trials for a given condition across different replicates    
    # First determine number of conditions    
    num_conditions = len(expt_time_data[0])    
    # Get the condition labels
    cond_labels = [df.condition_type for df in expt_time_data[0]]    
    cond_summ_dict = {}
        
    # For each condition type:
    for cond_ind in range(num_conditions):        
        cond_df_list = []        
        # For each replicate
        for rep_ind, replicate in enumerate(expt_time_data):            
            # Add the dataframe for the given condition to a list
            cond_df_list.append(expt_time_data[rep_ind][cond_ind])            
        # Concatenate all the dataframes in the condition list and add to
        # a condition summary dictionary
        cond_summ_dict[cond_labels[cond_ind]] = pd.concat(cond_df_list)               
    return cond_summ_dict


def set_boxplot_colors(data_to_plot, boxplot_object, line_list=None, line_colors=None, mean_colors = None, mean_edge_colors = None):
    """
    Function to set boxplot colors
    
    Passing plt line objects via line_list will allow the boxplot colors to 
    match the color scheme of previously plotted data
    
    Passing line_colors (hex colors in a list i.e. ["#800000", "#006700", "#0000ff"]) 
    will allow setting main boxplot colors 
    
    Passing mean_colors allows specification of the color for the mean point
    Passing mean_edge_colors allows specification of mean point edge color
    
    (NOTE: For all color options make sure there are are the same number of colors as there are datapoints to be plotted!)
    
    """    
    for indx, data in enumerate(data_to_plot, start=0):
        if line_list:                
            lcolors = line_list[indx].get_color()
        elif line_colors:
            lcolors = line_colors[indx]
        else:
            print "Warning!! No line_colors or line_list was specified for the boxplot! It will come out looking weird!!"
            break      
        
        if mean_colors:
            mcolors = mean_colors
        else:
            mcolors = ["#800000", "#006700", "#0000ff"]
            
        if mean_edge_colors:
            medges = mean_edge_colors
        else:
            medges = ["white", "white", "white"]    
            
        boxplot_object['boxes'][indx].set(color=lcolors)
        plt.setp(boxplot_object['caps'][2*indx:2*indx+2], color='none')
        plt.setp(boxplot_object['whiskers'][2*indx:2*indx+2], color=lcolors, linestyle='solid')
        boxplot_object['means'][indx].set(color=lcolors, markerfacecolor=mcolors[indx], markeredgecolor = medges[indx], marker="o", mew=1.1)             
        #outlier 'flier' colors are implemented with markeredgecolor not "color"
        boxplot_object['fliers'][indx].set(markeredgecolor=lcolors, markersize=8.0) 
                
        
def plot_data_summary(grand_dataset, norm = False, dv = 'total_time'):
    """
    Function that plots a data summary for each experimental replicate
    (1 replicate = 1 training and 1 testing session for control and treatment flies)
    
    Plots each individual lane (or fly) and their times 
    alongside a box and whisker summary of all flies for a trial
    """
    expt_time_data = process_data(grand_dataset, 3, norm, dv = dv)
    
    for indx, expt in enumerate(expt_time_data):
        
        fig = plt.figure(figsize=(11, 8.5))  
        fig.suptitle("Summarized Data for Expt Replicate {}".format(indx+1), fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88)             
        #hspace = 0.35
        outer_grid = gridspec.GridSpec(2, 2, hspace = 0.45, wspace = 0.2)        
        
        for indx2, cond_df in enumerate([expt[i] for i in [2,0,3,1]]):
            cond_label = cond_df.condition_type
            
            expt_set_cond = getattr(cond_df.faa_instance, cond_label)
            expt_set_cond_path = getattr(expt_set_cond, "expt_path")
            
            # expt_set_cond_path will take on a value something like:
            # u'2014-10-28 Test PM1 Air Dur 7.5 Data'
            expt_set_cond_date = os.path.basename(expt_set_cond_path).split(" ")[0]
            expt_set_cond_dur = os.path.basename(expt_set_cond_path).split(" ")[-2]
            
            cond_title_str = str(' '.join(cond_label.split("_")) 
                            + " - {} min - {}".format(expt_set_cond_dur, expt_set_cond_date) 
                            + '\n')
            
            titleax = plt.subplot(outer_grid[indx2])
            titleax.set_title(cond_title_str, fontweight='bold', linespacing = 0.5)
            titleax.axis('off')
            
            inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, 
            subplot_spec=outer_grid[indx2], wspace = 0.1, hspace = 0.2) 
                        
            ax = plt.Subplot(fig, inner_grid[0])         

            #setting the color cycle using colorbrewer2.org
            ax.set_color_cycle(['#e41a1c', '#4daf4a', '#377eb8'])            
            
            series = [series for series in list(cond_df.columns.values) if "tunnel" not in series]                                          
            ax.hold(True)      
            
            line_list = []
            #Plot for each trial the datapoints for the 6 tunnels
            for indx3, trial in enumerate(series, start=1):                                
                # plt.plot(x, y, marker, label, markerwidth, markersize, z-order)
                line, = ax.plot(range(1,7), cond_df["{}".format(trial)].values, '_', label="Trial {}".format(indx), mew=3, ms=15, zorder=0)
                line_list.append(line)
                
            ax.set_xlim(0.25, 6.75)
            if norm is True:
                ax.set_ylim(-1, 20)
            else:
                ax.set_ylim(-25, 325)
            ax.set_xlabel("Tunnel (Fly)", fontweight='bold')           
            
            #turn off x and y ticks on top and right sides
            ax.tick_params(top="off",right="off", bottom="off")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)             
            
            if indx2 == 0 or indx2 == 2:
                ax.set_ylabel("Time to End Chamber (sec)", fontweight='bold')          
            #lines = line1.get_lines()        
            #ax.legend(lines, [l.get_label() for l in lines], scatterpoints=1, numpoints=1, bbox_to_anchor=(1.05, 1.025), loc=2)                      
            #ax.grid(axis='x')            
            ax.hold(False)            
            fig.add_subplot(ax)

            #Reorganize the Pandas dataframe into a format that can be plotted easily with matplotlib functions
            data_to_plot = []                                
            for column in cond_df.columns.values:
                if not np.isnan(cond_df[column].tolist()).all():
                    if "tunnel" not in column:
                        data = np.array(cond_df[column].tolist())[~np.isnan(cond_df[column].tolist())]
                        data_to_plot.append(data)
            
            ax2 = plt.Subplot(fig, inner_grid[1])
            ax2.hold(True)
            
            #frame.boxplot(ax = ax, showmeans=True, labels=labels, return_type = 'axes')
            box = ax2.boxplot(data_to_plot, showmeans= True, patch_artist=True, medianprops=dict(color='white', linewidth=1.2))
            set_boxplot_colors(data_to_plot, box, line_list=line_list)
            
            ax2.tick_params(top="off",right="off", bottom="off")     
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)               
            ax2.set_yticklabels([])            
                     
            if norm is True:
                ax2.set_ylim(-1, 20)
            else:
                ax2.set_ylim(-25, 325)
            ax2.set_xlabel("Trial", fontweight='bold')
            ax.hold(False)
            fig.add_subplot(ax2)
                    
        #fig.savefig(u'C:/Users/Nicholas/Desktop/Summarized Data for Expt Replicate {}.pdf'.format(indx+1), format='pdf')        
 

def plot_single_expt(cond_df, expt_set, norm = False):
    """
    placeholder for some refactoring that needs to be done for the plotting functions
    """
    pass

               
def plot_grand_data_summary(grand_dataset, norm=False, dv = 'total_time'):
    """
    Takes the processed grand summary data (from process_grand_data_summary() function) 
    and plots it in a boxplot style
    """ 
    summ_data_dict = process_grand_data_summary(grand_dataset, norm, dv = dv)    
    fig = plt.figure(figsize=(11, 8.5))      
    fig.suptitle("Summarized Data", fontsize=12, fontweight='bold')
    fig.subplots_adjust(top=0.90)            
    outer_grid = gridspec.GridSpec(2, 2, hspace = 0.25, wspace = 0.15)
    
    sorted_keys = summ_data_dict.keys()
    sorted_keys.sort()  
    
    for indx, cond in enumerate([sorted_keys[i] for i in [2,0,3,1]]):                
        cond_df = summ_data_dict[cond]                
        #drop the tunnel column as it's not needed currently
        cond_df = cond_df.drop("tunnel", 1)

        #Reorganize the Pandas dataframe into a format that can be plotted easily with matplotlib functions        
        data_to_plot = []                                
        for column in cond_df.columns.values:
            if not np.isnan(cond_df[column].tolist()).all():
                if "tunnel" not in column:
                    data = np.array(cond_df[column].tolist())[~np.isnan(cond_df[column].tolist())]
                    data_to_plot.append(data)
                            
        ax = plt.Subplot(fig, outer_grid[indx])
        #ax.set_color_cycle(['#e41a1c', '#4daf4a', '#377eb8']) 
        ax.set_title(' '.join(cond.split('_')), fontweight='bold', fontsize=10)
        if norm is True:
            ax.set_ylim(-1, 20)
        else:
            ax.set_ylim(-25, 325)          
        ax.tick_params(top="off",right="off", bottom="off") 
        ax.tick_params(axis='x', pad=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)   
        
        if (indx == 0 or indx == 2):
            ax.set_ylabel("Time to End Chamber (sec)", fontweight='bold')
                        
        if indx == 2 or indx == 3:
            ax.set_xlabel("Trial", fontweight='bold')
                                                                                       
        box = ax.boxplot(data_to_plot, showmeans=True, patch_artist=True, medianprops=dict(color='white', linewidth=1.2), widths=0.20)        
        set_boxplot_colors(data_to_plot, box, line_colors=['#e41a1c', '#4daf4a', '#377eb8'])

        fig.add_subplot(ax)
    plt.tight_layout            
                    

def plot_trial_by_trial_grand_summary(grand_dataset, norm=False, dv = 'total_time'):
    """
    Plots trial by trial grand summary of the data
    """
    summ_data_dict = process_grand_data_summary(grand_dataset, norm, dv=dv)  
    
    fig = plt.figure(figsize=(11, 8.5))      
    fig.suptitle("Summarized Data", fontsize=12, fontweight='bold')
    fig.subplots_adjust(top=0.90)            
    outer_grid = gridspec.GridSpec(2, 2, hspace = 0.25, wspace = 0.15)
    
    sorted_keys = summ_data_dict.keys()
    sorted_keys.sort()  
    
    for indx, cond in enumerate([sorted_keys[i] for i in [2,0,3,1]]):                
        cond_df = summ_data_dict[cond]
        #Create a tunnel column dropped version of dataframe
        dropped_df = cond_df.drop("tunnel", 1)
        
        ax = plt.Subplot(fig, outer_grid[indx])
        #ax.set_color_cycle(['#e41a1c', '#4daf4a', '#377eb8']) 
        ax.set_title(' '.join(cond.split('_')), fontweight='bold', fontsize=10)
        if norm is True:
            ax.set_ylim(-1, 20)
        else:
            ax.set_ylim(-25, 325)
        ax.set_xlim(0.5, len(dropped_df.columns)+0.5)
        ax.set_xticks(range(1,len(dropped_df.columns)+1))
        ax.tick_params(top="off",right="off", bottom="off") 
        ax.tick_params(axis='x', pad=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)   
        
        if (indx == 0 or indx == 2):
            ax.set_ylabel("Time to End Chamber (sec)", fontweight='bold')
                        
        if indx == 2 or indx == 3:
            ax.set_xlabel("Trial", fontweight='bold')
            
        #Errorbar plot command!
        (plotlines, caplines, barlines) = ax.errorbar(x=list(range(1,len(dropped_df.columns)+1)), y=dropped_df.mean(), yerr=dropped_df.sem(), capsize=0, lw=2.5, elinewidth=1.5, color="black")
        
        #We want to modify each of our errorbar lines so that they are blue
        for barline in barlines:
            barline.set_color('b')
        
        #plot individual datapoints
        for indx, trial in enumerate(dropped_df.columns, start=1):
            y = dropped_df[trial]
            #adds some noise to x-axis to prevent overlap
            x = np.random.normal(indx, 0.08, size=len(y))
            ax.plot(x, y, 'bo', ms=4, alpha=0.3)        
        
        fig.add_subplot(ax)        
    
    #fig.savefig(u'C:/Users/Nicholas/Desktop/Summarized Data for Expt Replicate.pdf', format='pdf')
                   
def plot_speed_comparison(grand_dataset, norm=False, dv ='total_time'):
    
    expt_time_data = process_data(grand_dataset, 3, norm, dv=dv)     
    all_expt = []
    
    for expt in expt_time_data:        
        expt_dict = {}
    
        for cond_df in expt:  
            cond_label = cond_df.condition_type                      
            series = [series for series in list(cond_df.columns.values) if "tunnel" not in series]            
            fast_cond_times = []
            slow_cond_times = []
            
            # If the series is empty we still want to put placeholders in
            if not series:
                fast_cond_times.append(pd.DataFrame())
                slow_cond_times.append(pd.DataFrame())
            
            # Iterate through all trials except the last one
            for indx, trial in enumerate(series[:len(series)-1]):                  
                next_trial = series[indx+1] 
                
                # Check if trial is filled with "NaN" values 
                if math.isnan(cond_df[trial].min()):
                    fast_cond_times.append(pd.DataFrame(columns=[trial, next_trial], index=[0]))
                    slow_cond_times.append(pd.DataFrame(columns=[trial, next_trial], index=[0]))
                else:
                    # Find the fastest tunnel/fly for the trial
                    curr_fastest = (cond_df[cond_df[trial] == cond_df[trial].min()].iat[0,0])
                    curr_fastest_time = cond_df[trial].min()  
                    # Find the slowest tunnel/fly for the trial
                    curr_slowest = (cond_df[cond_df[trial] == cond_df[trial].max()].iat[0,0])
                    curr_slowest_time = cond_df[trial].max()
                        
                    # Find out what happened to the fastest fly in the next trial
                    # .loc indexing is 1 higher than actual tunnel values
                    next_fastest_time = cond_df[next_trial].loc[curr_fastest-1]                
                                   
                    # Find out what happened to the slowest fly in the next trial
                    next_slowest_time = cond_df[next_trial].loc[curr_slowest-1]
                    
                    temp_fast_df = pd.DataFrame({trial:[curr_fastest_time], next_trial:[next_fastest_time]})                
                    temp_slow_df = pd.DataFrame({trial:[curr_slowest_time], next_trial:[next_slowest_time]})                
                    
                    fast_cond_times.append(temp_fast_df)
                    slow_cond_times.append(temp_slow_df)
            #Example: will give {'fast':[(T1, T2), (T2, T3)], 'slow':[] 
            expt_dict[cond_label] = {'fast':fast_cond_times, 'slow':slow_cond_times}            
        all_expt.append(expt_dict)
        
    ## ----------- End of this set of For Loops ---------------- ##    
        
    fast_dict = defaultdict(list)
    slow_dict = defaultdict(list)        

    # Go through generated all_expt variable and do some unpacking
    for d in all_expt:
        for key, value in d.iteritems():
            for key2, value2 in value.iteritems( ):                
                if key2 == 'fast':
                    fast_dict[key].append(value2)
                if key2 == 'slow':
                    slow_dict[key].append(value2)
  
    # For loops to do plotting
    for indx, d in enumerate([fast_dict, slow_dict]):
        if indx == 0:
            titlestr = "Fastest Fly Per Trial and Subsequent Trial Time"
        elif indx == 1:
            titlestr = "Slowest Fly Per Trial and Subsequent Trial Time"
 
        fig = plt.figure(figsize=(11, 8.5))  
        fig.suptitle(titlestr, fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.9)             
        outer_grid = gridspec.GridSpec(2, 2, hspace = 0.2, wspace = 0.15)

        sorted_keys = d.keys()
        sorted_keys.sort()      
                       
        for indx, key in enumerate([sorted_keys[i] for i in [2,0,3,1]]):    
            value = d[key]                    
            unzipped = zip(*value)    
            catenated = [pd.concat(item) for item in unzipped]
            
            # Check to make sure catenated isn't empty
            if catenated:
                
                inner_grid = gridspec.GridSpecFromSubplotSpec(1, len(catenated), 
                             subplot_spec=outer_grid[indx], wspace = 0.3)           
                
                for indx2, frame in enumerate(catenated):
                    
                    ax = plt.Subplot(fig, inner_grid[indx2])
                    ax.set_title('\n'.join(key.split('_')), fontweight='bold', fontsize=10)
                    if norm is True:
                        ax.set_ylim(-1, 20)
                    else:
                        ax.set_ylim(-25, 325)
                    
                    if (indx == 0 or indx == 2) and indx2 == 0:
                        ax.set_ylabel("Time to End Chamber (sec)", fontweight='bold')
                        
                    if indx == 2 or indx == 3:
                        ax.set_xlabel("Trial", fontweight='bold')
     
                    labels = [string.lstrip('t') for string in frame.columns.values]                                                                                   
                    frame.boxplot(ax = ax, showmeans=True, labels=labels, return_type = 'axes')
                    
                    fig.add_subplot(ax)

                
def main():
    pass

if __name__ == '__main__':
    main()