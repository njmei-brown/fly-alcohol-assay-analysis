# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:19:28 2015

@author: Nicholas Mei

"""

import os

import faa_io as FAA

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import scipy.signal as sig
import numpy as np
import operator
import pandas as pd

#turn of matplotlib interactive plotting mode
plt.ioff()

debug_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/EtOH New Dose Expts Part 14/EtOH 85 - rev - 10 minutes set 3'

debug_group = [u'C:/Users/Nicholas/Desktop/FlyBar Analysis/EtOH New Dose Expts Part 14/EtOH 85 - rev - 10 minutes set 1', u'C:/Users/Nicholas/Desktop/FlyBar Analysis/EtOH New Dose Expts Part 14/EtOH 85 - rev - 10 minutes set 2']

def threshold_cluster_indices(cluster_indxs, min_cluster_size):
    cluster_sizes = cluster_indxs[1::2] - cluster_indxs[0::2]   
    #Need to repeat the thresholded cluster sizes once more to cover both odd and even indices
    bool_thresh = np.repeat([cluster_sizes > min_cluster_size], 2)
    thresholded_cluster_indxs = cluster_indxs[bool_thresh]
    return thresholded_cluster_indxs

def plot_cluster_span(axis, cluster_indxs, span_color):
    if len(cluster_indxs) >= 2: 
        for indx in np.arange(len(cluster_indxs)/2):
            #if cluster_indxs[1::2][indx]-cluster_indxs[0::2][indx] > min_cluster_size:
            axis.axvspan(cluster_indxs[1::2][indx], cluster_indxs[0::2][indx], edgecolor='None', alpha=0.5, facecolor=span_color)

def calculate_features(cluster_indxs, min_cluster_size):
    cluster_diffs = cluster_indxs[1::2]-cluster_indxs[0::2]
    feature_count = np.sum(cluster_diffs > min_cluster_size)
    feature_duration = np.sum(cluster_diffs[cluster_diffs > min_cluster_size])     
    return feature_count, feature_duration

def process_set_dataframe(set_path):

    expt_set = FAA.load_set(set_path)
                         
    trial_rows = ['Tunnel 1', 'Tunnel 2', 'Tunnel 3', 'Tunnel 4', 'Tunnel 5', 'Tunnel 6']
    
    condition_list = sorted([key for key in dir(expt_set) if ("Train" in key) or ("Test" in key)])
    condition_df_dict = {}
    
    for condition in condition_list:
    
        trial_list = sorted([key for key in dir(getattr(expt_set, condition)) if "Trial" in key])    
        trial_df_dict = {}    
        
        get_expt_path = operator.attrgetter('.'.join((condition, "expt_path")))
        expt_path = get_expt_path(expt_set)
                
        # expt_set_cond_path will take on a value something like:
        # u'2014-10-28 Test PM1 Air Dur 7.5 Data'
        expt_date = os.path.basename(expt_path).split(" ")[0]
        
        for trial_num, trial in enumerate(trial_list, start=1):     
    
            tunnel_df_list = []
    
            for tunnel in np.arange(1, 7): 
                
                print("Processing: {} {} tunnel {}".format(condition, trial, tunnel))        
                get_dataframe = operator.attrgetter('.'.join((condition, trial, "Raw_Walk_Table")))       
                raw_walk_table = get_dataframe(expt_set)
                
                #do median filtering to clean up noise in y position (the only one we care about for the turning around plotting)
                #filter_size = 5
                #median_filtered = sig.medfilt(raw_walk_table[raw_walk_table['tunnel'] == 4]['fly_y'], filter_size)
                
                unfiltered = raw_walk_table[raw_walk_table['tunnel'] == tunnel]['fly_y']   
    
                fig, ax = plt.subplots()
                ax.plot(unfiltered)
                ax.set_title("{} {} Tunnel: {}".format(condition, trial, tunnel))
                ax.set_ylabel('Y position in millimeters (mm)')
                ax.set_xlabel('Timepoint number')
                
                #next let's take position deltas of y-position in time
                diff = np.diff(unfiltered)
                
                #then obtain a boolean array to find regions where difference values were negative 
                #(i.e. regions where slope is negative and the fly is moving away from end-chamber will become TRUE)
                advance_bool_array = (diff > 0.08)
                retreat_bool_array = (diff < -0.08)
                pause_bool_array = (diff <= 0.08) - (diff <= -0.08)
                
                #we need to append a false to the end as well as the start of the bool array in case it ends/starts with "T,T,T" or something like that
                #this will catch the edge cases where a start index will have no end or an end index has no start
                advance_clust_bool_array = np.append(advance_bool_array, [False])
                advance_clust_bool_array = np.insert(advance_clust_bool_array, 0, [False])        
                
                retreat_clust_bool_array = np.append(retreat_bool_array, [False])
                retreat_clust_bool_array = np.insert(retreat_clust_bool_array, 0, [False])
                
                pause_clust_bool_array = np.append(pause_bool_array, [False])
                pause_clust_bool_array = np.insert(pause_clust_bool_array, 0, [False])
                
                #if we then take the deltas of the boolean array we can get start and end indexes for clusters of "TRUE"
                advance_cluster_indxs = np.diff(advance_clust_bool_array).nonzero()[0]
                retreat_cluster_indxs = np.diff(retreat_clust_bool_array).nonzero()[0]        
                pause_cluster_indxs = np.diff(pause_clust_bool_array).nonzero()[0]
                
                #let's only consider retreats or pauses that were longer than 3 timepoints long
                #To achieve this we subtract odd from even elemnts (i.e. end-start indexes)
                min_advance_cluster_size = 3
                min_retreat_cluster_size = 3
                min_pause_cluster_size = 6 
                    
                number_of_advances, dur_of_advances = calculate_features(advance_cluster_indxs, min_advance_cluster_size)
                number_of_retreats, dur_of_retreats = calculate_features(retreat_cluster_indxs, min_retreat_cluster_size)
                number_of_pauses, dur_of_pauses = calculate_features(pause_cluster_indxs, min_pause_cluster_size)
                
                total_expt_timepoints = len(unfiltered)        
                
                advance_percent = 100 * dur_of_advances/float(total_expt_timepoints) 
                retreat_percent = 100 * dur_of_retreats/float(total_expt_timepoints)
                pause_percent = 100 * dur_of_pauses/float(total_expt_timepoints)
                
                #print number_of_advances
                #print "Advance duration: {} (Percentage of total: {})".format(dur_of_advances, advance_percent)        
                #print number_of_retreats
                #print "Retreat duration: {} (Percentage of total: {})".format(dur_of_retreats, retreat_percent)        
                #print number_of_pauses
                #print "Pause duration: {} (Percentage of total: {})".format(dur_of_pauses, pause_percent)
                
                thresh_advance_cluster_indxs = threshold_cluster_indices(advance_cluster_indxs, min_advance_cluster_size)
                thresh_retreat_cluster_indxs = threshold_cluster_indices(retreat_cluster_indxs, min_retreat_cluster_size)
                thresh_pause_cluster_indxs = threshold_cluster_indices(pause_cluster_indxs, min_pause_cluster_size)
                                                
                plot_cluster_span(ax, thresh_advance_cluster_indxs, span_color='g')        
                plot_cluster_span(ax, thresh_retreat_cluster_indxs, span_color='b')        
                plot_cluster_span(ax, thresh_pause_cluster_indxs, span_color='r')   
                
                tunnel_dict = {'Advance duration': dur_of_advances, 
                               'Advance percentage': advance_percent,
                               'Advance cluster indices': thresh_advance_cluster_indxs,
                               'Retreat duration': dur_of_retreats, 
                               'Retreat percentage': retreat_percent, 
                               'Retreat cluster indices': thresh_retreat_cluster_indxs,
                               'Pause duration': dur_of_pauses, 
                               'Pause percentage': pause_percent, 
                               'Pause cluster indices': thresh_pause_cluster_indxs,
                               'Total experimental timepoints': total_expt_timepoints,
                               'Figure': fig}
                
                tunnel_df_list.append(tunnel_dict)
                #plt.waitforbuttonpress()
                
                plt.close(fig)
            
            trial_frame = pd.DataFrame(tunnel_df_list, index=trial_rows)
            trial_frame.trial = trial
            trial_frame.condition = condition
            trial_frame.date = expt_date
            
            trial_df_dict['Trial {}'.format(trial_num)] = trial_frame
        
        condition_df_dict[' '.join(condition.split('_'))] = trial_df_dict
            
    expt_set_df = pd.DataFrame.from_dict(condition_df_dict)
    
    expt_set_df.set_path = expt_set.set_path
    
    return expt_set_df

#%%
def plot_experiment_summaries(expt_set_df, show_plots=True, save_plots = False):
    for cond in sorted(expt_set_df.columns):
        num_rows = len(expt_set_df[cond].index)
        
        fig = plt.figure(figsize=(10,5))
        
        #create a grid with one extra row so we can put legend in it
        grid = gridspec.GridSpec(num_rows + 1, 6)
        
        #code for the legend
        lgd_ax = plt.subplot(grid[-1:, :])
        blue = mpatches.Patch(color='b', label='Retreating')
        green = mpatches.Patch(color='g', label='Advancing')
        red = mpatches.Patch(color='r', label='Pausing')
        lgd = plt.legend(handles=[blue, green, red], ncol=3, loc='center')
        lgd_ax.add_artist(lgd)
        lgd_ax.axis('off')
        fig.add_subplot(lgd_ax)
        
        expt_date = expt_set_df[cond][expt_set_df[cond].index[0]].date
        
        fig.suptitle("{}: {}".format(cond, expt_date), fontsize=14,)
                            
        for trial_indx, trial in enumerate(sorted(expt_set_df[cond].index)):
            try:
                for tunnel_indx, tunnel in enumerate(expt_set_df[cond][trial].index):
                    #print "{} {} {}:\n".format(cond, trial, tunnel)
                    
                    current_ax = plt.subplot(grid[trial_indx, tunnel_indx])
                    current_ax.set_ylim([0,1])
                    current_ax.set_xlim([0,100])
                    current_ax.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
             
                    if tunnel_indx is 0:
                        current_ax.set_ylabel(trial, rotation=0, labelpad=25, fontsize=12)       
             
                    if trial_indx is 0:
                        current_ax.set_title(tunnel, fontsize=12)               
               
                    for clust_indx_indx, cluster_indx_name in enumerate(['Advance cluster indices', 'Pause cluster indices', 'Retreat cluster indices']):
                        palette = ['g', 'r', 'b']
                        normalized_cluster_indices = expt_set_df[cond][trial][cluster_indx_name][tunnel] * 100/float(expt_set_df[cond][trial]['Total experimental timepoints'][tunnel])
                        #print normalized_cluster_indices
                        plot_cluster_span(current_ax, normalized_cluster_indices, palette[clust_indx_indx])
                        
                    fig.add_subplot(current_ax)
            except:
                print('Warning! {} {} has no tunnel information!'.format(cond, trial))
                     
        #make some room at the top of the figure for the figure suptitle
        grid.tight_layout(fig, rect=[0, 0.05, 1, 0.95])
        
        if save_plots is True:
            fig_name = "{} {} RAP analysis.png".format(expt_date, cond)
            savepath = os.path.join(expt_set_df.set_path, fig_name)
            fig.savefig(savepath, format='png', dpi=250, transparent=True)
    if show_plots is True:
        plt.show()

#plot_experiment_summaries(expt_set_df, show_plots = False, save_plots = True)
#%%    

def concatenate_expt_sets(expt_set_group=debug_group):
    """
    concatenate data across multiple replicates so that we can do summary
    statistics as well as summary plots
    
    expt_set_group should be a list of paths to experiment sets
    """    
    expt_set_dfs = [process_set_dataframe(expt_path) for expt_path in expt_set_group]
    
    #because our expt_set_df is a dataframe containing dataframes pd.concat
    #will only superficially concatenate the conditions
    cond_concatenated_df = pd.concat(expt_set_dfs)
    
    cond_concatenated_dict = {}   
    #so we have to delve deeper in the dataframe structure and separately
    #concatenate the inner nested dataframes
    for cond in cond_concatenated_df.columns:
        
        trial_concatenated_list = []    
        
        for trial in sorted(list(set(cond_concatenated_df.index))):
            
            #we also need to check the case where we only pass 1 trial set to the function
            if trial in cond_concatenated_df.index.get_duplicates():            
            
                listed_concat = list(cond_concatenated_df[cond][trial])
                #There might be conditions without all x number of trials
                #they will show up as np.nan entries so we filter them out
                nan_filtered_list = [df for df in listed_concat if df is not np.nan]
        
                #if we've filtered out np.nan trials then we might not have two items 
                #to combine so we need to check!!
                if len(nan_filtered_list) > 1:        
                    trial_concatenated_df = pd.concat(nan_filtered_list)
                    trial_concatenated_list.append(trial_concatenated_df)
                else:
                    trial_concatenated_list.append(nan_filtered_list[0])            
            else:
                trial_concatenated_list.append(cond_concatenated_df[cond][trial])
        
        cond_concatenated_dict[cond] = trial_concatenated_list
        
    concatenated_expt_set_dict = cond_concatenated_dict
    
    return concatenated_expt_set_dict

#%%
def plot_rap_summary(expt_set_group, show_plots=True, save_plots = False):
    
    concatenated_expt_set_dict = concatenate_expt_sets(expt_set_group)
    
    fig = plt.figure(figsize=(8,11.5)) 
    
    for feat_indx, feature in enumerate(['Retreat percentage', 'Advance percentage', 'Pause percentage']):
        
        #Training and testing column (2)
        #Retreat, Advance, and Pause rows (3) + 1 smaller row for legend
        grid = gridspec.GridSpec(4, 2, height_ratios=[1,1,1,0.00001])
        grid.update(hspace=0.75, wspace=0.4)
        
        etoh_concentration_label = [name for name in concatenated_expt_set_dict.keys() if "EtOH" in name][0].split(" ")[-1]
        
        #code for the legend
        lgd_ax = plt.subplot(grid[-1:, :])
        air = mpatches.Patch(color="#bababa", label='Air')
        etoh = mpatches.Patch(color="#377eb8", label=etoh_concentration_label)
        lgd = plt.legend(handles=[air, etoh], ncol=2, loc='center', borderpad=0.75)
        lgd_ax.add_artist(lgd)
        lgd_ax.axis('off')
        fig.add_subplot(lgd_ax)
        
        plt.hold(True)    
        
        for cond in sorted(concatenated_expt_set_dict.keys()):
            #gives you in a list: [Trial 1, Trial 2, Trial 3] data for a specific condition and feature
            feature_dsets = [trial_df[feature].dropna() for trial_df in concatenated_expt_set_dict[cond]]
            
            if 'Train' in cond:
                plot_column_indx = 0
                ax_title = 'Training Trials'
            elif 'Test' in cond:
                plot_column_indx = 1
                ax_title = 'Testing Trials'
            if 'Air' in cond:
                boxplot_offset = -10
                box_color = "#bababa"
                box_mean_color = "#7b7b7b"
                box_mean_edge_color = "white"
            elif 'EtOH' in cond:
                boxplot_offset = 10
                box_color = "#377eb8"
                box_mean_color = "#0000ff"
                box_mean_edge_color = "white"
            
            x_locs = np.array([50*indx for indx, dset in enumerate(feature_dsets, start=1)])      
    
            current_ax = plt.subplot(grid[feat_indx, plot_column_indx])
            box = current_ax.boxplot(feature_dsets, positions = x_locs+boxplot_offset, 
                                     widths=10, showmeans= True, patch_artist=True, 
                                     medianprops=dict(color='white', linewidth=1.2))
            
            current_ax.set_title("{} for\n {}".format(feature, ax_title), y=1.02, fontsize=14)
            current_ax.set_xlim([x_locs[0]-25, x_locs[-1]+25])
            current_ax.set_ylim([-5,105])
            
            x_labels = ["Trial {}".format(indx) for indx, dset in enumerate(feature_dsets, start=1)]                  
            current_ax.set_xticks(x_locs)
            current_ax.set_xticklabels(x_labels)
            current_ax.tick_params(axis='x', pad=10)
            current_ax.set_ylabel("Percentage", labelpad=5)
            
            current_ax.spines['right'].set_visible(False)
            current_ax.spines['top'].set_visible(False)
                  
            for indx, series in enumerate(feature_dsets):
                box['boxes'][indx].set(color=box_color)
                plt.setp(box['caps'][2*indx:2*indx+2], color='none')
                plt.setp(box['whiskers'][2*indx:2*indx+2], color=box_color, linestyle='solid')
                box['means'][indx].set(color=box_color, markerfacecolor=box_mean_color, markeredgecolor = box_mean_edge_color, marker="o", mew=1.1)             
                #outlier 'flier' colors are implemented with markeredgecolor not "color"
                box['fliers'][indx].set(markeredgecolor=box_color, markersize=8.0)         
                    
            current_ax.tick_params(axis='both', top='off', right='off', bottom='off')
            
        plt.hold(False)
        fig.add_subplot(current_ax)
        #grid.tight_layout(fig)      
    
    if save_plots is True:
        fig_name = "RAP summary analysis.png"
        fig.savefig(fig_name, format='png', dpi=300, transparent=True)
    if show_plots is True:
        plt.show()

plt.show()

#%%
#for feat_indx, feature in enumerate(['Retreat percentage', 'Advance percentage', 'Pause percentage']):
#    fig = plt.figure(figsize=(8,11.5))
#    
#    #Training and testing column (2)
#    #Retreat, Advance, and Pause rows (3)
#    #create a grid with one extra row so we can put legend in it
#    grid = gridspec.GridSpec(3 + 1, 2)
#    
#    #code for the legend
#    lgd_ax = plt.subplot(grid[-1:, :])
#    blue = mpatches.Patch(color='b', label='Retreating')
#    green = mpatches.Patch(color='g', label='Advancing')
#    red = mpatches.Patch(color='r', label='Pausing')
#    lgd = plt.legend(handles=[blue, green, red], ncol=3, loc='center')
#    lgd_ax.add_artist(lgd)
#    lgd_ax.axis('off')
#    fig.add_subplot(lgd_ax)
#    
#    expt_date = expt_set_df[cond][expt_set_df[cond].index[0]].date
#    
#    fig.suptitle("{}: {}".format(cond, expt_date), fontsize=14,)
#    
#    for cond in concatenated_expt_set_df.columns:
#        


