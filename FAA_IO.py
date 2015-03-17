#!/usr/bin/env python 
"""
Class to load in data produced by the 
Fly Alcohol Assay

By Nicholas Mei

Last Updated 2/13/15
"""
import os
import os.path

import glob

import pandas as pd
import numpy as np

import Tkinter as tk
import tkFileDialog

def chooseDir(cust_text):
    root = tk.Tk()
    try:
        root.tk.call('console','hide')
    except tk.TclError:
        pass
    
    baseFilePath = "C:\Users\Nicholas\Desktop\FlyBar Analysis"
    directory_path = tkFileDialog.askdirectory(parent = root, title=cust_text, initialdir= baseFilePath)
    root.destroy()
    
    return directory_path

# Class that adds attributes for data pathnames  
class PATH():   
    def __init__(self, paths):
        for attrname, path in paths:
            setattr(self, attrname, path)
        
# Class that loads a single trial
class load_trial():
    def __init__(self, trial_path = None):
        if not trial_path:
            trial_path = chooseDir("Please select an single trial directory to load")
        if trial_path:
                  
            data_paths = []
            
            try:
                analyzedWalkPath = glob.glob(os.path.join(trial_path, "Analyzed-Walkway*"))[0]
                data_paths.append(("Analyzed_Walkway", analyzedWalkPath))        
            except:
                print "Could not find the Analyzed-Walkway.txt file for {}! Skipping!\n".format(trial_path)
            
            try:
                summarizedWalkPath = glob.glob(os.path.join(trial_path, "Summarized-Walkway*"))[0]
                data_paths.append(("Summarized_Walkway", summarizedWalkPath))
                self.Summarized_Walk_Table = self.summary_readin(summarizedWalkPath)
            except:
                print "Could not find the Summarized-Walkway.txt file for {}! Skipping!\n".format(trial_path)
            
            try:
                analyzedEndPath = glob.glob(os.path.join(trial_path, "Analyzed-EndChamber*"))[0]
                data_paths.append(("Analyzed_End", analyzedEndPath))
            except:
                print "Could not find the Analyzed-Endchamber.txt file for {}! Skipping!\n".format(trial_path)
            
            try:
                summarizedEndPath = glob.glob(os.path.join(trial_path, "Summarized-EndChamber*"))[0]
                data_paths.append(("Summarized_End", summarizedEndPath))
                self.Summarized_End_Table = self.summary_readin(summarizedEndPath, End_Flag = True) 
            except:
                print "Could not find the Summarized-Endchamber.txt file for {}! Skipping!\n".format(trial_path)
            
            self.PATHS = PATH(data_paths)
            
    # Pandas Summary File Readin Function
    def summary_readin(self, dataPath = None, End_Flag = None):
        """
        Function that reads in the .txt summariy files generated from the 
        fly alcohol assay into a pandas dataframe
        """        
        if dataPath:
            data = pd.read_csv(dataPath)
                        
            # Initialize empty pandas dataframe template
            columns = ["tunnel", "total_time", "total_distance", "mean_velocity", "mean_angular_velocity"]

            if End_Flag:
                #24 entries corresponding to 6 lanes * 4 periods (Total, Air before, EtOH, Air After)
                num_entries = 24
                columns.append("status")
            else:
                num_entries = 6
            
            rep = num_entries / 6
            index = np.arange(0, num_entries)
            template = pd.DataFrame(columns=columns, index= index)
            # Fill column data with values as we already know them
            template["tunnel"] = np.tile(np.arange(1,7), rep)
            
            inters = np.intersect1d(np.arange(1,7), data["tunnel"].values)
            
            
            if End_Flag:
                template["status"] = np.concatenate((np.tile("Total", 6), 
                                                     np.tile("AirBefore", 6), 
                                                     np.tile("Ethanol", 6), 
                                                     np.tile("AirAfter", 6)))
                                                     
                for indx, val in data.iterrows():
                    #Find the template location indx that matches up to current data row value data
                    template.loc[(template['tunnel'] == val['tunnel']) & (template["status"] == val['status'])] = val.values
                
            else:                         
                #Insert existing lane values into empty template
                for indx, val in enumerate(inters):
                    template.loc[val-1] = data.loc[indx]
                                
            return(template)
        
# Class that loads in a single experiment        
class load_expt():
    
    def __init__(self, expt_path = None):
        if not expt_path:
            expt_path = chooseDir("Please select a single experiment directory to load")
        if expt_path:
            self.expt_path = expt_path            
            expt_name = os.path.basename(expt_path)

            # Get experiment details from directory name
            date, exptCondition, time, exptType, _, duration, _ = expt_name.split(" ")            
            
            # Get tentative trial directories from experiment directory path
            tentTrialDirs = glob.glob(os.path.join(expt_path, "Trial*"))
            
            if tentTrialDirs:
                for indx, trial in enumerate(tentTrialDirs, start = 1):
                    # Create an instance of the "Trial" class for each trial
                    setattr(self, "Trial_{}".format(indx), load_trial(trial))
                    
# A class that reads in an entire experiment set
# I.E. Air Train/Test vs. EtOH Train/Test (4 days worth of experiments)
class load_set(): 
    
    # Initilization to convert dictionary into object attributes
    def __init__(self, set_path= None):
        
        if not set_path:
            set_path = chooseDir('Please select a single "experiment set" directory to load')
        if set_path:
            self.set_path = set_path
            # Find all the subdirectories that may contain experiments
            tentExperimentDirs = [DIR for DIR in os.listdir(set_path) if os.path.isdir(os.path.join(set_path,DIR))]
            
            # Filter out non Training or Test folders
            exptDirectories = [name for name in tentExperimentDirs if ("Training" in name or "Test" in name)]
                       
            for exptDir in exptDirectories:
                
                # Parse out:
                # date: YYYY-MM-D
                # exptCondition: Training or Test
                # time: PM1 or PM2 or something else
                # exptType: Air or EtOH
                # duration: Amount of time 
                date, exptCondition, time, exptType, _, duration, _ = exptDir.split(" ")
                setattr(self, str(exptCondition + "_" + exptType), load_expt(os.path.join(set_path, exptDir)))
                
def main():
    pass
    
if __name__ == '__main__':
    main()