#!/usr/bin/env python 
"""
Class to load in data produced by the 
Fly Alcohol Assay

By Nicholas Mei
"""
import sys
import os
import os.path

import glob

import pandas as pd
import numpy as np

#If we are using python 2.7 or under
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
      
#If we are using python 3.0 or above
elif sys.version_info[0] > 3:
    import tkinter as tk
    import tkinter.filedialog as filedialog

def chooseDir(cust_text):
    root = tk.Tk()
    try:
        root.tk.call('console','hide')
    except tk.TclError:
        pass
    
    baseFilePath = "C:\Users\Nicholas\Desktop\FlyBar Analysis"
    directoryPath = filedialog.askdirectory(parent = root, title=cust_text, initialdir= baseFilePath)
    root.destroy()
    
    return directoryPath

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
            
            self.data_paths = []
            
            #Walkway relevant data                       
            self.Raw_Walk_Table = self.readin(trial_path, "Raw-Walkway")
            self.Analyzed_Walk_Table = self.readin(trial_path, "Analyzed-Walkway")          
            self.Summarized_Walk_Table = self.readin(trial_path, "Summarized-Walkway")

            #EndChamber relevant data  
            self.Raw_End_Table = self.readin(trial_path, "Raw-EndChamber")            
            self.Analyzed_End_Table = self.readin(trial_path, "Analyzed-EndChamber")
            self.Summarized_End_Table = self.readin(trial_path, "Summarized-EndChamber")
            
            self.PATHS = PATH(self.data_paths)
            
            delattr(self, "data_paths")
            
    # General FAA file readin function
    def readin(self, trial_path, searchString):
        try:
            filePath = glob.glob(os.path.join(trial_path, "{}*".format(searchString)))[0]
            self.data_paths.append((("_".join(searchString.split("-"))), filePath))
            
            if "Summarized" in searchString:
                if "End" in searchString:
                    return self.summary_readin(filePath, End_Flag = True)
                else:
                    return self.summary_readin(filePath)            
            else:
                return pd.read_csv(filePath)                
        except:
            print("Could not find or read the {} file for {}! Skipping!\n".format(searchString, trial_path))
        
    # Pandas Summary File Readin Function
    def summary_readin(self, dataPath = None, End_Flag = None):
        """
        Function that reads in the .txt summariy files generated from the 
        fly alcohol assay into a pandas dataframe
        
        Need a special readin function for the summary because sometimes if a
        lane times out we would still like to have the data be a NaN rather than
        completely omitted. (can be used for categorical analyses such as did 
        fly make it to the end or not)
        
        End_Flag should be toggled to true if you're trying to read in the
        EndChamber summary file as it has extra time periods compared to walkway
        data
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
            # This function can fail miserably depending on your naming scheme for the experiment folder 
            # Typical naming scheme I use:
            # "Dur" and "Data" are not parsed
            # "YYYY-MM-DD exptCondition timeOfDay exptType Dur treatmentDuration Data"
            # I.E. "2015-03-10 Test PM2 EtOH80 Dur 10 Data"
            date, exptCondition, time, exptType, _, duration, _ = expt_name.split(" ")            
            
            # Get tentative trial directories from experiment directory path
            tentTrialDirs = glob.glob(os.path.join(expt_path, "Trial*"))
            
            if tentTrialDirs:
                for indx, trial in enumerate(tentTrialDirs, start = 1):
                    # Create an instance of the "Trial" class for each trial
                    setattr(self, "Trial_{}".format(indx), load_trial(trial))
                    
# A class that reads in an entire experiment set
# I.E. Air Train/Test vs. EtOH Train/Test (2 days worth of experiments)
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
                
                # This function can fail miserably depending on your naming scheme for the experiment folder 
                # Typical naming scheme I use:
                # "Dur" and "Data" are not parsed
                # "YYYY-MM-DD exptCondition timeOfDay exptType Dur treatmentDuration Data"
                # I.E. "2015-03-10 Test PM2 EtOH80 Dur 10 Data"                                
                # Parse out:
                # date: YYYY-MM-DD
                # exptCondition: Training or Test
                # time: PM1 or PM2 or something else
                # exptType: Air or EtOH
                # duration: Amount of time flies are treated to EtOH or Air
                date, exptCondition, time, exptType, _, duration, _ = exptDir.split(" ")
                setattr(self, str(exptCondition + "_" + exptType), load_expt(os.path.join(set_path, exptDir)))
                
def main():
    pass
    
if __name__ == '__main__':
    main()