
# coding: utf-8

# In[2]:

#In Python, there is no lack of performance if we import several times the same module and / or package
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import datetime


# In[31]:

class Temperatures(object):
    """A dataframe storing all temperatures we know
    Given the path name of the directory containing all Meteo France gzip files we want to use"""
    
    def __init__(self):
        "constructor"
        import pandas as pd
        self.df =  pd.DataFrame()
    
    def clean_temperatures(self) :
        list_outremer = [71805, 81408, 81415, 81405, 81401, 78894, 78890, 78897, 78922, 78925, 89642, 
                 61998, 61996, 61997, 61980, 61976, 61968, 67005, 61970, 61972]
        list_corsica = [7790,7761]
        #Convert temperature from Kelvin to Celsius degrees :
        self.df.loc[self.df.t == 'mq','t'] = np.nan
        self.df.t = pd.to_numeric(self.df.t)
        self.df.t = self.df.t - 273.15
        #Convert dates to DateTime format
        self.df.date = pd.to_datetime(self.df.date, format='%Y%m%d%H%M%S')
        #Remove stations that are not in continental France ("Outre-mer" and Corsica)
        self.df = self.df[(self.df.numer_sta.isin(list_outremer)== False) & (self.df.numer_sta.isin(list_corsica)==False)]
        #Compute temperature average
        grouped_temp =  self.df.groupby('date').t.mean()
        #Replace self.df by the average temperatures dataframe
        self.df = pd.DataFrame(grouped_temp).reset_index()
        #Add a column giving the hour
        self.df['hour'] =  self.df.date.apply(lambda u: u.strftime('%H:%M'))
        self.df['date'] = self.df.date.apply(lambda u: u.date())
        #Pivot to get the same dataframe shape as electricity
        self.df = self.df.pivot(index='date',columns='hour', values='t')
        self.df['date'] = self.df.index
        self.df = self.df.reset_index(drop=True)
    
    def get_temperatures(self, mypath):
        """given a path name (do not forget '/' at the end) of the directory containing all Meteo France gzip files we want to use,
        Unzip the gzip files
        And insert them into the Temperatures instance"""
        import numpy as np
        from os import listdir
        from os.path import isfile, join
        
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for file_name in onlyfiles :
            df_temp = pd.read_table(mypath+file_name,compression='gzip', sep = ';')
            df_temp = df_temp[['numer_sta', 'date', 't']]
            self.df = self.df.append(df_temp, ignore_index=True)
        self.clean_temperatures()
        
        
            


# In[131]:

class Electricity(object):
    """A dataframe storing the consolidated half-hourly electricity load in France (except outre-mer and Corsica)
    Given the path name of the directory containing all RTE subdirectories we want to use"""
    
    def __init__(self):
        "constructor"
        import pandas as pd
        self.df =  pd.DataFrame()
    
    def clean_electricity(self) :
        import datetime
        #Get holiday dates in France using the library called workalendar
        from workalendar.europe import France
        cal = France()
        #Convert dates to DateTime format
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        #Create the DayTypes mentioned in the article (except DayType 8)
        #/!\ We chose an arbitrary order to create the day types, since they are not separated sets
        #Add normal weekdays (DayType 1)
        self.df['Day_type'] = self.df['Date'].apply(lambda u : (u.date(), 1))
        #Add Mondays (DayType 0)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],0) if x[0].weekday()==0 else x)
        #Add Fridays (DayType 2)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],2) if x[0].weekday()==4 else x)
        #Add Saturdays (DayType 3)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],3) if x[0].weekday()==5 else x)
        #Add Sundays (DayType 4)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],4) if x[0].weekday()==6 else x)
        #Add Bank holidays (DayType 6)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],6) if cal.is_holiday(x[0])==True else x)
        #Add Before Bank holidays (DayType 5)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],5) if cal.is_holiday(x[0] + datetime.timedelta(days=1))==True else x)
        #Add After Bank holidays (DayType 7)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : (x[0],7) if cal.is_holiday(x[0] - datetime.timedelta(days=1))==True else x)
        self.df['Day_type'] = self.df['Day_type'].apply(lambda x : x[1]) 
        self.df['Date'] = self.df['Date'].apply(lambda u: u.date())
    
    def get_electricity_data(self, mypath):
        """given a path name (do not forget '/' at the end) of the directory containing all RTE folders,
        Read the Excel files
        And insert them into the Electricity instance"""
        import numpy as np
        from os import listdir
        from os.path import isfile, join
        import warnings
        
        onlyfiles = [f + "/" + listdir(mypath + f)[0] for f in listdir(mypath)]
        for file_name in onlyfiles :
            df_temp = pd.read_excel(mypath+file_name, skiprows=range(17),  header = 1)
            #Drop empty lines separating each month on the Excel sheet
            df_temp = df_temp.dropna(axis = 0).reset_index(drop = True)
            #Check that df_temps has 365 rows (/!\ There can be leap years containing 366 days, that is why only a warning is raised here)
            if (len(df_temp) > 365) or (len(df_temp) < 365) :
                warnings.warn("WARNING : There is more than 365 days or less than 365 days in dataframe")
                print(file_name)
                print('Number of rows', len(df_temp))
            self.df = self.df.append(df_temp, ignore_index=True)
        self.clean_electricity()

