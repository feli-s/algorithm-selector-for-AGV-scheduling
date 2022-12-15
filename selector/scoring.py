import pandas as pd
import numpy as np

#Exact methods
#CP Optimizer
cpo = pd.read_csv('Validation_FJSSP_CPO_30.csv', sep=";")
#OR Tools
ortools = pd.read_csv('Validation_ORTOOLS_FJSSP_30.csv', sep=";")

#Dataframe to store winning algorithms per instance
df = cpo.drop(['solve_time', 'cmax', 'solver'], axis=1)
df['solver'] = ""
#Score performance of algorithms
pd.set_option('display.max_columns', None)

cpo["solve_time"] = cpo["solve_time"].str.replace(',', '.').astype(float)
ortools["solve_time"] = ortools["solve_time"].str.replace(',', '.').astype(float)
cpo["solve_time"] = pd.to_numeric(cpo["solve_time"])
ortools["solve_time"] = pd.to_numeric(ortools["solve_time"])

counter = 0
for x in df['instance']:
    if cpo['cmax'][counter] < ortools['cmax'][counter]:
        df['solver'][counter] = 'CPO'
    elif cpo['cmax'][counter] == ortools['cmax'][counter]:
        if cpo['solve_time'][counter] < ortools['solve_time'][counter]:
            df['solver'][counter] = 'CPO'
        else:
            df['solver'][counter] = 'ORTOOLS'
    else:
        df['solver'][counter] = 'ORTOOLS'
    counter += 1

algorithm_scoring = df.to_csv('scoring_FJSSP_30.csv')
algorithm_scoring_xls = df.to_excel('scoring_FJSSP_30.xlsx')





