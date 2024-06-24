# This file is used to group the data from all the .csv file created by the
# various runs of the program.
# It will create a new .csv file easier to read and plot from the notebook.

# imports
import pandas as pd

## Read all the files:

# Read all the data files
onenodes_1win_get_1200 = pd.read_csv('1nodes-1win-get-1200.csv')
onenodes_1win_get_12k = pd.read_csv('1nodes-1win-get-12000.csv')
onenodes_1win_put_1200 = pd.read_csv('1nodes-1win-put-1200.csv')
onenodes_1win_put_12k = pd.read_csv('1nodes-1win-put-12000.csv')
onenodes_2win_get_1200 = pd.read_csv('1nodes-2win-get-1200.csv')
onenodes_2win_get_12k = pd.read_csv('1nodes-2win-get-12000.csv')
onenodes_2win_put_1200 = pd.read_csv('1nodes-2win-put-1200.csv')
onenodes_2win_put_12k = pd.read_csv('1nodes-2win-put-12000.csv')
onenode_nowin_1200 = pd.read_csv('1nodes-two-sided-comm-none-1200.csv')
onenode_nowin_12k = pd.read_csv('1nodes-two-sided-comm-none-12000.csv')

twonodes_1win_get_1200 = pd.read_csv('2nodes-1win-get-1200.csv')
twonodes_1win_get_12k = pd.read_csv('2nodes-1win-get-12000.csv')
twonodes_1win_put_1200 = pd.read_csv('2nodes-1win-put-1200.csv')
twonodes_1win_put_12k = pd.read_csv('2nodes-1win-put-12000.csv')
twonodes_2win_get_1200 = pd.read_csv('2nodes-2win-get-1200.csv')
twonodes_2win_get_12k = pd.read_csv('2nodes-2win-get-12000.csv')
twonodes_2win_put_1200 = pd.read_csv('2nodes-2win-put-1200.csv')
twonodes_2win_put_12k = pd.read_csv('2nodes-2win-put-12000.csv')
twonode_nowin_1200 = pd.read_csv('2nodes-two-sided-comm-none-1200.csv')
twonode_nowin_12k = pd.read_csv('2nodes-two-sided-comm-none-12000.csv')

fournodes_1win_get_1200 = pd.read_csv('4nodes-1win-get-1200.csv')
fournodes_1win_get_12k = pd.read_csv('4nodes-1win-get-12000.csv')
fournodes_1win_put_1200 = pd.read_csv('4nodes-1win-put-1200.csv')
fournodes_1win_put_12k = pd.read_csv('4nodes-1win-put-12000.csv')
fournodes_2win_get_1200 = pd.read_csv('4nodes-2win-get-1200.csv')
fournodes_2win_get_12k = pd.read_csv('4nodes-2win-get-12000.csv')
fournodes_2win_put_1200 = pd.read_csv('4nodes-2win-put-1200.csv')
fournodes_2win_put_12k = pd.read_csv('4nodes-2win-put-12000.csv')
fournode_nowin_1200 = pd.read_csv('4nodes-two-sided-comm-none-1200.csv')
fournode_nowin_12k = pd.read_csv('4nodes-two-sided-comm-none-12000.csv')

eightnodes_1win_get_1200 = pd.read_csv('8nodes-1win-get-1200.csv')
eightnodes_1win_get_12k = pd.read_csv('8nodes-1win-get-12000.csv')
eightnodes_1win_put_1200 = pd.read_csv('8nodes-1win-put-1200.csv')
eightnodes_1win_put_12k = pd.read_csv('8nodes-1win-put-12000.csv')
eightnodes_2win_get_1200 = pd.read_csv('8nodes-2win-get-1200.csv')
eightnodes_2win_get_12k = pd.read_csv('8nodes-2win-get-12000.csv')
eightnodes_2win_put_1200 = pd.read_csv('8nodes-2win-put-1200.csv')
eightnodes_2win_put_12k = pd.read_csv('8nodes-2win-put-12000.csv')
eightnode_nowin_1200 = pd.read_csv('8nodes-two-sided-comm-none-1200.csv')
eightnode_nowin_12k = pd.read_csv('8nodes-two-sided-comm-none-12000.csv')

sixteennodes_1win_get_1200 = pd.read_csv('16nodes-1win-get-1200.csv')
sixteennodes_1win_get_12k = pd.read_csv('16nodes-1win-get-12000.csv')
sixteennodes_1win_put_1200 = pd.read_csv('16nodes-1win-put-1200.csv')
sixteennodes_1win_put_12k = pd.read_csv('16nodes-1win-put-12000.csv')
sixteennodes_2win_get_1200 = pd.read_csv('16nodes-2win-get-1200.csv')
sixteennodes_2win_get_12k = pd.read_csv('16nodes-2win-get-12000.csv')
sixteennodes_2win_put_1200 = pd.read_csv('16nodes-2win-put-1200.csv')
sixteennodes_2win_put_12k = pd.read_csv('16nodes-2win-put-12000.csv')
sixteennode_nowin_1200 = pd.read_csv('16nodes-two-sided-comm-none-1200.csv')
sixteennode_nowin_12k = pd.read_csv('16nodes-two-sided-comm-none-12000.csv')


## Create new columns to sum up the data from the different runs

onenodes_1win_get_1200['nodes'] = 1
onenodes_1win_get_1200['windows'] = 1
onenodes_1win_get_1200['operation'] = 'get'
onenodes_1win_get_1200['size'] = 1200
onenodes_1win_get_12k['nodes'] = 1
onenodes_1win_get_12k['windows'] = 1
onenodes_1win_get_12k['operation'] = 'get'
onenodes_1win_get_12k['size'] = 12000
onenodes_1win_put_1200['nodes'] = 1
onenodes_1win_put_1200['windows'] = 1
onenodes_1win_put_1200['operation'] = 'put'
onenodes_1win_put_1200['size'] = 1200
onenodes_1win_put_12k['nodes'] = 1
onenodes_1win_put_12k['windows'] = 1
onenodes_1win_put_12k['operation'] = 'put'
onenodes_1win_put_12k['size'] = 12000
onenodes_2win_get_1200['nodes'] = 1
onenodes_2win_get_1200['windows'] = 2
onenodes_2win_get_1200['operation'] = 'get'
onenodes_2win_get_1200['size'] = 1200
onenodes_2win_get_12k['nodes'] = 1
onenodes_2win_get_12k['windows'] = 2
onenodes_2win_get_12k['operation'] = 'get'
onenodes_2win_get_12k['size'] = 12000
onenodes_2win_put_1200['nodes'] = 1
onenodes_2win_put_1200['windows'] = 2
onenodes_2win_put_1200['operation'] = 'put'
onenodes_2win_put_1200['size'] = 1200
onenodes_2win_put_12k['nodes'] = 1
onenodes_2win_put_12k['windows'] = 2
onenodes_2win_put_12k['operation'] = 'put'
onenodes_2win_put_12k['size'] = 12000
onenode_nowin_1200['nodes'] = 1
onenode_nowin_1200['windows'] = 0
onenode_nowin_1200['operation'] = 'none'
onenode_nowin_1200['size'] = 1200
onenode_nowin_12k['nodes'] = 1
onenode_nowin_12k['windows'] = 0
onenode_nowin_12k['operation'] = 'none'
onenode_nowin_12k['size'] = 12000

twonodes_1win_get_1200['nodes'] = 2
twonodes_1win_get_1200['windows'] = 1
twonodes_1win_get_1200['operation'] = 'get'
twonodes_1win_get_1200['size'] = 1200
twonodes_1win_get_12k['nodes'] = 2
twonodes_1win_get_12k['windows'] = 1
twonodes_1win_get_12k['operation'] = 'get'
twonodes_1win_get_12k['size'] = 12000
twonodes_1win_put_1200['nodes'] = 2
twonodes_1win_put_1200['windows'] = 1
twonodes_1win_put_1200['operation'] = 'put'
twonodes_1win_put_1200['size'] = 1200
twonodes_1win_put_12k['nodes'] = 2
twonodes_1win_put_12k['windows'] = 1
twonodes_1win_put_12k['operation'] = 'put'
twonodes_1win_put_12k['size'] = 12000
twonodes_2win_get_1200['nodes'] = 2
twonodes_2win_get_1200['windows'] = 2
twonodes_2win_get_1200['operation'] = 'get'
twonodes_2win_get_1200['size'] = 1200
twonodes_2win_get_12k['nodes'] = 2
twonodes_2win_get_12k['windows'] = 2
twonodes_2win_get_12k['operation'] = 'get'
twonodes_2win_get_12k['size'] = 12000
twonodes_2win_put_1200['nodes'] = 2
twonodes_2win_put_1200['windows'] = 2
twonodes_2win_put_1200['operation'] = 'put'
twonodes_2win_put_1200['size'] = 1200
twonodes_2win_put_12k['nodes'] = 2
twonodes_2win_put_12k['windows'] = 2
twonodes_2win_put_12k['operation'] = 'put'
twonodes_2win_put_12k['size'] = 12000
twonode_nowin_1200['nodes'] = 2
twonode_nowin_1200['windows'] = 0
twonode_nowin_1200['operation'] = 'none'
twonode_nowin_1200['size'] = 1200
twonode_nowin_12k['nodes'] = 2
twonode_nowin_12k['windows'] = 0
twonode_nowin_12k['operation'] = 'none'
twonode_nowin_12k['size'] = 12000

# 4 nodes
fournodes_1win_get_1200['nodes'] = 4
fournodes_1win_get_1200['windows'] = 1
fournodes_1win_get_1200['operation'] = 'get'
fournodes_1win_get_1200['size'] = 1200
fournodes_1win_get_12k['nodes'] = 4
fournodes_1win_get_12k['windows'] = 1
fournodes_1win_get_12k['operation'] = 'get'
fournodes_1win_get_12k['size'] = 12000
fournodes_1win_put_1200['nodes'] = 4
fournodes_1win_put_1200['windows'] = 1
fournodes_1win_put_1200['operation'] = 'put'
fournodes_1win_put_1200['size'] = 1200
fournodes_1win_put_12k['nodes'] = 4
fournodes_1win_put_12k['windows'] = 1
fournodes_1win_put_12k['operation'] = 'put'
fournodes_1win_put_12k['size'] = 12000
fournodes_2win_get_1200['nodes'] = 4
fournodes_2win_get_1200['windows'] = 2
fournodes_2win_get_1200['operation'] = 'get'
fournodes_2win_get_1200['size'] = 1200
fournodes_2win_get_12k['nodes'] = 4
fournodes_2win_get_12k['windows'] = 2
fournodes_2win_get_12k['operation'] = 'get'
fournodes_2win_get_12k['size'] = 12000
fournodes_2win_put_1200['nodes'] = 4
fournodes_2win_put_1200['windows'] = 2
fournodes_2win_put_1200['operation'] = 'put'
fournodes_2win_put_1200['size'] = 1200
fournodes_2win_put_12k['nodes'] = 4
fournodes_2win_put_12k['windows'] = 2
fournodes_2win_put_12k['operation'] = 'put'
fournodes_2win_put_12k['size'] = 12000
fournode_nowin_1200['nodes'] = 4
fournode_nowin_1200['windows'] = 0
fournode_nowin_1200['operation'] = 'none'
fournode_nowin_1200['size'] = 1200
fournode_nowin_12k['nodes'] = 4
fournode_nowin_12k['windows'] = 0
fournode_nowin_12k['operation'] = 'none'
fournode_nowin_12k['size'] = 12000

# 8 nodes
eightnodes_1win_get_1200['nodes'] = 8
eightnodes_1win_get_1200['windows'] = 1
eightnodes_1win_get_1200['operation'] = 'get'
eightnodes_1win_get_1200['size'] = 1200
eightnodes_1win_get_12k['nodes'] = 8
eightnodes_1win_get_12k['windows'] = 1
eightnodes_1win_get_12k['operation'] = 'get'
eightnodes_1win_get_12k['size'] = 12000
eightnodes_1win_put_1200['nodes'] = 8
eightnodes_1win_put_1200['windows'] = 1
eightnodes_1win_put_1200['operation'] = 'put'
eightnodes_1win_put_1200['size'] = 1200
eightnodes_1win_put_12k['nodes'] = 8
eightnodes_1win_put_12k['windows'] = 1
eightnodes_1win_put_12k['operation'] = 'put'
eightnodes_1win_put_12k['size'] = 12000
eightnodes_2win_get_1200['nodes'] = 8
eightnodes_2win_get_1200['windows'] = 2
eightnodes_2win_get_1200['operation'] = 'get'
eightnodes_2win_get_1200['size'] = 1200
eightnodes_2win_get_12k['nodes'] = 8
eightnodes_2win_get_12k['windows'] = 2
eightnodes_2win_get_12k['operation'] = 'get'
eightnodes_2win_get_12k['size'] = 12000
eightnodes_2win_put_1200['nodes'] = 8
eightnodes_2win_put_1200['windows'] = 2
eightnodes_2win_put_1200['operation'] = 'put'
eightnodes_2win_put_1200['size'] = 1200
eightnodes_2win_put_12k['nodes'] = 8
eightnodes_2win_put_12k['windows'] = 2
eightnodes_2win_put_12k['operation'] = 'put'
eightnodes_2win_put_12k['size'] = 12000
eightnode_nowin_1200['nodes'] = 8
eightnode_nowin_1200['windows'] = 0
eightnode_nowin_1200['operation'] = 'none'
eightnode_nowin_1200['size'] = 1200
eightnode_nowin_12k['nodes'] = 8
eightnode_nowin_12k['windows'] = 0
eightnode_nowin_12k['operation'] = 'none'
eightnode_nowin_12k['size'] = 12000

# 16 nodes
sixteennodes_1win_get_1200['nodes'] = 16
sixteennodes_1win_get_1200['windows'] = 1
sixteennodes_1win_get_1200['operation'] = 'get'
sixteennodes_1win_get_1200['size'] = 1200
sixteennodes_1win_get_12k['nodes'] = 16
sixteennodes_1win_get_12k['windows'] = 1
sixteennodes_1win_get_12k['operation'] = 'get'
sixteennodes_1win_get_12k['size'] = 12000
sixteennodes_1win_put_1200['nodes'] = 16
sixteennodes_1win_put_1200['windows'] = 1
sixteennodes_1win_put_1200['operation'] = 'put'
sixteennodes_1win_put_1200['size'] = 1200
sixteennodes_1win_put_12k['nodes'] = 16
sixteennodes_1win_put_12k['windows'] = 1
sixteennodes_1win_put_12k['operation'] = 'put'
sixteennodes_1win_put_12k['size'] = 12000
sixteennodes_2win_get_1200['nodes'] = 16
sixteennodes_2win_get_1200['windows'] = 2
sixteennodes_2win_get_1200['operation'] = 'get'
sixteennodes_2win_get_1200['size'] = 1200
sixteennodes_2win_get_12k['nodes'] = 16
sixteennodes_2win_get_12k['windows'] = 2
sixteennodes_2win_get_12k['operation'] = 'get'
sixteennodes_2win_get_12k['size'] = 12000
sixteennodes_2win_put_1200['nodes'] = 16
sixteennodes_2win_put_1200['windows'] = 2
sixteennodes_2win_put_1200['operation'] = 'put'
sixteennodes_2win_put_1200['size'] = 1200
sixteennodes_2win_put_12k['nodes'] = 16
sixteennodes_2win_put_12k['windows'] = 2
sixteennodes_2win_put_12k['operation'] = 'put'
sixteennodes_2win_put_12k['size'] = 12000
sixteennode_nowin_1200['nodes'] = 16
sixteennode_nowin_1200['windows'] = 0
sixteennode_nowin_1200['operation'] = 'none'
sixteennode_nowin_1200['size'] = 1200
sixteennode_nowin_12k['nodes'] = 16
sixteennode_nowin_12k['windows'] = 0
sixteennode_nowin_12k['operation'] = 'none'
sixteennode_nowin_12k['size'] = 12000

## Concatenate all the dataframes

all_data = pd.concat([onenodes_1win_get_1200, onenodes_1win_get_12k, onenodes_1win_put_1200, onenodes_1win_put_12k, onenodes_2win_get_1200, onenodes_2win_get_12k, onenodes_2win_put_1200, onenodes_2win_put_12k, onenode_nowin_1200, onenode_nowin_12k, twonodes_1win_get_1200, twonodes_1win_get_12k, twonodes_1win_put_1200, twonodes_1win_put_12k, twonodes_2win_get_1200, twonodes_2win_get_12k, twonodes_2win_put_1200, twonodes_2win_put_12k, twonode_nowin_1200, twonode_nowin_12k, fournodes_1win_get_1200, fournodes_1win_get_12k, fournodes_1win_put_1200, fournodes_1win_put_12k, fournodes_2win_get_1200, fournodes_2win_get_12k, fournodes_2win_put_1200, fournodes_2win_put_12k, fournode_nowin_1200, fournode_nowin_12k, eightnodes_1win_get_1200, eightnodes_1win_get_12k, eightnodes_1win_put_1200, eightnodes_1win_put_12k, eightnodes_2win_get_1200, eightnodes_2win_get_12k, eightnodes_2win_put_1200, eightnodes_2win_put_12k, eightnode_nowin_1200, eightnode_nowin_12k, sixteennodes_1win_get_1200, sixteennodes_1win_get_12k, sixteennodes_1win_put_1200, sixteennodes_1win_put_12k, sixteennodes_2win_get_1200, sixteennodes_2win_get_12k, sixteennodes_2win_put_1200, sixteennodes_2win_put_12k, sixteennode_nowin_1200, sixteennode_nowin_12k])

# Aggregate the data
all_data_grouped = all_data.groupby(['size', 'what', 'nodes', 'windows', 'operation']).agg({'time': ['mean']}).reset_index()
# rename the mean column
all_data_grouped.columns = ['size', 'what', 'nodes', 'windows', 'operation', 'time']


# Save all the data to a new .csv file

all_data_grouped.to_csv('all_data_grouped.csv', index=False)