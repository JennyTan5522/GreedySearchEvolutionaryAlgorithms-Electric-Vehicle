import random
import numpy as np
from scipy.spatial import distance

'''
DETAILS DATA SECTION of the instance files
============================================
DETAILS of the instance files

Specification part
------------------
All entries in this section are of the form <keyword> : <value>. Below we give a list of all avaiable keywords.

NAME: <string>
Identifies the data file

TYPE: <string>
Specifies the type of the data, i.e., EVRP

COMMENT: <string>
Additional comments

OPTIMAL_VALUE: <integer>
Identifies either the optimal value, upper bound or best known value

VEHICLES: <integer>
It is the minimum number of EVs that can be used

DIMENSION: <integer>
It is the total number of nodes, including customers and depots

ENERGY_CAPACITY: <integer>
Specifies the EV battery capacity

ENERGY_CONSUMPTION: <decimal>
Specifies the energy consumption of the EV when traversing arcs

STATIONS: <integer>
It is the number of charging stations

CAPACITY: <integer>
Specifies the EV cargo capacity

EDGE_WEIGHT_TYPE: <string>
EUC_2D Weights are Euclidean distances in 2-D

EOF: 
Terminates the input data

Data part
----------------
The instance data are given in the corresponding data sections following the specification part. Each data 
begins with the corresponding keyword. The length of the sections depends on the type of the data.

NODE_COORD_SECTION: 
Node coordinates are given in this section. Each line is of the form 
<integer> <real> <real>
The integers give the number of the respective node and the real numbers give the associate coordinates

DEMAND_SECTION:
Customer delivey demands are given in this section. Each line is of the form
<integer> <integer>
The first integer give the number of the respective customer node and the real give its delivery demand. The demand 
of the depot node is always 0.

STATION_COORD_SECTION:
Contains a list of all the recharging station nodes

DEPOT_SECTION:
Contains a list of the depot nodes. This list is terminated by a -1.
'''

class EVRP:
    '''Implementaion of the electric vehicle routing'''

    #def clusteringMethod(self):

    def generate_distanceMatrix(self):
        '''Compute the distance matrix of the problem instance by using euclidean distance'''
        
        #Create a matrix based on the node size
        distanceMatrix=np.zeros((len(self.NODE),len(self.NODE)))
        
        for i in range(self.ACTUAL_PROBLEM_SIZE):
            x1=(self.NODE[i][1],self.NODE[i][2])
            for j in range(self.ACTUAL_PROBLEM_SIZE):
                x2=(self.NODE[j][1],self.NODE[j][2])
                distanceMatrix[i][j]=distance.euclidean(x1,x2)
            print('-------------------------------------------')
        return distanceMatrix

    def read_problems(self,filename:str):
        #TODO This one mayb can chg to RANDOM read if got time 
        '''Read the problem instance and generate the initial object vector'''
        with open(filename,'r') as f:
            data=f.read().splitlines()  
            
        #Store NODE and DEMAND as a dictionary for better access
        self.NODE={}
        self.DEMAND={}

        for idx,line in enumerate(data):
            record=line.split(':')
            record[0]=record[0].strip()

            if (record[0]=='OPTIMAL_VALUE'):
                self.OPTIMUM=float(record[1].strip())

            if (record[0]=='VEHICLES'):
                self.MIN_VEHICLES=int(record[1].strip())

            if (record[0]=='DIMENSION'):
                self.PROBLEM_SIZE=int(record[1].strip()) 

            if (record[0]=='STATIONS'):
                self.NUM_OF_STATIONS=int(record[1].strip())

            if (record[0]=='CAPACITY'):
                self.MAX_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CAPACITY'):
                self.BATTERY_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CONSUMPTION'):
                self.ENERGY_CONSUMPTION=float(record[1].strip())

            if (record[0]=='NODE_COORD_SECTION'):
                NUM_OF_CUSTOMERS=PROBLEM_SIZE-1
                ACTUAL_PROBLEM_SIZE=PROBLEM_SIZE+NUM_OF_STATIONS
                idx+=1
                for i in range(ACTUAL_PROBLEM_SIZE):
                    node_data=data[idx+i]
                    node_data=node_data.split(' ')
                    #Save node as tuple (index,x,y)
                    NODE[int(node_data[0])]=(int(node_data[1]),int(node_data[2]))

            if (record[0]=='DEMAND_SECTION'):
                idx+=1
                for i in range(PROBLEM_SIZE):
                    DEMAND[int(data[idx+i].split(' ')[0])]=int(data[idx+i].split(' ')[1])
                     
        #Generate distance matrix
        self.distanceNodeMatrix=self.generate_distanceMatrix()

        # print(f'OPTIMAL_VALUE: {self.OPTIMUM}')
        # print(f'MIN_VEHICLES: {self.MIN_VEHICLES}')
        # print(f'PROBLEM_SIZE: {self.PROBLEM_SIZE}')
        # print(f'NUM_OF_STATIONS: {self.NUM_OF_STATIONS}')
        # print(f'MAX_CAPACITY: {self.MAX_CAPACITY}')
        # print(f'BATTERY_CAPACITY: {self.BATTERY_CAPACITY}')
        # print(f'ENERGY_CONSUMPTION: {self.ENERGY_CONSUMPTION}')
        # print(f'NUM_OF_CUSTOMERS: {self.NUM_OF_CUSTOMERS}')
        # print(f'ACTUAL_PROBLEM_SIZE: {self.ACTUAL_PROBLEM_SIZE}')
        # print(f'NODE: {self.NODE}')
        # print(f'DEMAND: {self.DEMAND}')
        # print(f'distanceMatrix: {self.distanceNodeMatrix}')



    def __init__(self):
        random.seed(42)
        filenames=['evrp-benchmark-set/E-n22-k4.evrp']
        self.read_problems(filenames[0])

      

 

if __name__ == "__main__":       
    EVRP()
  


    