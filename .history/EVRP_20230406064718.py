import random
import numpy as np
from scipy.spatial import distance

'''

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
        return distanceMatrix

    def read_problems(self,filename:str):
        '''Read the problem instance and generate the initial object vector'''
        with open(filename,'r') as f:
            data=f.read().splitlines()
        
        self.NODE=[]
        self.DEMAND=[]
        print(data[0])

        for idx,line in enumerate(data):
            record=line.split(':')
            record[0]=record[0].strip()

            # if (record[0]=='Name'):
            #     self.

            if (record[0]=='OPTIMAL_VALUE'):
                #OPTIMAL_VALUE=optimal value, upper bound or best known value
                self.OPTIMUM=float(record[1].strip())

            if (record[0]=='VEHICLES'):
                #VEHICLES=Number of EV we can be used
                self.MIN_VEHICLES=int(record[1].strip())

            if (record[0]=='DIMENSION'):
                #PROBLEM_SIZE=DEPOT(1)+NUMBER_OF_CUSTOMERS
                self.PROBLEM_SIZE=int(record[1].strip()) 

            if (record[0]=='STATIONS'):
                #STATIONS=Number of charging stations
                self.NUM_OF_STATIONS=int(record[1].strip())

            if (record[0]=='CAPACITY'):
                #CAPACITY=Specifies the EV cargo capacity
                self.MAX_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CAPACITY'):
                #ENERGY_CAPACITY=BATTERY CAPACITY
                self.BATTERY_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CONSUMPTION'):
                #ENERGY_CONSUMPTION=Specifies the energy consumption of the EV when traversing arcs
                self.ENERGY_CONSUMPTION=float(record[1].strip())

            if (record[0]=='NODE_COORD_SECTION'):
                #NODE_COORD_SECTION=Includes all the 
                self.NUM_OF_CUSTOMERS=self.PROBLEM_SIZE-1
                self.ACTUAL_PROBLEM_SIZE=self.PROBLEM_SIZE+self.NUM_OF_STATIONS
                idx+=1
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    node_data=data[idx+i]
                    node_data=node_data.split(' ')
                    #Save node as tuple (index,x,y)
                    self.NODE.append((int(node_data[0]),int(node_data[1]),int(node_data[2])))

            if (record[0]=='DEMAND_SECTION'):
                idx+=1
                for i in range(self.PROBLEM_SIZE):
                    self.DEMAND.append(data[idx+i].split(' ')[1])
            

                     
        #Generate distance matrix
        self.distanceMatrix=self.generate_distanceMatrix()

        print(f'OPTIMAL_VALUE: {self.OPTIMUM}')
        print(f'MIN_VEHICLES: {self.MIN_VEHICLES}')
        print(f'PROBLEM_SIZE: {self.PROBLEM_SIZE}')
        print(f'NUM_OF_STATIONS: {self.NUM_OF_STATIONS}')
        print(f'MAX_CAPACITY: {self.MAX_CAPACITY}')
        print(f'BATTERY_CAPACITY: {self.BATTERY_CAPACITY}')
        print(f'ENERGY_CONSUMPTION: {self.ENERGY_CONSUMPTION}')
        print(f'NUM_OF_CUSTOMERS: {self.NUM_OF_CUSTOMERS}')
        print(f'ACTUAL_PROBLEM_SIZE: {self.ACTUAL_PROBLEM_SIZE}')
        print(f'NODE: {self.NODE}')
        print(f'DEMAND: {self.DEMAND}')
        print(f'distanceMatrix: {self.distanceMatrix}')



    def __init__(self):
        random.seed(42)
        filenames=['evrp-benchmark-set/E-n22-k4.evrp']
        self.read_problems(filenames[0])

      

 

if __name__ == "__main__":       
    EVRP()
  


    