import random
import numpy as np
from scipy.spatial import distance

class EVRP:
    '''Implementaion of the electric vehicle routing'''

    def calculateTotalDistance(self,routes:list):
        return np.sum([self.distanceMatrix[routes[i]-1][routes[i+1]-1] for i in range(len(routes)-1)])

    def totalDemandInRoute(self,route:list):
        '''Find the total demand of the customers in that particular route'''
        return np.sum([self.DEMAND[cust] for cust in route])
    
    def nearestCustomers(self,customer:int):
        #Extract distance of depot+customers
        sortCustIdxMatrix=np.argsort(self.distanceMatrix[customer-1][:self.NUM_OF_CUSTOMERS+1]) 
        #Exclude customer itself (distance=0) and depot, we can straight jump to second record
        return [sortCustIdxMatrix[i]+1 for i in range(1,sortCustIdxMatrix.shape[0]) if sortCustIdxMatrix[i]!=0]
    
    def euclidean_distance(self,node1:int,node2:int): 
        '''Compute and return the euclidean distance of 2 coordinates'''
        coor1=(self.NODE[node1][0],self.NODE[node1][1]) #(x,y)
        coor2=(self.NODE[node2][0],self.NODE[node2][1])
        return distance.euclidean(coor1,coor2)
    
    def compute_distances(self,matrix):
        '''Compute the distance matrix of the problem instance'''
        for i in range(self.ACTUAL_PROBLEM_SIZE): #Eg:Loop from index 0(node1) to index 29(node30)
            for j in range(self.ACTUAL_PROBLEM_SIZE):
                matrix[i][j]=self.euclidean_distance(i+1,j+1)
        return matrix
    
    def generate_2D_distance_matrix(self):
        '''Generate 2D distance matrix and find the distance between 2 points'''
        #Initialize 2D array
        matrix=np.zeros((len(self.NODE),len(self.NODE)))
        #Calculate euclidean distance based on 2 points
        distanceMatrix=self.compute_distances(matrix)
        return distanceMatrix 
    
    def swap(self,route,i,j):
        temp=route[j]
        route[j]=route[i] 
        route[i]=temp
        return route
    
    def local2Opt(self,existingRoute:list):
        '''
        
        '''
        existingDistance=self.calculateTotalDistance(existingRoute)
        stop=False
        while(stop==False):
            stop=True
            for i in range(len(existingRoute)):
                for j in reversed(range(i+1,len(existingRoute))):
                    #Exchange i and j location
                    newRoute=existingRoute.copy()
                    newRoute=self.swap(newRoute,i,j)
                    
                    #Find the total distance of newRoute
                    newRouteDistance=self.calculateTotalDistance(newRoute)
                    
                    #If the total distance of existingDistance is better than before then we swap the target
                    if (newRouteDistance < existingDistance):
                        #Update currentRoute to newRoute
                        existingRoute=newRoute.copy()
                        
                        #Update currentDistance to newDistance
                        existingDistance=newRouteDistance
                        
                        stop=False
        return existingRoute


    def balancingApproach(self):
        '''
        Step 2:
        Customers assigned in the last route are the non-clustered customers,
        the are remanining customers so their geo locations are not close-set.
        The customers on the last route will be less than other routes, even a single customer.
        Use a Balanced Approach to ensure the distance of customers and increase the number of customers in the last route.
        '''
        #1. Randomly select a customer (customer A form the last route).
        lastRoute=self.finalCluster[-1]
        customerA=lastRoute[random.randint(0,len(lastRoute)-1)]

        #2. Select in turn the customers from other routes such which is the closest to A.
        # Find all the nearest distance from A
        nearestA=self.nearestCustomers(customerA)
        # Make sure that the node taken not from the lastRoute
        nearestA=[customer for customer in nearestA if customer not in lastRoute] 

        #3. The chosen customers must satisfy the 2 following conditions - 3a and 3b
        #Loop all nearest customers from A
        currentLastRouteDemand=self.totalDemandInRoute(lastRoute)
        for cust in nearestA:
            '''
            (3a) Initial sum of the total capacity of the route and the capacity of the chosen the EV's 
                maximal carrying capacity Pmax customers does not exceed. 
            '''
            if (self.DEMAND[cust]+currentLastRouteDemand<self.MAX_CAPACITY):
                #The list of that particular cust route with the absence of that cust
                nearestRouteToAList=[cluster for cluster in self.finalCluster if cust in cluster][0]
                
                #lastRouteNewCustList=last route + 1 new nearest customer
                lastRouteExpandList=lastRoute+[cust]
                afterRemoveCustRoute=[i for i in [cluster for cluster in self.finalCluster if cust in cluster][0] if i!=cust]
                
                #capacityRouteA=old last route - the clsuter of the nearest route to A 
                capacityRouteA=abs(self.totalDemandInRoute(lastRoute)-self.totalDemandInRoute(nearestRouteToAList))
                #capacityRouteB=expand last route (add with new nearest cust) - the clsuter of the nearest route to A (but exclude that cust)
                capacityRouteB=abs(self.totalDemandInRoute(lastRouteExpandList)-self.totalDemandInRoute(afterRemoveCustRoute))
                '''
                (3b) Total capacity difference (delta) is less than before.
                If the expand route capacity B (capacityRouteB) less than old route (capacityRouteA), 
                then we'll expand the nearest cust to the last route, else remain it -> To make sure we'll get the min capacity diff
                '''
                if(capacityRouteB<=capacityRouteA):
                    #Update the currentLastRouteDemand
                    currentLastRouteDemand+=self.DEMAND[cust]
                    
                    #Remove the cust from the current cluster and replace the latest cluster in the self.finalCluster
                    for idx,cluster in enumerate(self.finalCluster):
                        if cust in cluster:
                            self.finalCluster[idx]=[i for i in cluster if i!=cust]
                    
                    #Expand last route
                    lastRoute.append(cust)
                    self.finalCluster[-1]=lastRoute
                            
        print(f'Step 2, balancing cluster: {self.finalCluster}') 
        return self.finalCluster    

    def clustering(self):
        '''
        Step 1:
        Implement the nearest neighbor method for clustering such that the cluster centers are uniformly distributed
        while the maximum power is not exceeded without exceeding the maximum carrying capacity of Ev Pmax.
        '''
        #List used to keep track uncluster customer node (Exclude depot+charging stations) eg:2-22
        unclusterNodeList=list(self.NODE.keys())[1:self.NUM_OF_CUSTOMERS+1] 

        #2D list to store each cluster members
        finalCluster=[] 

        #1.Randomly select a customer as seedPoint in a cluster (Customer node range 2-22)
        seedPoint=random.randint(2,self.NUM_OF_CUSTOMERS+1) 

        #Remove seedPoint from unclusterNodeList
        unclusterNodeList.remove(seedPoint)

        #currentCluster: Keep track the current cluster's members based on the seedPoint given (seedPoint+cluster members)
        currentCluster=[seedPoint] 

        #currentCapacityDemand: Keep track of the current capacity demands of the given customer node 
        currentCapacityDemand=self.DEMAND[seedPoint]

        #Nearest distance from seedPoint, return list of customers based on the shortest distance
        availableCustNode=self.nearestCustomers(seedPoint)

        idx=0
        while(len(availableCustNode)>0):
            #2.2-Check whether if the currentCapacity <= maxCapacity, if yes then only update currentCapacity + remove node from unclusterNodeList
            cust=availableCustNode[idx]

            if(currentCapacityDemand+self.DEMAND[cust] <= self.MAX_CAPACITY):
                #Update currentCapacity
                currentCapacityDemand+=self.DEMAND[cust]  
                    
                #Add node into currentCluster
                currentCluster.append(cust)
                    
                #Remove members from availableNode
                availableCustNode.remove(cust)
            else:
                #Add currentCluster to finalCluster
                finalCluster.append(currentCluster)

                #Create a new cluster based on new seedPoint
                while(True):
                    seedPoint=random.randint(2,self.NUM_OF_CUSTOMERS+1)
                        
                    #Make sure the seedPoint is in the availableCustNode
                    if (seedPoint in availableCustNode):
                        availableCustNode.remove(seedPoint)
                        
                        #Create new currentCluster
                        currentCluster=[seedPoint]
                            
                        #Update currentCapacityDemand based on the newSeedPoint
                        currentCapacityDemand=self.DEMAND[seedPoint]
                        break
                        
        #Update the last currentCluster to finalCluster
        finalCluster.append(currentCluster) 
        print(f'Step 1, clustering cluster: {finalCluster}')
        return finalCluster

    def read_problems(self,filename:str):
        '''Read the problem instance and generate the initial object vector'''
        with open(filename,'r') as f:
            data=f.read().splitlines()  

        #Store NODE and DEMAND as a dictionary {number:value}
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
                self.NUM_OF_CUSTOMERS=self.PROBLEM_SIZE-1
                self.ACTUAL_PROBLEM_SIZE=self.PROBLEM_SIZE+self.NUM_OF_STATIONS
                idx+=1
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    node_data=data[idx+i]
                    node_data=node_data.split(' ')
                    
                    #Save node as tuple (index,x,y)
                    self.NODE[int(node_data[0])]=(int(node_data[1]),int(node_data[2]))

            if (record[0]=='DEMAND_SECTION'):
                idx+=1
                for i in range(self.PROBLEM_SIZE):
                    self.DEMAND[int(data[idx+i].split(' ')[0])]=int(data[idx+i].split(' ')[1])
            
            if (record[0]=='STATIONS_COORD_SECTION'):
                self.STATIONS_COORD_SECTION=[]
                idx+=1
                for i in range(self.ACTUAL_PROBLEM_SIZE-self.PROBLEM_SIZE):
                    self.STATIONS_COORD_SECTION.append(int(data[idx+i].strip()))
                     
        #Generate distance matrix
        self.distanceMatrix=self.generate_2D_distance_matrix()

        self.displayParam(False)
    
    def displayParam(self,display:bool):
        if display==True:
            print(f'OPTIMAL_VALUE          : {self.OPTIMUM}')
            print(f'MIN_VEHICLES           : {self.MIN_VEHICLES}')
            print(f'PROBLEM_SIZE           : {self.PROBLEM_SIZE}')
            print(f'NUM_OF_STATIONS        : {self.NUM_OF_STATIONS}')
            print(f'MAX_CAPACITY           : {self.MAX_CAPACITY}')
            print(f'BATTERY_CAPACITY       : {self.BATTERY_CAPACITY}')
            print(f'ENERGY_CONSUMPTION     : {self.ENERGY_CONSUMPTION}')
            print(f'NUM_OF_CUSTOMERS       : {self.NUM_OF_CUSTOMERS}')
            print(f'ACTUAL_PROBLEM_SIZE    : {self.ACTUAL_PROBLEM_SIZE}')
            print(f'NODE                   : {self.NODE}')
            print(f'DEMAND                 : {self.DEMAND}')
            print(f'STATIONS_COORD_SECTION : {self.STATIONS_COORD_SECTION}')


    def __init__(self):
        random.seed(42)
        filenames=['evrp-benchmark-set/E-n22-k4.evrp']
        self.read_problems(filenames[0])
        self.finalCluster=self.clustering()
        self.finalCluster=self.balancingApproach()
        # print(self.finalCluster)


if __name__ == "__main__":       
    EVRP()
  


    