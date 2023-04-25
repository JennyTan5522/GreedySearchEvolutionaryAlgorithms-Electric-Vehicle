import copy
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class EVRP:
    '''Implementaion of the electric vehicle routing'''

    def __init__(self,filename,display:bool,random_state=42):
        random.seed(random_state)
        self.read_problems(filename)
        if display:
            self.displayParam()

    def swap(self,route,i,j):
        route[i],route[j]=route[j],route[i]
        return route
    
    def plotRoute(self,route):
        x=[]
        y=[]
        for cluster in route:
            for node in cluster:
                x.append(self.NODE[node][0])
                y.append(self.NODE[node][1])
        plt.scatter(x,y)
        plt.plot(x,y)
        plt.show()

    def calculateTotalDistance(self,routes:list):
        '''Calculate total distance of the particular route'''
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
        return distance.euclidean(self.NODE[node1],self.NODE[node2])
    
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
    
    def nearestChargingStations(self,customer:int):
        '''Find the nearest charging stations based on the current customer's location'''
        sortCustIdxMatrix=np.argsort(self.distanceMatrix[customer-1][self.NUM_OF_CUSTOMERS+1:]) 
        return np.array(list(range(self.NUM_OF_CUSTOMERS+1,self.ACTUAL_PROBLEM_SIZE+1)))[sortCustIdxMatrix] 

    def findChargingStation(self, balancedCluster:list):
        '''
        If the vehicle cannot find such a set of charging stations, it will go back to find a station from customer ci-1 to ci 
        and repeat until it can find a satisfactory station or back to depot.
        '''
        #Complete route include depot + customers + charging stations
        completeRoute = []
        balancedClusterComplete = copy.deepcopy(balancedCluster)
        
        for route in balancedClusterComplete:
            #Add depot to front and end,depot location=1
            route.insert(0,1)
            route.insert(len(route),1)
            
            #Index to keep track each customer in the current route
            idx = 0
            #Current route that include depot+customer
            finalRoute = []
            #A list of battery level at each station to keep track the consumption rate
            batteryLvlAtEachStation = []
            forceToInsert = False
            
            #Loop through each customer to check whether there is enough battery level to move from customer ci to ci+1
            while (idx<len(route)):
                #Idx==0 means that it is starts from the first customer, then the currentBatteryLevel is full
                if idx == 0:  
                    currentBatteryLvl = self.BATTERY_CAPACITY
                    #Update finalRoute and batteryLvlAtEachStation
                    finalRoute.append(route[idx])
                    batteryLvlAtEachStation.append(currentBatteryLvl)
                    #Move to next customer
                    idx+=1 
                #From second customer check whether currentBatteryLvl can support battery consumption from ci to ci+1 
                elif (idx >= 0):
                    #Calc battery consumption from prev to curr station
                    batteryConsumption = self.distanceMatrix[finalRoute[-1]-1][route[idx]-1]*self.ENERGY_CONSUMPTION
                    
                    #If current battery level is lower than battery consumption then it means that it have the enuf energy to move to next customer
                    if (batteryConsumption < currentBatteryLvl) and (not forceToInsert):
                        #Deduct current battery level
                        currentBatteryLvl = batteryLvlAtEachStation[-1] - batteryConsumption
                        #Update finalRoute and batteryLvlAtEachStation
                        finalRoute.append(route[idx])
                        batteryLvlAtEachStation.append(currentBatteryLvl)
                        #Move to next customer
                        idx+=1 
                    #If not enough battery to move from ci to ci+1    
                    else:   
                        #Find available charging stations from far to nearest 
                        stations=list(reversed(list(self.nearestChargingStations(route[idx]))))
                        
                        #Loop through each station
                        for i, s in enumerate(stations):
                            #Find battery consumption from customer ci-1 to ci
                            batteryConsumption=self.distanceMatrix[finalRoute[-1]-1][s-1]*self.ENERGY_CONSUMPTION
                            #Find a set of charging stations that are enough to support currentBatteryLvl
                            if batteryConsumption < currentBatteryLvl:
                                stations=stations[i:]
                                break
                        #No available stations found to suport currentBatteryLvl
                        else:  
                            stations = []
                        
                        #List of stations found to support currentBatteryLvl
                        if len(stations) > 0:  
                            #Update list of charging stations to finalRoute
                            finalRoute=finalRoute+stations
                            #Update battery capacity based on the available charging stations
                            for _ in range(len(stations)):
                                batteryLvlAtEachStation.append(self.BATTERY_CAPACITY)
                            
                            #Update current customer to finalRoute
                            finalRoute.append(route[idx])
                            #Find the batteryConsumption of LastChargingStation and the next customer
                            batteryConsumption = self.distanceMatrix[stations[-1]-1][route[idx]-1]*self.ENERGY_CONSUMPTION  
                            #Deduct battery level
                            currentBatteryLvl = batteryLvlAtEachStation[-1] - batteryConsumption
                            #Update currentBatterLvl to batteryLvlEachStation
                            batteryLvlAtEachStation.append(currentBatteryLvl)
                            forceToInsert = False
                            #Move to next station
                            idx+=1 

                        #No available stations found
                        else: 
                            finalRoute.pop()
                            batteryLvlAtEachStation.pop()
                            currentBatteryLvl = batteryLvlAtEachStation[-1]
                            forceToInsert = True
                            idx-=1
                else:
                    finalRoute = route
                    
            # print("finish insert: ", finalRoute)
            #Append finalRoute to complete Route
            completeRoute.append(finalRoute)
        return completeRoute
    
    #Swap last or last 2
    def local2Opt(self,existingRoute:list):
        existingDistance=self.calculateTotalDistance(existingRoute)
        stop=False
        while(stop==False):
            stop=True
            for i in range(len(existingRoute)):
                for j in reversed(range(i+1,len(existingRoute))):
                    #print(f'i is {i}:{existingRoute[i]}; j is {j}:{existingRoute[j]}')
                    #print(f'Existing route : {existingRoute}')
                    
                    #Exchange i and j location
                    newRoute=existingRoute.copy()
                    newRoute=self.swap(newRoute,i,j)
                    
                    #Find the total distance of newRoute
                    newRouteDistance=self.calculateTotalDistance(newRoute)
                    
                    #print(f'New route : {newRoute}, new distance : {newRouteDistance}')
                
                    #If the total distance of existingDistance is better than before then we swap the target
                    if (newRouteDistance < existingDistance):
                        #print('===SWAP===')
                        #Update currentRoute to newRoute
                        existingRoute=newRoute.copy()
                        #Update currentDistance to newDistance
                        existingDistance=newRouteDistance
                        #print(existingRoute,existingDistance)
                        stop=False
                    #print('\n---------------------------------------')
            #print('====================')
        return existingRoute

    def balancingApproach(self,initialCluster):
        '''
        Step 2:
        Customers assigned in the last route are the non-clustered customers,
        the are remanining customers so their geo locations are not close-set.
        The customers on the last route will be less than other routes, even a single customer.
        Use a Balanced Approach to ensure the distance of customers and increase the number of customers in the last route.
        '''
        #1. Randomly select a customer (customer A form the last route).
        lastRoute=initialCluster[-1]
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
            #TODO if a and b codition tgt, else straight break the loop
            '''
            (3a) Initial sum of the total capacity of the route and the capacity of the chosen the EV's 
                maximal carrying capacity Pmax customers does not exceed. 
            '''
            if (self.DEMAND[cust]+currentLastRouteDemand<self.MAX_CAPACITY):
                #The list of that particular cust route with the absence of that cust
                nearestRouteToAList=[cluster for cluster in initialCluster if cust in cluster][0]
                
                #lastRouteNewCustList=last route + 1 new nearest customer
                lastRouteExpandList=lastRoute+[cust]
                afterRemoveCustRoute=[i for i in [cluster for cluster in initialCluster if cust in cluster][0] if i!=cust]
                
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
                    for idx,cluster in enumerate(initialCluster):
                        if cust in cluster:
                            initialCluster[idx]=[i for i in cluster if i!=cust]
                    
                    #Expand last route
                    lastRoute.append(cust)
                    initialCluster[-1]=lastRoute
                            
        # print(f'Step 2, balancing cluster: {initialCluster}') 
        return initialCluster   

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

                #Create a new cluster based on new seedPoint #TODO put one while loop 
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
        #print(f'Step 1, clustering cluster: {finalCluster}')
        return finalCluster

    def read_problems(self,filename):
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

    def displayParam(self):
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

    #     self.finalCluster=self.clustering()
    #     self.finalCluster=self.balancingApproach()
    #     for i in range(len(self.finalCluster)-1):
    #         self.finalCluster[i]=self.local2Opt(self.finalCluster[i])
        
    #     print(self.finalCluster)
    #     print(f'Step 3, local 2-opt cluster: {self.finalCluster}') 


if __name__ == "__main__":       
    EVRP()
  


    