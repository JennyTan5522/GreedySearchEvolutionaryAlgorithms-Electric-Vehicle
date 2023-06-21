import copy
import random
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
import matplotlib.pyplot as plt

class EVRP:
    '''
    Implementaion of the electric vehicle routing
    '''

    def __init__(self,filename,display:bool,random_state=42):
        random.seed(random_state)
        self.read_problems(filename)
        if display:
            self.displayParam()

    def read_problems(self,filename):
        '''
        Description: This function reads the problem instance and generate the initial object vector.
                        NODE and DEMAND stored as dictionary; CUSTOMERS and STATIONS stored as list
                        NODE  ={customer node: (x,y)}
                        DEMAND={customer node: customer demand}
                        Generate distance matrix for each node.

        Input: (filename) - filename for read

        Output: None
        '''
        print(filename)
        with open(filename,'r') as f:
            data=f.read().splitlines()  
        
        self.PROBLEM_NAME=filename.split('/')[1]

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
                self.NUM_OF_CUSTOMERS=self.PROBLEM_SIZE-1
                self.CUSTOMERS=[]
                for i in range(1,self.NUM_OF_CUSTOMERS+1):
                    self.CUSTOMERS.append(i+1)

            if (record[0]=='STATIONS'):
                self.NUM_OF_STATIONS=int(record[1].strip())

            if (record[0]=='CAPACITY'):
                self.MAX_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CAPACITY'):
                self.BATTERY_CAPACITY=int(record[1].strip())

            if (record[0]=='ENERGY_CONSUMPTION'):
                self.ENERGY_CONSUMPTION=float(record[1].strip())

            if (record[0]=='NODE_COORD_SECTION'):
                self.ACTUAL_PROBLEM_SIZE=self.PROBLEM_SIZE+self.NUM_OF_STATIONS
                idx+=1
                for i in range(self.ACTUAL_PROBLEM_SIZE):
                    node_data=data[idx+i]
                    node_data=node_data.split(' ')
                    self.NODE[int(node_data[0])]=(int(node_data[1]),int(node_data[2]))

            if (record[0]=='DEMAND_SECTION'):
                idx+=1
                for i in range(self.PROBLEM_SIZE):
                    if int(data[idx+i].split(' ')[1])==0:
                        self.DEPOT=int(data[idx+i].split(' ')[0])
                    self.DEMAND[int(data[idx+i].split(' ')[0])]=int(data[idx+i].split(' ')[1])
                
            if (record[0]=='STATIONS_COORD_SECTION'):
                self.STATIONS=[]
                idx+=1
                for i in range(self.ACTUAL_PROBLEM_SIZE-self.PROBLEM_SIZE):
                    self.STATIONS.append(int(data[idx+i].strip()))

        self.distanceMatrix=self.generate_2D_distance_matrix()

    def displayParam(self):
        '''
        Description: Print the problem objects of the current file.

        Input: None

        Output: None
        '''
        print(f'PROBLEM NAME           : {self.PROBLEM_NAME}')
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
        print(f'DEPOT                  : {self.DEPOT}')
        print(f'CUSTOMERS              : {self.CUSTOMERS}')
        print(f'STATIONS               : {self.STATIONS}')

    def print_solution(self,subroute:list,solutionName:str):
        '''
        Description: Print the details of each solution.
                     Subroute (idx) (distance, customer demand in the subroute): depot | node1 | node2 | node3 | depot

        Input: (subroute)     - List containing the sequence of nodes visited
               (solutionName) - current solution name for printing purpose. Eg: Initial solution, Local 2-opt solution..

        Output: None
        '''
        print('-'*70)
        print(solutionName)
        
        route_distance=0.0
        copy_route=copy.deepcopy(subroute)
        for subroute in copy_route:
            if subroute[0]!=1:
                subroute.insert(0,1)
                subroute.insert(len(subroute),1)
            route_distance+=float(self.calcSubrouteDistance(subroute))
    
        print('Route length: '+str(f'{route_distance:.2f}'))
        for i, subroute in enumerate(copy_route):
            subroute_details=''
            for node in subroute:
                subroute_details+=f'{str(node):2}'+' | '
            print(f'Subroute {i}'+ ' ('+f'{self.calcSubrouteDistance(subroute):7.2f}' + f',{str(self.calcSubrouteDemand(subroute)):4}): {subroute_details}')
        
        print('-'*70)
   
    def generate_2D_distance_matrix(self):
        '''
        Description: This function is to generate 2D distance matrix and find the distance between 2 points

        Input: None

        Output: Return a list of nearest customers based on ascending order
        '''
        matrix=np.zeros((len(self.NODE),len(self.NODE)))
        distanceMatrix=self.calcMatrixDistance(matrix)
        return distanceMatrix 
    
    def calcNodeDistance(self,from_node:int,to_node:int): 
        '''
        Description: Compute euclidean distance of 2 nodes.

        Input: (from_node) - Start from which node
               (to_node)   - End to which node

        Output: Return distance of 2 nodes
        '''
        return distance.euclidean(self.NODE[from_node],self.NODE[to_node])
    
    def calcMatrixDistance(self,matrix):
        '''
        Description: Compute the distance between all nodes

        Input: (matrix) - 2D matrix containing nodes/coords

        Output: Return matrix
        '''
        for i in range(self.ACTUAL_PROBLEM_SIZE): 
            for j in range(self.ACTUAL_PROBLEM_SIZE):
                matrix[i][j]=self.calcNodeDistance(i+1,j+1)
        return matrix
    
    def calcSubrouteDistance(self,subroute:list):
        '''
        Description: Calculate total distance of the particular route

        Input: (subroute) - 1D list containing the sequence of nodes visited, eg: [1,2,3,4,1]

        Output: Return distance of the subroute
        '''
        return np.sum([self.distanceMatrix[subroute[i]-1][subroute[i+1]-1] for i in range(len(subroute)-1)])

    def calcRouteDistance(self,complete_route:list):
        '''
        Description: Calculate total distance of the complete route

        Input: (complete_route) - 2D list containing few subroutes, eg: [[1,2,3,4,1],[1,5,6,7,8,1],[1,9,10,1]]

        Output: Return distance of the complete route
        '''
        return np.sum([self.calcSubrouteDistance(subroute) for subroute in complete_route])

    def calcSubrouteDemand(self,subroute:list):
        '''
        Description: Find the total demand of the customers in that particular route

        Input: (subroute) - List containing the sequence of nodes visited

        Output: Return total demand in the subroute
        '''
        totalDemand=0
        for cust in subroute:
            if cust in self.STATIONS:
                continue
            totalDemand+=self.DEMAND[cust]
        return totalDemand

    def calcBatteryConsumption(self,from_node:int,to_node:int):
        '''
        Description: Calculate battery consumption of each route
                     Battery consumption = (cunsumption rate of the EV) * (distance of 2 nodes)

        Input: (from_node) - Start from which node
               (to_node)   - End to which node

        Output: Return battery consumption from particular node to particular node
        '''
        return self.ENERGY_CONSUMPTION * self.distanceMatrix[from_node-1][to_node-1]
    
    def capacityValidation(self,subroute:list):
        '''
        Description: This function used to ensure the customer capacity in subroute is valid (not exceed the max customer capcity)

        Input: (subroute) - List containing the sequence of nodes visited

        Output: Return True if cust demand < max cap, False if cust demand > max cap
        '''
        if self.calcSubrouteDemand(subroute) > self.MAX_CAPACITY:
            return False
        return True

    def routeCapacityValidation(self,complete_route:list):
        '''
        Description: This function used to ensure the customer capacity in complete route is valid (not exceed the max customer capcity)

        Input: (complete_route) - 2D list containing few subroutes

        Output: Return True if cust demand < max cap, False if cust demand > max cap
        '''
        for subroute in complete_route:
            if self.capacityValidation(subroute)==False:
                return False
        return True
    
    def batteryValidation(self,subroute:list):
        '''
        Description: This function used to ensure the battery capacity in subroute is valid (not exceed the max battery capcity)
                     The battery capacity is full and battery consumption is 0 when depart from the depot or after insert charging station(s)

        Input: (subroute) - 1D list containing the sequence of nodes visited

        Output: Return True if battery capacity is valid, False if battery capacity invalid (<0)
        '''
        currentBatteryLevel = self.BATTERY_CAPACITY
        
        for idx in range(len(subroute)-1):
            if subroute[idx] in self.STATIONS or subroute[idx] == self.DEPOT:
                currentBatteryLevel = self.BATTERY_CAPACITY
             
            batteryConsumption = self.calcBatteryConsumption(subroute[idx], subroute[idx+1])
            currentBatteryLevel -= batteryConsumption 

            if (currentBatteryLevel < 0):
                return False

        return True 
   
    def initial_route_validation(self,initalRoute:list):
        '''
        Description: Check for the initial cluster is valid or not 
                        1. Each customer only occur once
                        2. All customer is in the initial route
                        3. Total battery in the subroute does not exceed MAX_BATTERY

        Input: (initialRoute) - 2D list contain a list of subroutes (Not include depot + charging stations)
        '''
        flatten_route=[cust for subroute in initalRoute for cust in subroute]
        if len(flatten_route) != len(set(flatten_route)):
            return False
        
        if set(flatten_route) ^ set(self.CUSTOMERS) != set():
            return False
        
        for subroute in initalRoute:
            if self.capacityValidation(subroute)==False:
                return False
        
        return True
   
    def complete_route_validation(self,completeRoute:list):
        '''
        Description: Check if a given complete route is valid or invalid
                     1. Check each customer only occur once
                     2. Check given route is all in customer list
                     3. Check whether each subroute start and end with depot
                     4. Check capacity demand and battery in the subroute
        
        Input: (completeRoute) - 2D complete route, inside contain a list of subroutes (Include depot + charging stations)
        
        Output: Returns true if the complete route is valid, returns false if invalid
        
        '''
        flatten_route=[cust for subroute in completeRoute for cust in subroute[1:-1]]
   
        route_no_charging=[]

        for node in flatten_route:
            if node not in self.STATIONS:
                route_no_charging.append(node)

        if len(route_no_charging)!=len(set(route_no_charging)):
            print('Route contain duplicate customers!')
            return False
             
        if set(route_no_charging) ^ set(self.CUSTOMERS) != set():
            print('Route does not contain all customers!')
            return False
            
        for subroute in completeRoute:
            if subroute[0]!=self.DEPOT and subroute[-1]!=self.DEPOT:
                print('Route must start and end with depot!')
                return False
            if self.capacityValidation(subroute)==False:
                print('Route capacity exceed max demand capacity!')
                return False
            if self.batteryValidation(subroute)==False:
                print('Route capacity exceed max battery capacity!')
                return False
                 
        return True
    
    def nearestCustomers(self,customer:int):
        '''
        Description: This function used to find the nearest customers based on the input node

        Input: (customer) - customer node 

        Output: Return a list of nearest customers based on ascending order
        '''
        sortCustIdxMatrix=np.argsort(self.distanceMatrix[customer-1][:self.NUM_OF_CUSTOMERS+1]) 
        return [sortCustIdxMatrix[i]+1 for i in range(1,sortCustIdxMatrix.shape[0]) if sortCustIdxMatrix[i]!=0]
      
    def clustering(self):
        '''
        Description: Implement the nearest neighbor method for clustering such that the cluster centers are uniformly distributed
                     while the maximum power is not exceeded without exceeding the maximum carrying capacity of Ev Pmax.

                     1. Choose seedpoint
                     2. Find the nearest neigbors based on the seedpoint
                     3. Add seedpoint into current cluster
                     4. Add seepoint's capacity into current capacity demand
                     5. If the current demand + current customer capacity demand not exceed the MAX CAPACITY, then add into the current cluster,
                        else if exceed add the current cluster into final cluster and start the new cluster 

        Output: Complete cluster
        '''
        flag_newCluster=True
        visit_cust=[] #visited customer
        finalCluster=[] 

        while(True):
            if flag_newCluster==True:
                seedPoint=random.choice(list(set(visit_cust) ^ set(self.CUSTOMERS)))
                currentCluster=[seedPoint] 
                visit_cust.append(seedPoint)
                currentCapacityDemand=self.DEMAND[seedPoint]
                nearCust=self.nearestCustomers(seedPoint)
                nearCust=[cust for cust in nearCust if cust not in visit_cust] 
                
            if nearCust==[]:
                break

            cust=nearCust[0]
            
            if(currentCapacityDemand + self.DEMAND[cust] <= self.MAX_CAPACITY):
                currentCapacityDemand+=self.DEMAND[cust]  
                currentCluster.append(cust)
                visit_cust.append(cust)
                flag_newCluster=False
                nearCust.remove(cust)
            else:
                finalCluster.append(currentCluster) 
                flag_newCluster=True
        
        finalCluster.append(currentCluster)
        return finalCluster
 
    def balancingCluster(self, initialCluster:list):
        '''
        Description : Customers assigned in the last route are the non-clustered customers, they are remanining customers so their geo locations are not close-set.
                      The customers on the last route will be less than other routes, even a single customer.
                      Use a Balanced Approach to ensure the distance of customers and increase the number of customers in the last route.

                      Eg: customers=[2,3,4,5,6,7,8,9], initial cluster=[[2,3],[4,5,6,7],[8,9]]

                      1. Randomly select a customer from last route, eg:8
                      2. Select the closest customers based on customer chosen at step 1, eg: [2,5,6,7,9,3,4,5], 2 is the nearest to 8
                      3. Loop the closest customer, append the closer customer into the last route
                         subrouteA=[8,9,2]
                         - if capacity of subrouteA does not exceed the max capacity, then find subrouteB 
                           ( a ) First, find the closer customer's cluster, eg: [2,3]
                           ( b ) Then remove the closer customer from that particular cluster, eg: subrouteB=[3]
                           ( c ) new = capacity([8,9,2]) - capacity([3]) > old = capacity([8,9]) - capacity ([2,3]), then update the last route 
        '''
        lastRoute=initialCluster[-1]

        near_cust=self.nearestCustomers(random.choice(lastRoute))
        near_cust=[customer for customer in near_cust if customer not in lastRoute]
            
        for cust in near_cust:
            subrouteA=lastRoute+[cust]
            capacity_A=self.calcSubrouteDemand(subrouteA)

            if capacity_A <= self.MAX_CAPACITY: 
                for idx,cluster in enumerate(initialCluster):
                    if cust in cluster:
                        cust_subroute=cluster
                        cust_idx=idx
                        break
                    
                subrouteB=copy.deepcopy(cust_subroute)
                subrouteB.remove(cust)

                capacity_old=abs(self.calcSubrouteDemand(lastRoute)-self.calcSubrouteDemand(cust_subroute))
                capacity_AB=abs(capacity_A-self.calcSubrouteDemand(subrouteB))

                if capacity_AB <= capacity_old:
                    initialCluster[-1]=subrouteA
                    initialCluster[cust_idx]=subrouteB
                    lastRoute=initialCluster[-1]
            else:
                break
        
        return initialCluster
 
    def swapTwo(self,subroute,i,j):
        '''
        Description: This function is to swap the location of two nodes (For local 2-opt purpose)

        Input: (subroute) - List containing the sequence of nodes visited
                   (i, j) - Index for swap

        Output: Return new subroute after swap 2 nodes
        '''
        subroute[i],subroute[j]=subroute[j],subroute[i]
        return subroute
    
    def ls_2opt(self,subroute:list):
        '''
        Description: This function starts with a subroute and then repeatedly swap 2 nodes
                     If the distance of the swapped node less than original subroute, then replace with the new subroute

        Input: (subroute) - List containing the sequence of nodes visited

        Output: Return a list of subroute with the shortest distance

        '''
        existingDistance=self.calcSubrouteDistance(subroute)
        for i in range(len(subroute)):
            for j in range(i+1,len(subroute)):
                newRoute=copy.deepcopy(subroute)
                newRoute=self.swapTwo(newRoute,i,j)                   
                newRouteDistance=self.calcSubrouteDistance(newRoute)
                if (newRouteDistance < existingDistance):
                    subroute=copy.deepcopy(newRoute)
                    existingDistance=newRouteDistance

        return subroute

    def swapThree(self,subroute,a,c,e):
        '''
        Description: This function recreates a route by disconnecting and reconnecting 3 edges ab, cd
        and ef (such that the result is still a complete and feasible subroute).
        
        Input: (subroute)  - List containing the sequence of nodes visited
            (a)            - Position of the first node in the list
            (c)            - Position of the second node in the list
            (e)            - Position of the third node in the list
        
        Output (subsubroute cost) - Total cost of the input subroute
        '''
        #Nodes are sorted to allow a simpler implementation
        a,c,e = sorted([a,c,e])
        b,d,f=a+1,c+1,e+1
        
        new_subroute=[]
        
        #Four different reconnections of subroutes are considered
        new_subroute.append(subroute[:a+1] + subroute[b:c+1][::-1] + subroute[d:e+1][::-1] + subroute[f:])
        new_subroute.append(subroute[:a+1] + subroute[d:e+1] + subroute[b:c+1] + subroute[f:])
        new_subroute.append(subroute[:a+1] + subroute[d:e+1] + subroute[b:c+1][::-1] + subroute[f:])
        new_subroute.append(subroute[:a+1] + subroute[d:e+1][::-1] + subroute[b:c+1] + subroute[f:])

        subroutes_distance=[self.calcSubrouteDistance(sub) for sub in new_subroute]
        
        return new_subroute[np.argsort(subroutes_distance)[0]]

    def ls_3opt(self,subroute:list):
        '''
        Description: This function applies the 3-opt to find a new subroute with a lower cost than the 
                     input subroute. The algorithm scans all nodes a,c,e and swaps three edges connecting the current 
                     subroute. All four different reconnections of the three edges are attempted and the algorithm is 
                     stopped at the first improvement. If an improvement is found the subroute is swapped and the new 
                     subroute is used in the evaluation of further improvements. The algorithm stops when no further 
                     improvement can be found by swapping three edges considering one of the 4 possibilities.
                    
        Input: (subroute) - List containing the sequence of nodes visited
        
        Output: (subroute) - List containing the new sequence of nodes visited
        '''
        size=len(subroute)
        existing_distance=self.calcSubrouteDistance(subroute)
        improve=0
        while(improve<=0):
            for a in range(1,size-2):
                for c in range(a+1,size-1):
                    for e in range(c+1, size):
                        new_subroute=self.swapThree(subroute,a,c,e)
                        new_distance=self.calcSubrouteDistance(new_subroute)
                        if (new_distance < existing_distance):
                            subroute=copy.deepcopy(new_subroute)
                            existing_distance=new_distance
                            improve=0
            improve+=1
        return subroute

    def nearestChargingStations(self,customer:int):
        '''
        Description: Find the nearest charging stations based on the current customer's location
        
        Input: (customer) - customer node 

        Output: Return the nearest charging station with the input customer node
        '''
        sortCustIdxMatrix=np.argsort(self.distanceMatrix[customer-1][self.NUM_OF_CUSTOMERS+1:]) 
        return np.array(list(range(self.NUM_OF_CUSTOMERS+2,self.ACTUAL_PROBLEM_SIZE+1)))[sortCustIdxMatrix] 
    
    def nearest_station(self,from_cust:int,to_cust:int,current_energy:float):
        '''
        Description: This function is used to find the nearest charging station from one customer to another customer.
                     When find the nearest charging station make sure that the consumption from customer to charging station is enough

        Output: Return best station
        '''
        best_station=-1
        stations=list(self.nearestChargingStations(to_cust))

        for s in stations:
            if self.calcBatteryConsumption(from_cust,s) < current_energy:
                best_station=s
                break
        
        return best_station
    
    def findChargingStation(self, orginalSubroute:list,routeIdx:int=None,print_results=False):
        '''
        Description: This function is used to find available charging stations when the battery capacity demand is not enough to move from one customer to another customer.

        Data:
        subroute         : Incomplete subroute that does not contain depot and charging stations.
        completeSubroute : Complete subroute that contains depot and charging stations.
        batteryLvl       : A list to keep track of the battery consumption when moving from one customer to another customer.
        Idx              : Index to keep track of each customer in the current subroute.

        solution format : (from node, to node) -> (current battery level, battery consumption, current battery level after battery consumption)

        Details:
        1. Loop through each customer to check whether there is enough battery level to move from customer ci to ci+1

        2. If it is the depot or first customer (idx==0), the currentBatteryLvl is the BATTERY_CAPACITY
            - Append the depot to the complete subroute 
            - Append currentBatteryLvl into batteryLevel

        3. From the second customer (idx>=0), check whether currentBatteryLvl can support battery consumption from the current customer to another customer.
            If the battery consumption is less than currentBatteryLvl 
                - Append the current customer to the  complete subroute 
                - Append currentBatteryLvl into batteryLevel

            Else if the battery level is not enough to support the current customer
                - Find the nearest charging station
                - If available: 
                    - Append the charging station to the complete subroute 
                    - Append BATTERY_CAPACITY into batteryLevel
                    - Append the customer to the complete subroute
                    - Append the currentBatteryLvl (from charging station to customer)

                - If no charging station is available:
                    - Remove the last customer from the complete subroute
                    - Remove the last battery consumption from the battery level
                    - Update the currentBatteryLvl to the last battery consumption in currentBatteryLvl
                    - Update force to insert 
                    - Then find a new charging to insert to the current customer

        '''
        subroute=copy.deepcopy(orginalSubroute)
        completeSubroute = []
        batteryLevel = []
        forceToInsert = False
        idx = 0
        
        #Add depot to front and back
        subroute.insert(0,self.DEPOT)
        subroute.insert(len(subroute),self.DEPOT)
      
        solution='Subroute '+str(routeIdx)+': \n'
        
        while (idx < len(subroute)):
            if(idx<0):
                return -1
            
            if idx == 0:  
                currentBatteryLvl = self.BATTERY_CAPACITY
                completeSubroute.append(subroute[idx])
                batteryLevel.append(currentBatteryLvl)
                idx+=1 

            elif (idx >= 0):
                batteryConsumption = self.calcBatteryConsumption(completeSubroute[-1],subroute[idx])

                if (batteryConsumption < currentBatteryLvl) and (not forceToInsert): #If enough battery
                    solution+=f' ({subroute[idx-1]:2},{subroute[idx]:2})' + f' -> ({currentBatteryLvl:5.2f}, {batteryConsumption:5.2f}'
                    currentBatteryLvl = batteryLevel[-1] - batteryConsumption
                    completeSubroute.append(subroute[idx])
                    batteryLevel.append(currentBatteryLvl)
                    solution+=f', {currentBatteryLvl:.2f})\n'
                    idx+=1 
                    
                else:   #If not enough battery   
                    best_station=self.nearest_station(completeSubroute[-1],subroute[idx],currentBatteryLvl)
                    solution+="      S'"+f"    ({best_station})" 
                 
                    if best_station!=-1:  #If find available charging station
                        completeSubroute=completeSubroute+[best_station]
                        batteryLevel.append(self.BATTERY_CAPACITY)
                        completeSubroute.append(subroute[idx])
                        batteryConsumption = self.calcBatteryConsumption(best_station,subroute[idx])
                        currentBatteryLvl = batteryLevel[-1] - batteryConsumption
                        solution+=f'\n ({best_station:2},{subroute[idx]:2})' + f' -> ({batteryLevel[-1]:5.2f}, {batteryConsumption:5.2f}, {currentBatteryLvl:5.2f})\n'
                        batteryLevel.append(currentBatteryLvl)
                        forceToInsert = False
                        idx+=1 

                    else: #If no available stations found
                        completeSubroute.pop()
                        batteryLevel.pop()
                        currentBatteryLvl = batteryLevel[-1]
                        forceToInsert = True
                        idx-=1
            else:
                completeSubroute = subroute
        
        if print_results==True:    
            print(solution)

        return completeSubroute

    # def insert_energy_stations(self, originalSubroute):
    #     #Complete route include depot + customers + charging stations
    #     subroute=copy.deepcopy(originalSubroute)
        
    #     #Add depot to front and end,depot location=1
    #     subroute.insert(0,1)
    #     subroute.insert(len(subroute),1)
        
    #     remaining_energy = dict()
    #     required_min_energy = dict()
    #     complete_subroute = []
        
    #     depotID = 1
    #     remaining_energy[depotID] = self.BATTERY_CAPACITY
        
    #     # At the current customer node, calculate the minimum energy required for an 
    #     # electric vehicle to reach the nearest charging station.
    #     for node in subroute:
    #         nearest_station = self.nearest_station(node, node, self.BATTERY_CAPACITY)
    #         required_min_energy[node] = self.calcBatteryConsumption(node, nearest_station)
        
    #     i = 0
    #     from_node = subroute[0]
    #     to_node = subroute[1]
        
    #     while i < len(subroute) - 1:
            
    #         # go ahead util energy is not enough for visiting the next node
    #         energy_consumption = self.calcBatteryConsumption(from_node, to_node)
    #         if energy_consumption <= remaining_energy[from_node]:
    #             remaining_energy[to_node] = remaining_energy[from_node] - energy_consumption
    #             complete_subroute.append(from_node)
    #             i += 1
    #             from_node = subroute[i]
    #             if i < len(subroute) - 1:
    #                 to_node = subroute[i + 1]
    #             continue
            
    #         find_charging_station = True
    #         # If there is enough energy, find the nearest station.
    #         # If there is not enough energy to reach the nearest station, go back to the previous node and find the next nearest station from there.
    #         while find_charging_station:
    #             while i > 0 and required_min_energy[from_node] > remaining_energy[from_node]:
    #                 i -= 1
    #                 from_node = subroute[i]
    #                 complete_subroute.pop()
    #             if i == 0:
    #                 return subroute
                
    #             to_node = subroute[i + 1]
    #             best_station = self.nearest_station(from_node, to_node, remaining_energy[from_node])
    #             if best_station == -1:
    #                 return subroute
                
    #             complete_subroute.append(from_node)
    #             from_node = best_station
    #             to_node = subroute[i + 1]
    #             remaining_energy[from_node] = self.BATTERY_CAPACITY
    #             required_min_energy[from_node] = 0
    #             find_charging_station = False                    
                        
    #     complete_subroute.append(subroute[-1])
    #     return complete_subroute

    # def clustering1(self):
    #     '''
    #     Step 1:
    #     Implement the nearest neighbor method for clustering such that the cluster centers are uniformly distributed
    #     while the maximum power is not exceeded without exceeding the maximum carrying capacity of Ev Pmax.
    #     '''
    #     #List used to keep track uncluster customer node (Exclude depot+charging stations) eg:2-22
    #     unclusterNodeList=list(self.NODE.keys())[1:self.NUM_OF_CUSTOMERS+1] 

    #     #2D list to store each cluster members
    #     finalCluster=[] 

    #     #1.Randomly select a customer as seedPoint in a cluster (Customer node range 2-22)
    #     seedPoint=random.randint(2,self.NUM_OF_CUSTOMERS+1) 

    #     #Remove seedPoint from unclusterNodeList
    #     unclusterNodeList.remove(seedPoint)

    #     #currentCluster: Keep track the current cluster's members based on the seedPoint given (seedPoint+cluster members)
    #     currentCluster=[seedPoint] 

    #     #currentCapacityDemand: Keep track of the current capacity demands of the given customer node 
    #     currentCapacityDemand=self.DEMAND[seedPoint]

    #     #Nearest distance from seedPoint, return list of customers based on the shortest distance
    #     availableCustNode=self.nearestCustomers(seedPoint)

    #     idx=0
    #     while(len(availableCustNode)>0):
    #         #2.2-Check whether if the currentCapacity <= maxCapacity, if yes then only update currentCapacity + remove node from unclusterNodeList
    #         cust=availableCustNode[idx]

    #         if(currentCapacityDemand+self.DEMAND[cust] <= self.MAX_CAPACITY):
    #             #Update currentCapacity
    #             currentCapacityDemand+=self.DEMAND[cust]  
                    
    #             #Add node into currentCluster
    #             currentCluster.append(cust)
                    
    #             #Remove members from availableNode
    #             availableCustNode.remove(cust)
    #         else:
    #             #Add currentCluster to finalCluster
    #             finalCluster.append(currentCluster)

    #             #Create a new cluster based on new seedPoint #TODO put one while loop 
    #             while(True):
    #                 seedPoint=random.randint(2,self.NUM_OF_CUSTOMERS+1)
                        
    #                 #Make sure the seedPoint is in the availableCustNode
    #                 if (seedPoint in availableCustNode):
    #                     availableCustNode.remove(seedPoint)
                        
    #                     #Create new currentCluster
    #                     currentCluster=[seedPoint]
                            
    #                     #Update currentCapacityDemand based on the newSeedPoint
    #                     currentCapacityDemand=self.DEMAND[seedPoint]
    #                     break
                        
    #     #Update the last currentCluster to finalCluster
    #     finalCluster.append(currentCluster) 
    #     #print(f'Step 1, clustering cluster: {finalCluster}')
    #     return finalCluster

    # def balancingCluster1(self,initialCluster:list):
    #     '''
    #     Description : Customers assigned in the last route are the non-clustered customers, they are remanining customers so their geo locations are not close-set.
    #                     The customers on the last route will be less than other routes, even a single customer.
    #                     Use a Balanced Approach to ensure the distance of customers and increase the number of customers in the last route.
    #     '''
    #     #1. Randomly select a customer (customer A form the last route).
    #     lastRoute=initialCluster[-1]
    #     customerA=lastRoute[random.randint(0,len(lastRoute)-1)]

    #     #2. Select in turn the customers from other routes such which is the closest to A.
    #     # Find all the nearest distance from A
    #     nearestA=self.nearestCustomers(customerA)
    #     # Make sure that the node taken not from the lastRoute
    #     nearestA=[customer for customer in nearestA if customer not in lastRoute] 

    #     #3. The chosen customers must satisfy the 2 following conditions - 3a and 3b
    #     #Loop all nearest customers from A
    #     currentLastRouteDemand=self.calcSubrouteDemand(lastRoute)
    #     for cust in nearestA:
    #         #TODO if a and b codition tgt, else straight break the loop
    #         '''
    #         (3a) Initial sum of the total capacity of the route and the capacity of the chosen the EV's 
    #             maximal carrying capacity Pmax customers does not exceed. 
    #         '''
    #         if (self.DEMAND[cust]+currentLastRouteDemand<self.MAX_CAPACITY):
    #             #The list of that particular cust route with the absence of that cust
    #             nearestRouteToAList=[cluster for cluster in initialCluster if cust in cluster][0]
                
    #             #lastRouteNewCustList=last route + 1 new nearest customer
    #             lastRouteExpandList=lastRoute+[cust]
    #             afterRemoveCustRoute=[i for i in [cluster for cluster in initialCluster if cust in cluster][0] if i!=cust]
                
    #             #capacityRouteA=old last route - the clsuter of the nearest route to A 
    #             capacityRouteA=abs(self.calcSubrouteDemand(lastRoute)-self.calcSubrouteDemand(nearestRouteToAList))
    #             #capacityRouteB=expand last route (add with new nearest cust) - the clsuter of the nearest route to A (but exclude that cust)
    #             capacityRouteB=abs(self.calcSubrouteDemand(lastRouteExpandList)-self.calcSubrouteDemand(afterRemoveCustRoute))
    #             '''
    #             (3b) Total capacity difference (delta) is less than before.
    #             If the expand route capacity B (capacityRouteB) less than old route (capacityRouteA), 
    #             then we'll expand the nearest cust to the last route, else remain it -> To make sure we'll get the min capacity diff
    #             '''
    #             if(capacityRouteB<=capacityRouteA):
    #                 #Update the currentLastRouteDemand
    #                 currentLastRouteDemand+=self.DEMAND[cust]
                    
    #                 #Remove the cust from the current cluster and replace the latest cluster in the self.finalCluster
    #                 for idx,cluster in enumerate(initialCluster):
    #                     if cust in cluster:
    #                         initialCluster[idx]=[i for i in cluster if i!=cust]
                    
    #                 #Expand last route
    #                 lastRoute.append(cust)
    #                 initialCluster[-1]=lastRoute
                            
    #     # print(f'Step 2, balancing cluster: {initialCluster}') 
    #     return initialCluster   
            
    # def nearest_station(self,from_node,to_node,energy): #energy=battery capacity
    #     min_length=float('inf')
    #     best_station=-1

    #     for v in self.STATIONS:
    #         length=self.distanceMatrix[v-1][to_node-1]
    #         if self.calcBatteryConsumption(from_node,v) <= energy:
    #             if min_length > length:
    #                 min_length=length
    #                 best_station=v

    #     return best_station
    
    def nearest_station_back(self,from_cust:int,to_cust:int,current_energy:float,required_energy:float):
        '''
        Description: This function is used to find the nearest charging station from one customer to another customer.
                    When find the nearest charging station make sure that the consumption from customer to charging station is enough

        Output: Return best station
        '''
        best_station=-1
        stations=list(self.nearestChargingStations(to_cust))

        for s in stations:
            if self.calcBatteryConsumption(from_cust,s) <= current_energy and self.calcBatteryConsumption(s,to_cust) + required_energy < self.BATTERY_CAPACITY:
                best_station=s
        return best_station
    
    def optimize_charging_station(self,subroute:list):
        '''
        Suppose the tour = [1,2,3,4,23,25,8,1]

        1. Check whether the subroute first and last is the depot 
        2. Make sure the tour's remaining energy is not less than 0 from depot..customers..depot
        3. Valid tour? 
        reversed tour = [1,8,23,25,4,3,2,1]
        
        From fist node,
            - if node is customer then calculate energy consumption. (energy consump from [1,8])
            - if node is charging station, calculate the distance from the current customer until the next customer beside charging station. (ori distance=[8,23,25,4])
            - Calculate delta L1= ori distance -  distance between 8 and 4
            - Now from node=8, loop from [4,3,2,1], calc energy consumption, if not enuf find charging stations, calc delta_L2 if delta_L2 < delta L1 then swap
        
        '''

        if subroute[0]!=self.DEPOT or subroute[-1]!=self.DEPOT and len(subroute) > 2:
            raise Exception("Subroute must start and end with depot")
            
        remaining_energy = dict()
        remaining_energy[self.DEPOT] = self.BATTERY_CAPACITY
        optimal_subroute = []

        for i in range(1, len(subroute)):
            if subroute[i] in self.STATIONS or subroute[i]==self.DEPOT:
                remaining_energy[subroute[i]] = self.BATTERY_CAPACITY
            else:
                previous_energy = remaining_energy[subroute[i - 1]]
                remaining_energy[subroute[i]] = previous_energy - self.calcBatteryConsumption(subroute[i - 1], subroute[i])
                if remaining_energy[subroute[i]] < 0:
                    #Skip invalid solution
                    return subroute
                    
        subroute = list(reversed(subroute))
        #print('reversed subroute: ',subroute)

        energy = self.BATTERY_CAPACITY
        i = 0

        """ Optimize from depot_R """
        # print('\nOptimize from depot_R')
        # print('='*20)
        while i < len(subroute) - 1:
            optimal_subroute.append(subroute[i])
            # print(i,subroute[i],energy)
            if subroute[i + 1] not in self.STATIONS:
                energy -= self.calcBatteryConsumption(subroute[i], subroute[i + 1])
                i += 1
                continue
            
            # print('Optimal subroute: ',optimal_subroute)

            # Calculate delta_L1
            # print('\nCalculate delta _L1')
            # print('='*20)
            from_node = subroute[i]
            num_stations_in_row = 0
            original_distance = 0
            # print('from node:',from_node)
            
            while subroute[i + 1 + num_stations_in_row] in self.STATIONS:
                original_distance += self.calcNodeDistance(from_node,subroute[i + 1 + num_stations_in_row])
                # print(f'from node: {from_node} to node: {subroute[i + 1 + num_stations_in_row]}, distance: {original_distance}')
                from_node = subroute[i + num_stations_in_row + 1]
                num_stations_in_row += 1
                
            next_customer_idx = i + num_stations_in_row + 1
            original_distance += self.calcNodeDistance(from_node,subroute[next_customer_idx])
            delta_L1 = original_distance - self.calcNodeDistance(subroute[i],subroute[next_customer_idx]) #calculate the charging distance
            # print('\nsubroute i : ',subroute[i])
            # print(f'from node: {from_node} to next customer: {subroute[next_customer_idx]}')
            # print(f'delta_L1: {delta_L1}')
            
            from_node = subroute[i]
            considered_nodes = []  
            tmp_energy = energy
            # print('\nfrom node:',from_node)
            # print(subroute[next_customer_idx:])
            
            for node in subroute[next_customer_idx:]:
                # print('considered nodes: ',node)
                considered_nodes.append(node)
                if node in self.STATIONS:
                    break
                tmp_energy -= self.calcBatteryConsumption(from_node, node)
                # print(f'from node: {from_node} to node: {node}, tmp energy: {tmp_energy}')
                # print('-'*65)
                if tmp_energy <= 0:
                    break
                from_node = node
            
            # print('considered nodes: ',considered_nodes)
            
            from_node = subroute[i]
            best_station = subroute[i + 1]
            best_station_index = 0
            # print('\nfrom node:',from_node,' best station:',best_station,'\n')
            
            for node in considered_nodes:
                to_node=node
                required_energy=remaining_energy[to_node]
                station=self.nearest_station_back(from_node,to_node,energy,required_energy)
                # print(f'to node: {to_node}, required energy: {required_energy}, station: {station}')
                if station != -1:
                    if self.calcNodeDistance(station, to_node) < self.calcNodeDistance(best_station, to_node):
                        delta_L2  = self.calcNodeDistance(from_node, station) + self.calcNodeDistance(station, to_node) - self.calcNodeDistance(from_node, to_node)
                        if delta_L2 < delta_L1:
                            delta_L1 = delta_L2
                            best_station = station
                            # print('best station: ',best_station)
                            
                from_node = to_node
                # print('from_node: ',from_node)
                # print('-'*40)
                
            optimal_subroute.extend(considered_nodes[:best_station_index])
            optimal_subroute.append(best_station)
            i = i + num_stations_in_row + best_station_index + 1
            energy = self.BATTERY_CAPACITY        
            
        optimal_subroute.append(self.DEPOT)
        return list(reversed(optimal_subroute))
 
    # def nearest_station_back(self, from_node, to_node, energy, required_energy):
    #     min_length = float("inf")
    #     best_station = -1

    #     for v in self.STATIONS:
    #         if self.calcBatteryConsumption(from_node, v) <= energy and \
    #             self.calcBatteryConsumption(v, to_node) + required_energy < \
    #                 self.BATTERY_CAPACITY:
    #             length1 = self.calcNodeDistance(v,from_node)
    #             length2 = self.calcNodeDistance(v,to_node)
    #             if min_length > length1 + length2:
    #                 min_length = length1 + length2
    #                 best_station = v

    #     return best_station
    
    # def greedy_optimize_station(self, tour):
    #     """
    #     * Note at this function, number of continuous charging stations S and S' is 1. But it can be more than 1.
    #     valid tour after inserting energy stations
    #     : depot_L -> c6 -> c5 -> c4 -> c3 -> S(S1 -> S2) -> c2 -> c1 -> depot_R
    #     Reverse tour
    #     : depot_R -> c1 -> c2 -> S(S1 -> S2) -> c3 -> c4 -> c5 -> c6 -> depot_L
    #     Replace S to other:
    #     step 1. from depot_R, get a subtour that vehicle reach farest from depot_R but not visit any charging station
    #         : depot_R -> c1 -> c2 -> c3 -> c4 - (not enough energy to reach c5) -> c5
    #         : delta_L1 = (d(c2, s1) + d(s1, s2) + d(s2, c3) - d(c2, c3))
    #     step 2: From c2->c3, c3->c4, c4->c5, find S' (>= 1 charging stations):
    #         : delta_L2 = d(c3, S') + d(S', c3) - d(c2, c3)
    #         : delta_L2 = d(c3, S') + d(S', c4) - d(c3, c4)
    #         : delta_L2 = d(c4, S') + d(S', c5) - d(c4, c5)
    #         if delta_L2 < delta_L1 then replace S with S'
    #         # see the paper: https://doi.org/10.1007/s10489-022-03555-8 for more details
    #     """
    #     if not tour[0]==self.DEPOT or not tour[-1]==self.DEPOT and len(tour) > 2:
    #         raise Exception("Tour must start and end with depot")
        
    #     remaining_energy = dict()
    #     depotID = tour[0]
    #     remaining_energy[depotID] = self.BATTERY_CAPACITY
    #     optimal_tour = []
        
    #     for i in range(1, len(tour)):
    #         if tour[i] in self.STATIONS or tour[i]==self.DEPOT:
    #             remaining_energy[tour[i]] = self.BATTERY_CAPACITY
    #         else:
    #             previous_energy = remaining_energy[tour[i - 1]]
    #             remaining_energy[tour[i]] = previous_energy - self.calcBatteryConsumption(tour[i - 1], tour[i])
    #             if remaining_energy[tour[i]] < 0:
    #                 # skip invalid solution
    #                 return tour
        
    #     tour = list(reversed(tour))
    #     energy = self.BATTERY_CAPACITY
    #     i = 0
        
    #     """ Optimize from depot_R """
    #     while i < len(tour) - 1:
    #         optimal_tour.append(tour[i])
    #         if not tour[i + 1] in self.STATIONS:
    #             energy -= self.calcBatteryConsumption(tour[i], tour[i + 1])
    #             i += 1
    #             continue
            
    #         # Calculate delta_L1
    #         from_node = tour[i]
    #         num_stations_in_row = 0
    #         original_distance = 0
            
    #         while tour[i + 1 + num_stations_in_row] in self.STATIONS:
    #             original_distance += self.calcNodeDistance(from_node,tour[i + 1 + num_stations_in_row])
    #             from_node = tour[i + num_stations_in_row + 1]
    #             num_stations_in_row += 1
            
    #         next_customer_idx = i + num_stations_in_row + 1
    #         original_distance += self.calcNodeDistance(from_node,tour[next_customer_idx])
    #         delta_L1 = original_distance - self.calcNodeDistance(tour[i],tour[next_customer_idx])
    #         from_node = tour[i]
    #         considered_nodes = []  
    #         tmp_energy = energy
    #         for node in tour[next_customer_idx:]:
    #             considered_nodes.append(node)
    #             if node in self.STATIONS:
    #                 break
    #             tmp_energy -= self.calcBatteryConsumption(from_node, node)
    #             if tmp_energy <= 0:
    #                 break
    #             from_node = node
            
    #         from_node = tour[i]
    #         best_station = tour[i + 1]
    #         best_station_index = 0
            
    #         for j, node in enumerate(considered_nodes):
    #             to_node = node
    #             required_energy = remaining_energy[to_node]
    #             station = self.nearest_station_back(from_node, to_node, energy, required_energy)
    #             if station != -1:
    #                 if self.calcNodeDistance(best_station, to_node) > self.calcNodeDistance(station, to_node):
    #                     delta_L2 = self.calcNodeDistance(from_node, station) + self.calcNodeDistance(station, to_node) \
    #                         - self.calcNodeDistance(from_node, to_node)
    #                     if delta_L2 < delta_L1:
    #                         delta_L1 = delta_L2
    #                         best_station = station

    #             from_node = to_node

    #         optimal_tour.extend(considered_nodes[:best_station_index])
    #         optimal_tour.append(best_station)
    #         i = i + num_stations_in_row + best_station_index + 1
    #         energy = self.BATTERY_CAPACITY

    #     optimal_tour.append(self.DEPOT)
    #     return list(reversed(optimal_tour))
    
    # def plotRoute(self,route):
    #     x=[]
    #     y=[]
    #     for cluster in route:
    #         for node in cluster:
    #             x.append(self.NODE[node][0])
    #             y.append(self.NODE[node][1])
    #     plt.scatter(x,y)
    #     plt.plot(x,y)
    #     plt.show()

 
if __name__ == "__main__":       
    EVRP()
