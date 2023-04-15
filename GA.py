from EVRP import EVRP
import numpy as np
import random
import copy
import itertools

class GA:
    def __init__(self,POP_SIZE,CROSS_RATE,MUT_RATE,filename,display,random_state=42):
        random.seed(random_state)
        self.evrp=EVRP(filename,display,random_state=random_state)
        # self.MAX_GENERATION=25000*self.evrp.ACTUAL_PROBLEM_SIZE
        self.MAX_GENERATION=10
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE=CROSS_RATE
        self.MUT_RATE=MUT_RATE

    def chromosome_init(self):
        '''Generating chromosomes based on 4.1 Clustering, 4.2 Balancing and 4.3 Local Search'''
        #Step 1: Clustering
        initialCluster=self.evrp.clustering()

        #Step 2: Balancing
        balancedCluster=self.evrp.balancingApproach(initialCluster)

        #Step 3: Local 2-opt search
        for i in range(len(balancedCluster)-1):
            balancedCluster[i]=self.evrp.local2Opt(balancedCluster[i])

        return balancedCluster

    def crossover(self,parent1:list,parent2:list):
        #Make sure that sub1, and sub2 is not a empty list
        while(True):
            #1. Randomly select a customerA in the parent individuals
            customerA=random.randint(2,self.evrp.NUM_OF_CUSTOMERS) 

            #2. Sub1 is a set of customers in the route that comtains customerA of parent1.
            #   Sub2 is a set of customers in the route that contains customerA
            sub1=[subroute for subroute in parent1 if customerA in subroute][0]
            sub2=[subroute for subroute in parent2 if customerA in subroute][0]
            sub2=[s for s in sub2 if s not in sub1]
            
            if (sub1==[] or sub2==[]):
                continue
            else:
                break

        #4.1 Create child1=concatenate(sub2,sub1)
        concat1=sub2.copy() + sub1.copy()
        count=0
        child1=copy.deepcopy(parent1)
        for i,route in enumerate(parent1):
            for j,gene in enumerate(route):
                if gene in sub1 or gene in sub2:
                    child1[i][j]=concat1[count]
                    count+=1

        #4.2 Create child2=concatenate(reverse(sub1),reverse(sub2))
        concat2=sub1[::-1]+sub2[::-1]
        count=0
        child2=copy.deepcopy(parent2)
        for i,route in enumerate(parent2):
            for j,gene in enumerate(route):
                if gene in sub1 or gene in sub2:
                    child2[i][j]=concat2[count]
                    count+=1
                    
        return child1,child2

    def mutation_hsm(self,finalCluster:list):
        #1. Choose a random customer, ci
        ci=random.randint(2,self.evrp.NUM_OF_CUSTOMERS) 

        #2. Find the nearest customer, cj from different routes that has the shortest distance to ci
        #2.1 Find different route cluster from ci
        diffRouteFromci=[cluster for cluster in finalCluster if ci not in cluster]
        #Convert 2d to 1d
        diffRouteFromci=[route for cluster in diffRouteFromci for route in cluster]

        #2.2 Select the nearest customer with cj that is from different route
        nearest=self.evrp.nearestCustomers(ci)

        for node in nearest:
            if node in diffRouteFromci:
                cj=node
                break

        #3. Exchange its position with the customer cj 
        idxCi=0
        idxCj=0
        #Find index of ci and cj
        for i,cluster in enumerate(finalCluster):
            for j,route in enumerate(cluster):
                if (finalCluster[i][j]==ci):
                    idxCi=(i,j)
                if (finalCluster[i][j]==cj):
                    idxCj=(i,j)
        #Swap ci and cj
        #Replace ci with cj
        finalCluster[idxCi[0]][idxCi[1]]=cj
        #Replace cj with ci
        finalCluster[idxCj[0]][idxCj[1]]=ci 
       
        return finalCluster


    def rouletteWheelSelection(self,ranked_population):
        '''
        normalized_fitness= 0.1,0.2,0.3,0.4,0.5
        accumulated_normalized_fiteness= 0.1,0.3,0.6,1.0,1.5
        prob=0.1/1.5,0.3/1.5,0.6/1.5
        '''
        #Get all population chromosome fitness
        fitness=[chromosome[0] for chromosome in ranked_population]
        #Normalized fitness
        normalized_fitness=[(float(i)-min(fitness))/(max(fitness)-min(fitness)) for i in fitness]
        #Cumulated
        cumulated_normalized_fitness=itertools.accumulate(normalized_fitness)
        cumulated_normalized_fitness=[a for a in cumulated_normalized_fitness]
        #Count probability
        probability=[f/sum(cumulated_normalized_fitness) for f in cumulated_normalized_fitness]
        #Find min prob
        probability=1-np.array(probability)
        #Choose chromosome based on population size, choose chromosome based on probability
        return np.random.choice([chromosome[1] for chromosome in ranked_population],size=self.POP_SIZE,p=probability)
    
    #Before fitness need to check validity -> Battery and capacity, if invalid, return false, no need count fitness (inf)
    def checkCapacity(self,chromosome:list):#Chromosome already include charging stations
        #Chromosome not include depot+charging stations
        for route in chromosome:
            if self.evrp.totalDemandInRoute(route)>self.evrp.MAX_CAPACITY:
                return False
        return True

    def checkBattery(self,chromosome:list):
        '''
        [1,2,3,5,25,2,1], when 1(idx=0) full capacity -> Battery consumption=0
        2 is not charging stations
        '''
        #ChromosomeComplete include depot+charging stations
        for route in chromosome:
            #Loop for every stations
            currentBatteryLevel=self.evrp.BATTERY_CAPACITY
            for idx in range(1,len(route)-1):
                #Cal batter consumption (Current and previous energy consumption)
                batteryConsumption=self.evrp.distanceMatrix[route[idx]-1][route[idx-1]-1]*self.evrp.ENERGY_CONSUMPTION
                currentBatteryLevel-=batteryConsumption

                if (currentBatteryLevel < 0):
                    return False
                
                #Check whether current node if charging station
                if route[idx] in self.evrp.STATIONS_COORD_SECTION:
                    currentBatteryLevel=self.evrp.BATTERY_CAPACITY
        
        return True #Valid route
          

    def fitness(self,chromosome:list):
        '''
        Calculate the chromosome fitness(depot+charging+cust) based on distance[i],distance[i+1] .. to n
        '''
        return np.sum([self.evrp.calculateTotalDistance(cluster) for cluster in chromosome])

    def newGeneration(self):
        ''''
        - Read file  
        - Generation population (200)
            5.2 Initialization
            - Clustering -> Balancing -> Local search 
            - Output=1 final cluster (=1 chromosome) -> Append chromosome into my population
            5.3 Crossover operator   
            5.4 Mutation
            Before selection then do fitness, selection based on fitness value
            5.5 Selection
        '''
        #Store the population's average fitness history
        history={}

        #var to save the best individual fitness value(shortest distance)
        best_individual=(float('inf'),None,None)

        #Step 1: Initialiting first population
        self.population=[self.chromosome_init() for _ in range(self.POP_SIZE)]

        #Iterate through the max generation
        for iter in range(self.MAX_GENERATION):
            #Step 2: Crossover
            #For every single chromosome then see whether need do crossover
            children=[]
            for parent1 in self.population:
                if random.uniform(0,1) <= self.CROSS_RATE:
                    #Choose another partner for crossover
                    parent2=random.choice(self.population)
                    child1,child2=self.crossover(parent1,parent2)
                    children.append(child1)
                    children.append(child2)

            #Step 3: Mutation -> Original population and children
            self.population=self.population+children
            for idx,chromosome in enumerate(self.population):
                if random.uniform(0,1) <= self.MUT_RATE:
                    self.population[idx]=self.mutation_hsm(chromosome)
            
            #Step 4: Roulette Selection
            '''Evaluate for every chromosome 
            -> Insert charging stations -> Check validity -> If no valid(inf); else return total distance(include depot+charging stations), 
            but return chromosome (not include depot+charging stations)'''
            ranked_population=[] #stored as tuple (fitness,chromosome(original),chromosome(charging stations+depot))
            for idx,chromosome in enumerate(self.population):
                #Check capacity demand and battery level
                if (self.checkCapacity(chromosome)):
                    #If capacity true then insert charging stations
                    chromosomeComplete=self.evrp.findChargingStation(chromosome)
                    #Check battery
                    if self.checkBattery(chromosomeComplete):
                        #Evaluate fitness
                        chromosome_fitness=self.fitness(chromosomeComplete)
                        #Append into ranked_pop
                        ranked_population.append((chromosome_fitness,chromosome,chromosomeComplete))
                #     #If battery not valid
                #     else:
                #         ranked_population.append((float('inf'),chromosome,chromosomeComplete))

                # else:
                #     ranked_population.append((float('inf'),chromosome,chromosome))#ori chromosome, ori chromosome
        
            #Sort based on the shortest distance    
            ranked_population.sort()
            self.population=self.rouletteWheelSelection(ranked_population)

            #Compare the current population's best individual with the current best individual
            if(ranked_population[0][0] < best_individual[0]):
                best_individual=ranked_population[0]

            #Update history
            history[iter]={'Avg':np.mean([ind[0] for ind in ranked_population]),'Best Individual':best_individual}

        return best_individual,history
            


       

if __name__=='__main__':
    MAX_GENERATION=25000
    POP_SIZE=10 #POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.1
    #POP_SIZE,CROSS_RATE,MUT_RATE,filename,display,random_state=42):
    ga=GA(POP_SIZE,CROSS_RATE,MUT_RATE,'evrp-benchmark-set/E-n22-k4.evrp',display=True,random_state=42)
    best_individual,history=ga.newGeneration()
    print(history)
