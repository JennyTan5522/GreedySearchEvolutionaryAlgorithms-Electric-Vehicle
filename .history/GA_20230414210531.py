from EVRP import EVRP
import numpy as np
import random
import copy

class GA:
    def __init__(self,MAX_GENERATION,POP_SIZE,CROSS_RATE,MUT_RATE,evrp:EVRP,random_state=42):
        random.seed(random_state)
        self.MAX_GENERATION=MAX_GENERATION
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE=CROSS_RATE
        self.MUT_RATE=MUT_RATE
        self.evrp=evrp

    def chromosome_init(self):
        '''Generating chromosomes based on 4.1 Clustering, 4.2 Balancing and 4.3 Local Search'''
        #Step 1: Clustering
        initialCluster=EVRP().clustering()

        #Step 2: Balancing
        balancedCluster=EVRP().balancingApproach(initialCluster)

        #Step 3: Local 2-opt search
        for i in range(len(balancedCluster)-1):
            balancedCluster[i]=EVRP().local2Opt(balancedCluster[i])
        # print(EVRP.finalCluster)
        print(f'Step 3, local 2-opt cluster: {balancedCluster}')

        #Step 4: TODO
        return balancedCluster

    def crossover(parent1:list,parent2:list):
        #Make sure that sub1, and sub2 is not a empty list
        while(True):
            #1. Randomly select a customerA in the parent individuals
            customerA=random.randint(2,EVRP().NUM_OF_CUSTOMERS) 

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
        concat1=sub2.copy()+sub1.copy()
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

    def mutation_hsm(finalCluster:list):
        #1. Choose a random customer, ci
        ci=random.randint(2,EVRP().NUM_OF_CUSTOMERS) 

        #2. Find the nearest customer, cj from different routes that has the shortest distance to ci
        #2.1 Find different route cluster from ci
        diffRouteFromci=[cluster for cluster in finalCluster if ci not in cluster]
        #Convert 2d to 1d
        diffRouteFromci=[route for cluster in diffRouteFromci for route in cluster]

        #2.2 Select the nearest customer with cj that is from different route
        nearest=EVRP().nearestCustomers(ci)

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

    def fitness(chromosome:list):
        '''
        Calculate the chromosome fitness(depot+charging+cust) based on distance[i],distance[i+1] .. to n
        '''
        return np.sum([EVRP().calculateTotalDistance(cluster) for cluster in chromosome])

    def newGeneration(self):
        ''''
        - Read file  
        - Generation population (200)
            5.2 Initialization
            - Clustering -> Balancing -> Local search 
            - Output=1 final cluster (=1 chromosome) -> Append chromosome into my population
            5.3 Crossover operator   
            5.4 Mutation
            5.5 Selection
        '''
        random.seed(42)

        history={}

        #var to save the best individual fitness value
        best_individual=(float('inf'),None)

        #Step 1: Initialiting first population
        self.population=[self.chromosome_init() for _ in range(self.POP_SIZE)]

        #Iterate through the max generation
        for iter in range(self.MAX_GENERATION):
            ranked_population=[] #stored as tuple (fitness,chromosome)

            #Step 3: Evaluating chromosome
            for chromosome in self.population:
                ranked_population.append((self.fitness(chromosome),chromosome))
            
            #Sort based on ascending order
            ranked_population.sort()

            #Compare the current population's best individual with the current best individual
            if(ranked_population[0][0] < bes)

 


       

if __name__=='__main__':
    evrp=EVRP()
    MAX_GENERATION=25000
    POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.1
    ga=GA(MAX_GENERATION,POP_SIZE,CROSS_RATE,MUT_RATE,EVRP,random_state=42)
    ga.newGeneration()
