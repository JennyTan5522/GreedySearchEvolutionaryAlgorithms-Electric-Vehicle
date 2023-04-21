from EVRP import EVRP
import numpy as np
import random
import copy
import itertools
import time
import re
import yaml

import warnings
warnings.filterwarnings('ignore')

class GA:
    def __init__(self,POP_SIZE,CROSS_RATE,MUT_RATE,filename,display,random_state=42):
        random.seed(random_state)
        self.evrp=EVRP(filename,display,random_state=random_state)
        self.MAX_GENERATION=1000#25000*self.evrp.ACTUAL_PROBLEM_SIZE #10
        #self.MAX_GENERATION=5
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
        #1. Randomly select a customerA in the parent individuals
        customerA=random.randint(2,self.evrp.NUM_OF_CUSTOMERS) 

        #2. Sub1 is a set of customers in the route that comtains customerA of parent1.
        #   Sub2 is a set of customers in the route that contains customerA
        sub1=[subroute for subroute in parent1 if customerA in subroute][0]
        sub2=[subroute for subroute in parent2 if customerA in subroute][0]
        sub2=[s for s in sub2 if s not in sub1]
        
        if (sub1==[] or sub2==[]):
            return parent1, parent2

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
        fitness=[-chromosome[0] for chromosome in ranked_population]
        #Normalized fitness
        normalized_fitness=[(float(i)-min(fitness))/(max(fitness)-min(fitness)) for i in fitness]
        #Cumulated
        cumulated_normalized_fitness=itertools.accumulate(normalized_fitness)
        cumulated_normalized_fitness=[a for a in cumulated_normalized_fitness]
        #Count probability
        probability=[f/sum(cumulated_normalized_fitness) for f in cumulated_normalized_fitness]
        # return new_population
        selectedChromosomeIdx = np.random.choice(range(len(ranked_population)),size=self.POP_SIZE,p=probability,replace=False)
        return [ranked_population[i][1] for i in selectedChromosomeIdx]
    
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
        #Store the population's average fitness history, key=iter, values=(avg,best_ind)
        history={}
        
        #key=incomplete chromosome, value=(complete chromosome, fitness), if check capacity not pass then save as inf
        chromosome_results={} 

        #var to save the best individual fitness value(shortest distance)
        best_individual=(float('inf'),None,None)

        #Step 1: Initialiting first population
        self.population=[self.chromosome_init() for _ in range(self.POP_SIZE)]

        #Record start time
        start_time=time.time()

        #Iterate through the max generation
        for iter in range(self.MAX_GENERATION):
            if (iter>0):
                #Perform Local 2-opt search on population
                for j,solution in enumerate(self.population):
                    for k,cluster in enumerate(solution):
                        self.population[j][k]=self.evrp.local2Opt(self.population[j][k])
  
            # Step 2: Crossover
            # For every single chromosome then see whether need do crossover
            children=[]

            #list of index 0-199 -> shuffle list, reshape the size become 100*2 -> 100 rows and 2 cols
            index=list(range(self.POP_SIZE))
            random.shuffle(index)
            index=np.reshape(index,(int(self.POP_SIZE/2),2))

            for parent1,parent2 in index:
                parent1=self.population[parent1]
                parent2=self.population[parent2]
                if random.uniform(0,1) <= self.CROSS_RATE:
                    if parent1!=parent2:      
                        child1,child2=self.crossover(parent1,parent2)
                        children.append(child1)
                        children.append(child2)

            self.population=self.population+children
            
            #Step 3: Mutation -> Original population and children
            for idx,chromosome in enumerate(self.population):
                #print('Chromosome: ',chromosome)
                if random.uniform(0,1) <= self.MUT_RATE:
                    self.population[idx]=self.mutation_hsm(chromosome)

            #Step 4: Roulette Selection
            '''Evaluate for every chromosome 
            -> Insert charging stations -> Check validity -> If no valid(inf); else return total distance(include depot+charging stations), 
            but return chromosome (not include depot+charging stations)'''
            ranked_population=[] #stored as tuple (fitness,chromosome(original),chromosome(charging stations+depot))
            
            for idx,chromosome in enumerate(self.population):
                chromosome_tuple=tuple([j for sub in chromosome for j in sub])
                #Some chromosme might be repeated -> no need evaluate
                #Check whether this chromosome already did before
                if chromosome_tuple in chromosome_results.keys():
                    #Check whether fitness is inf-> Pass append to ranked pop; not pass check battery n capacity then no need put in ranked pop
                    if chromosome_results[chromosome_tuple][1]!=float('inf'):
                        ranked_population.append((chromosome_fitness,chromosome,chromosomeComplete))
                else:
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
                            #Add key-pair values into chromoome_results history
                            chromosome_results[chromosome_tuple]=(chromosomeComplete,chromosome_fitness)
                        else:
                            chromosome_results[chromosome_tuple]=(chromosome,float('inf'))
                    else:
                        #Put inside history 
                        chromosome_results[chromosome_tuple]=(chromosome,float('inf'))

            #Sort based on the shortest distance    
            ranked_population.sort(reverse=True)
            self.population=self.rouletteWheelSelection(ranked_population)
          
            #Compare the current population's best individual with the current best individual
            if(ranked_population[-1][0] < best_individual[0]):
                best_individual=copy.deepcopy(ranked_population[-1])

            #Update history
            print(f'Iter: {iter+1}, Best Individual: {best_individual[0]}')
            history[iter+1]={'Avg':str(np.mean([ind[0] for ind in ranked_population])),'Best Individual':str(best_individual)}
        
        #Record end time, run time-start time
        run_time=time.time()-start_time

        return run_time,best_individual,history

if __name__=='__main__':
    POP_SIZE=200 #POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.1
    #POP_SIZE,CROSS_RATE,MUT_RATE,filename,display,random_state=42):
    filename=['evrp-benchmark-set/E-n22-k4.evrp']

    for idx,file in enumerate(filename):
        ga=GA(POP_SIZE,CROSS_RATE,MUT_RATE,file,display=False,random_state=42)
        run_time,best_individual,history=ga.newGeneration()
        print(f'>> Running Time        :{run_time:.2f}')
        print(f'>> Approximation Ratio :{best_individual/ga.evrp.OPTIMUM}')

        results={}
        results['File Name']=file
        results['Running Time']=run_time
        results['Best Chromosome']=best_individual
        results['Approximation Ratio']=best_individual/ga.evrp.OPTIMUM

        #Save generation history into file
        file=file.strip('.evrp')
        benchmark_name=re.match(r'-(\w.*)/(\w.*)',file)
        save_filename='history_'+benchmark_name.group(1)+benchmark_name.group(2)+'.yml'
        with open(save_filename,'w') as f:
            yaml.dump(history,f)
            yaml.dump(results,f)
          
  