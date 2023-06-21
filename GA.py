from EVRP import EVRP
import numpy as np
import random
import copy
import itertools
import time
import re
import yaml
import os
from matplotlib import pyplot as plt
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

class GA:
    def __init__(self,MAX_GEN,POP_SIZE,CROSS_RATE,MUT_RATE,filename,display,fileIdx,random_state=42):
        random.seed(random_state)
        self.evrp=EVRP(filename,display,random_state=random_state)
        self.MAX_GENERATION=MAX_GEN*self.evrp.ACTUAL_PROBLEM_SIZE 
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE=CROSS_RATE
        self.MUT_RATE=MUT_RATE
        self.filename=filename
        self.fileIdx=fileIdx

    def chromosome_init(self,save_filename=None):
        '''
        Description: Generating chromosome based on Clustering, Balancing, Local 2-Opt Search and Local 3-Opt Search
        '''
        initialCluster=self.evrp.clustering()

        #balancedCluster=self.evrp.balancingCluster(initialCluster)
        balancedCluster=copy.deepcopy(initialCluster)
        balancedCluster=self.evrp.balancingCluster(balancedCluster)
        #print(balancedCluster)

        #Plot initial route
        #Add depot infront and at the end
        # initialRoute=copy.deepcopy(initialCluster)
        # for i,subroute in enumerate(initialRoute):
        #     initialRoute[i]=[1]+subroute+[1]
        # self.evrp.print_solution(balancedCluster,'Initial Solution')
        # path='C://Users//User//Desktop//EVRP//Code//figures//step_1_initial_solution.png'
        # ga.plot(initialRoute,path,'Initial Solution')
        
        # two_opt_route=copy.deepcopy(balancedCluster)
        # for i in range(len(two_opt_route)-1):
        #     two_opt_route[i]=self.evrp.ls_2opt(two_opt_route[i])
    
        # self.evrp.print_solution(two_opt_route,'Local 2-opt Solution')

        # two_opt_route_depot=copy.deepcopy(two_opt_route)
        # for i,subroute in enumerate(two_opt_route_depot):
        #     two_opt_route_depot[i]=[1]+subroute+[1]
        # path='C://Users//User//Desktop//EVRP//Code//figures//'+save_filename+'_step_2_ls2opt_solution.jpeg'
        # ga.plot(two_opt_route_depot,path,'Local 2-Opt Solution')

        # three_opt_route=copy.deepcopy(two_opt_route)
        # for i in range(len(three_opt_route)-1):
        #     three_opt_route[i]=self.evrp.ls_3opt(three_opt_route[i])
        #self.evrp.print_solution(three_opt_route,'Local 3-opt Solution')

        #print(self.evrp.initial_route_validation(three_opt_route))

        # three_opt_route_depot=copy.deepcopy(three_opt_route)
        # for i,subroute in enumerate(three_opt_route_depot):
        #     three_opt_route_depot[i]=[1]+subroute+[1]
        # path='C://Users//User//Desktop//EVRP//Code//figures//'+save_filename+'_step_3_ls3opt_solution.jpeg'
        # ga.plot(three_opt_route_depot,path,'Local 3-Opt Solution')
        
        #Step 4: Insert charging stations to route T
        # chargingRoute=copy.deepcopy(three_opt_route)
        # for idx,subroute in enumerate(chargingRoute):
        #     # print('-'*70)
        #     chargingRoute[idx]=self.evrp.findChargingStation(subroute,idx,print_results=False)
        # self.evrp.print_solution(chargingRoute,'Insert Charging Route Solution')

        # newChargingRoute=copy.deepcopy(chargingRoute)
        # station=False
        # for idx,subroute in enumerate(newChargingRoute):
        #     # print('-'*70)
        #     for node in subroute:
        #         if node in self.evrp.STATIONS:
        #             station=True
        #             break
        #     if station:
        #         newChargingRoute[idx]=self.evrp.greedy_optimize_station(subroute)
        #         station=False
        # self.evrp.print_solution(newChargingRoute,'Optimize Charging Route Solution')

        #print(self.evrp.complete_route_validation(chargingRoute))
        
        return balancedCluster

    def crossover(self,parent1:list,parent2:list):
        '''
        Description: This method is used for generate new offspring
                     1. Randomly select a customer 
                     2. Find 2 subroute that both parents belong to
                     3. child1=sub2+sub1, child2=reverse(sub1)+reverse(sub2)

                     parent1=[[5,9,8,4],[2,3],[6,1,7]]
                     parent2=[[6,2,3,9],[4,8],[7,1,5]]

                     sub1=[2,3], sub2=[6,9]

                     child1=[6, 9, 2, 3], child2=[3, 2, 9, 6])

                     after exchg: ([[5, 6, 8, 4], [9, 2], [3, 1, 7]], [[3, 2, 9, 6], [4, 8], [7, 1, 5]])

        Input (parent1, parent2): chromosome that consists of 2D list containing subroutes
        '''
        customerA=random.randint(2,self.evrp.NUM_OF_CUSTOMERS) 

        sub1=[subroute for subroute in parent1 if customerA in subroute][0]
        sub2=[subroute for subroute in parent2 if customerA in subroute][0]
        sub2=[s for s in sub2 if s not in sub1]
        
        if (sub1==[] or sub2==[]):
            return parent1, parent2

        #Create child1
        concat1= copy.deepcopy(sub2) + copy.deepcopy(sub1)
        count=0
        child1=copy.deepcopy(parent1)
        for i,route in enumerate(parent1):
            for j,gene in enumerate(route):
                if gene in sub1 or gene in sub2:
                    child1[i][j]=concat1[count]
                    count+=1

        #Create child2
        concat2=sub1[::-1]+sub2[::-1]
        count=0
        child2=copy.deepcopy(parent2)
        for i,route in enumerate(parent2):
            for j,gene in enumerate(route):
                if gene in sub1 or gene in sub2:
                    child2[i][j]=concat2[count]
                    count+=1
                    
        return child1,child2

    # def mutation_hsm(self,finalCluster:list):
    #     '''
    #     Description: Choose a random customer and exchange its position with the customer frrom diff route that has the nearest distance with teh selected customer
    #     '''
    #     #1. Choose a random customer, ci
    #     ci=random.randint(2,self.evrp.NUM_OF_CUSTOMERS) 

    #     #2. Find the nearest customer, cj from different routes that has the shortest distance to ci
    #     #2.1 Find different route cluster from ci
    #     diffRouteFromci=[cluster for cluster in finalCluster if ci not in cluster]
    #     #Convert 2d to 1d
    #     diffRouteFromci=[route for cluster in diffRouteFromci for route in cluster]

    #     #2.2 Select the nearest customer with cj that is from different route
    #     nearest=self.evrp.nearestCustomers(ci)

    #     for node in nearest:
    #         if node in diffRouteFromci:
    #             cj=node
    #             break

    #     #3. Exchange its position with the customer cj 
    #     idxCi=0
    #     idxCj=0
    #     #Find index of ci and cj
    #     for i,cluster in enumerate(finalCluster):
    #         for j,route in enumerate(cluster):
    #             if (finalCluster[i][j]==ci):
    #                 idxCi=(i,j)
    #             if (finalCluster[i][j]==cj):
    #                 idxCj=(i,j)
    #     #Swap ci and cj
    #     #Replace ci with cj
    #     finalCluster[idxCi[0]][idxCi[1]]=cj
    #     #Replace cj with ci
    #     finalCluster[idxCj[0]][idxCj[1]]=ci 
       
    #     return finalCluster
    
    def mutation_hsm(self,chromosome:list):
        '''
        Description: Choose a random customer and exchange its position with the customer from diff route that has the nearest distance with the selected customer
                     1. Select a random customer, A
                     2. Choose the nearest customer from A but from different subroute, B
                     3. Exchange the position of A and B
                     
                     ** Note: After exchange the position, we will ensure that the new route is valid. 
        '''
        while(True):
            cust=random.choice(self.evrp.CUSTOMERS) 
            near_cust=self.evrp.nearestCustomers(cust)
           
            A=[(idx,subroute) for idx,subroute in enumerate(chromosome) if cust in subroute][0]
            idx_A=A[0]
            cust_subroute=A[1]

            for cust in cust_subroute:
                if cust in near_cust:
                    near_cust.remove(cust)

            exchange_cust=near_cust[0]

            new_subroute_A=copy.deepcopy(cust_subroute)
            new_subroute_A.remove(cust)
            new_subroute_A+=[exchange_cust]

            B=[(idx,subroute) for idx,subroute in enumerate(chromosome) if exchange_cust in subroute][0]
            idx_B=B[0]
            exchg_subroute=B[1]
            new_subroute_B=copy.deepcopy(exchg_subroute)
            new_subroute_B.remove(exchange_cust)
            new_subroute_B+=[cust]

            chromosome[idx_A]=new_subroute_A
            chromosome[idx_B]=new_subroute_B
            
            if self.evrp.initial_route_validation(chromosome)==True:
                break
    
        return chromosome

    def rouletteWheelSelection(self,ranked_population,duplicate=True):
        '''
        Description: Use roulette wheel selection to select the best individual. This method ensures that individuals with 
                     good fitness will have a high probability of choosing. The best individual then add to new population
                     to maintain the best existing traits.

        Input: (ranked_population) - A population that contains chromosome which is sort based on fitness value

        Output: Return new population
        
        normalized_fitness= 0.1,0.2,0.3,0.4,0.5
        accumulated_normalized_fiteness= 0.1,0.3,0.6,1.0,1.5
        prob=0.1/1.5,0.3/1.5,0.6/1.5

        Details: 1. Get all population chromosom fitness
                 2. Calculate normalized fitness
                 3. Calculate cumuated fitness
                 4. Calculate probability
                 5. Select chromosome based on prob
        
        Output: Return the selected chromosome

        '''
        fitness=[-chromosome[0] for chromosome in ranked_population]
        normalized_fitness=[(float(i)-min(fitness))/(max(fitness)-min(fitness)) for i in fitness]
        cumulated_normalized_fitness=itertools.accumulate(normalized_fitness)
        cumulated_normalized_fitness=[a for a in cumulated_normalized_fitness]
        probability=[f/sum(cumulated_normalized_fitness) for f in cumulated_normalized_fitness]
        selectedChromosomeIdx = np.random.choice(range(len(ranked_population)),size=self.POP_SIZE,p=probability,replace=duplicate) #replace=False or True
        return [ranked_population[i][1] for i in selectedChromosomeIdx]

    def fitness(self,subroute:list):
        '''
        Description: Calculate the chromosome fitness(depot+charging+cust) based on distance[i],distance[i+1] .. to n

        Input: (subroute)     - List containing the sequence of nodes visited

        Output: Returns distance of the subroute (Distance=Fitness, the shorter the better)
        '''
        return self.evrp.calcSubrouteDistance(subroute)
         
    # def newGeneration(self,save_filename:str,print_iter=False):
    #     '''
    #     Description: Generate a new generation that containing number of chromosomes based on POP_SIZE

    #     Input: (print_iter) - Print each iteration chromosome' results

    #     Output: (run_time)        - End time minus start time, total time when performing new generation
    #             (best_individual) - Best chromosome in the new generation
    #             (history)         - Chromosome history in the new generation

    #     Data: 
    #     history            - Store the population's average fitness history -> {key=iter: value=(avg,best_ind)}
        
    #     chromosome_results - Store chromosomes in the new population        -> {key=incomplete chromosome: value=(complete chromosome, fitness)} 
    #                          -> Complete chromosome include depot, incomplete chromosome does not include depot 
    #                          -> If fitnes=inf means capacity is invalid
        
    #     best_individual    - Save the best chromosome in the new population -> {key=fitness: value=(chromosome (ori), chromosome(charging+depot)}
        
    #     ranked_population  - Store all chromosomes fitness based on descending order (long distance to short distance) -> {key=fitness: value=(chromosome (ori), chromosome(charging+depot)}

    #     Details:
    #     - Initiating first population
    #     - Record start time
    #     - After generate chromosomes, iterate through each chromosome perform 
    #         1. Local 2-opt search
    #         2. Crossover
    #         3. Mutation
    #         4. Insert charging stations
    #         5. Roulette wheel selection
    #         6. Evaluate chromosome
    #     - Record end time
    #     '''
    #     history={}
    #     chromosome_results={} 
    #     best_individual=(float('inf'),None,None)

    #     if (self.POP_SIZE%2!=0):
    #         self.POP_SIZE+=1

    #     #Initiate
    #     self.population=[self.chromosome_init() for _ in range(self.POP_SIZE)]

    #     # file=self.filename.strip('.evrp')
    #     # benchmark_name=re.match(r'-(\w.*)/(\w.*)',file)
    #     # save_filename=str(self.fileIdx+1)+' - '+benchmark_name.group(1)+benchmark_name.group(2)
    #     # title=benchmark_name.group(1)+benchmark_name.group(2)
        
    #     # plot_route=copy.deepcopy(self.population[0])
    #     # for i,subroute in enumerate(plot_route):
    #     #     plot_route[i]=[1]+subroute+[1]
    #     # path='C://Users//User//Desktop//EVRP//Code//figures//'+save_filename+'_step1_initial_solution.jpeg'
    #     # self.plot(plot_route,path,'Initial Solution')
        
    #     start_time=time.time()

    #     for iter in range(self.MAX_GENERATION):
    #         elitism_population=[]
    #         children=[]
    #         if iter==0: 
    #             elitism_population=self.population

    #             index=list(range(ga.POP_SIZE))
    #             random.shuffle(index)
    #             index=np.reshape(index,(int(ga.POP_SIZE/2),2))
    #             for parent1,parent2 in index:
    #                 parent1=ga.population[parent1]
    #                 parent2=ga.population[parent2]
    #                 if random.uniform(0,1) <= ga.CROSS_RATE:     
    #                     child1,child2=ga.crossover(parent1,parent2)
    #                 else:
    #                     child1,child2=parent1,parent2
                    
    #                 if self.evrp.routeCapacityValidation(child1):
    #                     if random.uniform(0,1) <= self.MUT_RATE:
    #                         child1=self.mutation_hsm(child1)    
    #                     children.append(child1)

    #                 if self.evrp.routeCapacityValidation(child2):
    #                     if random.uniform(0,1) <= self.MUT_RATE:
    #                         child1=self.mutation_hsm(child2)
    #                     children.append(child2)
  
    #         else:
    #             n_parents_count=self.POP_SIZE-int(self.POP_SIZE*0.2)

    #             if n_parents_count%2 !=0:
    #                 n_parents_count+=1
    #                 elitism_population=[ind[1] for ind in ranked_population[:int(self.POP_SIZE*0.2)-1]]
    #             else:
    #                 elitism_population=[ind[1] for ind in ranked_population[:int(self.POP_SIZE*0.2)]] 
                
    #             n_parents=list(range(n_parents_count)) 

    #             for i in n_parents: 
    #                 parent1=self.rouletteWheelSelection(ranked_population[int(self.POP_SIZE*0.2):])[0]
    #                 parent2=self.rouletteWheelSelection(ranked_population[int(self.POP_SIZE*0.2):])[1]

    #                 if random.uniform(0,1) <= self.CROSS_RATE:  
    #                     child1,child2=self.crossover(parent1,parent2)
    #                 else:
    #                     child1,child2=parent1,parent2
                    
    #                 if self.evrp.routeCapacityValidation(child1):
    #                     if random.uniform(0,1) <= self.MUT_RATE:
    #                         child1=self.mutation_hsm(child1)
    #                     children.append(child1)

    #                 if self.evrp.routeCapacityValidation(child2):
    #                     if random.uniform(0,1) <= self.MUT_RATE:
    #                         child1=self.mutation_hsm(child2)
    #                     children.append(child2)

    #         # if (iter>0):
    #         #     for j,solution in enumerate(self.population):
    #         #         for k,cluster in enumerate(solution):
    #         #             self.population[j][k]=self.evrp.ls_2opt(self.population[j][k])
            
    #         '''
    #         Crossover
    #         ===========
    #         - Generate parents by reshaping the population size become (POP_SIZE/2) rows, 2 cols -> If generate prob less than cross_rate -> Generate new child
    #         - Make sure the generated new child is valid, if invalid then remove the children.
    #         - Append the children into population.
    #         '''
    #         # for route in children:
    #         #     if self.evrp.initial_route_validation(route)==False:
    #         #         children.remove(route)

    #         self.population=elitism_population+children
            
    #         for j,solution in enumerate(self.population):
    #             for k,cluster in enumerate(solution):
    #                 self.population[j][k]=self.evrp.ls_2opt(self.population[j][k])

    #         '''
    #         Mutation
    #         ===========
    #         - If generate prob less than MUT_RATE, current chromsome will undergo mutation.
    #         '''
    #         # for idx,chromosome in enumerate(self.population):
    #         #     if random.uniform(0,1) <= self.MUT_RATE:
    #         #         self.population[idx]=self.mutation_hsm(chromosome)
            
    #         '''
    #         Insert charging stations
    #         =========================
    #         - Population contains chromosomes. A chromosome is a 2D list that contains many subroutes.
    #         1. Loop through the population
    #              Loop through the chromosome in the population
    #                 If the subroute in the chromosome already exist in the chromosome results, no need validate capacity & find charging stations
    #                 - If fitness !=inf -> Update current fitness and current chromosome to complete chromsome from the chromsome results

    #                 Else if the subroute not exist in the chromosome results
    #                 - Validate capacity -> Find charging station -> Validate battery  
    #                    Valid ? -> Yes: Update current fitness, current chromosome and chromsome results
    #                            ->  No: Update fitness to inf and no need add into the population
    #         '''
    #         ranked_population=[] 

    #         for idx,chromosome in enumerate(self.population):
    #             currentFitness=0
    #             currentChromosome=[]
    #             addIntoPopulation=True
    #             for subroute in chromosome:
    #                 if str(subroute) in chromosome_results.keys(): #If subroute already exist 
    #                     if chromosome_results[str(subroute)][1]!=float('inf'):
    #                         currentFitness+=chromosome_results[str(subroute)][1]
    #                         currentChromosome.append(chromosome_results[str(subroute)][0])
    #                     else: 
    #                         addIntoPopulation=False
    #                         break 
    #                 else: #If subroute not exist
    #                     if (self.evrp.capacityValidation(subroute)): #If capacity valid
    #                         subrouteComplete=self.evrp.findChargingStation(subroute,routeIdx=None,print_results=False) 
    #                         if(subrouteComplete!=-1):
    #                             if self.evrp.batteryValidation(subrouteComplete): #If battery valid
    #                                 subroute_fitness=self.fitness(subrouteComplete)
    #                                 currentFitness+=subroute_fitness
    #                                 currentChromosome.append(subrouteComplete)
    #                                 chromosome_results[str(subroute)]=(subrouteComplete,subroute_fitness)
    #                             else:
    #                                 chromosome_results[str(subroute)]=(subroute,float('inf'))
    #                                 addIntoPopulation=False
    #                                 break
    #                         else:
    #                             chromosome_results[str(subroute)]=(subroute,float('inf'))
    #                             addIntoPopulation=False
    #                             break
    #                     else:
    #                         chromosome_results[str(subroute)]=(subroute,float('inf'))
    #                         addIntoPopulation=False
    #                         break

    #             if addIntoPopulation: 
    #                 if self.evrp.complete_route_validation(currentChromosome): #Validate complete route, if true update ranked pop
    #                     ranked_population.append((currentFitness,chromosome,currentChromosome))
    #                 else:
    #                     print(currentChromosome)
  
    #         #print('ranked population')
    #         ranked_population.sort(reverse=True)

    #         self.population=self.rouletteWheelSelection(ranked_population,duplicate=False)
            
    #         if(ranked_population[-1][0] < best_individual[0]):
    #             best_individual=copy.deepcopy(ranked_population[-1])

    #         # print('update history')
    #         #Update history
    #         avg=np.mean([ind[0] for ind in ranked_population])
    #         history[iter+1]={'Avg':str(avg),'Best Individual':str(best_individual)}
            
    #         #print('print')
    #         if print_iter==True:
    #             print(f'Pop size: {len(self.population)}, Iter: {iter+1}, Best Individual: {best_individual[0]:.4f}, Avg: {avg:.4f}')

    #     run_time=time.time()-start_time

    #     return run_time,best_individual,history
        
    def newGeneration(self,save_filename:str,print_iter=False):
        history={}
        chromosome_results={}
        pop_history={} 
        best_individual=(float('inf'),None,None)

        if (self.POP_SIZE%2!=0):
            self.POP_SIZE+=1

        #Initiate
        self.population=[self.chromosome_init() for _ in range(self.POP_SIZE)]
        
        start_time=time.time()

        for iter in range(self.MAX_GENERATION):
            print([values for values in pop_history.values()])

            for chromosome in self.population:
                flatten_chromosome=tuple([gene for subroute in chromosome for gene in subroute])
                if flatten_chromosome in pop_history.keys():
                    pop_history[flatten_chromosome]+=1
                    if pop_history[flatten_chromosome] >= 10:
                        #Perform SA
                        pass
                else:
                    pop_history[flatten_chromosome]=1

            elitism_population=[]
            children=[]
            if iter==0: 
                elitism_population=self.population

                index=list(range(self.POP_SIZE))
                random.shuffle(index)
                index=np.reshape(index,(int(self.POP_SIZE/2),2))
                for parent1,parent2 in index:
                    parent1=self.population[parent1]
                    parent2=self.population[parent2]
                    if random.uniform(0,1) <= self.CROSS_RATE:     
                        child1,child2=self.crossover(parent1,parent2)
                    else:
                        child1,child2=parent1,parent2
                    
                    if self.evrp.routeCapacityValidation(child1):
                        if random.uniform(0,1) <= self.MUT_RATE:
                            child1=self.mutation_hsm(child1)    
                        children.append(child1)

                    if self.evrp.routeCapacityValidation(child2):
                        if random.uniform(0,1) <= self.MUT_RATE:
                            child1=self.mutation_hsm(child2)
                        children.append(child2)
  
            else:
                n_parents_count=self.POP_SIZE-int(self.POP_SIZE*0.2)

                if n_parents_count%2 !=0:
                    n_parents_count+=1
                    elitism_population=[ind[1] for ind in ranked_population[:int(self.POP_SIZE*0.2)-1]]
                else:
                    elitism_population=[ind[1] for ind in ranked_population[:int(self.POP_SIZE*0.2)]] 
                
                n_parents=list(range(n_parents_count)) 

                for i in n_parents: 
                    parent1=self.rouletteWheelSelection(ranked_population[int(self.POP_SIZE*0.2):])[0]
                    parent2=self.rouletteWheelSelection(ranked_population[int(self.POP_SIZE*0.2):])[1]

                    if random.uniform(0,1) <= self.CROSS_RATE:  
                        child1,child2=self.crossover(parent1,parent2)
                    else:
                        child1,child2=parent1,parent2
                    
                    if self.evrp.routeCapacityValidation(child1):
                        if random.uniform(0,1) <= self.MUT_RATE:
                            child1=self.mutation_hsm(child1)
                        children.append(child1)

                    if self.evrp.routeCapacityValidation(child2):
                        if random.uniform(0,1) <= self.MUT_RATE:
                            child1=self.mutation_hsm(child2)
                        children.append(child2)

            self.population=elitism_population+children

            ranked_population=[] 

            for idx,chromosome in enumerate(self.population):
                currentFitness=0
                currentChromosome=[]
                addIntoPopulation=True
                for subroute in chromosome:
                    if str(subroute) in chromosome_results.keys(): #If subroute already exist 
                        if chromosome_results[str(subroute)][1]!=float('inf'):
                            currentFitness+=chromosome_results[str(subroute)][1]
                            currentChromosome.append(chromosome_results[str(subroute)][0])
                        else: 
                            addIntoPopulation=False
                            break 
                    else: #If subroute not exist
                        if (self.evrp.capacityValidation(subroute)): #If capacity valid
                            subrouteComplete=self.evrp.findChargingStation(subroute,routeIdx=None,print_results=False) 
                            if(subrouteComplete!=-1):
                                if self.evrp.batteryValidation(subrouteComplete): #If battery valid
                                    subroute_fitness=self.fitness(subrouteComplete)
                                    currentFitness+=subroute_fitness
                                    currentChromosome.append(subrouteComplete)
                                    chromosome_results[str(subroute)]=(subrouteComplete,subroute_fitness)
                                else:
                                    chromosome_results[str(subroute)]=(subroute,float('inf'))
                                    addIntoPopulation=False
                                    break
                            else:
                                chromosome_results[str(subroute)]=(subroute,float('inf'))
                                addIntoPopulation=False
                                break
                        else:
                            chromosome_results[str(subroute)]=(subroute,float('inf'))
                            addIntoPopulation=False
                            break

                if addIntoPopulation: 
                    if self.evrp.complete_route_validation(currentChromosome): #Validate complete route, if true update ranked pop
                        ranked_population.append((currentFitness,chromosome,currentChromosome))
  
            ranked_population.sort(reverse=True)

            self.population=self.rouletteWheelSelection(ranked_population,duplicate=False)
            
            if(ranked_population[-1][0] < best_individual[0]):
                best_individual=copy.deepcopy(ranked_population[-1])

            avg=np.mean([ind[0] for ind in ranked_population])
            history[iter+1]={'Avg':str(avg),'Best Individual':str(best_individual)}
            
            if print_iter==True:
                print(f'Pop size: {len(self.population)}, Iter: {iter+1}, Best Individual: {best_individual[0]:.4f}, Avg: {avg:.4f}')

        run_time=time.time()-start_time

        return run_time,best_individual,history
   
    def plotHistory(self,history:dict,title,figureSaveName):
        best_ind=[float(re.findall(r'\w.*',history[i]['Best Individual'].split(',')[0])[0]) for i in history.keys()]
        avg=[float(history[avg]['Avg']) for avg in history.keys()]
        plt.figure(figsize=(20,10))
        plt.title(f'Results of {len(history)} MAX GENERATIONS'+' ('+title+')')
        plt.xlabel('Max Generations')
        plt.ylabel('Fitness')
        plt.plot(best_ind,label='Best Individual')
        plt.plot(avg,label='Average Fitness')
        plt.legend()
        plt.grid()
        plt.savefig(figureSaveName)
        plt.show()

    
    def plot(self,route:list,path:None,name:None):
        '''
        Description: Plot the soultion of the vehicle routing problem on a scatter plot.

        Input: (subroute)  - List containing the sequence of nodes visited
        
        Output: None
        '''
        _,ax=plt.subplots()

        for i in self.evrp.NODE:
            if i==1:#If node is depot
                ax.scatter(self.evrp.NODE[i][0],self.evrp.NODE[i][1],c='red',marker='s',s=30,alpha=0.5,label='Depot')
            elif i in self.evrp.CUSTOMERS: #If node is customer
                ax.scatter(self.evrp.NODE[i][0],self.evrp.NODE[i][1],c='green',marker='o',s=30,alpha=0.5,label='Customer')
            elif i in self.evrp.STATIONS: #If node is charging stations
                ax.scatter(self.evrp.NODE[i][0],self.evrp.NODE[i][1],c='blue',marker='^',s=30,alpha=0.5,label='Station')
            else:
                raise ValueError('Invalid None Type')

        #Set title and labels
        ax.set_title(f'Problem {self.evrp.PROBLEM_NAME} - {name}')

        handles, labels=plt.gca().get_legend_handles_labels()
        by_label=OrderedDict(zip(labels,handles))
        plt.legend(by_label.values(),by_label.keys(),loc='upper right')

        for subroute in route:
            for i in range(len(subroute)-1):
                first_node=(self.evrp.NODE[subroute[i]][0],self.evrp.NODE[subroute[i]][1])
                second_node=(self.evrp.NODE[subroute[i+1]][0],self.evrp.NODE[subroute[i+1]][1])
                plt.plot([first_node[0],second_node[0]],[first_node[1],second_node[1]],color='black',linewidth=0.5,linestyle='--')
        
        if path is not None:
            plt.savefig(path)
            plt.show()
            plt.close()
    
if __name__=='__main__':
    POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.85
    MAX_GEN=60

    # filename=['evrp-benchmark-set/E-n22-k4.evrp','evrp-benchmark-set/E-n23-k3.evrp','evrp-benchmark-set/E-n30-k3.evrp','evrp-benchmark-set/E-n33-k4.evrp',
    #            'evrp-benchmark-set/E-n51-k5.evrp','evrp-benchmark-set/E-n76-k7.evrp',
    #            'evrp-benchmark-set/E-n101-k8.evrp']
    filename=['evrp-benchmark-set/E-n22-k4.evrp']

    ga=GA(MAX_GEN,POP_SIZE,CROSS_RATE,MUT_RATE,filename[0],display=False,fileIdx=0,random_state=10)

    for idx,file in enumerate(filename):
        ga=GA(MAX_GEN,POP_SIZE,CROSS_RATE,MUT_RATE,file,display=False,fileIdx=idx,random_state=10)
        
        #ga.chromosome_init(save_filename=None)
       
        file=file.strip('.evrp')
        benchmark_name=re.match(r'-(\w.*)/(\w.*)',file)
        print('\n')
        save_filename=str(idx+1)+' - '+benchmark_name.group(1)+benchmark_name.group(2)
        print(save_filename)
    
        run_time,best_individual,history=ga.newGeneration(save_filename,print_iter=True)

        print('======================================')
        print(f'>> Running Time        : {run_time:.2f}s')
        print(f'>> Approximation Ratio : {best_individual[0]/ga.evrp.OPTIMUM:.2f}')

        results={}
        results['FileName']=file
        results['Running Time']=run_time
        results['Best Individual']=str(best_individual)
        results['Approximation Ratio']=str(best_individual[0]/ga.evrp.OPTIMUM)

        #Save generation history into file
        title=benchmark_name.group(1)+benchmark_name.group(2)
        path='C://Users//User//Desktop//EVRP//Code//figures//'+save_filename+'_historyPlot.jpeg'
        ga.plotHistory(history,title,figureSaveName=path)

        path='C://Users//User//Desktop//EVRP//Code//figures//'+save_filename+'_bestPlot.jpeg'
        ga.plot(best_individual[2],path=path,name='Best Plot')

        with open('C://Users//User//Desktop//EVRP//Code//results//'+save_filename+'.yml','w') as f:
            yaml.dump(results,f)
            yaml.dump(history,f)

        print('Complete '+file)
        print('-----------------------------------------------')
            
          
  