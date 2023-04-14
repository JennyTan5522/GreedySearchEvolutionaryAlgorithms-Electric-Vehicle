from EVRP import EVRP
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

    def initialization(self):
        #TODO do I need to move 4.1,4.2 and 4.3 to this GA part
        '''Generating chromosomes based on 4.1 Clustering, 4.2 Balancing and 4.3 Local Search'''
        pass

def crossover(parent1,parent2):
    #1. Randomly select a customerA in the parent individuals
    customerA=random.randint(2,NUM_OF_CUSTOMERS) 

    print(customerA)
    #2. Sub1 is a set of customers in the route that comtains customerA of parent1.
    #   Sub2 is a set of customers in the route that contains customerA
    sub1=[subroute for subroute in parent1 if customerA in subroute][0]
    sub2=[subroute for subroute in parent2 if customerA in subroute][0]
    sub2=[s for s in sub2 if s not in sub1]
    print(sub1,sub2)

    #4.1 Create child1=concatenate(sub2,sub1)
    concat1=sub2.copy()+sub1.copy()
    count=0
    child1=copy.deepcopy(parent1)
    for i,route in enumerate(parent1):
        for j,gene in enumerate(route):
            if gene in sub1 or gene in sub2:
                print('Route:',i+1,'idx pos:',j,gene,concat1[count])
                child1[i][j]=concat1[count]
                count+=1

    #4.2 Create child2=concatenate(reverse(sub1),reverse(sub2))
    concat2=sub1[::-1]+sub2[::-1]
    count=0
    child2=parent2.copy()
    for i,route in enumerate(parent2):
        for j,gene in enumerate(route):
            if gene in sub1 or gene in sub2:
                print('Route:',i+1,'idx pos:',j,gene,concat2[count])
                child2[i][j]=concat2[count]
                count+=1
                
    return child1,child2

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
        filenames=['evrp-benchmark-set/E-n22-k4.evrp']
        EVRP.read_problems(filenames[0])
        EVRP.finalCluster=EVRP.clustering()
        EVRP.finalCluster=EVRP.balancingApproach()
        for i in range(len(EVRP.finalCluster)-1):
            EVRP.finalCluster[i]=EVRP.local2Opt(EVRP.finalCluster[i])
        print(EVRP.finalCluster)
        print(f'Step 3, local 2-opt cluster: {EVRP.finalCluster}') 
       
        pass


if __name__=='__main__':
    evrp=EVRP()
    MAX_GENERATION=25000
    POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.1
    ga=GA(MAX_GENERATION,POP_SIZE,CROSS_RATE,MUT_RATE,EVRP,random_state=42)
    ga.newGeneration()
