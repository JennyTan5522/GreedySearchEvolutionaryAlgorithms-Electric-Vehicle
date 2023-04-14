from EVRP import EVRP

import random

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

    def newGeneration(self):
        ''''
        5.2 Initialization
            - Read file -> 
            - 
        '''
       
        pass


if __name__=='__main__':
    evrp=EVRP()
    MAX_GENERATION=25000
    POP_SIZE=200
    CROSS_RATE=0.95
    MUT_RATE=0.1
   
