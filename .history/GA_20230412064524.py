from EVRP import EVRP

import random

class GA:
    def __init__(self,MAX_GENERATION,POP_SIZE,CROSS_RATE,GEN_MUT_RATE,CX_RATE,evrp:EVRP,random_state=42):
        random.seed(random_state)
        self.MAX_GENERATION=MAX_GENERATION
        self.POP_SIZE=POP_SIZE
        self.CROSS_RATE=CROSS_RATE
        self.GEN_MUT_RATE=GEN_MUT_RATE
        self.CX_RATE=CX_RATE
        self.evrp=evrp

    def generateChromosome(self):
        #TODO do I need to move 4.1,4.2 and 4.3 to this GA part
        '''Generating chromosomes based on 4.1 Clustering, 4.2 Balancing and 4.3 Local Search'''
        pass