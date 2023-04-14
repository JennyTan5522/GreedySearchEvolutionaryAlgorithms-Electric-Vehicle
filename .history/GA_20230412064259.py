from EVRP import EVRP

import random

class GA:
    def __init__(self,MAX_GENERATION,POP_SIZE,CROSS_RATE,GEN_MUT_RATE,CX_RATE,evrp:EVRP,random_state=42):
        random.seed(random_state)
        self.MAX_GENERATION=MAX_GENERATION
        self.