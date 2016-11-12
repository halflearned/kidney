import numpy as np
import itertools as it
from collections import namedtuple
import pandas as pd
import scipy.stats as ss

class KidneyExchange:
    
    Pair = namedtuple('Pair', ['patient', 'donor'])
    
    over_demanded = [("a","o"), ("b","o"), ("ab","o"), ("ab","a"),("ab","b")]
    under_demanded = [("o","a"), ("o","b"),("o","ab"),("a","ab"),("b","ab")]
    self_demanded = [("o","o"),("a","a"),("b","b"),("ab","ab")]
    recip_demanded = [("a","b"),("b","a")]

    types = ["o", "a", "b", "ab"]
    #probs = [0.441, 0.42, 0.099, 0.04]
    probs=  [ 0.4814, 0.3373,  0.1428,  0.0385]

    def __init__(self, init_size = 30):
        self.init_size = init_size
        self.types = ["a","b","ab","o"]
        self.pairs = [self.Pair(*p) for p in it.product(self.types, self.types)]
        self.state = pd.Series(data = 0, index = self.pairs)
        self.exchanges = self.list_two_way_exchanges()   
        self.poisson_params = self.get_poisson_params()
        self.action_space = np.arange(len(self.exchanges))
        self.limit = np.inf
        self.done = False
                
    def can_receive(self, patient, donor):
        """
        Binary relation denoted by ► in Unver (2010) 
        """
        if patient == "ab":
            return True
        if donor == "o":
            return True
        elif patient == donor:
            return True
        else:
            return False
        
    def list_two_way_exchanges(self):
        """
        Computes all possible two-way exchanges. 
        Make this state-dependent and available for k-way exchanges later.
        """
        two_way_exchanges = []
        for p1 in self.pairs:
            for p2 in self.pairs:
                if self.can_receive(p1.patient, p2.donor) and self.can_receive(p2.patient, p1.donor):
                    if (p2, p1) not in two_way_exchanges:
                        two_way_exchanges += [(p1, p2)]
        return two_way_exchanges
    
    def get_poisson_params(self):
        """
        For now, just a bunch of lambda \sim Unif[0,1]
        """
        return pd.Series(data = np.random.uniform(0, .5, size = len(self.state)), index = self.pairs)
    
    def add_new_pairs(self):
        """
        Draws number of new patients from the Poisson(dt*lambda) process
        """
        #k = np.random.randint(16)
        #self.state[k] += 1
        self.state += ss.poisson(self.poisson_params).rvs()
        return self.state
    
    def match(self, pair1, pair2):
        """
        Removes patients from state if available, None otherwise
        """
        self.state[pair1] -= 1
        self.state[pair2] -= 1
        return self.state
    
    def get_action(self, i):
        if i > len(self.exchanges) - 1:
            return None
        return self.exchanges[i]
       
    def reset(self, populated = True):
        self.state = pd.Series(data = 0, index = self.pairs)
        if populated:
            blood_pairs = np.random.choice(self.types, p  = self.probs, size = 2*self.init_size).reshape(-1,2)
            data = pd.Series([(pat, don) for pat, don in blood_pairs]).value_counts()
            for pair in data.index:
                self.state[pair] = data[pair]
        return self.get_state()
    
    def get_state(self):
        return self.state.values.reshape(1, -1)
    
    def available_actions(self, s = None):
        if not s: s = self.state
        avail = []
        for i, (pair1, pair2) in enumerate(self.exchanges):
            if pair1 == pair2:
                avail.append(s[pair1] > 1)
            else:
                avail.append(s[pair1] > 0 and s[pair2] > 0)
        return np.array(avail).reshape(1, -1)
    
    def step(self, a, new_pairs = False):
        # Check if action indeed available
        avail = self.available_actions().flatten()
        assert avail[a], "Attempted unavailable action"

        # Match pair
        pair = self.get_action(a)
        self.match(*pair)
  
        # Update state if toggled
        if new_pairs:
            self.add_new_pairs()
            
        # Recheck if game is over
        avail = self.available_actions()
        zero_avail_actions = np.sum(avail) == 0
        burst = np.any(self.state > self.limit) 
        done = burst or zero_avail_actions
        next_state = self.get_state()

        # Reward positive if not burst, else negative
        reward = +1 if not burst else -1
        return  next_state, reward, done, avail

    def get_state(self):
        return self.state.values.reshape(1, -1)
    
    def get_max_patients(self):
        """
        Following Roth, Sonmez, Unver (2007)
        See: https://www2.bc.edu/~unver/research/kidney-exchange-survey.pdf page 17
        """
        max_patients = 0
        for pair in self.over_demanded:
            max_patients += 2*self.state[pair]
        for pair in self.self_demanded:
            max_patients += 2*(self.state[pair]//2)
        max_patients += 2*np.minimum(self.state[("a","b")], self.state[("b","a")])
        return max_patients

    def how_demanded(self, p):
        if p in self.over_demanded:
            return "over_demanded"
        elif p in self.under_demanded:
            return "under_demanded"
        elif p in self.self_demanded:
            return "self_demanded"
        elif p in self.recip_demanded:
            return "recip_demanded"

    def random_matching(self, iterations = 100):
        match_prob = []
        for _ in range(iterations):
            state = self.reset(populated = True)
            avail = self.available_actions()
            max_patients = self.get_max_patients()
            terminal = False
            matched = 0
            while not terminal:
                avail = avail.flatten()
                actions = np.arange(len(self.exchanges))[avail]
                a = np.random.choice(actions)
                next_state, reward, terminal, avail = self.step(a, False)
                matched += 2
            match_prob += [matched / max_patients]
        return match_prob



                    
