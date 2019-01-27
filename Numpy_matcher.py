# import pandas as pd
import numpy as np
from timeit import timeit


# Data from https://www.kaggle.com/c/santa-gift-matching/data
# Accessed via kaggle API

# children = np.array(pd.read_csv("child_wishlist_v2.csv", header=None, index_col=0))
# gifts = np.array(pd.read_csv("gift_goodkids_v2.csv", header=None, index_col=0))

class Matcher:
    '''Class for matching two arrays, a and b that each have preferences for some members
    of another array.'''

    def __init__(self, a_prefs, b_prefs):
        self.a_prefs = a_prefs
        self.b_prefs = b_prefs
        self.a_rows = len(a_prefs)
        self.b_rows = len(b_prefs)
        self.a_cols = len(a_prefs[0])
        self.b_cols = len(b_prefs[0])
        self.rematch = np.arange(self.a_rows)
        self.a_pref_index = np.zeros(self.a_rows, dtype=int)
        self.b_set = np.arange(self.b_rows)
        self.round = 0
        self.matches = np.zeros(self.a_rows, dtype=int)
        self.b_match_counter = np.zeros((self.b_rows, 2), dtype=int)
        self.b_match_counter[:, 0] = self.b_set

    def numpy_match(self):
        '''All numpy with various optimistaions'''
        # a proposes to the first member of b on their list that they have not proposed
        # to already
        for a in self.rematch:
            index = self.a_pref_index[a]
            if index < self.a_cols: pref = self.a_prefs[a, index]
            
            # Code now usess a set difference with the list of gifts
            # instead of creating an entire array of non-preferences to utilise
            else: pref = np.setdiff1d(self.b_set, self.a_prefs[a])[index - self.a_cols]
            self.matches[a] = pref
            self.a_pref_index[a] += 1

        # Each b can has up to n preferences, and can be matched up to n times, hence we can
        # remove a match from b if a is not in b's preference list. 
        # I have aimed to avoid looping through the data multiple times, 
        # as this would scale poorly.

        ## In v0, I was not aware of bincount function, just doing a value count was
        ## taking 80 percent of the function time to run
        self.b_match_counter[:, 1] = np.bincount(self.matches)                     

        # Now we need to get the b's with too many proposals
        excess = np.where(self.b_match_counter[:,1] > self.b_rows)[0]

        # We're going to get rid of any excess matches that aren't in b's preference list
        # at all until it has the maximum number of proposals. This works as long as the 
        # number of each b available is the same as the number of preferences of each b.
        self.rematch, i = np.zeros_like(self.rematch), 0
        for b in excess:

            # identify overmatched a and b's preferences
            overmatched, overmatched_prefs = np.where(self.matches == b)[0], self.b_prefs[b, :]

            for a in overmatched:
                if np.sum(overmatched_prefs == a) == 0:
                    self.rematch[i] = a
                    i += 1
                    self.b_match_counter[b, 1] -= 1
                if self.b_match_counter[b, 1] == self.b_rows: break

        # Now we record which a need rematching for the next loop
        self.rematch = self.rematch[:i]
        self.round += 1
    
    def stable_match(self):
        while(True): 
            self.numpy_match()
            print(f"Round {self.round} complete. Unmatched total: {len(self.rematch)}\t\t", 
                    end="\r")
            if len(self.rematch) == 0: 
                print("\nStable match found")
                print("Rounds taken:", self.round) 
                break
    
    def valid_match(self):
        '''Testing function to identify whether the match is valid. This is achieved by
        calculating how times each member of b has been matched - if the match is valid
        then this will be equal to the number of b preferences'''
        self.b_match_counter[:, 1] = np.bincount(self.matches)
        if np.max(self.b_match_counter[:, 1]) == self.b_cols and np.min(self.b_match_counter[:, 1] == self.b_cols): 
            print("Current match is valid")
            return True
        else: 
            print("Current match is not valid")
            return False
    
    def is_stable(self):
        '''This method checks to see if any a who is currently matched with b can find
        a match with b prime where a likes b prime more than b and b prime likes a more than
        their current match with a prime'''
        for i, prefs in enumerate(self.a_prefs):
            # Check to see how happy a is with their current match, if it's not in their list
            # then set the rank to be 1 below their last preference
            try: rank = np.argwhere(prefs == self.matches[i, 1])[0][0]
            except IndexError: rank = self.a_cols
            
            # Remove own match
            matches = np.concatenate((self.matches[:i], self.matches[i+1:]))
            for match in matches:
                
                # Find the rank of b' in their preferences, if it's not there, then move on
                try: b_rank = np.argwhere(prefs == match[1])[0][0]
                except IndexError: continue
                
                # If a likes b' more than b then we need to check if b' likes a more than a'
                # If a isn't even a preference of b' then it's a hopeless swap
                # If a' isn't even a preference of b' then the solution is not stable
                if b_rank < rank:
                    b = np.int(match[1])
                    b_pref = self.b_prefs[b]
                    try: b_rank_a = np.argwhere(b_pref == i)[0][0]
                    except IndexError: continue
                    
                    try: b_rank_aprime = np.argwhere(b_pref == match[0])[0][0]
                    except IndexError:
                        print(f"Match unstable: a[{i}] can propose to b[{b}] who is currently paired with a[{int(match[0])}]")
                        return False
                    
                    if b_rank_a < b_rank_aprime:
                        print(f"Match unstable: a[{i}] can propose to b[{b}] who is currently paired with a[{int(match[0])}]")
                        return False
            
        print("Current match is stable")
        return True


def MiniMatchData(nchildren, ngifts, randomstate=0):
    np.random.seed(randomstate)
    prefs = int(round(ngifts ** (2/3)))
    children = np.int_(np.zeros((nchildren, prefs)))
    for i in range(nchildren):
        arr = np.arange(ngifts)
        np.random.shuffle(arr)
        arr = arr[:(prefs)]
        children[i] = arr

    gifts = np.int_(np.zeros((ngifts, ngifts)))
    for i in range(ngifts):
        arr = np.arange(nchildren)
        np.random.shuffle(arr)
        arr = arr[:(ngifts)]
        gifts[i] = arr
    
    return children, gifts

def TestMatch(a, b):
    '''The algorithm should produce a stable matching, where no a can leave their b 
    for b prime even if they prefer b prime, because b prime prefers a prime over a'''
    matcher = Matcher(a, b)
    matcher.stable_match()
    assert matcher.is_stable()
    assert matcher.valid_match()
 
def benchmark(a, b, function="stable_numpy"):
    matcher = Matcher(a, b)
    if function == "numpy": matcher.numpy_match()
    elif function == "stable_numpy": matcher.stable_match()
    
if __name__ == "__main__":
    c, g = MiniMatchData(round(6 ** 6), round(6 ** 3), randomstate=63)
    TestMatch(c, g)
