import pandas as pd
import numpy as np
from numba import jit

# Data from https://www.kaggle.com/c/santa-gift-matching/data
# Accessed via kaggle API

# children = np.array(pd.read_csv("child_wishlist_v2.csv", header=None, index_col=0))
# gifts = np.array(pd.read_csv("gift_goodkids_v2.csv", header=None, index_col=0))

# The jitclass is still in early development, couldn't get it to work, so we're going to 
# remove the class

@jit(nopython=True)
def numba_stable_match(a_prefs, b_prefs):
    '''Numba matching function'''
    a_rows, a_cols = np.int64(a_prefs.shape[0]), np.int64(a_prefs.shape[1])
    b_rows, b_cols = np.int64(b_prefs.shape[0]), np.int64(b_prefs.shape[1])
    rematch, a_pref_index = np.arange(a_rows), np.zeros(a_rows, dtype=np.int64)
    matches, match_round = np.zeros(a_rows, dtype=np.int64), np.int64(0)
    b_match_counter = np.zeros((b_rows, 2), dtype=np.int64)
    b_match_counter[:, 0] = np.arange(b_rows)
    b_set = np.arange(b_rows)
    
    while(True):
        # a proposes to the first member of b on their list that they have not proposed
        # to already
        for a in rematch:
            index = a_pref_index[a]
            if index < a_cols: 
                matches[a] = a_prefs[a, index]
            else:
                k = np.int64(index - a_cols + 1)
                set_diff = np.zeros(b_rows - a_cols + 1, dtype=np.int64)
                set_diff[0] = a
                x = np.bincount(np.concatenate((b_set, a_prefs[a])))
                x = np.where(x == 1)[0]
                set_diff[1:] = x
                matches[a] = set_diff[k]
            a_pref_index[a] += 1

        # Each b can has up to n preferences, and can be matched up to n times, hence we can
        # remove a match from b if a is not in b's preference list. 
        # I have aimed to avoid looping through the data multiple times, 
        # as this would scale poorly.

        ## In v0, I was not aware of bincount function, just doing a value count was
        ## taking 80 percent of the function time to run
        b_match_counter[:, 1] = np.bincount(matches)

        # Now we need to get the b's with too many proposals
        excess, i = np.where(b_match_counter[:,1] > b_rows)[0], np.int64(0)

        # We're going to get rid of any excess matches that aren't in b's preference list
        # at all until it has the maximum number of proposals. This works as long as the 
        # number of each b available is the same as the number of preferences of each b.
        rematch = np.zeros_like(rematch)
        for b in excess:

            # identify overmatched a and b's preferences
            overmatched, overmatched_prefs = np.where(matches == b)[0], b_prefs[b, :]

            for a in overmatched:
                if np.sum(overmatched_prefs == a) == 0:
                    rematch[i] = a
                    i += 1
                    b_match_counter[b, 1] -= 1
                if b_match_counter[b, 1] == b_rows: break

        # Now we record which a need rematching for the next loop
        rematch = rematch[:i]
        if len(rematch) == 0: 
                print("\nStable match found")
                print("Rounds taken:", match_round) 
                break
        match_round += 1
        print(match_round)
        print(len(rematch))
    return matches

def valid_match(matches, b_prefs):
    '''Testing function to identify whether the match is valid. This is achieved by
    calculating how times each member of b has been matched - if the match is valid
    then this will be equal to the number of b preferences'''
    b_rows, b_cols = np.int64(b_prefs.shape[0]), np.int64(b_prefs.shape[1])
    b_match_counter = np.int_(np.zeros((b_rows, 2)))
    b_match_counter[:, 0] = np.arange(b_rows)
    b_match_counter[:, 1] = np.bincount(matches)
    if np.max(b_match_counter[:, 1]) == b_cols and np.min(b_match_counter[:, 1] == b_cols): 
        print("Current match is valid")
        return True
    else: 
        print("Current match is not valid")
        return False

def is_stable(matches, a_prefs, b_prefs):
    '''This method checks to see if any a who is currently matched with b can find
    a match with b prime where a likes b prime more than b and b prime likes a more than
    their current match with a prime'''
    for i, prefs in enumerate(a_prefs):
        # Check to see how happy a is with their current match, if it's not in their list
        # then set the rank to be 1 below their last preference
        try: rank = np.argwhere(prefs == matches[i])[0][0]
        except IndexError: rank = len(a_prefs)

        for match in matches:
            
            # Remove own match
            match = np.concatenate((matches[:i], matches[i+1:]))
            
            # Find the rank of b' in their preferences, if it's not there, then move on
            try: b_rank = np.argwhere(prefs == match[1])[0][0]
            except IndexError: continue
            
            # If a likes b' more than b then we need to check if b' likes a more than a'
            # If a isn't even a preference of b' then it's a hopeless swap
            # If a' isn't even a preference of b' then the solution is not stable
            if b_rank < rank:
                b = np.int(match[1])
                b_pref = b_prefs[b]
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
    prefs = np.int64(round(ngifts ** (2/3)))
    children = np.zeros((nchildren, prefs), dtype=np.int64)
    for i in range(nchildren):
        arr = np.arange(ngifts, dtype=np.int64)
        np.random.shuffle(arr)
        arr = arr[:(prefs)]
        children[i] = arr

    gifts = np.zeros((ngifts, ngifts), dtype=np.int64)
    for i in range(ngifts):
        arr = np.arange(nchildren, dtype=np.int64)
        np.random.shuffle(arr)
        arr = arr[:(ngifts)]
        gifts[i] = arr
    
    return children, gifts

def TestMatch(a, b):
    '''The algorithm should produce a valid and stable matching, where all a's and b's are
    fully matched, and no a can leave their b for b prime even if they prefer b prime, 
    because b prime prefers a prime over a'''
    match = numba_stable_match(a, b)
    assert valid_match(match, b)
    assert is_stable(match, a, b)
        
def benchmark(a, b):
    numba_stable_match(a, b)

if __name__ == "__main__":
    c, g = MiniMatchData(round(6 ** 6), round(6 ** 3), randomstate=63)
    TestMatch(c, g)