import numpy as np
import math

def sh_n(min_budget, max_budget, eta):
    s_max = int(np.floor(math.log(max_budget/min_budget, eta)))
    s_range = np.flip(range(s_max+1))
    for s in s_range:
        n = int(np.ceil((s_max+1) * eta**s / (s+1)))
        budget = eta**s * max_budget
        print("Run %d configurations with an initial budget of %d.\n" % (n, budget))
