import numpy as np
from math import pi, sqrt, cos, sin
from functools import reduce
import re
from os import remove

def mu_for_vector(v, w) -> bool:
    """
    This function is criterion of if two vecter is mutually unbiased.
    Please put two same dimensional vector, v and w, as parameter.
    """
    d = w.shape[0]
    if abs(np.inner(v.conjugate(), w)) - 1/sqrt(d) < 1e-12:
        return True
    return False

def mu_for_bases(M, N) -> bool:
    """
    This function is criterion of if two bases is mutually unbiased.
    Please put two same dimensional square matrix, M and N, as parameter.
    """
    d = M.shape[0]
    for i in range(d):
        for j in range(d):
            if not mu_for_vector(M[:, i], N[:, j]):
                return False
    return True

def mu_new_into_MUB(MUB, A) -> bool:
    """
    MUB is a list with matrices are already MU,
    and this function is checking if matrix A is MU to all matrices of MUB.
    """
    for B in MUB:
        if not mu_for_bases(A, B):
            return False
    return True

def already_in(L: list, A):
    for l in L:
        if np.array_equiv(l, A):
            return True
    return False

def is_MUB(MUB) -> bool:
    l = [MUB[0]]
    for B in MUB:
        if mu_new_into_MUB(l, B):
            l.append(B)
        else:
            return False
    return True

def diag(tup):
    """
    Turn a one dimensional tuple or list into a diagonal matrix.
    """
    d = len(tup)
    array = [[0 for _ in range(d)] for _ in range(d)]
    for i in range(d):
        array[i][i] = tup[i]
    return np.array(array)

def generate_candidate(test_val_list, length, initial = []):
    """
    Generating all
    """
    def append(previous, test_val_list, length):
        new_list = []
        for item in previous:
            for test_val in test_val_list:
                new_list.append(item + [test_val])
        if len(new_list[0]) == length:
            return new_list
        else:
            return append(new_list, test_val_list, length)
    return append([initial], test_val_list, length)

def generate_D_list(omegas, p, r):
    def repetition(l):
        for item in l:
            if l.count(item) > 1:
                return False
        return True
    pos = generate_candidate([i for i in range(1, p**r)], p**(r-2))
    for l in pos:
        l.sort()
    positions = list(filter(repetition, [list(t) for t in set(tuple(l) for l in pos)]))
    permutations = generate_candidate(omegas, p**(r-2))
    D_list = []
    for pos in positions:
        for per in permutations:
            D = [1 for _ in range(p**r)]
            for i in range(p**(r-2)):
                D[pos[i]] = per[i]
            D_list.append(D)
    return D_list

def turn_into_pair(L, num):
    l = len(L)
    result = [[i] for i in L]
    t = 1
    with open(f"temp{t}.txt", "w") as f:
      for item in result:
        print(item, file = f)
      
    
    while t < num:
        with open(f"temp{t}.txt", "r") as f:
            with open(f"temp{t+1}.txt", "w") as f2:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    item = line_to_list(line)
                    idx = L.index(item[-1])
                    for i in range(idx+1, l):
                        print(item + [L[i]], file = f2)
        t += 1
        remove(f"temp{t-1}.txt")
    return t

def line_to_list(line):
    pattern = r"(\d+)"
    return list(map(int, re.findall(pattern, line)))
    
p, r = 2, 3
n = p**r # dimension n = p^r

I2 = np.array([(1, 0), (0, 1)], dtype = complex)
F2 = 1/sqrt(2)*np.array([(1, 1), (1, -1)], dtype = complex)
H2 = 1/sqrt(2)*np.array([(1, 1), (1j, -1j)], dtype = complex)
mubp = [F2, H2] # forming mub in dimension p

# omegas = [(cos(2*pi/p) + 1j*sin(2*pi/p))**i for i in range(1, p)] # this is for general case
omegas = [-1] # this is entry of diagonal matrix for p=2
D_list = generate_D_list(omegas, p, r) # find all possible diagonal matrix
t = turn_into_pair([i for i in range(len(D_list))], p**(r-1)-1)
number_of_succ = 0

with open('result_of_MUB.txt', 'w') as f_mub:
    with open('result_of_D.txt', 'w') as f_D:
        candidate_list = list(map(lambda l: reduce(np.kron, l), generate_candidate(mubp, r)))
        with open(f'temp{t}.txt', 'r') as temp:
            while True:
                line = temp.readline()
                if not line:
                    break
                index_of_D_list = line_to_list(line)
                mubn = [reduce(np.kron, [mubp[0] for _ in range(r)])]
                for M in candidate_list:
                    if already_in(mubn, M):
                        continue
                    if mu_new_into_MUB(mubn, M):
                        mubn.append(M)
                    else:
                        for i in index_of_D_list:
                            if mu_new_into_MUB(mubn, np.matmul(diag(D_list[i]), M)):
                                mubn.append(np.matmul(diag(D_list[i]), M))
                                break        
                    
                if len(mubn) == p**r:
                    number_of_succ += 1
                    for i in index_of_D_list:
                        print(D_list[i], file=f_D)
                    print("\n --- \n", file=f_D)
                        
                    print(f"Mutually Unbiased bases of C^{p**r}", file=f_mub)
                    for (i, B) in enumerate(mubn):
                        print(f"M_{i} = \n{1/B[0, 0]*B}", file=f_mub)
                
remove(f"temp{t}.txt")
print(number_of_succ)