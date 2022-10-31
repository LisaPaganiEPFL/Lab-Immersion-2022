import numpy as np
import random
from numpy import linalg as LA
import time
from numpy.linalg import inv
import copy
from numba import njit
from joblib import Parallel, delayed

#Through this code, we consider a population with N_demes demes, and N_types types of individuals (if we have mutants and wild-types, N_types=2)

#We model the population composition through an array, e.g. [[1, 99], [0,100]] has 2 demes ; the first one has 1 mutant and 99 wild-types, the second has 100 wild-types.

#The graph model and migration probabilities is given by a migration matrix

#______________________________________Migration matrices________________________________________________

#These codes simply define migration matrices for each type of graph

def define_clique(N_demes,m):
    migration_matrix=np.zeros((N_demes,N_demes))
    for i in range(N_demes):
        for j in range(N_demes):
            if not i==j :
                migration_matrix[i,j]=m
        migration_matrix[i,i]=1-(N_demes-1)*m
    return migration_matrix

def define_cycle(N_demes,mA, mC):
    migration_matrix=np.zeros((N_demes,N_demes))
    for i in range(N_demes-1):
        migration_matrix[i,i+1]=mA
        migration_matrix[i+1,i]=mC
    migration_matrix[N_demes-1,0]=mA
    migration_matrix[0,N_demes-1]=mC
    for j in range(N_demes):
        migration_matrix[j,j]=1-mA-mC
    return migration_matrix

#The option "equal_contribution" is True if all demes send the same number of individuals in average. It is False if they all receive the same number in average.
#True --> sum on lines = 1
#False --> sum on columns = 1
def define_star(N_demes,mI,mO,equal_contribution=True):
    migration_matrix=np.zeros((N_demes,N_demes))
    for i in range(1,N_demes):
        migration_matrix[i,0]=mI
        migration_matrix[0,i]=mO
        if equal_contribution:
            migration_matrix[i,i]=1-mI
        else:
            migration_matrix[i,i]=1-mO
    if equal_contribution:
        migration_matrix[0,0]=1-(N_demes-1)*mO 
    else:
        migration_matrix[0,0]=1-(N_demes-1)*mI 
    return migration_matrix

def define_grid(D,mN,mS,mE,mW):
    migration_matrix=np.zeros((D,D))
    for i in range(D):
        migration_matrix[i,i//int(np.sqrt(D))*int(np.sqrt(D))+((i+1)%int(np.sqrt(D)))]=mE
        migration_matrix[i,i//int(np.sqrt(D))*int(np.sqrt(D))+((i-1)%int(np.sqrt(D)))]=mW
        migration_matrix[i,(i+int(np.sqrt(D)))%D]=mS
        migration_matrix[i,(i-int(np.sqrt(D)))%D]=mN
        migration_matrix[i,i]=1-(mE+mS+mN+mW)
    return migration_matrix

#_________________________________________Growth and dilution/migration_____________________________________________

#Deterministic growth event for t time steps
#fitnesses = list with the mutant and wild-type fitnesses
#in_numbers = initial numbers before the growth phase

@njit
def growth_event(in_numbers,fitnesses,t):
    return (in_numbers * np.exp(fitnesses * t)).astype(np.float64)

#Dilution and migration event
#in_numbers = initial numbers at the end of the growth phase
#Nmin = bottleneck number

@njit
def dilution_migration_event(in_numbers,migration_matrix,Nmin_table):
    N_demes, N_types= np.shape(in_numbers)

    #Numbers that will be kept at the end of the migration/dilution phase
    new_numbers=np.zeros((N_demes, N_types), dtype=np.int64)

    #Migrants from a deme to another
    migrants_ij=np.empty(2, dtype=np.int64)
    
    #Sample whatever comes out of deme i
    for i in range(N_demes):

        #Total number of individuals on deme i
        Ni=np.sum(in_numbers[i,:])
        if Ni<1:
            print('extinct deme', i)
        #Ratio of mutants in deme i
        p=in_numbers[i,0]/Ni

        #Migration towards deme j
        for j in np.arange(N_demes):
            
            mij=migration_matrix[i,j]

            #migrants from i to j, mutants and wild-types
            p0=max(min(Nmin_table[i]*p*mij/Ni,1),0)
            p1=max(min(Nmin_table[i]*(1-p)*mij/Ni,1),0)
            migrants_ij[0]=np.random.binomial(Ni, p0,1)[0]
            migrants_ij[1]=np.random.binomial(Ni, p1, 1)[0]

            #Updating the numbers in j with the migrants that arrived from i
            new_numbers[j, 0]+=migrants_ij[0]
            new_numbers[j, 1]+=migrants_ij[1]

    return new_numbers

#_________________________________________Extinction or fixation____________________________________________________________

#Checking if the mutant is completely extinct
@njit
def extinct_mutant(numbers):
    N_demes = numbers.shape[0]
    for i in range(N_demes):
        if numbers[i,0]>0:
            return False
    return True

#Checking if the wild-type is completely extinct
@njit
def extinct_wild(numbers):
    N_demes, N_types= numbers.shape
    for i in range(N_demes):
        for j in range(N_types -1):
            if numbers[i,j+1]>0:
                return False
    return True

#____________________________________Complete simulation for one trajectory_____________________________________________

#We do a number of cycles starting from inital numbers in_numbers ; until the mutant is fixed or dies out
#nb_cycles : a priori number of how many cycles we do before observing fixation or extinction
#Nmin = bottleneck
#growth_factor = t the growth parameter

@njit
def cycle(in_numbers, migration_matrix, fitnesses, nb_cycles, growth_factor, Nmin, start_follow_numbers, size_follow_numbers, start_cycle, print_frequency, save_dynamics=False, dilution_std=None, fitness_range=None):
    N_demes, N_types= np.shape(in_numbers)
    s = fitnesses[0] - 1
    #Setting up array to track numbers, can be saved
    #Its size = size_follow_parameters
    if start_follow_numbers is None:
        follow_numbers=np.zeros((size_follow_numbers,N_demes,N_types), dtype=np.int64)
    else:
        follow_numbers=start_follow_numbers.copy()

    #Boolean that says if the mutant is fixed or extinct at the end
    fixation=True
    numbers=in_numbers.copy()

    #Booleans that check if mutants or wild-types are extinct
    keep_going=True

    for i in range(nb_cycles):
        end_cycle = nb_cycles
        #One cycle: growth then dilution/migration
        if fitness_range is None:  
            numbers1=growth_event(numbers,fitnesses,growth_factor)
            
        else:
            fit = np.array([np.random.uniform(1+s-fitness_range,1+s+fitness_range,1)[0],1.])
            numbers1=growth_event(numbers,fit,growth_factor)

            
        if dilution_std is None:
            Nmin_table = np.array([Nmin]*N_demes)
        else:
            Nmin_table = np.random.normal(loc=Nmin, scale=dilution_std, size=N_demes).astype(np.int64)
            Nmin_table[np.where(Nmin_table<=0)] = 1
        
        numbers=dilution_migration_event(numbers1,migration_matrix,Nmin_table)

        #Saving new numbers in the tracking array
        if (start_cycle+i)%print_frequency==0 and ((i+start_cycle)/print_frequency)<size_follow_numbers:
            follow_numbers[int(i+start_cycle), :, :]=numbers

        #Checking if mutants are extinct
        if extinct_mutant(numbers):
            keep_going=False
            fixation=False
            end_cycle=i+start_cycle
            follow_numbers[int((i+start_cycle))+1:]=numbers
            break
        #Checking if wild-types are extinct
        if extinct_wild(numbers):
            keep_going=False
            fixation=True
            #print("Fixation occurred at cycle ", i+start_cycle)
            end_cycle=i+start_cycle
            follow_numbers[int((i+start_cycle))+1:]=numbers
            break
    
    #If mutants are not extinct or fixed at the end of nb_cycles cycles, we keep going
    if keep_going:
        follow_numbers, end_cycle, fixation= cycle(numbers, migration_matrix, fitnesses, nb_cycles, growth_factor, Nmin, follow_numbers, size_follow_numbers, start_cycle+end_cycle, print_frequency, save_dynamics, dilution_std, fitness_range)
    return follow_numbers, end_cycle, fixation

#________________________________________Fixation probability computed on several simulations_______________________________________________


#We compute the fixation probability starting from in_numbers, on nb_sim simulations.
#For each simulation we use the function above, and see if the mutant is fixed or not

def fixation_probability(in_numbers, folder, migration_matrix, fitnesses, nb_sim, nb_cycles, growth_factor, Nmin, size_follow_numbers=10000, print_frequency=1, save_dynamics=False, dilution_std=None, fitness_range=None):
    #Counter for fixation trajectories
    fix_count=0

    #Keeping track of fixation or extinction cycle
    fix_cycle=np.zeros(nb_sim)
    ex_cycle=np.zeros(nb_sim)

    for i in range(nb_sim):
        start_cycle=0
        start_follow_numbers=None
        follow_numbers, end_cycle, fixation = cycle(in_numbers, migration_matrix, fitnesses, nb_cycles, growth_factor, Nmin, start_follow_numbers, size_follow_numbers, start_cycle, print_frequency, save_dynamics, dilution_std, fitness_range)
        
        if fixation :
            fix_count+=1
            fix_cycle[i] = end_cycle
            if save_dynamics:
                #We save first 10000 fixation trajectories
                if fix_count<10000:
                    np.savez(folder+"/fix_"+str(fix_count),follow_numbers)
        else:
            ex_cycle[i] = end_cycle
            if save_dynamics:
                #We save first 10000 fixation trajectories
                if i-fix_count<10000:    
                    np.savez(folder+"/ex_"+str(i-fix_count),follow_numbers)

    #Number of extinctions
    ex_count=nb_sim-fix_count

    if fix_count>0:
        average_fixation_cycle = np.sum(fix_cycle)/fix_count
    else :
        average_fixation_cycle = 0
    if ex_count>0:
        average_extinction_cycle = np.sum(ex_cycle)/ex_count
    else:
        average_extinction_cycle=0

    #Probability of fixation
    proba = fix_count/nb_sim

    return average_extinction_cycle,average_fixation_cycle, proba   
