import numpy as np
import parameters as pm

# ALl functions related to the connectivity matrix and the connection strengths

# Read in important parameters from parameters.py
# phasemax = pm.phasemax
# phasemin = pm.phasemin
# Estrength = pm.Estrength
# Istrength = pm.Istrength
# esumadd = pm.esumadd
# Iboost = pm.Iboost
# Icutoff = pm.Icutoff
# Edes_cutoff = pm.Edes_cutoff
# firstEcutof = pm.firstEcutof
# Easc_cutoff = pm.Easc_cutoff
# Espread = pm.Espread




def phase_to_conn(phase,phasemax,phasemin):
    """ Defines the relationship between desired phase and connection strength
    Currently configured for inhibition only, i.e. inhibition if 'out of phase'
    and no connection if 'in phase'.
    What phase relationship is considered 'in phase' and 'out of phase' is
    determined by the phasemax and phasemin parameters in parameters.py

    Input:
        -phase: a float between 0 and 1, representing a desired phase relationship
            phase = 0.5 is exactly out of phase
    Output:
        -float: represnting either inhibition (-1) or no connection (0)
    """
    if phase > phasemax: return 0.0
    elif phase < phasemin: return 0.0
    else: return -1.0




def eibalanced(mat, params):
    """ Normalizes the sum of the incoming E connections to the incoming I connections for each neuron
    esumadd (from parameters.py) is added to the E connection total before balancing with the I total

    Inputs:
        -mat: a connectivity matrix, to have its strengths normalized
        -params: model parameters
    Outputs:
        -balancedmat: a connectivity matrix that is a balanced version of mat
    """

    N = params["Numsegs"]
    T = params["Numtypes"]

    balancedmat = np.zeros([N, 2, T, N, 2, T])
    for n1 in range(N):
        for a1 in range(2):
            for p1 in range(T):
                esum, isum, tsum = 0, 0, 0
                for n2 in range(N):
                    for a2 in range(2):
                        for p2 in range(T):
                            c = mat[n2, a2, p2, n1, a1, p1]
                            tsum += np.abs(c)
                            if c > 0: esum += c        
                            if c < 0: isum += np.abs(c)
                esum += params["esumadd"]
                for n2 in range(N):
                    for a2 in range(2):
                        for p2 in range(T):
                            c = mat[n2, a2, p2, n1, a1, p1]
                            if c > 0:
                                if esum > 0:
                                    balancedmat[n2, a2, p2, n1, a1, p1]  = tsum*mat[n2, a2, p2, n1, a1, p1]/(2.0*esum)
                            elif c < 0:
                                if isum > 0:
                                    balancedmat[n2, a2, p2, n1, a1, p1]  = tsum*mat[n2, a2, p2, n1, a1, p1]/(2.0*isum)
    return balancedmat




def buildmatrix(params):
    """ Builds the connectivity matrix for the network based on hyperparameters

    Inputs:
        -params: model parameters
    """
    N = params['Numsegs']
    T = params['Numtypes']

    mat = np.zeros([N, 2, T, N, 2, T])
    for n1 in range(N):
        for a1 in range(2):
            for p1 in range(T):
                for n2 in range(N):
                    for a2 in range(2):
                        for p2 in range(T):
                            c = connection(n1, a1, p1, n2, a2, p2, params=params)
                            mat[n1, a1, p1, n2, a2, p2] = c

    if params["eibalanced"]: mat = eibalanced(mat, N, T) # normalize incoming E and I connections equal for each neuron
    return mat

def connection(n1, a1, p1, n2, a2, p2, params): # source: 1 , target: 2
    """ Finds the connection strength between two neurons
    
    Inputs:
        -n1: integer, the segment number of the source neuron
        -a1: integer 0 or 1, the side of the source neuron (left or right)
        -p1: integer, representing the type of the source neuron
        -n2: integer, the segment number of the target neuron
        -a2: integer 0 or 1, the side of the target neuron (left or right)
        -p2: integer, representing the type of the target neuron
        -params: model parameters

    Intermediates:
        -N: integer, number of neurons in the model
        -speedmix: float between 0 and 1, determing the relative size of the intra- and inter- speed module connections
        -cutoff: an integer representing the maximum distance (measured in number of segments) between neurons to have a connection
    Outputs:
        -float: representing the connection strength between the two neurons
    """
    N = params["Numsegs"]
    speedmix = params["speedmix"]
    cutoff = params["cutoff"]
    phasemax = params["phasemax"]
    phasemin = params["phasemin"]

    segdist = np.abs(n1-n2)
    if  segdist > cutoff: return 0.0
    if [n1, a1, p1] == [n2, a2, p2]: return 0.0 # No self-connections

    # if segdist > Icutoff: return 0.0 # Check global as well as type-specific cutoffs

    if a1 == a2: altdiff = 0  # altdiff = 0 if on same side (Ipsi), 1 if on opposite sides (Contra)
    else: altdiff = 1

    if n1 < n2: # direction is either 'asc', 'desc', or 'inseg'
        descdiff = n2 - n1
        dphase = (descdiff/N) + (altdiff/2)
        direction = 'desc'
    else:
        descdiff = n1 - n2
        dphase = (descdiff/N) + (altdiff/2)
        dphase = 1 - dphase
        if descdiff == 0: direction = 'inseg'
        else: direction = 'asc'
    
    dphase = dphase%1

    factor = 1.0
    conn = 0.0
    ablation = 1.0

    # Type specific factors 
    if 'E' in params["typenames"][p1]: # E outgoing
        ablation = params["Eablation"]
        if 'fast' in params["typenames"][p1]: Edes_cutoff = params["Edes_cutoff_fast"]
        elif 'slow' in params["typenames"][p1]: Edes_cutoff = params["Edes_cutoff_slow"]
        else: Edes_cutoff = params["Edes_cutoff"]
        if altdiff == 0:
            if (np.abs(n1-n2) < Edes_cutoff) and (direction != 'asc') and (np.abs(n1-n2) > params["firstEcutoff"]):
                conn = 1.0
            elif np.abs(n1-n2) < params["Easc_cutoff"]:
                conn = 1.0
            else:
                conn = 0.0
        else: conn = 0.0
        
        # Check if we have an E specific speedmix, otherwise use the default speedmix
        if 'Espeedmix' in params:
            speedmix = params['Espeedmix']
            
        factor *= params["Estrength"] # Estrength and Espread are redundant, so just keep Estrength

    elif 'I' in params["typenames"][p1]:
        if np.abs(n1-n2) > params["Icutoff"]: return 0.0
        conn = phase_to_conn(dphase,phasemax,phasemin) # turn a desired phase relationship into a connection strength

        if 'ipsi' in params["typenames"][p1]: # I ipsi outgoing
            if 'asc' in params["typenames"][p1]: # I ipsi asc
                ablation = params["Iipsiascablation"]
                if direction  == "desc":
                    conn = 0.0
            elif 'des' in params["typenames"][p1]: # I ipsi des
                ablation = params["Iipsidesablation"]
                if direction  == "asc":
                    conn = 0.0
            else:
                ablation = params["Iipsiablation"]

            if altdiff == 0:
                factor *= params["Istrength"]
            else: conn = 0.0

        elif 'contra' in params["typenames"][p1]: # I contra outgoing
            ablation = params["Icontraablation"]
            if altdiff == 1:
                factor *= params["Istrength"]
            else: conn = 0.0
        
        if 'Ispeedmix' in params:
            speedmix = params['Ispeedmix']

        
    if 'fast' in params["typenames"][p1] and 'fast' in params["typenames"][p2]: factor *= 1-speedmix
    if 'fast' in params["typenames"][p1] and 'slow' in params["typenames"][p2]: factor *= speedmix
    if 'slow' in params["typenames"][p1] and 'slow' in params["typenames"][p2]: factor *= 1-speedmix
    if 'slow' in params["typenames"][p1] and 'fast' in params["typenames"][p2]: factor *= speedmix
                          
    if 'modasym' in params:
        ma = params['modasym']
        if 'fast' in params["typenames"][p1] and 'slow' in params["typenames"][p2]: factor *= (1+ma)
        if 'slow' in params["typenames"][p1] and 'fast' in params["typenames"][p2]: factor *= (1-ma)
    
    return factor*conn*params["Cstrength"]*ablation



def driveshape(i, N):
    """ For spatially shaping the tonic drive

    Inputs:
        -i: integer, the segment number of the neuron
        -N: integer, the total number of segments in the model
    Outputs:
        -float: representing the strength of the tonic drive for the neuron
    """
    result = 1 - pm.driveslope*i/N
    if result < 0: return 0.0
    else: return result