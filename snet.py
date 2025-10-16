import numpy as np
#import parameters as pm
import connectome as cn

# Defines a class for a spinal network model, and a couple functions for the simulating the model


class SpinalNetwork():
    """ Class for a model of a spinal network
    Contains hyperpermaters of the model, and contains methods for simulating the model

    params: dictionary with the parameters of the model
    N: number of segments
    T: number of neuron types
    rates: 3D array initial rates to start the simulation
    W: 6D numpy array of connection strengths. W[n1,a1,t1,n2,a2,t2] is the connection
        from the neuron in segment n1 on side a1 of type t1 
        to the neuron in segment n2 on side a2 of type t2
    mtcs: 3D array of membrane time constants
    drives: 3D array of tonic drive strengths

    fullrates: 4D array of the firing rates of the model over time. 
        fullrates[i,n,a,t] is the firing rate at time step i of the neuron of type t on side a of segment n
    
    update_model: updates model to match the parameters in params (run before simulating)
    timestep: increments the model by one timestep
    simulation: runs the whole simulation, calling timestep many times
    """
    def __init__(self, params):
        self.params = params
        self.N = params['Numsegs']
        self.T = params['Numtypes']

        # Generate random initial rates on interval [0,0.1)
        rng = np.random.default_rng(self.params['rngseed'])
        self.rates = 0.1*rng.random((self.N, 2, self.T))

        # Build the connectome
        self.W = cn.buildmatrix(self.params)

        # Set membrane time constants
        self.mtcs = 1.0*np.ones([self.N, 2, self.T])
        for p in range(self.T):
            if 'fast' in self.params['typenames'][p]: self.mtcs[:, :, p] = params["fastmtc"]*np.ones([self.N, 2])
            elif 'slow' in self.params['typenames'][p]: self.mtcs[:, :, p] = params["slowmtc"]*np.ones([self.N, 2])
            else: self.mtcs[:, :, p] = params["fastmtc"]*np.ones([self.N, 2]) # If there are no fast or slow types, default to the value for the fast population

        # Set drives (Ben's method drives both E and I neurons)
        self.drives = 1.0*np.ones([self.N, 2, self.T])
        if "drivefdrop" in self.params:
            for n in range(self.N):
                drivef_n = 1.0 + (self.params['drivefdrop']-1.0)*(n/(self.N-1)) # This factor ensures a linear decrease in drive from '[x]drivemod' at n=0 to '[x]drivemod' * 'drivefdrop' at n=N-1
                for a in range(2):
                    for p in range(self.T):
                        if 'fast' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["fastdrivemod"] * drivef_n # for now, apply the same fractional decrease in drive to all populations
                        elif 'slow' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["slowdrivemod"] * drivef_n
        else:
            for n in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        if 'fast' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["fastdrivemod"]
                        elif 'slow' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["slowdrivemod"]

    def update_model(self):
        """Updates the model to reflect the current values of self.params
            Run this after changing parameters and before simulating when doing parameter sweeps
        """
        self.N = self.params['Numsegs']
        self.T = self.params['Numtypes']

        # Generate random initial weights on interval [0,0.1) (if rngseed is unchanged, these will be the same for every run)
        rng = np.random.default_rng(self.params['rngseed'])
        self.rates = 0.1*rng.random((self.N, 2, self.T))

        # Build connectome
        self.W = cn.buildmatrix(self.params)

        # Set membrane time constants
        self.mtcs = 1.0*np.ones([self.N, 2, self.T])
        for p in range(self.T):
            if 'fast' in self.params['typenames'][p]: self.mtcs[:, :, p] = self.params["fastmtc"]*np.ones([self.N, 2])
            elif 'slow' in self.params['typenames'][p]: self.mtcs[:, :, p] = self.params["slowmtc"]*np.ones([self.N, 2])
            else: self.mtcs[:, :, p] = self.params["fastmtc"]*np.ones([self.N, 2]) # If there are no fast or slow types, default to the value for the fast population

        # Set drives (Ben's method drives both E and I neurons)
        self.drives = 1.0*np.ones([self.N, 2, self.T])
        if "drivefdrop" in self.params:
            for n in range(self.N):
                drivef_n = 1.0 + (self.params['drivefdrop']-1.0)*(n/(self.N-1)) # This factor ensures a linear decrease in drive from '[x]drivemod' at n=0 to '[x]drivemod' * 'drivefdrop' at n=N-1
                for a in range(2):
                    for p in range(self.T):
                        if 'fast' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["fastdrivemod"] * drivef_n # for now, apply the same fractional decrease in drive to all populations
                        elif 'slow' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["slowdrivemod"] * drivef_n
        else:
            for n in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        if 'fast' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["fastdrivemod"]
                        elif 'slow' in self.params['typenames'][p]: self.drives[n, a, p] = self.params["slowdrivemod"]


    def timestep(self, tstep, t, timedelay_off=False, delayts=2, Cstrength=0.5):
        """Increments the simulation by 1 time step
        Results are written into self.rates

        Inputs:
            -tstep: float, numerical timestep in ms (assumed to be 0.1 throughout)
            -t: integer,  representing the current time step of the simulation
            -timedelay_off: boolean, turns off the distance-based time delay if true
            -delayts: scale of the time delay
            -Cstrength: overall connection strength
        Outputs:
            -none
        """
        newrates = np.zeros([self.N, 2, self.T])

        if False:  # BL's version had lots of for loops
            # contribution from other neurons
            for i in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        mtc = self.mtcs[i, a, p]
                        newrates[i, a, p] += self.drives[i, a, p]/mtc
                        for i2 in range(self.N):
                            for a2 in range(2):
                                for p2 in range(self.T):
                                    w = self.W[i2, a2, p2, i, a, p]
                                    dts = int(delaymag(i, i2, timedelay_off = timedelay_off)* delayts)
                                    #dts = int(delaymag(i, a, p, i2, a2, p2)*self.delayts)
                                    if t-dts >= 0: sourcerate = self.fullrates[t-dts, i2, a2, p2]
                                    else: sourcerate = 0 # assumes all neurons were inactive for t < 0
                                    newrates[i, a, p] += self.Cstrength*w*sourcerate/mtc
            # decay term
            for i in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        newrates[i,a,p] = actfunc(newrates[i,a,p])
                        newrates[i,a,p] -= self.fullrates[t-1, i, a, p]/self.mtcs[i,a,p]

        else: # JM's version:                            
            # contribution from other neurons
            newrates += self.drives/self.mtcs
            for i in range(self.N):
                for i2 in range(self.N):
                    dts = int(delaymag(i, i2, timedelay_off=timedelay_off) * delayts)
                    if t-dts >= 0: sourcerate = self.fullrates[t-dts, :, :, :] 
                    else: sourcerate = np.zeros_like(self.fullrates[0,:,:,:]) # assumes all neurons were inactive for t < 0
                    newrates[i,:,:] += Cstrength * np.tensordot(self.W[i2,:,:,i,:,:], 
                                                                     sourcerate[i2,:,:], 
                                                                     axes=([0,1], [0,1])) / self.mtcs[i,:,:]
            # decay term
            newrates = actfunc(newrates)
            newrates -= self.fullrates[t-1, :, :, :] / self.mtcs  # decay term

        newrates = newrates*tstep
        newrates = self.fullrates[t-1] + newrates
        newrates[newrates<0] = 0
        self.fullrates[t] = newrates

    def timestep_diffdelay(self, tstep, t, fast_mask, slow_mask, timedelay_off=False, fastdelayts=2, slowdelayts=4, Cstrength=0.5):
        """Increments the simulation by 1 time step
        Results are written into self.rates

        Inputs:
            -tstep: float, numerical timestep in ms (assumed to be 0.1 throughout)
            -t: integer,  representing the current time step of the simulation
            -timedelay_off: boolean, turns off the distance-based time delay if true
            -delayts: scale of the time delay
            -Cstrength: overall connection strength
        Outputs:
            -none
        """
        newrates = np.zeros([self.N, 2, self.T])

        if False:  # BL's version had lots of for loops
            # contribution from other neurons
            for i in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        mtc = self.mtcs[i, a, p]
                        newrates[i, a, p] += self.drives[i, a, p]/mtc
                        for i2 in range(self.N):
                            for a2 in range(2):
                                for p2 in range(self.T):
                                    w = self.W[i2, a2, p2, i, a, p]
                                    dts = int(delaymag(i, i2, timedelay_off = timedelay_off)* delayts)
                                    #dts = int(delaymag(i, a, p, i2, a2, p2)*self.delayts)
                                    if t-dts >= 0: sourcerate = self.fullrates[t-dts, i2, a2, p2]
                                    else: sourcerate = 0 # assumes all neurons were inactive for t < 0
                                    newrates[i, a, p] += self.Cstrength*w*sourcerate/mtc
            # decay term
            for i in range(self.N):
                for a in range(2):
                    for p in range(self.T):
                        newrates[i,a,p] = actfunc(newrates[i,a,p])
                        newrates[i,a,p] -= self.fullrates[t-1, i, a, p]/self.mtcs[i,a,p]

        else:
            # contribution from other neurons
            newrates += self.drives/self.mtcs
            for i in range(self.N):
                for i2 in range(self.N):
                    delmag = delaymag(i, i2, timedelay_off=timedelay_off)
                    fdts = int(delmag * fastdelayts)
                    sdts = int(delmag * slowdelayts)
                    sourcerate = np.zeros_like(self.fullrates[0,:,:,:])
                    if t-fdts >= 0: sourcerate += self.fullrates[t-fdts, :, :, :] * fast_mask
                    if t-sdts >= 0: sourcerate += self.fullrates[t-sdts, :, :, :] * slow_mask
                    else: sourcerate = np.zeros_like(self.fullrates[0,:,:,:]) # assumes all neurons were inactive for t < 0
                    newrates[i,:,:] += Cstrength * np.tensordot(self.W[i2,:,:,i,:,:], 
                                                                     sourcerate[i2,:,:], 
                                                                     axes=([0,1], [0,1])) / self.mtcs[i,:,:]
            # decay term
            newrates = actfunc(newrates)
            newrates -= self.fullrates[t-1, :, :, :] / self.mtcs  # decay term

        newrates = newrates*tstep
        newrates = self.fullrates[t-1] + newrates
        newrates[newrates<0] = 0
        self.fullrates[t] = newrates


    def simulation(self):
        """Runs the simulation from t=0 to t=T, with a numerical timestep of tstep
        Results are recorded in self.rates
        
        Intermediates:
            -tstep: float, numerical timestep in ms (assumed to be 0.1 throughout)
            -simtime: float, representing the total time of the simulation (in ms)
        Outputs:
            -none
        
        """
        tstep = self.params["tstep"]
        simtime = self.params["simtime"]

        taxis = np.arange(0, simtime, tstep)
        numsteps = len(taxis)

        self.fullrates = np.zeros([numsteps, self.N, 2, self.T])
        self.fullrates[0, :, :, :] = self.rates

        Cstrength = self.params["Cstrength"]

        if ("fastdelayts" in self.params) and ("slowdelayts" in self.params):
            timedelay_off = self.params["timedelay_off"]
            fastdelayts = self.params["fastdelayts"]
            slowdelayts = self.params["slowdelayts"]
            fast_mask = np.zeros_like(self.fullrates[0,:,:,:],dtype=np.int32)
            slow_mask = np.zeros_like(self.fullrates[0,:,:,:],dtype=np.int32)
            for i,typename in enumerate(self.params['typenames']):
                if "slow" in typename:
                    slow_mask[:,:,i] += 1
                elif 'fast' in typename: 
                    fast_mask[:,:,i] += 1
                else:
                    raise KeyError("Type {0} with name '{1}' does not specify a speed, so we cannot assign a delay".format(i,typename))
            
            for t in range(1,numsteps):
                self.timestep_diffdelay(tstep, t, 
                                        fast_mask, 
                                        slow_mask, 
                                        timedelay_off=timedelay_off, 
                                        fastdelayts=fastdelayts,
                                        slowdelayts=slowdelayts,
                                        Cstrength=Cstrength)


        else:
            timedelay_off = self.params["timedelay_off"]
            delayts = self.params["delayts"]
        

            for t in range(1, numsteps):
                self.timestep(tstep, t, 
                              timedelay_off = timedelay_off, 
                              delayts = delayts, 
                              Cstrength = Cstrength)
    
    # def setdrives(self, fastdrivemod, slowdrivemod):
    #     """ Changes the tonic drive strengths for the model

    #     Inputs:
    #         -fastdrivemod: float, representing the strength of the tonic drive to fast neurons
    #         -slowdrivemod: float, representing the strength of the tonic drive to slow neurons
    #     Outputs:
    #         -none
    #     """
    #     self.fastdrive = fastdrivemod
    #     self.slowdrive = slowdrivemod
    #     for n in range(self.N):
    #         for a in range(2):
    #             for p in range(self.T):
    #                 # Ben's version sent tonic input to all populations:
    #                 if 'fast' in self.params['typenames'][p]: self.drives[n, a, p] = fastdrivemod
    #                 elif 'slow' in self.params['typenames'][p]: self.drives[n, a, p] = slowdrivemod
                    
    #                 # JM's version sends tonic input to E populations only in 6pop version:
    #                 #if pm.typenames[p] in ['E fast', 'I fast']:
    #                 #    self.drives[n, a, p] = fastdrivemod
    #                 #elif pm.typenames[p] in ['E slow', 'I slow']:
    #                 #    self.drives[n, a, p] = slowdrivemod
    #                 #else:
    #                 #    self.drives[n, a, p] = 0

    # def setmtcs(self, fastmtc, slowmtc):
    #     """ Changes the membrane time constants of the model

    #     Inputs:
    #         -fastmtc: float, representing the membrane time constant of the fast neurons
    #         -slowmtc: float, representing the membrane time constant of the slow neurons
    #     Outputs:
    #         -none
    #     """
    #     self.fastmtc = fastmtc
    #     self.slowmtc = slowmtc
    #     for p in range(self.T):
    #         if 'fast' in self.params['typenames'][p]: self.mtcs[:, :, p] = fastmtc*np.ones([self.N, 2])
    #         elif 'slow' in self.params['typenames'][p]: self.mtcs[:, :, p] = slowmtc*np.ones([self.N, 2])


def delaymag(n1, n2, timedelay_off=False):
    """ Calculates the delay magnitude between two neurons

    Inputs:
        -n1: integer, segment number of the source neuron
        -n2: integer, segment number of the target neuron
    Outputs:
        -float, representing the delay magnitude between the two neurons
    
    """
    if timedelay_off: return 1.0 # must go back at least 1 numerical timestep
    return 1.0 + np.abs(n1-n2)



def actfunc(x):
    """Activation function for neurons in the model
    
    Inputs:
        -x: float, representing a neuron's input
    Outputs:
        -float, representing the resulting neuron's activity
    
    """
    #if x < 0: return 0
    #else: return x
    return x * (x > 0)