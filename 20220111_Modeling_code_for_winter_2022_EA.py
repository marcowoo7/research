# =============================================================================
# Import libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
import time

plt.close('all')


# =============================================================================
# Function for acoustic contrast factor
# =============================================================================
def ACF(rhop,rhof,betap,betaf):
    phi = (5*rhop - 2*rhof)/(2*rhop + rhof) - (betap/betaf)
    
    return phi


# =============================================================================
# Acoustic radiation force. This function takes an array of particle positions 
# and outputs an array of dimensional forces on each particle
# =============================================================================
def acousticrad(x,n,W,a,betaf,p0,phi):
    #Solve for wavenumber as a function of nodes
    k = n*np.pi/W
    
    #Initialize force array    
    Fp = np.zeros(len(x))
    
    for i in range(len(x)):
        Fp[i] = np.pi*phi*k*pow(a,3)*betaf*pow(p0,2)*np.sin(2*k*x[i])/3
        
    return Fp


# =============================================================================
# Scattering force of each particle due to every other particle. Inputs listed
# below:
#     x:            x coordinates of all particles
#     n:            number of nodes
#     a:            particle radius
#     W:            channel width
#     Nx:           number of particles  
#     betaf:        compressibility of the fluid
#     betap:        compressibility of the particle (active material)
#     rhof:         density of the fluid
#     rhop:         density of the particle (active material)
#     p0:           acoustic pressure
#    
# Output of this function is an array with the scattering force on each particle
# =============================================================================
def scattering(x,n,a,W,Nx,betaf,betap,rhof,rhop,p0):
    #Define constants
    k = n*np.pi/W
    w2 = pow(k,2)/(betaf*rhof)
    
    #Initialize an array to hold the Fs terms for each particle
    Fs = np.zeros(Nx)
    
    #Sum over all particles to find net force on each particle due to scattering
    for i in range(Nx):
        for j in range(Nx):
            if i != j:
                if abs(i-j) <= 15:
                    #Set up coordinate system
                    dij = x[j] - x[i]
                    xij = np.average([x[j],x[i]])
                
                    #Calculate squares of pressure and velocity
                    p2 = pow(p0,2)*pow(np.cos(k*xij),2)/2
                    v2 = pow(p0,2)*betaf*pow(np.sin(k*xij),2)/(2*rhof)
            
                    #Calculate scattering force
                    term1 = 2*pow(rhop-rhof,2)*v2/(6*rhof*pow(dij,4))
                    term2 = w2*rhof*pow(betap - betaf,2)*p2/(9*pow(dij,2))                
                    Fs_hold = 4*np.pi*pow(a,6)*(term1-term2)
                
                    if (dij > 0):
                        Fs[i] -= Fs_hold
                    elif (dij < 0):
                        Fs[i] += Fs_hold
                    
    return Fs


# =============================================================================
# System of ODEs to solve
# x is an array of particle positions, and the function iterates new position
# values until there are position values for all discrete time steps
# =============================================================================
def position(x,t,acousticrad,scattering,n,W,a,betaf,betap,rhof,rhop,p0,phi,Nx,eta):
    Fp = acousticrad(x,n,W,a,betaf,p0,phi)
    Fs = scattering(x,n,a,W,Nx,betaf,betap,rhof,rhop,p0)
    
    drag = np.zeros(Nx)
    for i in range(Nx):
        drag[i] = (Fp[i] + Fs[i])/(6*np.pi*a*eta)
    
    return drag


# =============================================================================
# An equation for converting the volume fraction to the number of particles 
# across the channel width, rounded to the nearest integer
# =============================================================================
def vf2np(vf,a,W):
    Np = int(np.round(W/a*(3*vf/(4*np.pi))**(1/3)))
    
    return Np


# =============================================================================
# A function that stores the material properties of electrode active materials
# and other common focusing materials
# =============================================================================
def material(mat_prop):
    if mat_prop == 1: #Alumina in epoxy and acetone
        rhop = 3.95*pow(10,3)       #kg/m^3
        rhof = 1.12*pow(10,3)       #kg/m^3
        betap = 1.6*pow(10,-12)     #1/Pa
        betaf = 2.0*pow(10,-10)     #1/Pa
    elif mat_prop ==2: #Silica in water
        rhop = 2.5*pow(10,3)        #kg/m^3
        rhof = 1.0*pow(10,3)        #kg/m^3
        betap = 2.72*pow(10,-10)    #1/Pa
        betaf = 4.54*pow(10,-10)    #1/Pa
    elif mat_prop ==3: #NMC in NMP and PVDF (battery materials)
        rhop = 4.5*pow(10,3)        #kg/m^3
        rhof = 1.03*pow(10,3)       #kg/m^3
        betap = 7.5*pow(10,-12)     #1/Pa
        betaf = 5.2*pow(10,-10)     #1/Pa

    return rhop,rhof,betap,betaf


# =============================================================================
# Function to solve for the number of nodes given the fluid material
# properties, channel width, and operational frequency
# =============================================================================
def f2n(rhof,betaf,W,f):
    c = np.sqrt(1/(rhof*betaf))
    n = round(2*f*W/c)
    
    return n


# =============================================================================
# Solve for initial particle position, this option spaces particles evenly in 
# the channel and adjusts positions if they are too close to an antinode. This
# is done because particles near the antinodes tend to get trapped and take a 
# long time to focus. In an experiment, particles likely have a velocity in the
# direction of the channel width, and this would cause them to move away from
# the antinode naturally
# =============================================================================
def init_pos_2(n,Np,W,a):
    dW = (W - Np*2*a)/(2*(Np+1))+a
    x0 = np.linspace(0+dW,W-dW,Np)
    
    #Solve for anti-nodes
    even = np.arange(0,2*n+1,2)
    antinode = np.zeros(len(even))
    
    for i in range(len(even)):
        antinode[i] = even[i]*W/(2*n)
        
    for i in range(len(antinode)):
        for j in range(len(x0)):
            test = antinode[i] - x0[j]
            if abs(test) < a/4:
                if test > 0:
                    x0[j] = x0[j] - a/4
                else:
                    x0[j] = x0[j] + a/4
    
    return x0


# =============================================================================
# Solves for the focused widths and spacing widths by using the roots of the 
# acoustic radiation force roots. The function returns the following:
#       Outer:    an array with the outer positions of the particles
#       Root:     the location of the nodes
#       avg_p:    the average focusing width
#       avg_s:    the average spacing between focusing widths
# =============================================================================
def solve_width(a,W,n,x0,Np,pos,M):
    #Solve for location of nodes across channel width
    odd = np.arange(1,2*n,2)
    root = np.zeros(len(odd))
    
    for i in range(len(odd)):
        root[i] = odd[i]*W/(2*n)
    
    #Define bin edges
    even = np.arange(0,2*n+1,2)
    edges = np.zeros(len(even))
    
    for i in range(len(even)):
        edges[i] = even[i]*W/(2*n)
    
    #Sort into bins based on edge of each focusing node
    split,bins = np.histogram(x0,edges)
    
    #Find particles at the edge (outer region) of each focused region
    outer = np.zeros(2*n)
    outer[0] = 1
    outer[-1] = Np
    count_1 = 0
    j = 0
    
    for i in range(len(outer)-1):
        if i != 0:
            if i%2 != 0:
                count_1 += split[j]
                j += 1
                outer[i] = count_1
            else:
                count_2 = count_1 + 1
                outer[i] = count_2
    
    #Solve for the focusing widths and spacing widths
    width = np.zeros((len(pos),2*n-1))
    
    for i in range(len(pos)):
        for j in range(2*n-1):
            width[i,j] = pos[i,int(outer[j+1]-1)] - pos[i,int(outer[j]-1)]
    
    #Solving for focusing width and spacing
    width_p_int = np.arange(0,2*n-1,2)
    width_s_int = np.arange(1,2*n-1,2)
    width_p = np.zeros((M,len(width_p_int)))
    width_s = np.zeros((M,len(width_s_int)))
    avg_p = np.zeros(M)
    avg_s = np.zeros(M)
    
    for i in range(len(width_p_int)):
        width_p[:,i] = (width[:,width_p_int[i]]+(2*a))*10**6
    
    for i in range(len(width_s_int)):
        width_s[:,i] = (width[:,width_s_int[i]]-(2*a))*10**6
    
    for i in range(M):
        avg_p[i] = np.average(width_p[i,:])
        avg_s[i] = np.average(width_s[i,:])   
        
    return outer,root,avg_p,avg_s


# =============================================================================
# Function that solves for the index of when particles are focused completely
# Inputs include
#       tol:   the percent difference desired between the final focused state 
#              and the reported critical focusing length
#       avg_p: average width of focused lines
#       avg_s: average spacing between focused lines
# The function returns the first index of when avg_p/avg_s is within the 
# tolerance of the final focused value
# =============================================================================
def ind_solve(tol,avg_p,avg_s):
    ratio = avg_p/avg_s
    end = ratio[-1]
    ind = []
    for i in range(len(ratio)):
        err = abs(ratio[i]-end)/end
        if err <= tol:
            ind.append(i)
    
    return ind[0]


# =============================================================================
# Function that solves for the index of when particles are focused completely.
# This function differs from ind_solve because it considers the position of 
# individual particles rather than the entire focused width.
# Inputs include
#       tol:     the percent difference desired between the final focused state 
#                and the reported critical focusing length (or index of this)
#       soln:    array with particle trajectories
#       Np:      the number of particles
# The function returns the following:
#       ind:     an array of indices (that correspond to each particle) that
#                indicate the point at which the particle is within the 
#                tolerance of the function
#       hit_tol: 
# =============================================================================
def ind_solve_idv(tol,soln,Np):
    #Initialize an array to hold indices
    ind = np.zeros(Np)
    
    #Iterate over particles and find indices for when particles are within 
    #the tolerance distance from their final focused position
    for i in range(Np):
        end = soln[-1,i]
        hold = []
        
        for j in range(np.shape(soln)[0]):
            err = abs(soln[j,i]-end)/end
            if err <= tol:
                hold.append(j)
        
        ind[i] = hold[0]
    
    #Create an array to keep track of when particles have hit this tolerance
    hit = np.zeros(np.shape(soln)[0])
    
    for i in range(Np):
        ind_range = np.arange(ind[i],np.shape(soln)[0],1)
        for j in range(len(ind_range)):
            hit[int(ind_range[j])]+=1
            
    hit_tol = [100*x/Np for x in hit]
    
    return ind, hit_tol


# =============================================================================
# Main, location where program is run
# =============================================================================
def main():
    #Timer
    start = time.time()
    
    #Define constants
    a = 15*pow(10,-6)               # Particle radius (m)
    W = 3*pow(10,-3)                # Channel width (m)
    vs = 0.001                      # Velocity of fluid (m/s)
    eta = 10                        # Viscosity (Pa-s)        
    p0 = 1*pow(10,6)                # Acoustic pressure (Pa) 
    vf = 0.03                       # Volume fraction (decimal)        
    Np = vf2np(vf,a,W)              # Number of particles
    
    #Specify which material properties are of interest
    #      1 = Alumina in Epoxy and Acetone
    #      2 = Silica in Water
    #      3 = NMC in NMP and PVDF
    mat_prop = 1
    rhop,rhof,betap,betaf = material(mat_prop)
    
    #Set up the format of plots
    font = {'fontname':'Helvetica'}         #font style
    FS = 28                                 #font size
    LW = 2                                  #line width
    NS = 24                                 #number size
    
    #Solve for acoustic contrast factor
    phi = ACF(rhop, rhof, betap, betaf)
    print(f"Acoustic contrast factor: {phi}")
    
    #Set time interval 
    t = np.arange(0,150,0.01)
    
# =============================================================================
#   Plot particle trajectory through channel
# =============================================================================
    #Frequency to node conversion
    f = 1.5*pow(10,6)            #Frequency (Hz)
    n = f2n(rhof,betaf,W,f)
    
    #Set initial particle positions
    x0 = init_pos_2(n,Np,W,a)
    
    #Solve for plug flow solution
    plug_soln = odeint(position,x0,t, args = (acousticrad,scattering,n,W,a,betaf,betap,rhof,rhop,p0,phi,Np,eta))

    #Solve for roots
    M = np.shape(plug_soln)[0]
    focus,root,avg_p,avg_s = solve_width(a,W,n,x0,Np,plug_soln,M)
    
    #Mapping the area taken up by particles- plug flow
    for i in range(Np):
        y1 = plug_soln[:,i]*10**2 - a*10**2
        y2 = plug_soln[:,i]*10**2 + a*10**2
        plt.figure(6)
        plt.plot(t*vs*10**2,y1,color='black')
        plt.plot(t*vs*10**2,y2,color='black')
        plt.fill_between(t*vs*10**2,y1,y2,color = 'black')
        plt.hlines(root*10**2,0,240,colors = 'b')
        plt.xlabel('Focusing Length (cm)',fontsize = FS,**font)
        plt.ylabel('Particle Position (cm)',fontsize = FS,**font)
        plt.tick_params(labelsize = NS)
        plt.xlim((-0.05,10))
        plt.ylim((0,W*10**2))


    
# =============================================================================
#   Plot of width ratio and pore width vs volume fraction for different nodes
# =============================================================================
    #Note to user: Set nodes of interest and volume fraction range below    
    n1 = f2n(rhof,betaf,W,3*pow(10,6))
    n2 = f2n(rhof,betaf,W,5*pow(10,6)) 
    n3 = f2n(rhof,betaf,W,8*pow(10,6))
    N = [n1,n2,n3]
    vf_range = [0.05,0.1,0.15]
    
    for i in range(len(N)):
        print(N[i])
        
        #Set up arrays for holding focusing width information
        ratio = np.zeros(len(vf_range))
        width_s = np.zeros(len(vf_range))
        
        for j in range(len(vf_range)):        
            #Calculate the number of particles and their starting positions
            Np = vf2np(vf_range[j],a,W)
            print(Np)
            x0 = init_pos_2(N[i],Np,W,a)
        
            #Solve for position
            plug_soln = odeint(position,x0,t, args = (acousticrad,scattering,N[i],W,a,betaf,betap,rhof,rhop,p0,phi,Np,eta))
            
            #Solve for roots
            M = np.shape(plug_soln)[0]
            focus,root,avg_p,avg_s = solve_width(a,W,N[i],x0,Np,plug_soln,M)
            
            #Put width values into array
            ratio[j] = avg_p[-1]/avg_s[-1]
            width_s[j] = avg_s[-1]
            
        plt.figure(2)
        plt.plot(vf_range,ratio,linewidth = 3)
        plt.xlim((0,0.25))
        plt.ylim((1,4))
        plt.tick_params(labelsize = NS)
        plt.xlabel('Volume Fraction',fontsize = FS,**font)
        plt.ylabel('Electrode:Pore Width',fontsize = FS,**font)
        plt.legend(['3 MHz','5 MHz','8 MHz'],loc = 2,prop={'size':NS})
        
        plt.figure(3)
        plt.plot(vf_range,width_s,linewidth = 3)
        plt.xlim((0,0.25))
        plt.ylim((5,140))
        plt.tick_params(labelsize = NS)
        plt.xlabel('Volume Fraction',fontsize = FS,**font)
        plt.ylabel('Pore Width (\u03bcm)',fontsize = FS,**font)
        plt.legend(['3 MHz','5 MHz','8 MHz'],loc = 3,prop={'size':NS})
    
    
    #Timer
    end = time.time()
    total = round((end - start)/60)
    print(f"Program runtime is {total} minutes.")
           
    
main()
        
        