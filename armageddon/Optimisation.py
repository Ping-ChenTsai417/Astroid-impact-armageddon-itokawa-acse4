import scipy.interpolate as intpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Optimization problem

'''
def meteor_data(file):
    """ The given results of the meteor event.
    Parameter
    -----------
    file: file path
    """
    data_csv = pd.read_csv(file)
    df=pd.DataFrame(data=data_csv)
    altitude_ = np.array(df.iloc[1:,0])
    energy = np.array(df.iloc[1:,1]) #* 4.184 * 1e12 # convert to joules
    return altitude_, energy

def RMSE(energy_guess, energy_target):
    """  Calculate the root mean square error of the optimized energy and target energy
    Parameter
    ------------
    energy_guess: array

    energy_arget: array
    """
    return np.sqrt(np.mean((energy_guess-energy_target)**2))

# loop through each possible r and Y
def get_opt_radius_Y(earth, rad, Y, height_ori, energy_ori, target_func):
    '''Optimize r and Y by looping guessed parameters within possible range.
        Possible range can be tuned manually.
        Parameters
        ----------
        earth: object of class 
            Planet in solver.py
        rad: array
            Guessed radian
        Y: array
            Guessed strength
        height_ori: array
            Given heights of the event
        energy_ori: array
            Given energy of the event
        target_func: function
            Interpolated function of the event data
        Returns
        -------
        outcome : 
            'radius_opt', 'strength_opt','rmse','height_guess_s' and 'energy_guess_s' are
            the optimized radius, optimized strength, rmse between optimised energy and target energy, array of
            optimized height, array of optimized strength.
     '''
    rmse_all = []
    tol = 5
 
    for r in rad:
        for s in Y:
            result = earth.solve_atmospheric_entry(radius = r, angle=18.3, strength = s, velocity=1.92e4, density=3300)
            outcome = earth.calculate_energy(result)

            energy_guess = outcome.dedz
            height_guess = outcome.altitude/1000
            

            # Slice optimized function to the same range as target one
            lower_bound = np.where(height_guess <= height_ori[0])[0][0]
            upper_bound = np.where(height_guess >= height_ori[-1])[0][-1]
            height_guess_s = height_guess[lower_bound:upper_bound]
            energy_guess_s = energy_guess[lower_bound:upper_bound]
            
            # Calculate optimal energy 
            energy_ori_s = target_func(height_guess_s)

            # Output energy rmse difference, put error into an array
            rmse = RMSE(energy_guess_s, energy_ori_s)
            rmse_all.append(rmse)

            if rmse < np.amin(rmse_all[:]) or np.allclose(rmse, np.amin(rmse_all[:])):
                radius_opt = r
                strength_opt = s

            elif rmse<tol:
                radius_opt = r
                strength_opt = s
                break
                
    return radius_opt, strength_opt,rmse, height_guess_s, energy_guess_s

def plot_Optimisation_radius_strength(filepath_, earth):
    '''
    Plot the optimized function vs. the target function of the event

    Parameter
    ------------
    filepath_: file path
    earth: object of the class Planet() in solver
    '''
     
    height_ori, energy_ori = meteor_data(filepath_) # insert filename
    target_func = intpl.interp1d(height_ori, energy_ori)
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(121)
    # Interpolate function
    target_func = intpl.interp1d(height_ori, energy_ori)
    # Plot target function
    ax.plot(height_ori, target_func(height_ori),'r',label = 'Target func')

    #Guess energy and height
    result = earth.solve_atmospheric_entry(radius=8.21, angle=18.3, strength=5e6, velocity=1.92e4, density=3300)
    outcome = earth.calculate_energy(result)

    energy_guess = outcome.dedz
    height_guess = outcome.altitude/1000

    # Plot guess function
    ax.plot(height_guess, energy_guess,label = 'Guess func')
    ax.legend()
    ax.grid(True)
    ax.set_ylabel('Energy Loss per Unit Height (kt TNT)')
    ax.set_xlabel('Altitude (km)')

    # Change guessed range for radius and strength
    radius_ = np.linspace(8.1, 8.3, 3)
    strength_ = np.linspace(4.9e6,5.3e6, 3)

    radius_opt, strength_opt, rmse_opt, height_guess_s, energy_guess_s = get_opt_radius_Y(earth, radius_, strength_ ,height_ori, energy_ori, target_func)

    ax1 = plt.subplot(122)
    ax1.plot(height_guess_s, energy_guess_s, label = 'Guess func')
    ax1.plot(height_ori, target_func(height_ori),'r', label = 'Target func')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylabel('Energy Loss per Unit Height (kt TNT)')
    ax1.set_xlabel('Altitude (km)')

    print('radius_opt:')
    print(radius_opt)
    print('strength_opt: ')
    print(strength_opt)

    return 


