import numpy as np
import pandas as pd

from armageddon import solver
import matplotlib.pyplot as plt

def solve_ensemble(planet, fiducial_impact, variables,
                   radians=False, rmin=8, rmax=12, 
                   nb_samples=181, plot=False):
    """
    Run asteroid simulation for a distribution of initial conditions and
    find the burst distribution

    Parameters
    ----------

    planet : object
        The Planet class instance on which to perform the ensemble calculation

    fiducial_impact : dict
        Dictionary of the fiducial values of radius, angle, strength, velocity
        and density

    variables : list
        List of strings of all impact parameters to be varied in the ensemble
        calculation

    rmin : float, optional
        Minimum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    rmax : float, optional
        Maximum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    nb_samples : int, optional
        sampling numbers of ensemble function. Default is 181.

    plot: logical, optional
        Optionally plot burst altitude results histogram.
        Default is False.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """

    # extract initial value from fiducial_impact
    r1 = fiducial_impact['radius']
    ang = fiducial_impact['angle']
    vel = fiducial_impact['velocity']
    densi = fiducial_impact['density']
    stren = fiducial_impact['strength']

    # create arrays of all elements, sample 51 times(default)
    radius_arr = np.empty(nb_samples)
    radius_arr.fill(r1)

    velocity_arr = np.empty(nb_samples)
    velocity_arr.fill(vel)

    density_arr = np.empty(nb_samples)
    density_arr.fill(densi)

    stren_arr = np.empty(nb_samples)
    stren_arr.fill(stren)

    angle_arr = np.empty(nb_samples)
    angle_arr.fill(ang)

    # fixed value to ensure get max value included in uniform distribution
    inc = 1e-4

    # get burst_altitude by calling impact() according to parameters of
    # radius, angle, strength, velocity and density
    def vary_general(radius, angle, strength, velocity, density):
        result, outcome = planet.impact(radius, velocity, density,
                                        strength, angle,
                                        init_altitude=85e3, ts=0.05,
                                        tmax=60, radians=radians)
        if outcome["outcome"] == "Cratering":
            return 0
        return outcome["burst_altitude"]

    for var in variables:
        if var == 'radius':

            # uniform distribution, interval: [8, 12]
            radius_arr = np.random.uniform(rmin, rmax + inc, nb_samples)

        if var == 'angle':

            # apply inverse transform sampling
            # the range of angle's CDF is [0, 1]
            y_list = np.random.uniform(0., 1 + inc, nb_samples)

            # inverse function of CDF
            angle_arr = np.arcsin(np.sqrt((y_list)))

            # arcsin() returns angle in radians
            if not radians:
                angle_arr = angle_arr * 180/np.pi

        if var == 'strength':

            # log(strength) follows uniform distribution, interval: [log(1000), log(1000000)]
            stren_arr = np.random.uniform(np.log(1.e3), np.log(1.e6) + inc, nb_samples)

            # Get rid of log()
            stren_arr = np.exp(stren_arr)

        if var == 'velocity':
            x_min = 0.171
            x_max = 50.5372
            vesc = 11.2
            x_ = np.linspace(x_min, x_max, 500)

            # calculate PDF according to CDF
            def pdf_vel(v):
                return np.sqrt(2/np.pi)*np.exp(-v**2/242)*v**2/1331

            # calculate min and max of pdf range
            y_ = pdf_vel(x_)
            p_min = 0.
            p_max = y_.max()

            accept_num = 0
            ran_num = []

            # get points we want
            while accept_num < nb_samples:
                # generate x and y values uniformly
                x_tri = np.random.uniform(x_min, x_max + inc)
                y_tri = np.random.uniform(p_min, p_max + inc)

                # if y value is bigger than it supposed to be, drop it
                if y_tri < pdf_vel(x_tri):
                    ran_num.append(x_tri)
                    accept_num = accept_num + 1

            # calculate impact velocity through escape velocity
            # and convert km/s to m/s
            velocity_arr = np.sqrt(vesc**2 + np.array(ran_num)**2) * 1000

        if var == 'density':
            x_min = 1000.
            x_max = 7264.9
            x_ = np.linspace(x_min, x_max, 500)

            def pdf_den(v):
                return np.exp(-(v - 3000)**2/2.e6) / (500*np.sqrt(2.*np.pi))

            y_ = pdf_den(x_)
            p_min = 0.
            p_max = y_.max()

            accept_num = 0
            ran_num = []

            while accept_num < nb_samples:
                x_tri = np.random.uniform(x_min, x_max + inc)
                y_tri = np.random.uniform(p_min, p_max + inc)

                if y_tri < pdf_den(x_tri):
                    ran_num.append(x_tri)
                    accept_num = accept_num + 1

            density_arr = np.array(ran_num)

    # generate a dataframe with columns of radius, angle, strength, velocity
    dataset = pd.DataFrame({'radius': radius_arr, 'angle': angle_arr,
                            'strength': stren_arr, 'velocity': velocity_arr,
                            'density': density_arr},
                           columns=['radius', 'angle', 'strength',
                                         'velocity', 'density'])

    # apply vary_general() function to all variables and get burst_altitude
    dataset["burst_altitude"] = dataset.apply(lambda x: vary_general(x.radius, x.angle,
                                                                     x.strength, x.velocity,
                                                                     x.density), axis=1)
    variables.append('burst_altitude')
    df = dataset.loc[:, variables]
    if plot:
        plotter(df, variables)
    return df

def plotter(df, variable_):
    fig = plt.figure(figsize=(18, 10))
    unique = df['burst_altitude'].tolist()
    plt.hist(unique, bins=len(unique))
    plt.title(" ,".join(variable_), size=18)
    plt.xlabel("$burst altitude (m)$", size=16)
    plt.ylabel("$Counts$", size=16)
    plt.show(block=True)