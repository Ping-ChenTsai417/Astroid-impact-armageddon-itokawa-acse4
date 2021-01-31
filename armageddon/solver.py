from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------

        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function ``rho = rho0 exp(-z/H)``.
            Options are ``exponential``, ``tabular``, ``constant`` and ``mars``

        atmos_filename : string, optional
            If ``atmos_func`` = ``'tabular'``, then set the filename of the table
            to be read in here.

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        Returns
        -------

        None
        """
        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0

        if atmos_func == 'exponential':
            self.rhoa = lambda z: self.rho0 * np.exp(-z/self.H)

        elif atmos_func == 'tabular':
            assert isinstance(
                atmos_filename, str), 'atmos_filename required for tabular atmosphere'
            data = pd.read_csv(atmos_filename, comment='#', delimiter=' ', names=[
                               'Altitude', 'Density', 'Scale'])

            # ensure data is sorted by altitude ascending for interpolation
            data.sort_values(by=['Altitude'], ascending=True)
            zmax = data.Altitude.iloc[-1]
            zmin = data.Altitude.iloc[0]
            altitude_values = data.Altitude.values
            density_values = data.Density.values

            # interpolate from data table (revert to exponential function if z outside range)
            self.rhoa = lambda z: np.select([(z >= zmin) & (z <= zmax), (z < zmin) | (z > zmax)], [
                                            np.interp(z, altitude_values, density_values), self.rho0 * np.exp(-z/self.H)])

        elif atmos_func == 'mars':
            self.rhoa = lambda z: (0.699 * np.exp(-0.00009*z)) / (0.1921 * np.select(
                [z >= 7000, z < 7000], [249.7 - 0.00222*z, 242.1 - 0.000998*z]))

        elif atmos_func == 'constant':
            self.rhoa = lambda z: rho0

        else:
            raise ValueError(
                'Valid atmos_func inputs are: "exponential", "tabular", "mars", "constant"')

    def system(self, t, state, strength, density):
        """Define the RHS vector f(t, u) of the system du/dt = f(t, u)
        """
        v, m, theta, z, _, r = state
        A = np.pi*r**2  # radius generally varies with time after break-up
        rhoa = self.rhoa(z)

        # u = [dv/dt, dm/dt, dtheta/dt, dz/dt, dx/dt, dr/dt]
        u = np.zeros_like(state)
        u[0] = -self.Cd*rhoa*A*v**2 / (2*m) + self.g*np.sin(theta)  # dv/dt
        u[1] = -self.Ch*rhoa*A*v**3/(2*self.Q)  # dm/dt
        u[2] = self.g*np.cos(theta)/v - self.Cl*rhoa * A*v / \
            (2*m) - (v*np.cos(theta) / (self.Rp+z))  # dtheta/dt
        u[3] = -v*np.sin(theta)  # dz/dt
        u[4] = v*np.cos(theta)/(1+z/self.Rp)  # dx/dt
        if rhoa * v**2 < strength:
            u[5] = 0
        else:
            u[5] = (7/2*self.alpha*rhoa/density)**0.5 * v  # dr/dt

        return u

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, ts=0.01, dt=0.05, tmax=120, radians=False, plot=False):
        """
        Solve the system of differential equations for a given impact event.
        Also calculates the kinetic energy lost per unit altitude and
        analyses the result to determine the outcome of the impact.

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        ts : float, optional
            The timestep for the solver, in s

        dt : float, optional
            The output timestep, in s

        tmax : float, optional
            Analysis termination time, in s. The solver will stop timestepping once
            tmax is reached, if the object has not impacted the ground yet.

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        plot : logical, optional
            Optionally plot time history results and energy depositon curve.
            Default is False 
        Returns
        -------

        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``, ``dedz``

        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.

            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.

            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        Result = self.solve_atmospheric_entry(
            radius, velocity, density, strength, angle, init_altitude, ts, dt, tmax, radians)
        Result = self.calculate_energy(Result)
        outcome = self.analyse_outcome(Result)

        if plot:
            self.plotter(Result, outcome)
        return Result, outcome

    def solve_atmospheric_entry(self, radius, velocity, density, strength, angle,
                                init_altitude=100e3, ts=0.01, dt=0.05, tmax=120, radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------

        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        ts : float, optional
            The timestep for the solver, in s

        dt : float, optional
            The output timestep, in s

        tmax : float, optional
            Analysis termination time, in s. The solver will stop timestepping once
            tmax is reached, if the object has not impacted the ground yet.

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input

        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``
        """
        # RK4 solver
        def RK4(f, u0, t0, t_max, dt, args=()):
            """ Implement RK4 time-stepping to solve du/dt = f(t, u), given the RHS vector f,
            initial condition u0, start time t0, termination time t_max, and the timestep dt
            """
            u = np.array(u0)
            t = np.array(t0)
            u_all = [u0]
            t_all = [t0]
            while t+dt < t_max:
                k1 = dt*f(t, u, *args)
                k2 = dt*f(t + 0.5*dt, u + 0.5*k1, *args)
                k3 = dt*f(t + 0.5*dt, u + 0.5*k2, *args)
                k4 = dt*f(t + dt, u + k3, *args)
                u = u + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
                u_all.append(u)
                t = t + dt
                t_all.append(t)
                if u[3] <= 0:
                    break  # terminate at ground
            return np.array(u_all), np.array(t_all)

        # initial condition
        v0 = velocity
        m0 = (4/3) * np.pi * radius**3 * density
        if radians:
            theta0 = angle
        else:
            theta0 = angle * np.pi / 180
        z0 = init_altitude
        x0 = 0
        r0 = radius
        state0 = np.array([v0, m0, theta0, z0, x0, r0])

        # run solver
        t0 = 0
        sol = RK4(self.system, state0, t0, tmax, ts, args=(strength, density))

        # convert angles back to degrees if specfied at input
        if not radians:
            sol[0][:, 2] = sol[0][:, 2] * 180 / np.pi

        # interpolate results at the output timestep
        if dt == ts:
            t_out = sol[1].T
            sol_out = sol[0].T
        else:
            t_sol = sol[1]
            N = floor(t_sol[-1] * 0.9999 / dt)
            t_out = np.hstack([np.linspace(t_sol[0], N*dt, N+1), t_sol[-1]])
            sol_out = np.array([np.interp(t_out, t_sol, sol[0][:, j])
                                for j in range(len(state0))])

        return pd.DataFrame({'velocity': sol_out[0, :],
                             'mass': sol_out[1, :],
                             'angle': sol_out[2, :],
                             'altitude': sol_out[3, :],
                             'distance': sol_out[4, :],
                             'radius': sol_out[5, :],
                             'time': t_out}, index=range(len(t_out)))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.
        Parameters
        ----------
        result : DataFrame
            A pandas DataFrame with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns
        -------
        Result : DataFrame
            Returns the DataFrame with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """
        # add the dedz column to the result DataFrame
        result = result.copy()
        mass = np.array(result.mass)
        velocity = np.array(result.velocity)
        altitude = np.array(result.altitude) / 1000
        energy = 1/2 * mass * velocity**2 / (4.184 * (10)**12)

        dedz = np.zeros_like(energy)
        dedz[0] = (energy[1] - energy[0]) / (altitude[1] - altitude[0])
        dedz[1:-1] = (energy[2:] - energy[:-2]) / \
            (altitude[2:] - altitude[:-2])
        dedz[-1] = (energy[-1] - energy[-2]) / (altitude[-1] - altitude[-2])

        result.insert(len(result.columns), 'dedz', np.array(dedz))
        return result

    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats
        Parameters
        ----------
        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        Returns
        -------
        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.
            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.
            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        index_max = result['dedz'].idxmax()
        burst_altitude = result.iloc[index_max]['altitude']  # (m)
        burst_peak_dedz = result.iloc[index_max]['dedz']  # (kt TNT per km)

        initial_energy = 1/2 * \
            result.iloc[0]['mass']*(result.iloc[0]['velocity']**2)  # (J)

        outcome = {}
        if burst_altitude > 5000:
            event = "Airburst"
            outcome["outcome"] = event

            burst_mass = result.iloc[index_max]['mass']  # (kg)
            burst_speed = result.iloc[index_max]['velocity']  # (m/s)
            burst_energy = 1/2 * burst_mass * burst_speed**2  # (J)
            burst_total_ke_lost = (initial_energy - burst_energy) / (4.184e12)

            outcome["burst_peak_dedz"] = burst_peak_dedz
            outcome["burst_altitude"] = burst_altitude
            outcome["burst_total_ke_lost"] = burst_total_ke_lost

        elif burst_altitude <= 0:
            event = "Cratering"
            outcome["outcome"] = event

            impact_time = result['time'].iloc[-1]  # (s)
            impact_mass = result['mass'].iloc[-1]  # (kg)
            impact_speed = result['velocity'].iloc[-1]  # (m/s

            outcome["impact_time"] = impact_time
            outcome["impact_mass"] = impact_mass
            outcome["impact_speed"] = impact_speed

        else:
            event = "Airburst and cratering"
            outcome["outcome"] = event

            burst_mass = result.iloc[index_max]['mass']  # (kg)
            burst_speed = result.iloc[index_max]['velocity']  # (m/s)
            burst_energy = 1/2 * burst_mass * burst_speed**2  # (J)
            burst_total_ke_lost = (initial_energy - burst_energy) / (4.184e12)

            outcome["burst_peak_dedz"] = burst_peak_dedz
            outcome["burst_altitude"] = burst_altitude
            outcome["burst_total_ke_lost"] = burst_total_ke_lost

            impact_time = result['time'].iloc[-1]  # (s)
            impact_mass = result['mass'].iloc[-1]  # (kg)
            impact_speed = result['velocity'].iloc[-1]  # (m/s

            outcome["impact_time"] = impact_time
            outcome["impact_mass"] = impact_mass
            outcome["impact_speed"] = impact_speed

        return outcome

    def plotter(self, Result, outcome):
        """ Optionally plot results if plot=True is set in the impact() method.
        Plots produced are: 
            - time histories of velocity, mass, angle, altitude, distance and radius
            - energy deposition curve
        """
        # Plot results time histories
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs = axs.reshape(-1)
        axs[0].plot(Result.time, Result.velocity)
        axs[1].plot(Result.time, Result.mass)
        axs[2].plot(Result.time, Result.angle)
        axs[3].plot(Result.time, Result.altitude)
        axs[4].plot(Result.time, Result.distance)
        axs[5].plot(Result.time, Result.radius)
        axs[0].set_title('velocity (m/s) vs time (s)', fontsize=16)
        axs[1].set_title('mass (kg) vs time (s)', fontsize=16)
        axs[2].set_title('angle (rad) vs time (s)', fontsize=16)
        axs[3].set_title('altitude (m) vs time (s)', fontsize=16)
        axs[4].set_title('distance (m) vs time (s)', fontsize=16)
        axs[5].set_title('radius (m) vs time (s)', fontsize=16)
        plt.tight_layout()

        # Plot energy deposition curve
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(Result.dedz, Result.altitude / 1e3)
        ax.set_xlabel('Energy per unit height [Kt/km]', fontsize=14)
        ax.set_ylabel('Altitude [km]', fontsize=14)
        plt.show()
