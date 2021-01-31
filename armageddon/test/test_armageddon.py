
from collections import OrderedDict
import inspect
import pandas as pd
import numpy as np

from pytest import fixture
from scipy.integrate import solve_ivp

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly


@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon


@fixture(scope='module')
def planet(armageddon):
    """Return a default planet (exponential atmosphere)"""
    return armageddon.Planet()


@fixture(scope='module')
def planet_simple(armageddon):
    """Return a simplified planet for analytical results comparison"""
    return armageddon.Planet(atmos_func='exponential', atmos_filename=None,
                             Cd=1., Ch=0, Q=1, Cl=0, alpha=0, Rp=np.inf,
                             g=0, H=8000., rho0=1.2)


@fixture(scope='module')
def input_data():
    input_data = {'radius': 10.,
                  'velocity': 20e3,
                  'density': 3000.,
                  'strength': 1e5,
                  'angle': 45.0,
                  'init_altitude': 100e3,
                  'dt': 0.05,
                  'radians': False
                  }
    return input_data


@fixture(scope='module')
def multi_input():
    multi_input = {'radius': [8., 9., 10., 11., 12.],
                   'velocity': [20.e3, 10.e3, 15.e3, 25.e3, 30.e3],
                   'density': [3000., 1500., 3500., 4000., 2500.],
                   'strength': [1e5, 1e3, 1e6, 1e4, 1e7],
                   'angle': [45., np.pi/6., 75., 15., np.pi/3.],
                   'init_altitude': [100e3, 100e3, 100e3, 100e3, 100e3],
                   'dt': [0.05, 0.05, 0.05, 0.05, 0.05],
                   'radians': [False, True, False, False, True]
                   }
    return [dict(zip(multi_input, i)) for i in zip(*multi_input.values())]


@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)

    return result


# @fixture(scope='module')
# def result_simple(planet_simple, input_data):
#     """Solve a default impact for the default planet"""

#     result_simple = planet_simple.solve_atmospheric_entry(**input_data)

#     return result_simple


def test_import(armageddon):
    """Check package imports"""
    assert armageddon


def test_planet_signature(armageddon):
    """Check planet accepts specified inputs"""
    inputs = OrderedDict(atmos_func='constant',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    _ = armageddon.Planet(**inputs)

    # call by position
    _ = armageddon.Planet(*inputs.values())


def test_attributes(planet):
    """Check planet has specified attributes."""
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)


def test_solve_atmospheric_entry(result, input_data):
    """Check atmospheric entry solve.

    Currently only the output type for zero timesteps."""

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    assert np.allclose(result.velocity.iloc[0], input_data['velocity'])
    assert np.allclose(result.angle.iloc[0], input_data['angle'])
    assert np.allclose(result.altitude.iloc[0], input_data['init_altitude'])
    assert np.allclose(result.distance.iloc[0], 0.0)
    assert np.allclose(result.radius.iloc[0], input_data['radius'])
    assert np.allclose(result.time.iloc[0], 0.0)


def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    print(energy)

    assert type(energy) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns


def test_analyse_outcome(planet, result):

    result = planet.calculate_energy(result.copy())
    outcome = planet.analyse_outcome(result)

    assert type(outcome) is dict


def test_ensemble(planet, armageddon):

    fiducial_impact = {'radius': 10.0,
                       'angle': 45.,
                       'strength': 1.e5,
                       'velocity': 2.1e4,
                       'density': 3000.}

    ensemble = armageddon.ensemble.solve_ensemble(planet,
                                                  fiducial_impact,
                                                  variables=['radius'], radians=False,
                                                  rmin=8, rmax=12)

    assert 'radius' in ensemble.columns
    assert 'burst_altitude' in ensemble.columns


def test_ODE_sol_scipy(planet, multi_input):
    for input_data in multi_input:
        result = planet.solve_atmospheric_entry(**input_data)

        # object inputs
        print(input_data)
        radius = input_data['radius']
        velocity = input_data['velocity']
        density = input_data['density']
        strength = input_data['strength']
        angle = input_data['angle']
        init_altitude = input_data['init_altitude']
        radians = input_data['radians']

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

        # use same time intervals as for solver.py results
        t = result.time.to_numpy()
        t0 = t[0]
        tf = t[-1]

        sol = solve_ivp(lambda t, y: planet.system(t, y, strength, density), [
                        t0, tf], state0, method='Radau', t_eval=t)

        if not radians:
            sol.y[2, :] = sol.y[2, :] * 180 / np.pi

        # # assert a maximum of 1% difference between solution norms to pass test
        # norm_scipy = np.linalg.norm(sol.y)
        # norm_solver = np.linalg.norm(result.drop(
        #     ['time'], axis=1).to_numpy()) #[:sol.y.shape[1], :])
        # assert np.abs(norm_scipy - norm_solver)/min(norm_scipy, norm_solver) < 1e-2
        for i in range(len(state0)):
            norm_scipy = np.linalg.norm(sol.y[i, :])
            norm_solver = np.linalg.norm(result.iloc[:, i].to_numpy())
            assert np.abs(norm_scipy - norm_solver) / \
                min(norm_scipy, norm_solver) < 1e-2


def test_ODE_sol_analytical(planet_simple, multi_input):
    for input_data in multi_input:
        result_simple = planet_simple.solve_atmospheric_entry(**input_data)

        # object inputs
        print(input_data)
        radius = input_data['radius']
        velocity = input_data['velocity']
        density = input_data['density']
        strength = input_data['strength']
        angle = input_data['angle']
        init_altitude = input_data['init_altitude']
        radians = input_data['radians']

        z0 = init_altitude # entry altitude
        if radians:
            th0 = angle  # entry angle
        else:
            th0 = angle * np.pi/180
        v0 = velocity  # entry velocity
        r = radius  # object radius
        rhom = density  # object density
        m = (4/3) * np.pi * r**3 * rhom  # object mass
        A = np.pi * r**2  # object cross-sectional area

        # define a constant in the solution
        x = planet_simple.H * (-planet_simple.Cd * A / (2*m)
                               * planet_simple.rho0 / np.sin(th0))

        # analytical solution gives v as a function of z
        z = result_simple.altitude
        anly_sol = v0 * \
            (np.exp(x * (np.exp(-z/planet_simple.H) - np.exp(-z0/planet_simple.H))))

        assert np.allclose(anly_sol, result_simple.velocity, atol=0, rtol=1e-2)
