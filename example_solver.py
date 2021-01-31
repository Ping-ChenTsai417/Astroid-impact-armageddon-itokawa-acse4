from armageddon import solver

'''
1. Initialize the Planet class with default inputs
'''
planet = solver.Planet()

'''
2. Define input data for the impact scenario
'''
input_data = {'radius': 10.,
              'velocity': 20e3,
              'density': 3000.,
              'strength': 1e5,
              'angle': 45.0,
              'init_altitude': 100e3,
              'ts': 0.01,
              'dt': 0.05,
              'radians': False
              }

''' 
3. Call the numerical solver and post-prcoessing functions. Plot and print some results
'''
Result, outcome = planet.impact(**input_data, plot=True)
print(outcome)
