from armageddon import solver, ensemble

##########################################################
# To use plotter() function, you need to execute following 
# code but change `fiduical_impact` and `variables`

fiducial_impact = {}
fiducial_impact['radius'] = 10.
fiducial_impact['angle'] = 45.
fiducial_impact['strength'] = 1.e5
fiducial_impact['velocity'] = 21.e3
fiducial_impact['density'] = 3000.
planet = solver.Planet()
variables = ['radius', 'angle', 'strength', 'velocity']
data = ensemble.solve_ensemble(planet=planet, fiducial_impact=fiducial_impact,
                               variables=variables, radians=False, plot=True)
