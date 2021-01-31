# ACSE-4-armageddon

Asteroids entering Earth’s atmosphere are subject to extreme drag forces that decelerate, heat and disrupt the space rocks. The fate of an asteroid is a complex function of its initial mass, speed, trajectory angle and internal strength. 

[Asteroids](https://en.wikipedia.org/wiki/Asteroid) 10-100 m in diameter can penetrate deep into Earth’s atmosphere and disrupt catastrophically, generating an atmospheric disturbance ([airburst](https://en.wikipedia.org/wiki/Air_burst)) that can cause [damage on the ground](https://www.youtube.com/watch?v=tq02C_3FvFo). Such an event occurred over the city of [Chelyabinsk](https://en.wikipedia.org/wiki/Chelyabinsk_meteor) in Russia, in 2013, releasing energy equivalent to about 520 [kilotons of TNT](https://en.wikipedia.org/wiki/TNT_equivalent) (1 kt TNT is equivalent to 4.184e12 J), and injuring thousands of people ([Popova et al., 2013](http://doi.org/10.1126/science.1242642); [Brown et al., 2013](http://doi.org/10.1038/nature12741)). An even larger event occurred over [Tunguska](https://en.wikipedia.org/wiki/Tunguska_event), an unpopulated area in Siberia, in 1908. 

This tool predicts the fate of asteroids entering Earth’s atmosphere for the purposes of hazard assessment.

### Installation Guide

To install the tool, run pip install on a machine that has Git installed on it:

```
python -m pip install git+https://github.com/acse-2019/acse-4-armageddon-itokawa.git
```

### User instructions

**solver**

In a Python 3.7 environment running from the top level of the tool directory, first import solver from armageddon:

```
from armageddon import solver
```

Next, initialise the `Planet` class with the desired attributes, e.g. using the default attributes:

```
planet = solver.Planet()
```

Finally, define the inputs for the impact scenario and call the `impact` method to analyse the impact:

```
Result, outcome = planet.impact(radius, velocity, density, strength, angle, init_altitude)
```

Results may also be plotted by adding the plot=True keyword argument to `impact`. An example python file [example_solver.py](./example_solver.py) is available which can be run from the command line using:

```
python -m example_solver
```

**ensemble**

In a Python 3.7 environment running from the top level of the tool directory, first import as follows:

```
from armageddon import solver
from armageddon.ensemble import solve_ensemble
```

Next, initialise the `Planet` class with the desired attributes, e.g. using the default attributes:

```
planet = solver.Planet()
```

Next, create a `fiducial_impact` dictionary and `variables` list which include all initial value 

```
fiducial_impact = {}
fiducial_impact['radius'] = `your_value`
fiducial_impact['angle'] = `your_value`
fiducial_impact['strength'] = `your_value`
fiducial_impact['velocity'] = `your_value`
fiducial_impact['density'] = `your_value`
variables = [`'your_string'`]
```

Note that only `outcome['burst_altitude']` will be picked. The program will return a dataframe includes
value of different input parameters and `burst_altitude`. So call the `solve_ensemble` function, you 
can plot the result out by setting plot=True

```
dataset = solve_ensemble(planet=planet, fiducial_impact=fiducial_impact, variables=variables, radians=False, plot=True)
```

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```
python -m sphinx docs html
```

then viewing the `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `Armageddon.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to checki its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```
python -m pytest armageddon
```
