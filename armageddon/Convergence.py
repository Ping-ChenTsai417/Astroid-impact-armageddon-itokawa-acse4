from armageddon import solver
import numpy as np
import matplotlib.pyplot as plt

input_data = {'radius': 10.,
              'velocity': 20e3,
              'density': 3000.,
              'strength': 1e5,
              'angle': 45.0,
              'init_altitude': 100e3,
              'dt': 0.05}

z0 = input_data['init_altitude']  # entry altitude
th0 = input_data['angle'] * np.pi/180  # entry angle
v0 = input_data['velocity']  # entry velocity
r = input_data['radius']  # object radius
rhom = input_data['density']  # object density
m = (4/3) * np.pi * r**3 * rhom  # object mass
A = np.pi * r**2  # object cross-sectional area

planet = solver.Planet(Cd=1., Ch=0, Q=1, Cl=0, alpha=0,
                       Rp=np.inf, g=0, H=8000., rho0=1.2)
result = planet.solve_atmospheric_entry(**input_data)

ts_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

error = []
for time_step in ts_list:
    results = planet.solve_atmospheric_entry(**input_data, ts=time_step)
    height_n = results.altitude.values
    velocity_n = results.velocity.values

    # define a constant in the solution
    x = planet.H * (-planet.Cd * A / (2*m) * planet.rho0 / np.sin(th0))

    # analytical solution gives v as a function of z
    z = height_n
    velocity_a = v0 * \
        (np.exp(x * (np.exp(-z/planet.H) - np.exp(-z0/planet.H))))

    e = np.abs(velocity_a[-1] - velocity_n[-1])
    error.append(e)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.loglog(ts_list, error, 'b.', markersize=12, label='RK4')
ax.set_xlabel('Time Step (s)')
ax.set_ylabel('Error')

start_fit = 2
end_fit = -3
line_fit = np.polyfit(np.log(ts_list[start_fit:end_fit]), np.log(
    error[start_fit:end_fit]), 1)
ax.loglog(ts_list, np.exp(line_fit[1]) * ts_list**(line_fit[0]),
          'k-', label='slope: {:.2f}'.format(line_fit[0]))
ax.legend(loc='best')
plt.show()
