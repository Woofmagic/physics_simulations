import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Axes, rc_context

class PlotCustomizer:
    def __init__(
            self,
            axes: Axes,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            zlabel: str = "",
            xlim = None,
            ylim = None,
            zlim = None,
            grid: bool = False):
        
        self._custom_rc_params = {
            'text.usetex': True,
            'font.family': 'serif, sans-serif',
            'font.size': 11.,
            'mathtext.fontset': 'dejavusans', # https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_fontfamily_example.html
            'xtick.direction': 'in',
            'xtick.major.size': 5,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 2.5,
            'xtick.minor.width': 0.5,
            'xtick.minor.visible': True,
            'xtick.top': True,
            'ytick.direction': 'in',
            'ytick.major.size': 5,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 2.5,
            'ytick.minor.width': 0.5,
            'ytick.minor.visible': True,
            'ytick.right': True,
        }

        self.axes_object = axes
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.grid = grid

        self._apply_customizations()

    def _apply_customizations(self):

        with rc_context(rc = self._custom_rc_params):

            # (1): Set the Title -- if it's not there, will set empty string:
            self.axes_object.set_title(self.title)

            # (2): Set the X-Label -- if it's not there, will set empty string:
            self.axes_object.set_xlabel(self.xlabel)

            # (3): Set the Y-Label -- if it's not there, will set empty string:
            self.axes_object.set_ylabel(self.ylabel)

            # (4): Set the X-Limit, if it's provided:
            if self.xlim:
                self.axes_object.set_xlim(self.xlim)

            # (5): Set the Y-Limit, if it's provided:
            if self.ylim:
                self.axes_object.set_ylim(self.ylim)

            # (6): Check if the Axes object is a 3D Plot that has 'set_zlabel' method:
            if hasattr(self.axes_object, 'set_zlabel'):

                # (6.1): If so, set the Z-Label -- if it's not there, will set empty string:
                self.axes_object.set_zlabel(self.zlabel)

            # (7): Check if the Axes object is 3D again and has a 'set_zlim' method:
            if self.zlim and hasattr(self.axes_object, 'set_zlim'):

                # (7.1): If so, set the Z-Limit, if it's provided:
                self.axes_object.set_zlim(self.zlim)

            # (8): Apply a grid on the plot according to a boolean flag:
            self.axes_object.grid(self.grid)

    def add_line_plot(self, x_data, y_data, label: str = "", color = None, linestyle = '-'):
        """
        Add a line plot to the Axes object:
        connects element-wise points of the two provided arrays.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        label: str

        color: str

        linestyle: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Just add the line plot:
            self.axes_object.plot(x_data, y_data, label = label, color = color, linestyle = linestyle)

            if label:
                self.axes_object.legend()

    def add_scatter_plot(self, x_data, y_data, radial_size: float = 1., label: str = "", color = None, marker = 'o'):
        """
        Add a scatter plot to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        radial_size: float
        
        label: str

        color: str 

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the scatter plot:
            self.axes_object.scatter(
                x_data,
                y_data,
                s = radial_size,
                label = label,
                color = color,
                marker = marker)

            if label:
                self.axes_object.legend()

def initialize_magnetic_spins(number_of_spins):

    if number_of_spins <= 0:
        raise ValueError("N must be a positive integer.")
    
    spins = np.random.choice([-1, 1], size = number_of_spins)

    return spins

def compute_total_energy(state_of_magnetic_spins):

    hamiltonian_energy_scale = 1.0
    hamiltonian_external_field = 0.0

    state_of_magnetic_spins = np.asarray(state_of_magnetic_spins)

    hamiltonian_energy = -hamiltonian_energy_scale * np.sum(state_of_magnetic_spins * np.roll(state_of_magnetic_spins, -1)) - hamiltonian_external_field * np.sum(state_of_magnetic_spins)

    return hamiltonian_energy

def compute_total_magnetization(state_of_magnetic_spins):

    magnetization = np.sum(state_of_magnetic_spins)

    return magnetization

def compute_boltzmann_probability(heat_bath_temperature_in_kelvin, energy_in_joules):
    return np.exp(-energy_in_joules / (heat_bath_temperature_in_kelvin))

def markov_chain_monte_carlo_thermal_evolution(state_of_variables, temperature = 1.4):
    _NUMBER_OF_ITERATIONS = 1000

    simulation_energies = []
    simulation_magnetizations = []

    for _ in range(_NUMBER_OF_ITERATIONS):

        randomly_chosen_spin_index = int(np.floor(len(state_of_variables) * np.random.random()))
        print(f"> Chosen to flip spin #{randomly_chosen_spin_index + 1}")

        current_energy = compute_total_energy(state_of_variables)
        print(f"> Total energy is: {current_energy}")

        current_magnetization = compute_total_magnetization(state_of_variables)
        print(f"> Total magnetization is: {current_magnetization}")

        simulation_energies.append(current_energy)
        simulation_magnetizations.append(current_magnetization)

        state_of_variables[randomly_chosen_spin_index] *= -1
        print(f"> Spin flipped: {state_of_variables}")

        new_energy_after_spin_flip = compute_total_energy(state_of_variables)
        print(f"> Total energy changed to: {new_energy_after_spin_flip}")

        energy_difference = new_energy_after_spin_flip - current_energy
        print(f"> Energy difference computed as: {energy_difference}")

        if energy_difference > 0:

            monte_carlo_probability = compute_boltzmann_probability(temperature, energy_difference)

            if np.random.uniform() >= monte_carlo_probability:

                state_of_variables[randomly_chosen_spin_index] *= -1
                print("> Spin flip rejected")

    average_energy_per_simulation = np.mean(simulation_energies)
    average_magnetization_per_simulation = np.mean(simulation_magnetizations)
    print(f"> Simulation average energy was {average_energy_per_simulation} at temperature {temperature}")

    return np.array(average_energy_per_simulation), np.array(average_magnetization_per_simulation)


np.random.seed(42)
_NUMBER_OF_SPINS = 100
ising_spin_state = initialize_magnetic_spins(_NUMBER_OF_SPINS)

print(ising_spin_state)
print(compute_total_energy(ising_spin_state))
print(compute_total_magnetization(ising_spin_state))

temperature_values = np.linspace(0.1, 5.0, 100)

average_energy_per_temperature = []
average_energy_density_per_temperature = []
average_magnetization_per_temperature = []

for temperature in temperature_values:
    average_energy_for_given_temperature, average_magnetization_for_given_temperature = markov_chain_monte_carlo_thermal_evolution(ising_spin_state, temperature = temperature)

    average_energy_per_temperature.append(average_energy_for_given_temperature)
    average_magnetization_per_temperature.append(average_magnetization_for_given_temperature)

energy_plot_figure = plt.figure(figsize = (13.5, 10))

energy_plot_axes = energy_plot_figure.add_subplot(1, 1, 1)

energy_plot = PlotCustomizer(
    energy_plot_axes,
    title = r"1D Ising Chain",
    xlabel = r"$T$",
    ylabel = r"$E$",
    grid = True)

energy_plot.add_scatter_plot(
    x_data = temperature_values,
    y_data = average_energy_per_temperature,
    radial_size = 5.,
    color = 'red'
)

plt.show()