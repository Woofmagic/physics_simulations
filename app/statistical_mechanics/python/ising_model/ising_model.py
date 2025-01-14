import numpy as np
import matplotlib.pyplot as plt

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

def markov_chain_monte_carlo_thermal_evolution(state_of_variables, temperature=1.4):
    _NUMBER_OF_ITERATIONS = 50

    simulation_energies = []
    simulation_magnetizations = []

    for iteration in range(_NUMBER_OF_ITERATIONS):

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
    print(f"> Simulation average energy was {average_energy_per_simulation} at temperature {temperature}")


np.random.seed(42)
_NUMBER_OF_SPINS = 5
ising_spin_state = initialize_magnetic_spins(_NUMBER_OF_SPINS)

print(ising_spin_state)
print(compute_total_energy(ising_spin_state))
print(compute_total_magnetization(ising_spin_state))

markov_chain_monte_carlo_thermal_evolution(ising_spin_state)