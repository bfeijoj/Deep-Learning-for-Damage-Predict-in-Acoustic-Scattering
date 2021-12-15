import numpy as np
import damageField as damf
from training import training
from predict import predict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# -------------------------------------------------------------------- Generate Data (COMMENTED EXAMPLE) ---------------------------------------------------------------------------------------------

#n_samples = 2

#training_damage, transformed_damage, covariance_matrix = damf.randomField(x = length_mesh, correlation_length = 140, stdev = 0.5, number_of_samples = n_samples)

#for ii in range(0, n_samples):

#	(u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z) = acousticScattering(inspecting_location = 'zz_3',
#																					     damage_distribution_zz_3 = transformed_damage[:, ii])

# ------------------------------------------------------------------------------- Training the Network ----------------------------------------------------------------------------------------------

interface = 'zz_3'
data_path = r'C:\Users\T-Gamer\Desktop\Doutorado\artigo scattering + neural network\Programas\Dados'
hidden_layers_parameters = [2, [1000, 1000], ['linear', 'relu']]
training(inspecting_location = interface, path = data_path, hidden_layers = hidden_layers_parameters)

# ----------------------------------------------------------------------------------- Damage Field --------------------------------------------------------------------------------------------------

damage = damf.parameterizedField(max_defect = 0.6)
interfacial_damage = damage.Gaussian(mean = 120, stdev = 0.8)

# ---------------------------------------------------------------------------------- Predict Field --------------------------------------------------------------------------------------------------

noise = 0.1
predicted_field = predict(noise_level = noise, damage_location = interface, damage_field = interfacial_damage)

# ----------------------------------------------------------------------------------- Plot Results --------------------------------------------------------------------------------------------------

x_plot = np.linspace(-1, 1, 700)
plt.figure(figsize = (12,8))
plt.plot(x_plot, interfacial_damage, label = 'True', color = 'b', linewidth = '3')
plt.plot(x_plot, predicted_field, label = 'Estimated', color = 'r', linewidth = '3', linestyle='dashed')
plt.title(f'Noise Level = {noise * 100}%', size = 36)
plt.xlabel('x', size = 30)
plt.legend(fontsize = 28, loc = "lower left")
plt.ylim([0, 1])
plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
plt.tick_params(axis='x', labelsize=28)
plt.tick_params(axis='y', labelsize=28)
plt.show()