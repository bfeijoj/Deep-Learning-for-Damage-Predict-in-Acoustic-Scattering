import numpy as np
import pandas as pd 
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy import stats
from scipy.stats import norm
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import damageField as damf
import acousticTensors as at
from QSA import Stiffness


def acousticScattering(inspecting_location,
				  damage_distribution_xx_2 = np.ones((700)),
				  damage_distribution_zz_2 = np.ones((700)),
				  damage_distribution_xx_3 = np.ones((700)),
				  damage_distribution_zz_3 = np.ones((700))):

	# -------------------------------------------------------------------------------- Reference values -------------------------------------------------------------------------------------------------

	h_ref = 1e-2
	rho_ref = 5e3
	lambda_ref = 2.5e3 ** 2 * rho_ref
	c_ref = np.sqrt(lambda_ref / rho_ref)

	scale_factor = 10e-2 / h_ref
	x = np.arange(- 50 * scale_factor, 50 * scale_factor, scale_factor / 7)
	dx = x[1] - x[0]
	lx = len(x)

	# ------------------------------------------------------------------------------------ Interfaces ---------------------------------------------------------------------------------------------------

	h_interface_2 = 100e-6 / h_ref
	vp_interface_2 = 2150 / c_ref
	vs_interface_2 = 1030 / c_ref
	rho_interface_2 = 1200 / rho_ref

	stiffness_2 = Stiffness(h_interface_2, rho_interface_2, vp_interface_2, vs_interface_2)
	K_specular_2 = stiffness_2.calculate(defect = 0)

	h_interface_3 = 100e-6 / h_ref
	vp_interface_3 = 2150 / c_ref
	vs_interface_3 = 1030 / c_ref
	rho_interface_3 = 1200 / rho_ref

	stiffness_3 = Stiffness(h_interface_3, rho_interface_3, vp_interface_3, vs_interface_3)
	K_specular_3 = stiffness_3.calculate(defect = 0)

	K_scattering_2 = np.zeros((3, lx))
	K_112 = np.zeros((lx))
	K_222 = np.zeros((lx))
	K_332 = np.zeros((lx))

	defect_xx_2 = damage_distribution_xx_2.reshape(700,)
	defect_yy_2 = np.ones((lx))
	defect_zz_2 = damage_distribution_zz_2.reshape(700,)

	K_scattering_2[0, :] = K_specular_2[0, 0] * defect_xx_2
	K_scattering_2[1, :] = K_specular_2[1, 1] * defect_yy_2
	K_scattering_2[2, :] = K_specular_2[2, 2] * defect_zz_2

	K_112 = K_scattering_2[0, :]
	K_222 = K_scattering_2[1, :]
	K_332 = K_scattering_2[2, :]

	K_scattering_3 = np.zeros((3, lx))
	K_113 = np.zeros((lx))
	K_223 = np.zeros((lx))
	K_333 = np.zeros((lx))

	defect_xx_3 = damage_distribution_xx_3.reshape(700,)
	defect_yy_3 = np.ones((lx))
	defect_zz_3 = damage_distribution_zz_3.reshape(700,)

	K_scattering_3[0, :] = K_specular_3[0, 0] * defect_xx_3
	K_scattering_3[1, :] = K_specular_3[1, 1] * defect_yy_3
	K_scattering_3[2, :] = K_specular_3[2, 2] * defect_zz_3

	K_113 = K_scattering_3[0, :]
	K_223 = K_scattering_3[1, :]
	K_333 = K_scattering_3[2, :]


	if inspecting_location == 'xx_2':
		incidence = 4.128510809424635
		freq = 102.8 * 1e3

	elif inspecting_location == 'xx_3':
		incidence = 9.2
		freq = 96.5 * 1e3

	elif inspecting_location == 'zz_2':
		incidence = 0
		freq = 128.4 * 1e3

	elif inspecting_location == 'zz_3':
		incidence = 0
		freq = 128.4 * 1e3


	# ------------------------------------------------------------------------------- Acoustic parameters -----------------------------------------------------------------------------------------------

	omega = freq * 2 * np.pi * h_ref / c_ref

	# ------------------------------------------------------------------------------- Inferior Half-Space -----------------------------------------------------------------------------------------------

	rho_inf = 1000 / rho_ref
	vp_inf = 1480 / c_ref

	# ------------------------------------------------------------------------------------- layer 1 -----------------------------------------------------------------------------------------------------

	rho_1 = 2700 / rho_ref
	vp_1 = 6320 / c_ref
	vs_1 = 3130 /c_ref
	mu_1 = vs_1 ** 2 * rho_1
	lambda_1 = vp_1 ** 2 * rho_1 - 2 * mu_1
	h_1 = 2e-2 / h_ref

	# ------------------------------------------------------------------------------------- layer 2 -----------------------------------------------------------------------------------------------------

	rho_2 = 8930 / rho_ref
	vp_2 = 4660 / c_ref
	vs_2 = 2660 /c_ref
	mu_2 = vs_2 ** 2 * rho_2
	lambda_2 = vp_2 ** 2 * rho_2 - 2 * mu_2
	h_2 = 1e-2 / h_ref

	# ------------------------------------------------------------------------------------- layer 3 -----------------------------------------------------------------------------------------------------

	rho_3 = 7900 / rho_ref
	vp_3 = 5790 / c_ref
	vs_3 = 3100 /c_ref
	mu_3 = vs_3 ** 2 * rho_3
	lambda_3 = vp_3 ** 2 * rho_3 - 2 * mu_3
	h_3 = 3e-2 / h_ref

	# ------------------------------------------------------------------------------- Superior Half-Space -----------------------------------------------------------------------------------------------

	rho_sup = 1000 / rho_ref
	vp_sup = 1480 / c_ref

	# -------------------------------------------------------------------------------- Acoustic properties ----------------------------------------------------------------------------------------------

	kp_inf = omega / vp_inf
	kp_1 = omega / vp_1
	kt_1 = omega / vs_1
	kp_2 = omega / vp_2
	kt_2 = omega / vs_2
	kp_3 = omega / vp_3
	kt_3 = omega / vs_3
	kp_sup = omega / vp_sup

	# ------------------------------------------------------------------------------------- Incidence ---------------------------------------------------------------------------------------------------

	Amplitude = 1

	alpha_max = 1 / (2 * dx)
	dalpha = alpha_max / ((lx / 2 + 1) - 1)
	alpha_plus = np.arange(0, alpha_max, dalpha)
	alpha_minus = np.arange(-alpha_max, 0, dalpha)
	alpha = np.append(alpha_minus, alpha_plus) * 2 * np.pi
	lalpha = len(alpha)

	xL_minus = -np.sqrt((x[0 : lx // 2]) ** 2) / (1 + np.tan(np.deg2rad(incidence)) ** 2)
	xL_plus = np.sqrt((x[lx // 2 : lx]) ** 2) / (1 + np.tan(np.deg2rad(incidence)) ** 2)
	xL = np.append(xL_minus, xL_plus)
	dxL = xL[1] - xL[0]
	lxL = len(xL)

	alphaL_max = 1 / (2 * dxL)
	dalphaL = alphaL_max / ((lxL / 2 + 1) - 1)
	alphaL_plus = np.arange(0, alphaL_max, dalphaL)
	alphaL_minus = np.arange(-alphaL_max, 0, dalphaL)
	alphaL = np.append(alphaL_minus, alphaL_plus) * 2 * np.pi
	lalphaL = len(alphaL)

	zL_minus = -xL[0 : lxL // 2] * np.tan(np.deg2rad(incidence))
	zL_plus = -xL[lxL // 2 : lxL] * np.tan(np.deg2rad(incidence))
	zL = np.append(zL_minus, zL_plus)
	dzL = zL[1] - zL[0]
	lzL = len(zL)

	pulse = np.zeros((lxL))

	for ii in range(0, lxL):
	    pulse[ii] = Amplitude

	pulse_fft = fftshift(fft(pulse))
	    
	aux1L_fft = np.zeros((lalphaL), dtype = complex)
	u_incidence1 = np.zeros((lalphaL), dtype = complex)
	u_incidence3 = np.zeros((lalphaL), dtype = complex)
	for ii in range (0, lzL):
	    for jj in range(0, lalphaL):
	        kpz_sup = np.sqrt(kp_sup ** 2 - alphaL[jj] ** 2, dtype = complex)
	        aux1L_fft[jj] = pulse_fft[jj] * np.exp(-1j * kpz_sup * zL[ii])
	    aux1L = ifft(ifftshift(aux1L_fft))
	    u_incidence1[ii] = aux1L[ii]* np.sin(np.deg2rad(incidence))
	    u_incidence3[ii] = -aux1L[ii] * np.cos(np.deg2rad(incidence))
	    
	u_plus_down_4 = np.zeros((3, len(x)), dtype = complex)
	u_plus_down_4[0, :] = u_incidence1
	u_plus_down_4[2, :] = u_incidence3
	    
	u_plus_down_4_fft = np.zeros((3, len(x)), dtype = complex)
	u_plus_down_4_fft[0, :] = fftshift(fft(u_plus_down_4[0, :]))
	u_plus_down_4_fft[1, :] = fftshift(fft(u_plus_down_4[1, :]))
	u_plus_down_4_fft[2, :] = fftshift(fft(u_plus_down_4[2, :]))

	# ------------------------------------------------------------------------------ Calculating the Tensors --------------------------------------------------------------------------------------------

	Zf_down_inf = np.zeros((3, 3, lalpha), dtype = complex)

	Z_up_1 = np.zeros((3, 3, lalpha), dtype = complex)
	Z_down_1 = np.zeros((3, 3, lalpha), dtype = complex)
	M_up_1 = np.zeros((3, 3, lalpha), dtype = complex)
	M_down_1 = np.zeros((3, 3, lalpha), dtype = complex)

	Z_up_2 = np.zeros((3, 3, lalpha), dtype = complex)
	Z_down_2 = np.zeros((3, 3, lalpha), dtype = complex)
	M_up_2 = np.zeros((3, 3, lalpha), dtype = complex)
	M_down_2 = np.zeros((3, 3, lalpha), dtype = complex)

	Z_up_3 = np.zeros((3, 3, lalpha), dtype = complex)
	Z_down_3 = np.zeros((3, 3, lalpha), dtype = complex)
	M_up_3 = np.zeros((3, 3, lalpha), dtype = complex)
	M_down_3 = np.zeros((3, 3, lalpha), dtype = complex)

	Zf_up_sup = np.zeros((3, 3, lalpha), dtype = complex)
	Zf_down_sup = np.zeros((3, 3, lalpha), dtype = complex)

	for ii in range(0, lalpha):

	    kpz_inf = np.sqrt(kp_inf ** 2 - alpha[ii] ** 2, dtype = complex)
	    zf_down_inf = rho_inf * omega / kpz_inf
	    Zf_down_inf[:, :, ii] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, zf_down_inf]])

	    Z_up_1[:, :, ii], Z_down_1[:, :, ii] = at.calculateImpedance(omega, alpha[ii], kp_1, kt_1, lambda_1, mu_1)
	    M_up_1[:, :, ii], M_down_1[:, :, ii] = at.calculatePropagationMatrix(alpha[ii], kp_1, kt_1, -h_1, -h_1)

	    Z_up_2[:, :, ii], Z_down_2[:, :, ii] = at.calculateImpedance(omega, alpha[ii], kp_2, kt_2, lambda_2, mu_2)
	    M_up_2[:, :, ii], M_down_2[:, :, ii] = at.calculatePropagationMatrix(alpha[ii], kp_2, kt_2, -h_2, -h_2)

	    Z_up_3[:, :, ii], Z_down_3[:, :, ii] = at.calculateImpedance(omega, alpha[ii], kp_3, kt_3, lambda_3, mu_3)
	    M_up_3[:, :, ii], M_down_3[:, :, ii] = at.calculatePropagationMatrix(alpha[ii], kp_3, kt_3, -h_3, -h_3)


	    kpz_sup = np.sqrt(kp_sup ** 2 - alpha[ii] ** 2, dtype = complex)
	    zf_down_sup = rho_sup * omega / kpz_sup
	    Zf_down_sup[:, :, ii] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, zf_down_sup]])
	    zf_up_sup = -rho_sup * omega / kpz_sup
	    Zf_up_sup[:, :, ii] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, zf_up_sup]])

	# ------------------------------------------------------------------------------------- Initial Guess -----------------------------------------------------------------------------------------------

	p0_guess_fft = np.zeros((1, 4 * lalpha), dtype = complex)
	u_guess_fft = np.zeros((3, lalpha), dtype = complex)
	cont = 0

	for ii in range(0, len(p0_guess_fft[0,:]), 4):

	    u_guess_fft[0, cont] = p0_guess_fft[0, ii] + p0_guess_fft[0, ii + 1] * 1j
	    u_guess_fft[1, cont] = 0
	    u_guess_fft[2, cont] = p0_guess_fft[0, ii + 2] + p0_guess_fft[0, ii + 3] * 1j

	    cont += 1

	# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	initial_cost = np.zeros((1))
	final_cost = np.zeros((1))
	u_solution_x_fft = np.zeros((lx))
	u_solution_y_fft = np.zeros((lx))
	u_solution_z_fft = np.zeros((lx))
	u_solution_x = np.zeros((lx))
	u_solution_z = np.zeros((lx))
	defect_distribuction = np.zeros((lx))
	defect_field = np.zeros((lx))
	defect_field_non_transformed = np.zeros((lx))

	# -------------------------------------------------------------------------- Iterative process - Direct Problem -------------------------------------------------------------------------------------

        
    # -------------------------------------------------------------------------------- Damaged Stiffness --------------------------------------------------------------------------------------------


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	u_plus_down_1_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_1_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_1_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_1_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_down_1_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_down_1_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_up_1_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_down_1_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_1_fft = np.zeros((3, lalpha), dtype = complex)

	u_plus_down_2_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_2_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_2_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_2_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_down_2_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_up_2_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_down_2_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_2_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_2_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_2_fft = np.zeros((3, lalpha), dtype = complex)

	u_plus_down_3_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_3_fft = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_3_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_3_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_down_3_fft = np.zeros((3, lalpha), dtype = complex)
	t_plus_up_3_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_down_3_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_3_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_3_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_3_fft = np.zeros((3, lalpha), dtype = complex)

	u_minus_down_4_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_4_fft = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_4_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_up_4_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_4_fft = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_4_fft = np.zeros((3, lalpha), dtype = complex)

	u_plus_down_1 = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_1 = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_1 = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_1 = np.zeros((3, lalpha), dtype = complex)

	u_plus_down_2 = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_2 = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_2 = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_2 = np.zeros((3, lalpha), dtype = complex)
	u_minus_down_2 = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_2 = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_2 = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_2 = np.zeros((3, lalpha), dtype = complex)

	u_plus_down_3 = np.zeros((3, lalpha), dtype = complex)
	u_plus_up_3 = np.zeros((3, lalpha), dtype = complex)
	u_plus_total_3 = np.zeros((3, lalpha), dtype = complex)
	t_plus_total_3 = np.zeros((3, lalpha), dtype = complex)
	u_minus_down_3 = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_3 = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_3 = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_3 = np.zeros((3, lalpha), dtype = complex)

	u_minus_down_4 = np.zeros((3, lalpha), dtype = complex)
	u_minus_up_4 = np.zeros((3, lalpha), dtype = complex)
	u_minus_total_4 = np.zeros((3, lalpha), dtype = complex)
	t_minus_total_4 = np.zeros((3, lalpha), dtype = complex)


    # ------------------------------------------------------------------------------ Interfaces 3 and 4 ---------------------------------------------------------------------------------------------


	for ii in range(0, lalpha):

		u_minus_up_4_fft[:, ii] = np.dot(np.linalg.inv(Zf_up_sup[:, :, ii] - Z_up_3[:, :, ii]), 
                                         (np.dot((Z_down_3[:, :, ii] - Zf_up_sup[:, :, ii]), u_guess_fft[:, ii]) +
                                          np.dot((Zf_up_sup[:, :, ii] - Zf_down_sup[:, :, ii]), u_plus_down_4_fft[:, ii])))

		u_plus_down_3_fft[:, ii] = np.dot(M_down_3[:, :, ii], u_guess_fft[:, ii])

		t_plus_down_3_fft[:, ii] = - 1j * omega * np.dot(Z_down_3[:, :, ii], u_plus_down_3_fft[:, ii])

		u_plus_up_3_fft[:, ii] = np.dot(M_up_3[:, :, ii], u_minus_up_4_fft[:, ii])

		t_plus_up_3_fft[:, ii] = -1j * omega * np.dot(Z_up_3[:, :, ii], u_plus_up_3_fft[:, ii])

	u_plus_total_3_fft = u_plus_up_3_fft + u_plus_down_3_fft
	t_plus_total_3_fft = t_plus_up_3_fft + t_plus_down_3_fft
	t_minus_total_3_fft = t_plus_total_3_fft

	for ii in range(0, 3, 2):

		u_plus_total_3[ii, :] = ifft(ifftshift(u_plus_total_3_fft[ii, :]))
		t_plus_total_3[ii, :] = ifft(ifftshift(t_plus_total_3_fft[ii, :]))
		t_minus_total_3[ii, :] = ifft(ifftshift(t_minus_total_3_fft[ii, :]))

	for ii in range(0, lx):

		u_minus_total_3[:, ii] = u_plus_total_3[:, ii] - np.dot(np.linalg.inv(np.diag([K_113[ii], K_223[ii], K_333[ii]])),
                                                                t_minus_total_3[:, ii])

	for ii in range(0, 3, 2):

		u_minus_total_3_fft[ii, :] = fftshift(fft(u_minus_total_3[ii, :])) 

	for ii in range(0, lalpha):

		u_minus_up_3_fft[:, ii] = np.dot(np.linalg.inv(Z_up_2[:, :, ii] - Z_down_2[:, :, ii]),
                                     (t_minus_total_3_fft[:, ii]) / (-1j * omega) - np.dot(Z_down_2[:, :, ii], 
                                                                                   u_minus_total_3_fft[:, ii]))

		u_minus_down_3_fft[:, ii] = u_minus_total_3_fft[:, ii] - u_minus_up_3_fft[:, ii]


    # ----------------------------------------------------------------------------------- Interface 2 -----------------------------------------------------------------------------------------------


	for ii in range(0, lalpha):

		u_plus_down_2_fft[:, ii] = np.dot(M_down_2[:, :, ii], u_minus_down_3_fft[:, ii])
		t_plus_down_2_fft[:, ii] = -1j * omega * np.dot(Z_down_2[:, :, ii], u_plus_down_2_fft[:, ii])
		u_plus_up_2_fft[:, ii] = np.dot(M_up_2[:, :, ii], u_minus_up_3_fft[:, ii])
		t_plus_up_2_fft[:, ii] = -1j * omega * np.dot(Z_up_2[:, :, ii], u_plus_up_2_fft[:, ii])

	u_plus_total_2_fft = u_plus_up_2_fft + u_plus_down_2_fft
	t_plus_total_2_fft = t_plus_up_2_fft + t_plus_down_2_fft
	t_minus_total_2_fft = t_plus_total_2_fft

	for ii in range(0, 3, 2):

		u_plus_total_2[ii, :] = ifft(ifftshift(u_plus_total_2_fft[ii, :]))
		t_plus_total_2[ii, :] = ifft(ifftshift(t_plus_total_2_fft[ii, :]))
		t_minus_total_2[ii, :] = ifft(ifftshift(t_minus_total_2_fft[ii, :]))

	for ii in range(0, lx):

		u_minus_total_2[:, ii] = u_plus_total_2[:, ii] - np.dot(np.linalg.inv(np.diag([K_112[ii], K_222[ii], K_332[ii]])),
                                                                t_minus_total_2[:, ii])


	for ii in range(0, 3, 2):

		u_minus_total_2_fft[ii, :] = fftshift(fft(u_minus_total_2[ii, :]))

	for ii in range(0, lalpha):

		u_minus_up_2_fft[:, ii] = np.dot(np.linalg.inv(Z_up_1[:, :, ii] - Z_down_1[:, :, ii]),
                                     (t_minus_total_2_fft[:, ii]) / (-1j * omega) - np.dot(Z_down_1[:, :, ii], 
                                                                                   u_minus_total_2_fft[:, ii]))
		u_minus_down_2_fft[:, ii] = u_minus_total_2_fft[:, ii] - u_minus_up_2_fft[:, ii]


    # ------------------------------------------------------------------------------------Interface 1 -----------------------------------------------------------------------------------------------


	for ii in range(0, lalpha):

		u_plus_down_1_fft[:, ii] = np.dot(M_down_1[:, :, ii], u_minus_down_2_fft[:, ii])
		t_plus_down_1_fft[:, ii] = -1j * omega * np.dot(Z_down_1[:, :, ii], u_plus_down_1_fft[:, ii])
		u_plus_up_1_fft[:, ii] = np.dot(M_up_1[:, :, ii], u_minus_up_2_fft[:, ii])
		t_plus_up_1_fft[:, ii] = -1j * omega * np.dot(Z_up_1[:, :, ii], u_plus_up_1_fft[:, ii])
		u_minus_down_1_fft[:, ii] = u_plus_up_1_fft[:, ii] + u_plus_down_1_fft[:, ii]
		t_minus_down_1_fft[:, ii] = -1j * omega * np.dot(Zf_down_inf[:, :, ii], u_minus_down_1_fft[:, ii])

	t_plus_total_1_fft = t_plus_up_1_fft + t_plus_down_1_fft

    # ------------------------------------------------------------------------- Assembling the residual vector --------------------------------------------------------------------------------------


	residual_vector = np.zeros((4 * lalpha, 1), dtype = complex)
	residual_aux1 = t_plus_total_1_fft[0, :] - t_minus_down_1_fft[0, :]
	residual_aux3 = t_plus_total_1_fft[2, :] - t_minus_down_1_fft[2, :]
	cont = 0

	for ii in range(0, lalpha):

		residual_vector[cont, 0] = np.real(residual_aux1[ii])
		residual_vector[cont + 1, 0] = np.real(residual_aux3[ii])
		residual_vector[cont + 2, 0] = np.imag(residual_aux1[ii])
		residual_vector[cont + 3, 0] = np.imag(residual_aux3[ii])
		cont += 4

	cost_function_0 = np.dot(residual_vector.T, residual_vector)


    # ------------------------------------------------------------------------- Assembling the jacobian matrix --------------------------------------------------------------------------------------


	jacobian_rows = 4 * lalpha
	jacobian_columns = 4 * lalpha
	jacobian_matrix = np.zeros((jacobian_rows, jacobian_columns), dtype = complex)
	epsilon = 1.0

	for kk in range(0, 4 * lalpha):


		p_fft = np.zeros((1, 4 * lalpha), dtype = complex)
		p_fft[0, kk] = p_fft[0, kk] + epsilon
		u_iter_fft = np.zeros((3, lalpha), dtype = complex)
		cont = 0

		for ii in range(0, len(p_fft[0,:]), 4):

			u_iter_fft[0, cont] = p_fft[0, ii] + p_fft[0, ii + 1] * 1j
			u_iter_fft[1, cont] = 0
			u_iter_fft[2, cont] = p_fft[0, ii + 2] + p_fft[0, ii + 3] * 1j

			cont += 1


        # --------------------------------------------------------------------------- Interfaces 3 and 4 --------------------------------------------------------------------------------------------


		for ii in range(0, lalpha):

			u_minus_up_4_fft[:, ii] = np.dot(np.linalg.inv(Zf_up_sup[:, :, ii] - Z_up_3[:, :, ii]), 
                                             (np.dot((Z_down_3[:, :, ii] - Zf_up_sup[:, :, ii]), u_iter_fft[:, ii]) +
                                              np.dot((Zf_up_sup[:, :, ii] - Zf_down_sup[:, :, ii]), u_plus_down_4_fft[:, ii])))

			u_plus_down_3_fft[:, ii] = np.dot(M_down_3[:, :, ii], u_iter_fft[:, ii])

			t_plus_down_3_fft[:, ii] = - 1j * omega * np.dot(Z_down_3[:, :, ii], u_plus_down_3_fft[:, ii])

			u_plus_up_3_fft[:, ii] = np.dot(M_up_3[:, :, ii], u_minus_up_4_fft[:, ii])

			t_plus_up_3_fft[:, ii] = -1j * omega * np.dot(Z_up_3[:, :, ii], u_plus_up_3_fft[:, ii])

		u_plus_total_3_fft = u_plus_up_3_fft + u_plus_down_3_fft
		t_plus_total_3_fft = t_plus_up_3_fft + t_plus_down_3_fft
		t_minus_total_3_fft = t_plus_total_3_fft

		for ii in range(0, 3, 2):

			u_plus_total_3[ii, :] = ifft(ifftshift(u_plus_total_3_fft[ii, :]))
			t_plus_total_3[ii, :] = ifft(ifftshift(t_plus_total_3_fft[ii, :]))
			t_minus_total_3[ii, :] = ifft(ifftshift(t_minus_total_3_fft[ii, :]))

		for ii in range(0, lx):

			u_minus_total_3[:, ii] = u_plus_total_3[:, ii] - np.dot(np.linalg.inv(np.diag([K_113[ii], K_223[ii], K_333[ii]])),
                                                                    t_minus_total_3[:, ii])

		for ii in range(0, 3, 2):

			u_minus_total_3_fft[ii, :] = fftshift(fft(u_minus_total_3[ii, :])) 

		for ii in range(0, lalpha):

			u_minus_up_3_fft[:, ii] = np.dot(np.linalg.inv(Z_up_2[:, :, ii] - Z_down_2[:, :, ii]),
                                         (t_minus_total_3_fft[:, ii]) / (-1j * omega) - np.dot(Z_down_2[:, :, ii], 
                                                                                       u_minus_total_3_fft[:, ii]))

			u_minus_down_3_fft[:, ii] = u_minus_total_3_fft[:, ii] - u_minus_up_3_fft[:, ii]


        # ------------------------------------------------------------------------------- Interface 2 -----------------------------------------------------------------------------------------------


		for ii in range(0, lalpha):

			u_plus_down_2_fft[:, ii] = np.dot(M_down_2[:, :, ii], u_minus_down_3_fft[:, ii])
			t_plus_down_2_fft[:, ii] = -1j * omega * np.dot(Z_down_2[:, :, ii], u_plus_down_2_fft[:, ii])
			u_plus_up_2_fft[:, ii] = np.dot(M_up_2[:, :, ii], u_minus_up_3_fft[:, ii])
			t_plus_up_2_fft[:, ii] = -1j * omega * np.dot(Z_up_2[:, :, ii], u_plus_up_2_fft[:, ii])

		u_plus_total_2_fft = u_plus_up_2_fft + u_plus_down_2_fft
		t_plus_total_2_fft = t_plus_up_2_fft + t_plus_down_2_fft
		t_minus_total_2_fft = t_plus_total_2_fft

		for ii in range(0, 3, 2):

			u_plus_total_2[ii, :] = ifft(ifftshift(u_plus_total_2_fft[ii, :]))
			t_plus_total_2[ii, :] = ifft(ifftshift(t_plus_total_2_fft[ii, :]))
			t_minus_total_2[ii, :] = ifft(ifftshift(t_minus_total_2_fft[ii, :]))

		for ii in range(0, lx):

			u_minus_total_2[:, ii] = u_plus_total_2[:, ii] - np.dot(np.linalg.inv(np.diag([K_112[ii], K_222[ii], K_332[ii]])),
                                                                    t_minus_total_2[:, ii])


		for ii in range(0, 3, 2):

			u_minus_total_2_fft[ii, :] = fftshift(fft(u_minus_total_2[ii, :]))

		for ii in range(0, lalpha):

			u_minus_up_2_fft[:, ii] = np.dot(np.linalg.inv(Z_up_1[:, :, ii] - Z_down_1[:, :, ii]),
                                         (t_minus_total_2_fft[:, ii]) / (-1j * omega) - np.dot(Z_down_1[:, :, ii], 
                                                                                       u_minus_total_2_fft[:, ii]))
			u_minus_down_2_fft[:, ii] = u_minus_total_2_fft[:, ii] - u_minus_up_2_fft[:, ii]


        # ------------------------------------------------------------------------------- Interface 1 -----------------------------------------------------------------------------------------------


		for ii in range(0, lalpha):

			u_plus_down_1_fft[:, ii] = np.dot(M_down_1[:, :, ii], u_minus_down_2_fft[:, ii])
			t_plus_down_1_fft[:, ii] = -1j * omega * np.dot(Z_down_1[:, :, ii], u_plus_down_1_fft[:, ii])
			u_plus_up_1_fft[:, ii] = np.dot(M_up_1[:, :, ii], u_minus_up_2_fft[:, ii])
			t_plus_up_1_fft[:, ii] = -1j * omega * np.dot(Z_up_1[:, :, ii], u_plus_up_1_fft[:, ii])
			u_minus_down_1_fft[:, ii] = u_plus_up_1_fft[:, ii] + u_plus_down_1_fft[:, ii]
			t_minus_down_1_fft[:, ii] = -1j * omega * np.dot(Zf_down_inf[:, :, ii], u_minus_down_1_fft[:, ii])

		t_plus_total_1_fft = t_plus_up_1_fft + t_plus_down_1_fft

        # ---------------------------------------------------------------------- Assembling the residual vector -------------------------------------------------------------------------------------


		residual_vector_jacobian = np.zeros((4 * lalpha, 1), dtype = complex)
		residual_aux1_jacobian = t_plus_total_1_fft[0, :] - t_minus_down_1_fft[0, :]
		residual_aux3_jacobian = t_plus_total_1_fft[2, :] - t_minus_down_1_fft[2, :]
		cont = 0

		for ii in range(0, lalpha):

			residual_vector_jacobian[cont, 0] = np.real(residual_aux1_jacobian[ii])
			residual_vector_jacobian[cont + 1, 0] = np.real(residual_aux3_jacobian[ii])
			residual_vector_jacobian[cont + 2, 0] = np.imag(residual_aux1_jacobian[ii])
			residual_vector_jacobian[cont + 3, 0] = np.imag(residual_aux3_jacobian[ii])
			cont += 4

		jacobian_matrix[:, kk] = (residual_vector_jacobian[:, 0] - residual_vector[:, 0]) / epsilon

    # -------------------------------------------------------------------------------- Solving the problem ------------------------------------------------------------------------------------------

	solution_vector = p0_guess_fft.T - np.dot(np.linalg.inv(jacobian_matrix), residual_vector)
	u_solution_fft = np.zeros((3, lalpha), dtype = complex)
	u_solution = np.zeros((3, lx), dtype = complex)

	cont = 0

	for ii in range(0, len(solution_vector), 4):

		u_solution_fft[0, cont] = solution_vector[ii, 0] + solution_vector[ii + 1, 0] * 1j
		u_solution_fft[2, cont] = solution_vector[ii + 2, 0] + solution_vector[ii + 3, 0] * 1j

		cont += 1

	for ii in range(0, 3, 2):
		u_solution[ii, :] = ifft(ifftshift(u_solution_fft[ii, :]))
    
	u_solution_x_fft = abs(u_solution_fft[0, :])
	u_solution_z_fft = abs(u_solution_fft[2, :])
	u_solution_x = abs(ifft(ifftshift(u_solution_fft[0, :])))
	u_solution_z = abs(ifft(ifftshift(u_solution_fft[2, :])))

	return u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z