import numpy as np

def calculateImpedance(omega, kx, kp, kt, lambda_, mu):
    
    kp_z = np.sqrt(kp ** 2 - kx ** 2, dtype = complex)
    kt_z = np.sqrt(kt ** 2 - kx ** 2, dtype = complex)
    
    A_aux_up = np.array([[kx, kp_z], [kt_z, -kx]], dtype = complex)
    A_aux_down = np.array([[-kx, kp_z], [kt_z, kx]], dtype = complex)
    
    C1_up = np.array([[-2 * mu * kp_z * kx / omega], [(mu / omega) * (kx ** 2 - kt_z ** 2)]], dtype = complex)
    C2_up = np.array([[-(lambda_ * kp ** 2 + 2 * mu * kp_z ** 2) / omega], [2 * mu * kt_z * kx / omega]], dtype = complex)
    C1_down = np.array([[-2 * mu * kp_z * kx / omega], [-(mu / omega) * (kx ** 2 - kt_z ** 2)]], dtype = complex)
    C2_down = np.array([[(lambda_ * kp ** 2 + 2 * mu * kp_z ** 2) / omega], [2 * mu * kt_z * kx / omega]], dtype = complex)
    
    z_11_up_z_13_up = np.dot(np.linalg.inv(A_aux_up), C1_up)
    z_11_up = z_11_up_z_13_up[0]
    z_13_up = z_11_up_z_13_up[1]
    z_12_up = 0
    z_21_up = 0
    z_22_up = -mu * kt_z / omega
    z_23_up = 0
    z_32_up = 0
    z_31_up_z_33_up = np.dot(np.linalg.inv(A_aux_up), C2_up)
    z_31_up = z_31_up_z_33_up[0]
    z_33_up = z_31_up_z_33_up[1]
    
    z_11_down_z_13_down = np.dot(np.linalg.inv(A_aux_down), C1_down)
    z_11_down = z_11_down_z_13_down[0]
    z_13_down = z_11_down_z_13_down[1]
    z_12_down = 0
    z_21_down = 0
    z_22_down = mu * kt_z / omega
    z_23_down = 0
    z_32_down = 0
    z_31_down_z_33_down = np.dot(np.linalg.inv(A_aux_down), C2_down)
    z_31_down = z_31_down_z_33_down[0]
    z_33_down = z_31_down_z_33_down[1]
    
    Z_up = np.array([[z_11_up, z_12_up, z_13_up],
                     [z_21_up, z_22_up, z_23_up],
                     [z_31_up, z_32_up, z_33_up]],
                      dtype = complex)
    
    Z_down = np.array([[z_11_down, z_12_down, z_13_down],
                       [z_21_down, z_22_down, z_23_down],
                       [z_31_down, z_32_down, z_33_down]],
                        dtype = complex)
    
    return Z_up, Z_down
    
def calculatePropagationMatrix(kx, kp, kt, h_up, h_down):
    
    kp_z = np.sqrt(kp ** 2 - kx ** 2, dtype = complex)
    kt_z = np.sqrt(kt ** 2 - kx ** 2, dtype = complex)
    
    gamma_up = np.zeros((3,3), dtype = complex)
    np.fill_diagonal(gamma_up,
                     [np.exp(1j * kp_z * h_up),
                      np.exp(1j * kt_z * h_up),
                      np.exp(1j * kt_z * h_up)])
            
    gamma_down = np.zeros((3,3), dtype = complex)
    np.fill_diagonal(gamma_down,
                     [np.exp(-1j * kp_z * h_down),
                      np.exp(-1j * kt_z * h_down),
                      np.exp(-1j * kt_z * h_down)])
    
    A_up = np.array([[kx / kp, -kt_z / kt, 0], 
                     [0, 0, 1], 
                     [kp_z / kp, kx / kt, 0]])
    A_down = np.array([[kx / kp, kt_z / kt, 0], 
                     [0, 0, 1], 
                     [-kp_z / kp, kx / kt, 0]])
    
    M_up = np.dot(np.dot(A_up, gamma_up), np.linalg.inv(A_up))
    M_down = np.dot(np.dot(A_down, gamma_down), np.linalg.inv(A_down))
    
    return M_up, M_down

def calculateFluidImpedance(omega, kx, kp, rho):
    
    kp_z = np.sqrt(kp ** 2 - kx ** 2, dtype = complex)
    zf = rho * omega / kp_z
    
    Z_up = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, -z_33_up]],
                      dtype = complex)
    
    Z_down = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, z_33_up]],
                      dtype = complex)
    
    return Z_up, Z_down
    
def calculateFluidPropagationMatrix(kx, kp, h_up, h_down):
    
    kp_z = np.sqrt(kp ** 2 - kx ** 2, dtype = complex)
    
    M_up = np.exp(1j * kp_z * h_up)
    M_down = np.exp(- 1j * kp_z * h_down)
    
    return M_up, M_down