import numpy as np

def randomField(correlation_length, stdev, number_of_samples, mean_random_field = 0, field_type = 'gaussian'):

    h_ref = 1e-2
    scale_factor = 10e-2 / h_ref
    x = np.arange(- 50 * scale_factor, 50 * scale_factor, scale_factor / 7)
    n_elements = len(x)
    
    if field_type == 'gaussian':
        covariance_field = lambda r, s :stdev ** 2 * np.exp(-1 * ((r - s) / correlation_length) ** 2)
    elif field_type == 'exponential':
        covariance_field = lambda r, s : stdev ** 2 * np.exp(-1 * abs(r - s) / correlation_length)
    else:
        print('error - field type not found')
        
    covariance_matrix = np.zeros((n_elements, n_elements))
        
    for ii in range(0, n_elements):
        for jj in range(ii, n_elements):
            covariance_matrix[ii, jj] = covariance_field(x[ii], x[jj])
    covariance_matrix = (covariance_matrix + covariance_matrix.T) - np.diag(np.diag(covariance_matrix))
        
    xi = np.random.randn(n_elements, number_of_samples)
    
    muY = mean_random_field * np.ones((n_elements, number_of_samples))
    Ag, Ug = np.linalg.eig(covariance_matrix)
    Ag = np.diag(Ag)
    Vg = np.dot(Ug, np.real(np.sqrt(Ag, dtype = complex)))
    random_field = muY + np.dot(Vg, xi)
    
    random_field_transformed = np.tanh(np.exp(-random_field))
    
    for ii in range(0, number_of_samples):
        random_field_transformed[:, ii] = (random_field_transformed[:, ii] - max(random_field_transformed[:, ii]))
    
    filter_center = (x[0] + x[-1]) / 2
    filter_sigma = 2
    filter_offset = x[-1] - 0.05 * (x[-1] - x[0])
    field_filter = np.zeros(x.shape)
    field_filter_2 = np.zeros(x.shape)

    for ii in range(0, len(x)):
        field_filter[ii] = np.exp(- np.max([0, abs(x[ii] - filter_center) - filter_offset]) / (2 * filter_sigma ** 2))

    damage_field = np.zeros(random_field.shape)
    damage_field_transformed = np.zeros(random_field.shape)
    
    for ii in range(len(x)):
        damage_field_transformed[ii, :] = field_filter[ii] * random_field_transformed[ii, :] + 1
        damage_field[ii, :] = field_filter[ii] * random_field[ii, :]
        
    
    
    return damage_field, damage_field_transformed, covariance_matrix


class parameterizedField:
    
    def __init__(self, max_defect = 0.99):
        self.max_defect = max_defect
        self.h_ref = 1e-2
        scale_factor = 10e-2 / self.h_ref
        self.x = np.arange(- 50 * scale_factor, 50 * scale_factor, scale_factor / 7)

    def Gaussian(self, mean = None, stdev = None):
        if mean is None:
            mean = (max(self.x) + min(self.x))/2
        if stdev is None:
            stdev = max(self.x)/100
        if stdev == 0:
            gaussian = np.zeros((len(self.x),)) + 1
            return gaussian

        gaussian = 1 - self.max_defect * np.exp(-((self.x - mean) / (stdev / self.h_ref) ) ** 2)
        return gaussian

    def Rectangular(self, x_min = None, x_max = None):
        if x_min is None:
            x_min = min(self.x)
        if x_max is None:
            x_max = max(self.x)
        rectangular = []
        for ii in self.x:
            if ii < x_min:
                rectangular.append(0)
            elif ii >= x_min and ii <= x_max:
                rectangular.append(self.max_defect)
            else:
                rectangular.append(0)
        rectangular = np.array(rectangular)
        rectangular = 1 - rectangular
                
        return rectangular