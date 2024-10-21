import numpy as np
    
class Covariance:
     
    def __init__(self, err, beta=None, **kwargs): 
        self.err = err
        self.cov_reset() # set up covariance matrix
        self.cov_cholesky = None # initialize
        self.beta=beta # uncertainty scaling factor 'beta' should be added to kwargs

    def __call__(self,params,**kwargs):
        self.cov_reset() # reset covariance matrix

    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2) # = True if is matrix

    def add_data_err_scaling(self, beta): # Scale uncertainty with factor beta
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2

    def add_model_err(self, model_err): # Add a model uncertainty term
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)

    def get_logdet(self): # log of determinant
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet
 
    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal
    
    def get_dense_cov(self): 
        if self.is_matrix:
            return self.cov
        return np.diag(self.cov) # get the errors from the diagonal