import numpy as np
from scipy.special import loggamma # gamma function

class LogLikelihood:

    def __init__(self,retrieval_object,scale_flux=True,scale_err=True):

        self.d_flux = retrieval_object.data_flux
        self.d_mask = retrieval_object.mask_isfinite
        self.scale_flux   = scale_flux
        self.scale_err    = scale_err
        self.N_d      = self.d_mask.sum() # number of degrees of freedom / valid datapoints
        self.N_params = retrieval_object.n_params
        self.alpha =2 # from Ruffio+2019
        self.N_phi=1 # number of scaling parameters
        
    def __call__(self, m_flux, Cov, **kwargs):

        self.ln_L   = 0.0
        self.chi2_0 = 0.0
        self.m_flux_phi = np.nan * np.ones_like(self.d_flux) # scaled model flux

        mask = self.d_mask # mask out nans
        N_d = mask.sum() # Number of (valid) data points
        d_flux = self.d_flux[mask] # data flux
        m_flux = m_flux[mask] # model flux
        
        if self.scale_flux: # Find the optimal phi-vector to match the observed spectrum
            self.m_flux_phi[mask],self.phi=self.get_flux_scaling(d_flux, m_flux, Cov)

        residuals_phi = (self.d_flux - self.m_flux_phi) # Residuals wrt scaled model
        inv_cov_0_residuals_phi = Cov.solve(residuals_phi[mask]) 
        chi2_0 = np.dot(residuals_phi[mask].T, inv_cov_0_residuals_phi) # Chi-squared for the optimal linear scaling
        logdet_MT_inv_cov_0_M = 0

        if self.scale_flux:
            inv_cov_0_M    = Cov.solve(m_flux) # Covariance matrix of phi
            MT_inv_cov_0_M = np.dot(m_flux.T, inv_cov_0_M)
            logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # (log)-determinant of the phi-covariance matrix

        if self.scale_err: 
            self.s2 = self.get_err_scaling(chi2_0, N_d) # Scale variance to maximize log-likelihood
        logdet_cov_0 = Cov.get_logdet()  # Get log of determinant (log prevents over/under-flow)

        # from Ruffio+2019
        self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi)+loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

        # Add this order/detector to the total log-likelihood
        self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
        self.chi2_0 += chi2_0

        # Reduced chi-squared
        self.chi2_0_red = self.chi2_0 / self.N_d

        return self.ln_L

    def get_flux_scaling(self, d_flux, m_flux, cov): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux.T, cov.solve(m_flux)) # Left-hand side
        rhs = np.dot(m_flux.T, cov.solve(d_flux)) # Right-hand side
        phi = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux, phi), phi # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_scaled, N):
        s2 = np.sqrt(1/N * chi_squared_scaled)
        return s2 # uncertainty scaling that maximizes log-likelihood