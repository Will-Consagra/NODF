import scipy
import numpy as np  

"""
Code is adapted from software package released in support of: 
    `Learning from uncertain curves: The 2-Wasserstein metric for Gaussian processes`
"""

def fixed_point_map(Ki, Covs, weights):
    sqrtKi = scipy.linalg.sqrtm(Ki);
    p = Covs.shape[0]
    n = len(weights)
    Tmat = np.zeros((p,p))
    for i in range(n):
        Tmat = Tmat + weights[i]*np.real(scipy.linalg.sqrtm(sqrtKi @ Covs[:,:,i] @ sqrtKi));

    Tmat_2 = (Tmat@Tmat).astype(np.float64)
    T = np.real(np.linalg.solve(sqrtKi, np.linalg.solve(sqrtKi.conj().T, Tmat_2.conj().T).conj().T)).astype(np.float64)
    return T 

def W2dist(mu0, mu1, Cov0, Cov1):
    """
    Compute 2-Wasserstein distance between Gps
    """
    cov_dist = np.trace(Cov0) +  np.trace(Cov1) - 2*np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(Cov0)@Cov1@scipy.linalg.sqrtm(Cov0)))
    mu_dist = np.linalg.norm(mu0 - mu1, ord=2)**2
    return np.real(np.sqrt(mu_dist + cov_dist))

def Wbarycenter(gps, weights, maxiter = 100, tol=1e-8):
    """
        This function computes the Barycenter of the set of posterior predictive GPs at location vector
        gps = list of (mus, covs)
        weights: (n,)
    """

    n = len(gps)
    p = len(gps[0][0])
    mus = np.zeros((p,n))
    Covs =  np.zeros((p,p,n))
    for e in range(n):
        m_e, K_e = gps[e];
        mus[:,e] = m_e
        Covs[:,:,e] = K_e;
    
    Ki = Covs[:,:,0]
    Kf = fixed_point_map(Ki, Covs, weights)
    iteration = 0
    while (W2dist(np.zeros(p), np.zeros(p), Ki, Kf) > tol) and (iteration < maxiter):
        Ki = Kf 
        Kf = fixed_point_map(Ki, Covs, weights)
        iteration += 1
    if iteration==maxiter:
        raise Exception("Barycenter did not converge.")

    mu_center = np.sum(np.multiply(mus, weights),1).reshape(-1,1)
    Cov_center =  Kf
    return mu_center, Cov_center

