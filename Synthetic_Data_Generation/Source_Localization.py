import numpy as np

def generate_hypergraph_diffusion(sc, n_samples, n_sources, source_upper, timesteps):

    # get the number of nodes
    n = sc.pts.shape[0]

    # compute the gso for the hypergraph (weighted by involvement in multiple hyperedges)
    gso = np.zeros((n,n))
    for he in sc.simplices:
        for ind in range(len(he) - 1):
            for jnd in range(ind+1, len(he)):
                gso[he[ind],he[jnd]] += 1
                gso[he[jnd],he[ind]] += 1
    
    # normalize gso
    # obtain eigenvalues
    eigenvalues, _ = np.linalg.eig(gso) 

    # normalize by eigenvalue with largest absolute value
    gso = gso / np.max(np.abs(eigenvalues))

    # initialize the tensor used to store the samples
    # shape is n_samples x n x time x 1 features
    z = np.zeros((n_samples, n, timesteps, 1))

    for i in range(n_samples):

        # pick n_sources at random from n nodes
        sources = np.random.choice(n, n_sources, replace=False)

        # define z_0 for each sample
        z[i, sources, 0, 0] = np.random.uniform(0,10, n_sources)

    # noise mean and variance
    mu = np.zeros(n)
    sigma = np.eye(n) * 1e-3

    for t in range(timesteps - 1):

        # generate noise
        noise = np.random.multivariate_normal(mu, sigma, n_samples)

        # generate z_t
        z[:, :, t + 1] = gso @ z[:, :, t] + np.expand_dims(noise, -1)
        
    # transpose dimensions so shape is n_samples x time x n x 1 feature
    z = z.transpose((0, 2, 1, 3))
    
    # squeeze feature dimension, as there is only 1 feature
    return z.squeeze()