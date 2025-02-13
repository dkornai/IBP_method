import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern
from torch.distributions import Poisson as Pois
from torch.distributions import Categorical as Categorical

import numpy as np

class UncollapsedGibbsIBP(nn.Module):
    def __init__(self, K, max_K, alpha, sigma2_a, phi, sigma2_n, epsilon, lambd):
        super(UncollapsedGibbsIBP, self).__init__()

        # idempotent - all are constant and have requires_grad=False
        self.K = K                              # current number of context features
        self.max_K = max_K                      # maximum number of context features
        self.alpha = torch.tensor(alpha)        # propensity to add new context features
        self.sigma2_a = torch.tensor(sigma2_a)  # variance of the force prior
        self.sigma2_n = torch.tensor(sigma2_n)  # variance of the observation noise
        self.epsilon = torch.tensor(epsilon)    # pixel noise (probability of spontaneous activation)
        self.lambd = torch.tensor(lambd)        # pixel noise (probability of successful activation, per active feature)
        self.phi = torch.tensor(phi)            # expected number of pixels activated by a new feature

    def gibbs(self, F, X, iters):
        """
        Main function to run the Gibbs sampler for the IBP model.
        """
        n_obs_F, n_features = F.size()
        n_obs_X, n_pixels   = X.size()
        assert n_obs_X == n_obs_F, "Number of observations in X and F must match"
        
        self.N:int = n_obs_X    # Number of data points
        self.D:int = n_features # Force dimension
        self.T:int = n_pixels   # Pixel dimension
        
        # Initalize Z, A, and Y from the priors
        Z = self.sample_Z_prior(self.N, self.K)
        A = self.sample_A_prior(self.K, self.D)
        Y = self.sample_Y_prior(self.K, self.T)

        As = []
        Zs = []
        Ys = []

        # Gibbs resampling
        for i in range(iters):
            print(f'iteration: {i}/{iters}', end='\r')
            
            A       = self.resample_A(F,Z)
            Y       = self.resample_Y(Z,X,Y)
            Z, A, Y = self.resample_Z(Z,F,X,A,Y)

            Z, A, Y = self.remove_allzeros_ZAY(Z,A,Y)

            As.append(A.clone().numpy())
            Zs.append(Z.clone().numpy())
            Ys.append(Y.clone().numpy())

        return As,Zs,Ys
    

    #### PRIORS ####
    def sample_A_prior(self,K,D):
        '''
        Initialise the force basis matrix A (K x D) with a sample sample from prior p(A_k)
        A_k ~ N(0, sigma^2_A @ I)        
        '''
        Ak_mean = torch.zeros(D)
        Ak_cov  = self.sigma2_a*torch.eye(D)
        p_Ak = MVN(Ak_mean, Ak_cov)
        
        A = torch.zeros(K,D)
        for k in range(K):
            A[k] = p_Ak.sample()
        
        return A

    def sample_Y_prior(self, K, T):
        '''
        Initialise the feature image matrix Y (K x T) with a sample from prior p(Y)
        Y_kd ~ Bern(phi)
        '''
        p_Ykd = Bern(self.phi)

        Y = torch.zeros(K,T)
        for k in range(K):
            for d in range(T):
                Y[k,d] = p_Ykd.sample()
        
        return Y

    def sample_Z_prior(self, N, K):
        '''
        Samples from the IBP prior that defines P(Z).

        First Customer i=1 takes the first Poisson(alpha/(i=1)) dishes
        Each next customer i>1 takes each previously sampled dish k
        independently with m_k/i where m_k is the number of people who
        have already sampled dish k. Z_ik=1 if the ith customer sampled
        the kth dish and 0 otherwise.
        '''
        Z = torch.zeros(N, K)
        total_dishes_sampled = 0
        
        for i in range(N):
            
            selected = torch.rand(total_dishes_sampled) < Z[:,:total_dishes_sampled].sum(dim=0) / (i+1.)
            
            Z[i][:total_dishes_sampled][selected]=1.0
            
            p_new_dishes = Pois(torch.tensor([self.alpha/(i+1)]))
            
            new_dishes = int(p_new_dishes.sample().item())
            
            if total_dishes_sampled + new_dishes >= K:
                new_dishes = K - total_dishes_sampled
            
            Z[i][total_dishes_sampled:total_dishes_sampled+new_dishes]=1.0
            
            total_dishes_sampled += new_dishes
        
        return self.left_order_form(Z)
    
    def left_order_form(self,Z):
        Z_numpy = Z.clone().numpy()
        twos = np.ones(Z_numpy.shape[0])*2.0
        twos[0] = 1.0
        powers = np.cumprod(twos)[::-1]
        values = np.dot(powers,Z_numpy)
        idx = values.argsort()[::-1]

        return torch.from_numpy(np.take(Z_numpy,idx,axis=1))



    #### RESAMPLE A FOR EXISTING FEATURES ####
    def resample_A(self, F, Z):
        '''
        Resample the force basis matrix A given the current feature matrix Z and the observed forces F.
        p(A|X,Z) = N(mu,cov) [implemented in `posterior_param_A`]
          let mu = (Z^T @ Z + (sigma_n^2/sigma_A^2)I)^{-1} @ Z^T @ F
          let Cov = sigma_n^2 (Z^T @ Z + (sigma_n^2/sigma_A^2)I)^{-1}
        '''
        mu, cov = self.posterior_param_A(F, Z)
        
        # Perform sampling
        A = torch.zeros(self.K,self.D)
        for k in range(self.K):
            p_A = MVN(mu[k,:],cov)
            A[k,:] = p_A.sample()
        
        return A
    
    def posterior_param_A(self, F, Z):
        """
        Calculate the posterior parameters (mean, covariance) for A given Z and F.

        p(A|X,Z) = N(mu,cov)
          let mu = (Z^T @ Z + (sigma_n^2/sigma_A^2)I)^{-1} @ Z^T @ F
          let Cov = sigma_n^2 (Z^T @ Z + (sigma_n^2/sigma_A^2)I)^{-1}
        """
        # (Z^T Z + (sigma_n^2/sigma_A^2) I)^{-1} is repeated, so compute it once
        temp = ((Z.T @ Z) + ((self.sigma2_n/self.sigma2_a)*torch.eye(self.K))).inverse()
        # Calculate mean and covariance
        mu  = temp @ Z.T @ F
        cov = self.sigma2_n * temp

        return mu, cov
    
    #### RESAMPLE Y FOR EXISTING FEATURES ####
    def resample_Y(self, Z, X, Y, start_idx=0):
        """
        Sample the feature-to-pixel activation matrix Y given the current feature matrix Z and the observed images X.
        
        P(Y_kt=a|Z, X, Y_-kt) [implemented in `posterior_param__Y_kt`]
        propto P(Y_kt=a) * P(X|Z, Y)|Y_kt=a 
        """
        # Iterate through the requested regions of Y
        for k in range(start_idx, self.K):
            for t in range(self.T):
                # Calculate the posterior probability of Y_kt = 1
                pY_kt_1 = self.posterior_param_Y_kt(Z, X, Y, k, t)
                # Sample the element back into Y from a Bernoulli distribution
                Y[k,t] = Bern(pY_kt_1).sample()

        return Y

    def posterior_param_Y_kt(self, Z, X, Y, k, t):
        """
        Calculate the posterior parameter for Y_kt, which is the probability P(Y_kt=1) given Z, X, and Y_-kt.
        
        P(Y_kt=a|Z, X, Y_-kt) 
        propto P(Y_kt=a) * P(X|Z, Y)|Y_kt=a
        P(Y_kt=a) = phi^a * (1-phi)^(1-a)
        P(X|Z, Y) implemented in `loglik_x__t_given_Zy`
        """
        # Calculate log posterior proportionals for Y_kt = 0 and Y_kt = 1 
        Y[k, t] = 0
        logp_Ykt_0 = torch.log(1 - self.phi)\
            + self.loglik_x__t_given_Zy(X[:, t:t+1], Z, Y[:, t:t+1])
        Y[k, t] = 1
        logp_Ykt_1 = torch.log(self.phi)    \
            + self.loglik_x__t_given_Zy(X[:, t:t+1], Z, Y[:, t:t+1])
        
        # Normalise to get the probabilities
        pYkt_0, pYkt_1 = normalise_bern_logpostprop(logp_Ykt_0, logp_Ykt_1)

        return pYkt_1


    def loglik_x__t_given_Zy(self, x__t, Z, y__t):
        """
        Calculate the log likelihood of x_:,t given Z and y_:,t
        
          let n_actfeat = Z_i,: @ y_:,t
        P(x_:,t|Z, y_:,t) = prod_i p(x_it|Z_i, y_it) 
        = prod_i [1 - ((1-lambda)^n_actfeat) * (1-epsilon)]^(x_it) * [((1-lambda)^n_actfeat) * (1-epsilon)]^(1-x_it)
        """
        # Calculate the number of active features
        n_actfeat = torch.matmul(Z, y__t)
        # Calculate the probability of x_it = 1
        p_x_1 = 1 - ((((1 - self.lambd)**n_actfeat))*(1 - self.epsilon))
        # Calculate the log likelihood
        loglik = torch.sum(x__t * torch.log(p_x_1) + (1 - x__t) * torch.log(1 - p_x_1))

        return loglik

    

    #### RESAMPLE Z ####
    def resample_Z(self,Z,F,X,A,Y):
        '''
        - Re-samples existing Z_ik by using p(Z_ik=1|Z_-ik,A,X)
        - Samples the number of new dishes that customer i takes
          corresponding to:
            - prior: p(k_new) propto Pois(alpha/N)
            - likelihood: p(X|Z_old,A_old,k_new)
            - posterior: p(k_new|X,Z_old,A_old)
        - Adds the columns to Z corresponding to the new dishes,
          setting those columns to 1 for customer i
        - Adds rows to A corresponds to the new dishes.
          - p(A_new|X,Z_new,Z_old,A_old) propto p(X|Z_new,Z_old,A_old,A_new)p(A_new)
        '''
        
        # Iterate over each data point
        for i in range(self.N):
            # Start collecting Z_ik to set to 0
            marked_Z_ik = torch.ones(self.K)

            # Iterate over each feature
            for k in range(self.K):
                # Count the number of objects that have feature k without the current object (Called m_-nk in the paper)
                m = Z[:,k].sum() - Z[i,k] 
                if m > 0:
                    # Sample Z_ik
                    pZ_ik_1 = self.posterior_param_Z_ik(F, X, A, Y, Z, i, k, m)
                    Z[i,k] = Bern(pZ_ik_1).sample()
                else:
                    # Mark Z_ik to be set to 0
                    marked_Z_ik[k] = 0
            
            # Zero marked Z_iks
            Z[i] *= marked_Z_ik
            
            # Sample the number of new features k_new
            k_new = self.sample_k_new_clipped(Z,F,X,A,Y,i)

            # If new features are drawn, add them to Z, A, and Y
            if k_new > 0:
                # Get new columns for Z
                Z_new = self.Z_new(k_new, self.N, i)
                # Get new rows for A
                A_new = self.A_new(F,Z_new,Z,A)
                # Add new columns to Z
                Z = torch.cat((Z,Z_new),dim=1)
                # Add new rows to A
                A = torch.cat((A,A_new),dim=0)
                # Get new rows for Y
                Y_new = self.Y_new(k_new, self.T)
                # Add new rows to Y
                Y = torch.cat((Y,Y_new),dim=0)
                # update K
                self.K += k_new
                # resample Y at the rows of the new features
                Y = self.resample_Y(Z, X, Y, start_idx=self.K-k_new)

        return Z, A, Y


    #### RESAMPLE Z FOR EXISTING FEATURES ####
    def posterior_param_Z_ik(self,F, X, A, Y, Z, i, k, m):
        '''
        Calculate the posterior parameter for P(Z_ik=1) given Z_-ik, F, X, A, Y, and i
        
        P(z_i,k=a|f_i,:,x_i,:,A,Y,Z_-ik)
        propto P(z_i,k=a) * P(x_i,:|z_i,:,Y) * p(f_i,:|z_i,:,A)
        P(z_i,k=a) = (m_(-nk)/N)^a * (1 - m_(-nk)/N)^(1-a)
          let m_(-nk) = sum_n Z_n,k - Z_i,k
        P(x_i,:|z_i,:,Y) implemented in `loglik_x_i__given_Yz`
        P(f_i,:|z_i,:,A) implemented in `loglik_f_i__given_Az`
        '''
        
        # Calculate priors for Z_ik = 0 and Z_ik = 1
        log_prior_Z_ik_0 = (1 - (m/(self.N))).log()
        log_prior_Z_ik_1 = (m/(self.N)).log()

        # Calculate log posterior proportionals for Z_ik = 0 and Z_ik = 1
        Z[i, k] = 0
        logp_Z_ik_0 = log_prior_Z_ik_0 \
            + self.loglik_x_i__given_Yz(X[i:i+1, :], Y, Z[i:i+1, :]) \
            + self.loglik_f_i__given_Az(F[i:i+1, :], A, Z[i:i+1, :])
        
        Z[i, k] = 1
        logp_Z_ik_1 = log_prior_Z_ik_1 \
            + self.loglik_x_i__given_Yz(X[i:i+1, :], Y, Z[i:i+1, :]) \
            + self.loglik_f_i__given_Az(F[i:i+1, :], A, Z[i:i+1, :])
        
        # Normalise to get the probabilities
        p_Z_ik_0, p_Z_ik_1 = normalise_bern_logpostprop(logp_Z_ik_0, logp_Z_ik_1)

        return p_Z_ik_1

    def loglik_x_i__given_Yz(self, x_i_, Y, z_i_):
        """
        Calculate the log likelihood of x_i,: given Y and z_i,:
        
          let n_actfeat = z_i,: @ y_:,t
        P(x_i,:| Y, z_i,:) 
        = prod_t p(x_it| z_i,: , y_i,:) 
        = prod_t [1 - ((1-lambda)^n_actfeat) * (1-epsilon)]^(x_it) * [((1-lambda)^n_actfeat) * (1-epsilon)]^(1-x_it)
        """
        # Calculate the number of active features
        n_actfeat = torch.matmul(z_i_, Y)
        # Calculate the probability of x_it = 1
        p_x_1 = 1 - ((1 - self.lambd) ** n_actfeat) * (1 - self.epsilon)
        # Calculate the log likelihood
        loglik = torch.sum(x_i_ * torch.log(p_x_1) + (1 - x_i_) * torch.log(1 - p_x_1))

        return loglik

    def loglik_f_i__given_Az(self, f_i_, A, z_i_):
        '''
        Calculate the log likelihood of f_i,: given z_i,: and A
          let n_actfeat = z_i,: @ y_:,t
        p(f_i,:|z_i,:,A) 
        = 1/([2*pi*sigma^2_n]^(D/2)) * exp(-1/(2 * sigma^2_n) * (f_i,:- z_i,: @ A)^T @ (f_i,:- z_i,: @ A))
        '''
        # Calculate log of 1/([2*pi*sigma_n^2]^(D/2))
        log_first_term = torch.tensor([1.0]).log() - (self.D/2.)*(2*np.pi*self.sigma2_n).log()
        # Calculate -1/2 sigma^2_n * (f_i,:- z_i,: @ A)^T @ (f_i,:- z_i,: @ A)
        log_second_term = (-(1/(2*self.sigma2_n)) * (f_i_-z_i_@A)@(f_i_-z_i_@A).T) # torch transpose reversed
        
        log_likelihood = log_first_term + log_second_term[0]

        return log_likelihood


    #### SAMPLE K_NEW ####
    def sample_k_new(self,Z,F,X,A,Y,i,truncation=10):
        '''
        Sample the number of new features k_new for the i-th data point.
        P(k_new|x_i,:,f_i,:,z_i,:,A,Y) 
        = P(k_new) * P(f_i,:|z_i,:,A,k_new) * P(x_i,:|z_i,:,Y,k_new)
        '''

        # Compute the prior over k_new
        p_k_new = Pois(torch.tensor([self.alpha/self.N]))        
        prior_poisson_probs = torch.zeros(truncation)

        # Compute the log likelihood of F with k_new equaling j
        F_log_likelihood    = torch.zeros(truncation)
        X_log_likelihood    = torch.zeros(truncation)

        for j in range(truncation):
            # Compute the prior probability of k_new equaling j
            prior_poisson_probs[j]  = p_k_new.log_prob(torch.tensor(j))
            # Compute the log likelihood of F with k_new equaling j
            F_log_likelihood[j]     = self.loglik_f_i__given_Az_Knew(F[i:i+1, :], A ,Z[i:i+1, :], j)
            # Compute the log likelihood of X with k_new equaling j
            X_log_likelihood[j]     = self.loglik_x_i__given_Yz_Knew(X[i:i+1, :], Y, Z[i:i+1, :], j)

        # Compute log posterior of k_new and exp/normalize
        log_sample_probs = prior_poisson_probs + F_log_likelihood + X_log_likelihood
        sample_probs = renormalize_log_probs(log_sample_probs)

        # Sample k_new from the posterior
        return Categorical(sample_probs).sample()
    
    def loglik_x_i__given_Yz_Knew(self, x_i_, Y, z_i_, k_new):
        """
        Calculate the log likelihood of x_i,: given Y and z_i,: while marginalizing over the new features k_new.
        
          let n_actfeat = z_i,: @ y_:,t
        P(x_i,:| Y, z_i,:) 
        = prod_t [1 - (1-epsilon) * ((1-lambda)^n_actfeat) * (1-lambda*phi)^K_new]^(x_it) * [(1-epsilon) * ((1-lambda)^n_actfeat) * (1-lambda*phi)^K_new]^(1-x_it)
        """
        # Calculate the number of active features
        n_actfeat = torch.matmul(z_i_, Y)
        # Calculate the probability of x_it = 1
        p_x_1 = 1 - (1 - self.epsilon) * ((1 - self.lambd) ** n_actfeat) * ((1 - self.lambd*self.phi) ** k_new)
        # Calculate the log likelihood
        loglik = torch.sum(x_i_ * torch.log(p_x_1) + (1 - x_i_) * torch.log(1 - p_x_1))

        return loglik
    
    def loglik_f_i__given_Az_Knew(self, f_i_, A, z_i_, k_new):
        '''
        Calculate the log likelihood of f_i,: given z_i,: and A and K_new
          let mu = z_i,: @ A
          let cov = sigma^2_n + k_new * sigma^2_a @ I
          let det(cov) = sigma^2_n + k_new * sigma^2_a 
        p(f_i,:|z_i,:,A) 
        = 1/[2 pi det(cov)]^{D/2} * exp(-1/(2 det(cov)) * (f_i,:- z_i,: @ A)^T (f_i,:- z_i,: @ A))
        '''

        # Calculate log of 1/([2*pi*sigma_n^2]^(D/2))
        log_first_term = torch.tensor([1.0]).log() - (self.D/2.)*(2*np.pi*(self.sigma2_n + k_new*self.sigma2_a)).log()
        # Calculate [-1/2 sigm * (f_i,:- z_i,: @ A)^T @ cov^{-1} @ (f_i,:- z_i,: @ A)]
        log_second_term = (-(1/(2*(self.sigma2_n + k_new*self.sigma2_a))) * (f_i_-z_i_@A) @ (f_i_-z_i_@A).T) # torch transpose reversed
        
        log_likelihood = log_first_term + log_second_term[0]

        return log_likelihood
    
    def sample_k_new_clipped(self,Z,F,X,A,Y,i,truncation=10):
        """
        Wrapper function for sample_k_new that clips the output to be between 0 and max_K - current_K.
        also avoids executing if K is already at max_K
        """
        # If K is not at max_K, sample k_new
        if self.K < self.max_K:
            # Decide how many new features to draw
            k_new = self.sample_k_new(Z,F,X,A,Y,i,truncation)

            # Limit such that current_k + k_new <= max_K
            k_new = np.clip(k_new, 0, self.max_K - self.K)
        else:
            k_new = 0

        return k_new


    #### NEW MATRIX SECTIONS ####
    def A_new(self,F,Z_new,Z,A):
        '''
        p(A_new | F, Z_new, Z, A) 
        propto p(F | Z_new, Z, A, A_new) * p(A_new)
        ~ N(mu,cov)
            let ones = knew x knew matrix of ones
            mu  = (ones + sig_n2/sig_a2 I)^{-1} @ Z_new_T @ (F - Z_old A_old)
            cov = sigma^2_n (ones + sig_n2/sig_A2 @ I)^{-1}
        '''
        k_new = Z_new.size()[1]
        # (ones + sig_n2/sig_a2 I)^{-1} is repeated, so compute it once
        temp = (torch.ones(k_new,k_new) + ((self.sigma2_n/self.sigma2_a) * torch.eye(k_new))).inverse()
        # Compute mean and covariance
        mu = temp @ Z_new.T @ (F - (Z@A))
        cov = self.sigma2_n * temp
        
        # Perform sampling
        A_new = torch.zeros(k_new,self.D)
        for d in range(self.D):
            p_A = MVN(mu[:,d],cov)
            A_new[:,d] = p_A.sample()
        
        return A_new
                    
    def Y_new(self, k_new, T):
        """
        When initialised, the new features are not active in any of the images.
        """
        return torch.zeros(k_new, T)
    
    def Z_new(self, k_new, N, i):
        """
        When initialised, the new features are only active in the i-th image.
        """
        Z_new = torch.zeros(N, k_new)
        Z_new[i,:] = 1.0

        return Z_new
    

    #### CLEANUP STEP ####
    def remove_allzeros_ZAY(self,Z,A,Y):
        """
        Remove columns (features) from Z that are not active, and also the corresponding rows from A and Y
        """
        to_keep = Z.sum(dim=0) > 0
        self.K = to_keep.sum()
        return Z[:, to_keep], A[to_keep, :], Y[to_keep, :]
    

    #### LIKELHOODS FOR FORWARD DATA GENERATION ####
    def F_likelihood_sample(self,Z,A):
        """
        Sample forces, given the feature matrix Z and the force basis matrix A.
        sample ~ p(F|Z,A) = N(ZA,sigma_n2 I)
        """
        # Calculate the mean of the Gaussian
        mu = Z @ A 
        N = mu.size()[0]
        D = mu.size()[1]

        # Set the covariance matrix
        cov = self.sigma2_n * torch.eye(D)

        # Perform sampling
        sample = torch.zeros_like(mu)
        for n in range(N):
            p_F = MVN(mu[n,:],cov)
            sample[n,:] = p_F.sample()
        
        return sample
    
    def X_likelihood_sample(self,Z,Y):
        """
        Sample images, given the feature matrix Z and the pixel activation matrix Y.
        sample ~ p(X|Z,Y) = Bernoulli(p)
          let p = 1 - (1-lambda)^(Z @ Y)*(1-epsilon)
        """
        # Calculate the number of features with active pixels at each location in each image
        n_actfeat = torch.matmul(Z, Y)
        
        # Calculate the probability of each pixel being active
        p_x_1 = 1 - (((1 - self.lambd) ** n_actfeat) * (1 - self.epsilon))

        # Perform sampling
        sample = torch.zeros_like(p_x_1)
        for i in range(p_x_1.size()[0]):
            for j in range(p_x_1.size()[1]):
                p_X = Bern(p_x_1[i,j])
                sample[i,j] = p_X.sample()

        return sample


        


def normalise_bern_logpostprop(log_postprop_0, log_postprop_1):
    """
    Get the normalised probabilities from two log posterior proportionals.
        let maxp = max(log_postprop_0, log_postprop_1)
        let minp = min(log_postprop_0, log_postprop_1)
        let logZ = maxp + log(1 + exp(minp - maxp))
      p0 = exp(log_postprop_0 - logZ)
      p1 = exp(log_postprop_1 - logZ)
    """
    maxp = torch.max(log_postprop_0, log_postprop_1)
    minp = torch.min(log_postprop_0, log_postprop_1)
    logZ = maxp + torch.log(1 + torch.exp(minp - maxp))

    p0 = torch.exp(log_postprop_0 - logZ)
    p1 = torch.exp(log_postprop_1 - logZ)

    return p0, p1

def renormalize_log_probs(log_probs):
    log_probs = log_probs - log_probs.max()
    likelihoods = log_probs.exp()
    return likelihoods / likelihoods.sum()

# Import cProfile
# inf = UncollapsedGibbsIBP(alpha=0.05, K=1, max_K=4, sigma_a=0.2, sigma_n=0.1, epsilon=0.05, lambd=0.99, phi=0.25)
# with cProfile.Profile() as pr:
#     As, Zs, Ys = inf.gibbs(F_dataset, X_dataset, 5)

# pr.print_stats(sort='cumtime')