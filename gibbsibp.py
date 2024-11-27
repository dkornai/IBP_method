import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Bernoulli as Bern
from torch.distributions import Poisson as Pois
from torch.distributions import Categorical as Categorical

import numpy as np

class UncollapsedGibbsIBP(nn.Module):
    ################################################
    ########### UNCOLLAPSED GIBBS SAMPLER ##########
    ################################################
    ### Depends on a few self parameters but could##
    ### be made a standalone script if need be #####
    ###############################################
    def __init__(self, alpha, K, max_K, sigma_a, sigma_n, epsilon, lambd, phi):
        super(UncollapsedGibbsIBP, self).__init__()

        # idempotent - all are constant and have requires_grad=False
        self.alpha = torch.tensor(alpha)
        self.K = torch.tensor(K)
        self.max_K = torch.tensor(max_K)
        self.sigma_a = torch.tensor(sigma_a)
        self.sigma_n = torch.tensor(sigma_n)
        self.epsilon = torch.tensor(epsilon)
        self.lambd = torch.tensor(lambd)
        self.phi = torch.tensor(phi)

    def init_A(self,K,D):
        '''
        Sample from prior p(A_k)
        A_k ~ N(0,sigma_A^2 I)
        '''
        Ak_mean = torch.zeros(D)
        Ak_cov = self.sigma_a.pow(2)*torch.eye(D)
        p_Ak = MVN(Ak_mean, Ak_cov)
        A = torch.zeros(K,D)
        for k in range(K):
            A[k] = p_Ak.sample()
        return A

    def init_Y(self, n_filters, n_pixels):
        '''
        Sample from prior p(Y_kd)
        Y_kd ~ Bern(epsilon)
        '''
        Y = torch.zeros(n_filters,n_pixels)
        for k in range(n_filters):
            for d in range(n_pixels):
                p_Ykd = Bern(self.epsilon)
                Y[k,d] = p_Ykd.sample()
        
        return Y

    def left_order_form(self,Z):
        Z_numpy = Z.clone().numpy()
        twos = np.ones(Z_numpy.shape[0])*2.0
        twos[0] = 1.0
        powers = np.cumprod(twos)[::-1]
        values = np.dot(powers,Z_numpy)
        idx = values.argsort()[::-1]
        return torch.from_numpy(np.take(Z_numpy,idx,axis=1))

    def init_Z(self,N=20):
        '''
        Samples from the IBP prior that defines P(Z).

        First Customer i=1 takes the first Poisson(alpha/(i=1)) dishes
        Each next customer i>1 takes each previously sampled dish k
        independently with m_k/i where m_k is the number of people who
        have already sampled dish k. Z_ik=1 if the ith customer sampled
        the kth dish and 0 otherwise.
        '''
        Z = torch.zeros(N,self.K)
        K = int(self.K.item())
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

    def remove_allzeros_ZAY(self,Z,A,Y):
        """
        Remove columns (features) from Z that are not active, and also the corresponding rows from A and Y
        """
        to_keep = Z.sum(dim=0) > 0
        return Z[:, to_keep], A[to_keep, :], Y[to_keep, :]



    def F_loglik_given_ZA(self,F,Z,A):
        '''
        p(F|Z,A) = 1/([2*pi*sigma_n^2]^(ND/2)) * exp([-1/(2*sigma_n^2)] tr((F-ZA)^T(F-ZA)))
        '''
        N = F.size()[0]
        D = F.size()[1]
        pi = np.pi
        sig_n2 = self.sigma_n.pow(2)
        one = torch.tensor([1.0])
        log_first_term = one.log() - (N*D/2.)*(2*pi*sig_n2).log()
        log_second_term = ((-1./(2*sig_n2)) * \
            torch.trace((F-Z@A).transpose(0,1)@(F-Z@A)))
        log_likelihood = log_first_term + log_second_term

        return log_likelihood

    # def X_loglik_given_ZY(self,X,Z,Y):
    #     '''
    #     p(X|Z,Y) = prod_n prod_d p(x_nd|Z,Y)
    #       let e_n = Z_n,:@Y_:,n
    #     p(x_nd=1|Z,Y) = (1 - (1-lamb)^e_n) * (1-epsilon)
    #     p(x_nd=0|Z,Y) = (1-lamb)^e_n * (1-epsilon)
    #     '''
    #     N, D, K = X.shape[0], X.shape[1], Z.shape[1]

    #     lamb = self.lambd
    #     ep = self.epsilon

    #     # Initialize the likelihood variable
    #     log_likelihood = 0.0

    #     # Loop over each image (or do this in a batch-wise fashion)
    #     for i in range(N):
    #         # Compute the effective feature activations for the i-th image
    #         e_n = torch.matmul(Z[i, :], Y) # size (1, D)
            
    #         # Calculate the log-likelihood for the i-th image
    #         log_likelihood += torch.sum(
    #             X[i, :] * torch.log(1 - ((1 - lamb) ** e_n) * (1 - ep)) +
    #             (1 - X[i, :]) * torch.log((1 - lamb) ** e_n * (1 - ep))
    #         )

    #     return log_likelihood

    def X_loglik_given_ZY(self, X, Z, Y):
        '''
        Vectorized computation of p(X|Z,Y).
        
        p(X|Z,Y) = prod_n prod_d p(x_nd|Z,Y)
        let e_n = Z_n,:@Y_:,n
        p(x_nd=1|Z,Y) = (1 - (1-lamb)^e_n) * (1-epsilon)
        p(x_nd=0|Z,Y) = (1-lamb)^e_n * (1-epsilon)
        '''
        lamb = self.lambd
        ep = self.epsilon

        # Compute e_n for all samples at once: Z @ Y has shape (N, D)
        e = torch.matmul(Z, Y)

        # Compute probabilities in a vectorized manner
        p_x1 = (1 - ((1 - lamb) ** e) * (1 - ep))  # Probabilities when x_nd = 1
               #(1 - ((1 - lamb) **e_n)* (1 - ep))
        p_x0 = ((1 - lamb) ** e * (1 - ep))      # Probabilities when x_nd = 0
               #((1 - lamb) ** e_n* (1 - ep))

        # Avoid log(0) by clamping probabilities
        #p_x1 = torch.clamp(p_x1, min=1e-12)
        #p_x0 = torch.clamp(p_x0, min=1e-12)

        # Compute log-likelihood in a vectorized manner
        log_likelihood = torch.sum(
            X * torch.log(p_x1) + (1 - X) * torch.log(p_x0)
        )

        return log_likelihood



    def resample_Z_ik(self,Z,F,X,A,Y,i,k):
        '''
        m = number of observations not including Z_ik containing feature k

        Prior: p(z_ik=1) = m / (N-1)
        
        Posterior combines the prior with the likelihood:
        p(z_ik=1|Z_-nk,F,X,A,Y) propto p(z_ik=1)p(X|Z,Y)p(F|Z,A)
        
        Z_ik is a Bernoulli RV with this posterior probability
        '''
        N,D = X.size()
        Z_k = Z[:,k]
        
        m = Z_k.sum() - Z_k[i] # Called m_-nk in the paper

        # Store the current value of Z_ik
        Z_ik = Z[i,k]

        # If Z_nk were 0
        #Z_if_0 = Z.clone()
        Z[i,k] = 0
        
        log_prior_if_0 = (1 - (m/(N))).log()                #Prior Maybe should use N-1 instead of N?
        F_log_likelihood_if_0 = self.F_loglik_given_ZA(F,Z,A) # Likelihood of F
        X_log_likelihood_if_0 = self.X_loglik_given_ZY(X,Z,Y) # Likelihood of X

        log_score_if_0 = log_prior_if_0 + F_log_likelihood_if_0 + X_log_likelihood_if_0

        # If Z_nk were 1
        #Z_if_1 = Z.clone()
        Z[i,k] = 1
        
        log_prior_if_1 = (m/(N)).log()                      # Prior

        F_log_likelihood_if_1 = self.F_loglik_given_ZA(F,Z,A) # Likelihood of F  
        X_log_likelihood_if_1 = self.X_loglik_given_ZY(X,Z,Y) # Likelihood of X
      
        log_score_if_1 = log_prior_if_1 + F_log_likelihood_if_1 + X_log_likelihood_if_1

        # Exp, Normalize, Sample
        # log_scores = torch.cat((log_score_if_0,log_score_if_1),dim=0)
        # probs = self.renormalize_log_probs(log_scores)
        # p_znk = Bern(probs[1])
        p0, p1 = self.renormalize_log_prob_pair(log_score_if_0, log_score_if_1)
        p_znk = Bern(p1)
        
        # Change Z back to the original value
        Z[i,k] = Z_ik

        return p_znk.sample() # 0 or 1


    def renormalize_log_probs(self,log_probs):
        log_probs = log_probs - log_probs.max()
        likelihoods = log_probs.exp()
        return likelihoods / likelihoods.sum()

    def renormalize_log_prob_pair(self, log_prob_0, log_prob_1):
        """
        Normalize two log probabilities in a pair. This is useful when we have two log probabilities that are
        extremely small and would underflow if exponentiated directly.
        """
        max_p = torch.max(log_prob_0, log_prob_1)
        log_Z = max_p + torch.log(torch.exp(log_prob_0 - max_p) + torch.exp(log_prob_1 - max_p))
        
        return torch.exp(log_prob_0 - log_Z), torch.exp(log_prob_1 - log_Z)

    def F_loglik_given_k_new(self,cur_F_minus_ZA,Z,D,i,j):
        '''
        cur_F_minus_ZA is equal to F - ZA, using Z without the
        extra j columns that are appended to compute the likelihood
        for X|k_new=j. We have to pass this in because Z is changed
        in a loop that calls this function.

        Z: each time this function is called in the loop one level up,
        Z has one more column. Z is N x (K + k_new=j) dimensional.

        D: F.size()[1]

        i: A few levels up from this function, we are looping through every datapoint,
        and for each datapoint, considering how many new features k_new it draws. We
        are considering the i^th datapoint.

        j: We are calculating the likelihood for F|k_new = j
        '''
        N,K=Z.size()
        cur_F_minus_ZA_T = cur_F_minus_ZA.transpose(0,1)
        sig_n = self.sigma_n
        sig_a = self.sigma_a

        if j==0:
            ret = 0.0
        else:
            w = torch.ones(j,j) + (sig_n/sig_a).pow(2)*torch.eye(j)
            # alternative: torch.potrf(a).diag().prod()
            w_numpy = w.numpy()
            sign,log_det = np.linalg.slogdet(w_numpy)
            log_det = torch.tensor([log_det],dtype=torch.float32)
            # Note this is in log space
            first_term = j*D*(sig_n/sig_a).log() - ((D/2)*log_det)

            second_term = 0.5* \
                torch.trace( \
                cur_F_minus_ZA_T @ \
                Z[:,-j:] @ \
                w.inverse() @ \
                Z[:,-j:].transpose(0,1) @ \
                cur_F_minus_ZA) / \
                sig_n.pow(2)
            ret = first_term + second_term

        return ret
    
    def X_loglik_given_k_new(self, Z, Y, X, i, orig_k, k_new):
        """
        Calculate the log-likelihood of observing X[i, :] given Z, Y, and a proposed new feature count k_new.

        Parameters:
        - i: int, index of the image row in X
        - k_new: int, proposed number of new features
        - Z: Tensor, binary matrix of shape (N, K) for current feature ownership
        - Y: Tensor, binary matrix of shape (K, d) for feature-to-pixel activations
        - X: Tensor, binary matrix of shape (N, d) for observed images
        - lamb: float, efficacy parameter for feature activation
        - ep: float, spontaneous activation probability for each pixel
        - p: float, probability of a new feature turning on a pixel

        Returns:
        - log_likelihood: Tensor, the computed log-likelihood value for this k_new
        """

        lamb = self.lambd
        ep = self.epsilon
        p = self.phi

        # Compute effective feature activations for the i-th image
        e = torch.matmul(Z[i, 0:orig_k], Y[0:orig_k, :])
        
        # Indices of pixels that are "on" and "off" in X[i, :]
        one_inds = [t for t in range(X.shape[1]) if X[i, t] == 1]
        zero_inds = [t for t in range(X.shape[1]) if t not in one_inds]
        
        # Compute eta values for "on" and "off" pixels
        eta_one = (1 - lamb) ** e[one_inds]
        eta_zero = (1 - lamb) ** e[zero_inds]
        
        # Calculate likelihood components for pixels that are "on" and "off"
        lhood_XiT = torch.sum(torch.log(1 - (1 - ep) * eta_one * ((1 - lamb * p) ** k_new)))
        lhood_XiT += torch.sum(torch.log((1 - ep) * eta_zero * ((1 - lamb * p) ** k_new)))
        
        return lhood_XiT

    def sample_k_new(self,Z,F,X,A,Y,i,truncation=10):
        '''
        i: The loop calling this function is asking this function
        "how many new features (k_new) should data point i draw?"

        truncation: When computing the un-normalized posterior for k_new|X,Z,A, we cannot
        compute the posterior for the infinite amount of values k_new could take on. So instead
        we compute from 0 up to some high number, truncation, and then normalize. In practice,
        the posterior probability for k_new is so low that it underflows past truncation=20.
        '''

        N,K = Z.size()
        D = X.size()[1]

        # # Check if we are at the maximum number of features
        # if K == self.max_K:
        #     return 0

        p_k_new = Pois(torch.tensor([self.alpha/N]))
        cur_F_minus_ZA = F - Z@A
        
        prior_poisson_probs = torch.zeros(truncation)
        F_log_likelihood = torch.zeros(truncation)
        X_log_likelihood = torch.zeros(truncation)

        for j in range(truncation):
            # Compute the prior probability of k_new equaling j
            prior_poisson_probs[j] = p_k_new.log_prob(torch.tensor(j))

            # Compute the log likelihood of F with k_new equaling j
            F_log_likelihood[j] = self.F_loglik_given_k_new(cur_F_minus_ZA,Z,D,i,j)

            # Compute the log likelihood of X with k_new equaling j
            X_log_likelihood[j] = self.X_loglik_given_k_new(Z,Y,X,i,K,j)

            # Add new column to Z for next feature
            zeros = torch.zeros(N)
            Z = torch.cat((Z,torch.zeros(N,1)),1)
            Z[i][-1]=1

        # Compute log posterior of k_new and exp/normalize
        log_sample_probs = prior_poisson_probs + F_log_likelihood + X_log_likelihood
        sample_probs = self.renormalize_log_probs(log_sample_probs)

        # Important: we changed Z for calculating p(k_new| ...) so we must take off the extra rows
        Z = Z[:,:-truncation]
        assert Z.size()[1] == K
        posterior_k_new = Categorical(sample_probs)
        return posterior_k_new.sample()

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

        N = F.size()[0]
        K = A.size()[0]
        
        # Iterate over each data point
        for i in range(N):
            # Resample existing Z_ik
            for k in range(K):
                Z[i,k] = self.resample_Z_ik(Z,F,X,A,Y,i,k)
            
            k_new = 0
            current_k = A.size()[0]
            if current_k < self.max_K:
            # Decide how many new features to draw
                k_new = self.sample_k_new(Z,F,X,A,Y,i)

                # Limit such that current_k + k_new <= max_K
                k_new = np.clip(k_new, 0, self.max_K - current_k)

            # If new features are drawn, add them to Z, A, and Y
            if k_new > 0:
                # Add new columns to Z
                Z = torch.cat((Z,torch.zeros(N,k_new)),1)
                for j in range(k_new):
                    Z[i][-(j+1)] = 1

                # Add new rows to A, based on Z and A
                A_new = self.A_new(F,k_new,Z,A)
                A = torch.cat((A,A_new),dim=0)

                # Add new rows to Y, based on Z and Y
                Y_new = self.Y_new(k_new,Y.size()[1])
                Y = torch.cat((Y,Y_new),dim=0)
                # resample Y_new at the new features
                Y = self.resample_Y(Z, X, Y, start_idx=K)

        return Z, A, Y

    def resample_A(self,F,Z):
        '''
        mu = (Z^T Z + (sigma_n^2 / sigma_A^2) I )^{-1} Z^T  X
        Cov = sigma_n^2 (Z^T Z + (sigma_n^2/sigma_A^2) I)^{-1}
        p(A|X,Z) = N(mu,cov)
        '''
        N,D = F.size()
        K = Z.size()[1]
        ZT = Z.transpose(0,1)
        ZTZ = ZT@Z
        I = torch.eye(K)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        mu = (ZTZ + (sig_n/sig_a).pow(2)*I).inverse()@ZT@F
        cov = sig_n.pow(2)*(ZTZ + (sig_n/sig_a).pow(2)*I).inverse()
        A = torch.zeros(K,D)
        for d in range(D):
            p_A = MVN(mu[:,d],cov)
            A[:,d] = p_A.sample()
        return A

    def A_new(self,F,k_new,Z,A):
        '''
        p(A_new | X, Z_new, Z_old, A_old) propto
            p(X|Z_new,Z_old,A_old,A_new)p(A_new)
        ~ N(mu,cov)
            let ones = knew x knew matrix of ones
            let sig_n2 = sigma_n^2
            let sig_A2 = sigma_A^2
            mu =  (ones + sig_n2/sig_a2 I)^{-1} Z_new_T (X - Z_old A_old)
            cov = sig_n2 (ones + sig_n2/sig_A2 I)^{-1}
        '''
        N,D = F.size()
        K = Z.size()[1]
        assert K == A.size()[0]+k_new
        ones = torch.ones(k_new,k_new)
        I = torch.eye(k_new)
        sig_n = self.sigma_n
        sig_a = self.sigma_a
        Z_new = Z[:,-k_new:]
        Z_old = Z[:,:-k_new]
        Z_new_T = Z_new.transpose(0,1)
        # mu is k_new x D
        mu = (ones + (sig_n/sig_a).pow(2)*I).inverse() @ \
            Z_new_T @ (F - Z_old@A)
        # cov is k_new x k_new
        cov = sig_n.pow(2) * (ones + (sig_n/sig_a).pow(2)*I).inverse()
        A_new = torch.zeros(k_new,D)
        for d in range(D):
            p_A = MVN(mu[:,d],cov)
            A_new[:,d] = p_A.sample()
        return A_new

    def resample_Y(self, Z, X, Y, start_idx=0):
        """
        Sample the feature-to-pixel activation matrix Y given the current feature matrix Z and the observed images X.
        """
        K = Z.size()[1]
        N, T = X.size()
        ep = self.epsilon
        lamb = self.lambd
        p = self.phi

        pY_a0 = torch.zeros(K, T)
        pY_a1 = torch.zeros(K, T)
        
        prior_Y_a0 = torch.log(1 - p)
        prior_Y_a1 = torch.log(p)

        for t in range(T):
            for k in range(start_idx, K):
                for a in [0, 1]:
                    Y[k, t] = a

                    e = torch.matmul(Z, Y[:, t])
                    # Compute the total log-likelihood
                    log_likelihood = torch.sum(
                        (X[:,t])  *torch.log(1-((1-lamb)**e)*(1-ep)) + \
                        (1-X[:,t])*torch.log(   (1-lamb)**e)*(1-ep)
                        )

                    if a == 0:
                        temp_lpY_a0 = prior_Y_a0 + log_likelihood
                    else:
                        temp_lpY_a1 = prior_Y_a1 + log_likelihood

                # Normalise in log space to avoid underflows
                # max_p = torch.max(temp_lpY_a0, temp_lpY_a1)
                # log_Z = max_p + torch.log(torch.exp(temp_lpY_a0 - max_p) + torch.exp(temp_lpY_a1 - max_p))
                # pY_a0[k, t] = torch.exp(temp_lpY_a0 - log_Z)
                # pY_a1[k, t] = torch.exp(temp_lpY_a1 - log_Z)
                pY_a0[k, t], pY_a1[k, t] = self.renormalize_log_prob_pair(temp_lpY_a0, temp_lpY_a1)

                # Sample the element
                p_Ykt = Bern(pY_a1[k, t])
                Y[k, t] = p_Ykt.sample()

        return Y
                
                
    def Y_new(self, k_new, D):
        
        return torch.zeros(k_new, D) 


    def gibbs(self, F, X, iters):
        n_obs_X, n_pixels = X.size()
        n_obs_F, n_features = F.size()
        assert n_obs_X == n_obs_F, "Number of observations in X and F must match"
        n_obs = n_obs_X

        K = self.K

        Z = self.init_Z(n_obs)
        A = self.init_A(K, n_features)
        Y = self.init_Y(K, n_pixels)

        As = []
        Zs = []
        Ys = []

        for i in range(iters):
            print(f'iteration: {i}/{iters}', end='\r')
            # Gibbs resampling
            A       = self.resample_A(F,Z)
            Y       = self.resample_Y(Z,X,Y)
            Z, A, Y = self.resample_Z(Z,F,X,A,Y)

            # cleanup
            Z, A, Y = self.remove_allzeros_ZAY(Z,A,Y)

            # save the samples to the chain
            As.append(A.clone().numpy())
            Zs.append(Z.clone().numpy())
            Ys.append(Y.clone().numpy())

        return As,Zs,Ys
