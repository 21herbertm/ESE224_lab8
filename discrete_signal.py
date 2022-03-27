# MELANIE HERBERT AND ALINA HO
# ESE 224
# Lab 8

import numpy as np
import cmath
from scipy import signal
    
# QUESTION 1.1 INNER PRODUCTS AND ORTHOGONALITY
# PYTHON CLASS TAKES IN TWO-D SIGNALS X AND Y
# OUTPUTS INNER PRODUCT

class inner_product_2D():
    def __init__(self, x, y):
        self.x_signal=x
        self.y_signal=y
        self.N_dimensions=np.shape(x)[0]
        # NumPy arrays have an attribute called shape that returns
        # a tuple with each index having the number of corresponding elements.
        '''
        For example: 
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        print(arr.shape)
        
        prints out: (2, 4)
        '''
        # WE DEFINE EACH SIGNAL, X AND Y, AS A N BY N MATRIX OF COMPLEX NUMBERS

# DEFINE THE METHOD OF THE CLASS IN ORDER TO COMPUTE THE INNER PRODUCT
    def compute_inner_product(self):
        prod = 0        
        for i in range(self.N):
            for j in range(self.N):
                prod = prod + self.x[i,j] * np.conj(self.y[i,j])
            
        return prod
    
# QUESTION 1.2 DISCRETE COMPLEX EXPONENTIALS
# WRITE A PYTHON CLASS
# INPUTS: FREQUENCIES K AND L , N=SIGNAL DURATION
# RETURNS: 3 MATRICES (N BY N), (1) COMPLEX VALUES (2) REAL PARTS (3) IMAGINARY

class complex_exponential_2D():
    def __init__(self, k_frequency, l_frequency, N_frequency):

        self.k_frequency=k_frequency
        self.l_frequency=l_frequency
        self.N_frequency=N_frequency

    def compute_complex_exponential_2D(self):
        e_kl=np.zeros([self.N_frequency, self.N_frequency], dtype=np.complex)
        for i in range(self.N_frequency):
            for j in range(self.N_frequency):
                e_kl[i,j] = 1/np.sqrt(self.N_frequency*self.N_frequency)*np.exp(-1j*2*cmath.pi*(self.k_frequency*i/self.N_frequency+self.l_frequency*j/self.N_frequency))

        # RETURNS: 3 MATRICES (N BY N), (1) COMPLEX VALUES (2) REAL PARTS (3) IMAGINARY
        return e_kl, np.real(e_kl), np.imag(e_kl)
    
    
# QUESTION 1.3 UNIT ENERGY 2-D SQUARE PULSE
# WRITE PYTHON CLASS
# INPUT SIZE N
    # L = SIZE SQUARE PULSE
# OUTPUT
    # 2D SQUARE PULSE AS A N X N MATRIX
    # N^2 = TOTAL NUMBER OF SAMPLES
    # IN CREATION OF: PLOT 2-D SQUARE PULSE FOR N= 32 AND L =4 IN MAIN
class square_pulse_2D():

    def __init__(self, N_dimension, L_square_pulse_size):

        self.N_dimension=N_dimension
        self.L_square_pulse_size=L_square_pulse_size
        self.samples=N_dimension*N_dimension

    def compute_square_pulse_2D(self):

        sq_pulse=np.zeros([self.N_dimension, self.N_dimension], dtype=np.float)
        for i in range(self.L_square_pulse_size):
            for j in range(self.L_square_pulse_size):
                sq_pulse[i,j] = 1/self.L_square_pulse_size/self.L_square_pulse_size
            
        return sq_pulse, self.samples
    
# QUESTION 1.4 TWO-DIMENSIONAL GAUSSIAN SIGNALS
# PYTHON CLASS
# INPUT: N, MU, SIGMA
# OUTPUT: TWO GAUSSIAN PULSES


class compute_gaussian_2D():
    def __init__(self, N, mu, sigma):
        
        self.N=N
        self.mu=mu
        self.samples=N*N
        self.sigma=sigma

    def compute_gaussian_pulse_2D(self):

        gaussian_pulse=np.zeros([self.N, self.N], dtype=np.float)
        for i in range(self.N):
            for j in range(self.N):
                gaussian_pulse[i,j] = np.exp(-((i-self.mu)*(i-self.mu)+(j-self.mu)*(j-self.mu))/2/self.sigma/self.sigma)
            
        return gaussian_pulse, self.samples
    
#  QUESTION 1.5 DFT IN TWO DIMENSIONS
# MODIFY PYTHON CLASS FROM THE ONE DIMENSIONAL DFT FROM LAB 2
class DFT_2D():

    # COMPUTES 2-D DFT IN TERMS OF THE INNER PRODUCT

    def __init__(self, x_signal_time):
        # INPUT: X AS A SIGNAL FROM THE TIME DOMAIN
        self.x_signal_time=x_signal_time
        self.M=np.shape(x_signal_time)[0]
        self.N=np.shape(x_signal_time)[1]

    def raw_solve(self):

        X = np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):
                for i in range(self.M):
                    for j in range(self.N):
                        X[m,n] = X[m,n] + self.X[i,j]/np.sqrt(self.M*self.N)*np.exp(-1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
            
        return X
    
    
# QUESTION 1.6 iDFT in TWO DIMENSIONS
# PYTHON CLASS
# INPUTS: N X N- DIMENSIONAL SIGNAL
# OUTPUTS: 2-D iDFT

class compute_iDFT_2D():
    def __init__(self, X_signal):

        self.X_signal=X_signal
        self.M=np.shape(X_signal)[0]
        self.N=np.shape(X_signal)[1]

        # X_signal has N^2 coefficients

    def solve2(self):
        """
        \\\\\ METHOD: Compute the iDFT of X with N^2/2 coefficients
        """
        x=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):              
                for i in range(int(self.M/2)+1):
                    for j in range(self.N):
                        x[m,n] = x[m,n] + self.X_signal[i,j]/np.sqrt(self.M*self.N)*np.exp(1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
                        if i != 0:
                            x[m,n] = x[m,n] + np.conj(self.X_signal[i,j])/np.sqrt(self.M*self.N)*np.exp(1j*2*cmath.pi*(-m*i/self.M-n*j/self.N))

        return x
    


class Convolution_2D():
    """
    2-D convolution
    """
    def __init__(self, x, y):
        """
        input signal x
        filter y
        """
        self.x=x
        self.y=y

    def solve(self):
        """
        \\\\\ METHOD: Compute 2-D convolution
        """
        filtered_signal = signal.convolve2d(self.x, self.y, boundary='symm', mode='same')
            
        return filtered_signal
