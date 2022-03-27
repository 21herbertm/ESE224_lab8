# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Lab 8

###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# Local files
import discrete_signal

###############################################################################
############################ CLASSES ##########################################
###############################################################################  
    
    
def question_1_3(N_dimension, L_square_pulse_size):

    #Question 1.3
    
    sq_2D = discrete_signal.square_pulse_2D(N_dimension, L_square_pulse_size)
    square_pulse_2D, num_samples = sq_2D.compute_square_pulse_2D()
    # compute_square_pulse_2D METHOD DEFINED IN DISCRETE SIGNAL CLASS
    
    plt.imshow(square_pulse_2D, cmap='gray')
    #PLT.IMSHOW FUNCTION FROM MATPLOTLIB LIBRARY IN ORDER TO DISPLAY DATA AS AN IMAGE
    plt.colorbar()
    #INDICATES COLOR SCALE
    plt.show()
    # DISPLAYS PLOT FOR SQUARE PULSE


def question_1_4(N, mu, sigma):

    #QUESTION 1.4 TWO-DIMENSIONAL GAUSSIAN SIGNALS

    signal_2D = discrete_signal.compute_gaussian_2D(N, mu, sigma)
    gaussian_signal_2D, num_samples = signal_2D.compute_gaussian_pulse_2D()
    
    plt.imshow(gaussian_signal_2D, cmap='gray')
    plt.colorbar()
    plt.show()   
    
    # PLOT THE TWO-DIMENSIOANL GAUSSIAN
    fig, axs = plt.subplots(2)
    axs[0].grid()
    axs[1].grid()
    fig.suptitle('Projection to a single dimension')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    # DISTINGUISH THE DIFFERENT AXIS
    axs[0].plot(np.sum(gaussian_signal_2D, axis=1))
    axs[0].set_ylabel('Signal')
    axs[1].plot(np.sum(gaussian_signal_2D, axis=0))
    axs[1].set_ylabel('Signal')

    
def q_815(N, mu, sigma):

    #Question 1.5
    
    signal_2D = discrete_signal.compute_gaussian_2D(N, mu, sigma)
    gaussian_signal_2D, num_samples = signal_2D.compute_gaussian_pulse_2D()
    
    gaussian_DFT= discrete_signal.DFT_2D(gaussian_signal_2D)
    gaussian_signal_DFT = gaussian_DFT.solve()
    
    plt.imshow(abs(gaussian_signal_DFT), cmap='gray')
    plt.colorbar()
    plt.show()


def q_816(N, mu, sigma):

    #Question 1.6
    
    signal_2D = discrete_signal.compute_iDFT_2D(N, mu, sigma)
    gaussian_signal_2D, num_samples = signal_2D.solve() 
    
    gaussian_DFT = discrete_signal.DFT_2D(gaussian_signal_2D)
    gaussian_signal_DFT = gaussian_DFT.solve()
    
    gaussian_iDFT = discrete_signal.compute_iDFT_2D(gaussian_signal_DFT)
    gaussian_signal_iDFT = gaussian_iDFT.solve2()
    
    plt.imshow(abs(gaussian_signal_iDFT), cmap='gray')
    plt.colorbar()
    plt.show()   
    
    
def q_821(N, L):

    #Question 2.1
    
#    img = cv2.imread('imgA.png')  
    img = mpimg.imread('imgB.png') 

    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()   
   
    
def q_822(N):
    
    img = mpimg.imread('imgB.png')     
    N = N
    mu = (N-1)/2
    sigma = (N-1)/6
    
    signal_2D = discrete_signal.compute_gaussian_pulse_2D(N, mu, sigma)
    gaussian_signal_2D, num_samples = signal_2D.compute_gaussian_pulse_2D()
    
    gaussian_con_2D = discrete_signal.Convolution_2D(img, gaussian_signal_2D)
    filtered_img = gaussian_con_2D.compute_gaussian_2D()

    plt.imshow(filtered_img, cmap='gray')
    plt.colorbar()
    plt.show()    
    
    
  

###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__': 
    
#    question_1_3(32, 4) # N= 32, L= 4

#    question_1_4(9, 2, 1)
#    question_1_4(65, 2, 10)
#    question_1_4(255, 128, 42)

    # N = 9, MU = 2, SIGMA= 1
    # N = 65, MU = 2, SIGMA= 10
    # N = 255, MU = 128, SIGMA= 42
    
#    q_815(10, 0, 2)
    
#    q_816(10, 0, 2)
    
    q_822(7)