import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well  
import matplotlib.pyplot as plt
import os

from matplotlib.patches import Ellipse

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = 4 # process model dimension
        self.dt = 0.1 # time increment
        self.q=0.1 # CT process noise variable for Kalman filter Q

    def F(self):
        # system matrix
        dt = self.dt
        F = np.matrix([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return F
    def Q(self):
        # DT process noise covariance Q

        ############
        dt = self.dt
        q = self.q
        A = np.matrix([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
        Qd = A @ A.T * q * (dt**3 / 3) + (A + A.T) * q * (dt**2 / 2) + A.T @ A * q * dt
        return Qd
        ############
    
    def H(self):
        # measurement matrix H

        H = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
        return H
    
    def predict(self, x, P):
        # predict state and estimation error covariance to next timestep
        F = self.F()
        x = F*x # state prediction
        P = F*P*F.transpose() + self.Q() # covariance prediction
        return x, P

    def update(self, x, P, z, R):
        # update state and covariance with associated measurement
        H = self.H() # measurement matrix
        gamma = z - H*x # residual
        S = H*P*H.transpose() + R # covariance of residual
        K = P*H.transpose()*np.linalg.inv(S) # Kalman gain
        x = x + K*gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K*H) * P # covariance update
        return x, P   

def draw_ellipse(ax, x, P, n_std=2.0):
    cov = P[0:2, 0:2]
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(float(x[0]), float(x[1])),
                  width=width, height=height,
                  angle=angle,
                  edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(ell)    
        
def run_filter():
    ''' loop over data and call predict and update'''
    np.random.seed(0) # make random values predictable
    
    # init filter
    KF = Filter()
    
    # init figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # init track state and covariance
    x = np.matrix([[0],
                [0],
                [0],
                [0]])
    # Define STDs for position and velocity(assumed same in x and y)
    #std_p=0.1
    std_p=5
    #std_v=2
    std_v = 10
    # covariance matrix
    P = np.matrix([[std_p**2, 0, 0, 0],
                    [0, std_p**2, 0, 0],      
                    [0, 0, std_v**2, 0],
                    [0, 0, 0, std_v**2]])
    
    # lists for error and covariance tracking
    errors_px, errors_py = [], []
    meas_errors_px, meas_errors_py = [], []
    cov_px, cov_py = [], []

    first_label = True
    # loop over measurements and call predict and update
    for i in range(1,6001):        
        
        # prediction
        x, P = KF.predict(x, P) # predict to next timestep
        
        # ground truth generation
        t = i * KF.dt
        gt = np.matrix([[t],
                        [0.1 * t**2]]) #TODO: implement ground truth generation
        
        # measurement generation
        sigma_z = 0.2 # measurement noise 
        z = np.matrix([[float(gt[0]) + np.random.normal(0, sigma_z)],
                       [float(gt[1]) + np.random.normal(0, sigma_z)]]) # generate noisy measurement
        R = np.matrix([[sigma_z**2, 0], # measurement noise covariance matrix
                            [0, sigma_z**2]])
        
        # update
        x, P = KF.update(x, P, z, R) # update with measurement
        
        error_px = float(x[0]) - float(gt[0])
        error_py = float(x[1]) - float(gt[1])
        meas_error_px = float(z[0]) - float(gt[0])
        meas_error_py = float(z[1]) - float(gt[1])

        errors_px.append(error_px)
        errors_py.append(error_py)
        meas_errors_px.append(meas_error_px)
        meas_errors_py.append(meas_error_py)
        cov_px.append(P[0, 0])
        cov_py.append(P[1, 1])

        # visualization    
        '''
        ax.scatter(float(x[0]), float(x[1]), color='green', s=40, marker='x', label='track')
        ax.scatter(float(z[0]), float(z[1]), color='blue', marker='.', label='measurement')
        ax.scatter(float(gt[0]), float(gt[1]), color='gray', s=40, marker='+', label='ground truth')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_xlim(0, 600)
        ax1.set_ylim(0, 40000)
        ax1.set_title("Full View with Confidence Ellipses")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")

        ax2.set_xlim(0, 20)
        ax2.set_ylim(0, 50)
        ax2.set_title("Zoomed-in View")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
            
        for ax in [ax1, ax2]:
            ax.scatter(float(x[0]), float(x[1]), color='green', s=10, marker='x', label='track' if first_label else '')
            ax.scatter(float(z[0]), float(z[1]), color='blue', marker='.', label='measurement' if first_label else '')
            ax.scatter(float(gt[0]), float(gt[1]), color='gray', s=10, marker='+', label='ground truth' if first_label else '')
            draw_ellipse(ax, x, P)
        first_label = False

    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        ax.legend(handle_list, label_list, loc='upper left')

    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/tracking_with_ellipses.png")

    fig1 = plt.figure()
    plt.plot(errors_px, label='Estimation error x')
    plt.plot(meas_errors_px, label='Measurement error x', linestyle='--')
    plt.title("Estimation vs Measurement Error in X")
    plt.xlabel("Timestep")
    plt.ylabel("Error [m]")
    plt.legend()
    plt.grid()
    fig1.tight_layout()
    fig1.savefig("outputs/estimation_vs_measurement_error.png")

    fig2 = plt.figure()
    plt.plot(cov_px, label='Covariance p_x')
    plt.plot(cov_py, label='Covariance p_y')
    plt.title("Covariance Diagonal Elements")
    plt.xlabel("Timestep")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()
    fig2.tight_layout()
    fig2.savefig("outputs/covariance_diagonal_elements.png")

    print("All plots saved in 'outputs/' folder.")
        

####################
# call main loop
run_filter()
