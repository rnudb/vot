import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        """
        The class will be initialized with six parameters:
         dt : time for one cycle used to estimate state (sampling time)
         u_x, u_y : accelerations in the x-, and y-directions respectively
         std_acc: process noise magnitude
         x_sdt_meas, y_sdt_meas :standard deviations of the measurement in the x- and y-directions
        respectively
         Control input variables u=[u_x, u_y]
         Initial state matrix: (𝑥ෞ௞)=[x0=0, y0=0, vx=0,vy=0]
         Matrices describing the system model A,B with respect to the sampling time dt (∆t) :
         Measurement mapping matrix H
         Initial process noise covariance matrix Q with respect to the standard deviation of
        acceleration (std_acc) σa :
         Initial measurement noise covariance R. Suppose that the measurements z (x, y) are both
        independent (so that covariance x and y is 0), and look only the variance in the x and y:
         Initialize covariance matrix P for prediction error as an identity matrix whose shape is the same as
        the shape of the matrix A
        """

        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas

        self.u = np.array([[self.u_x], [self.u_y]])
        self.x_k = np.array([[0], [0], [0], [0]])
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.B = np.array(
            [[self.dt**2 / 2, 0], [0, self.dt**2 / 2], [self.dt, 0], [0, self.dt]]
        )

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = (
            np.array(
                [
                    [self.dt**4 / 4, 0, self.dt**3 / 2, 0],
                    [0, self.dt**4 / 4, 0, self.dt**3 / 2],
                    [self.dt**3 / 2, 0, self.dt**2, 0],
                    [0, self.dt**3 / 2, 0, self.dt**2],
                ]
            )
            * self.std_acc**2
        )
        self.R = np.array([[self.x_sdt_meas**2, 0], [0, self.y_sdt_meas**2]])

        self.P = np.identity(self.A.shape[0])

    def predict(self):
        """
        This function does the prediction of the state estimate 𝑥ො௞
        ି and the error prediction 𝑃௞
        ି. This task also
        call the time update process (u) because it projects forward the current state to the next time step.
         Update time state
         Calculate error covariance
        """
        self.x_k = self.A.dot(self.x_k) + self.B.dot(self.u)
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
        return self.x_k

    def update(self, z):
        """
        This, function takes measurements 𝑧௞ as input (centroid coordinates x,y of detected circles)
         Compute Kalman gain
         Update the predicted state estimate 𝑥ෞ௞ and predicted error covariance 𝑃௞
        """

        K = self.P.dot(self.H.T).dot(
            np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.R)
        )
        self.x_k = self.x_k + K.dot(z - self.H.dot(self.x_k))
        self.P = (np.identity(self.P.shape[0]) - K.dot(self.H)).dot(self.P)
        return self.x_k
