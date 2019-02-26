import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)

        #### TODO ####
        # update self.x, self.P
        self.x = g
        self.P = Gx.dot(self.P).dot(Gx.T) + dt * Gu.dot(self.Q).dot(Gu.T)
        ##############

    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        #### TODO ####
        # update self.x, self.P
        P = self.P
        sig = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(sig))
        self.x = self.x + K.dot(z[:,0])
        # print(self.x.shape)
        # print(K.dot(z).shape)
        self.P = P - K.dot(sig).dot(K.T)
        ##############

    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x

        #### TODO ####
        # compute g, Gx, Gu
        xdot = v*np.cos(th)
        ydot = v*np.sin(th)
        
        if np.abs(om) < 1e-6:
            gx = x + xdot*dt 
            gy = y + ydot*dt
            gth = th + om*dt
            Gx = np.array([ [1, 0, -v*np.sin(th)*dt ],
                            [0, 1, v*np.cos(th)*dt ],
                            [0, 0, 1] ])
            Gu = np.array([ [np.cos(th)*dt, -v*dt*dt*np.sin(th)],
                            [np.sin(th)*dt, v*dt*dt*np.cos(th)],
                            [0, dt] ])           
        else:
            gx = x + v/om * (np.sin(th+om*dt)-np.sin(th))
            gy = y + v/om * (np.cos(th)-np.cos(th+om*dt))
            gth = th + om*dt
            Gx = np.array([ [1, 0, v/om * (np.cos(th+om*dt) - np.cos(th)) ],
                            [0, 1, v/om * (np.sin(th+om*dt) - np.sin(th))],
                            [0, 0, 1] ])
            Gu = np.array([ [1 /om * (np.sin(th+om*dt)-np.sin(th)), -v/(om**2) * (np.sin(th+om*dt)-np.sin(th)) + v/om *( np.cos(th+ om*dt)*dt)],
                            [1/om * (np.cos(th)-np.cos(th+om*dt)), -v/(om**2) * (np.cos(th)-np.cos(th+om*dt)) + v/om *(np.sin(th+om*dt)*dt) ],
                            [0, dt] ])
        g = np.stack((gx,gy,gth),axis = 0)
        
        ##############

        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m

        #### TODO ####
        # compute h, Hx
        xc,yc,thc = self.tf_base_to_camera #camera coordinates/pose relative to base
        x,y,th = self.x #base coordinates relative to world
        #we want parameters in terms of camera (alphac, rc)
        #currently have parameters in terms of world (alpha,r)
        #need to transform parameters from world to camera
        rot = np.array([ [cos(th), -sin(th)],
                        [sin(th), cos(th)] ])
        dx,dy = rot.dot([xc,yc])
        alphaprime = alpha-thc-th
        h = np.array([alphaprime, r - (x+dx)*np.cos(alpha) - (y+dy)*np.sin(alpha)])
        derivh2 = -cos(alpha)*(-xc*sin(th)-yc*cos(th)) - sin(alpha)*(xc*cos(th)-yc*sin(th))
        Hx = np.array([ [0,0,-1],
                        [-np.cos(alpha), -np.sin(alpha), derivh2 ]])
        
        ##############

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        g = self.g
        valid = g**2
        P = self.P
        v_list = []
        R_list = []
        H_list = []
        d_list = []
        for i in range(len(rawR)):
            for j in range((self.map_lines.shape[1])):
                m = self.map_lines[:,j]
                h, Hx = self.map_line_to_predicted_measurement(m)
                v = rawZ[:,i]-h
                S = Hx.dot(P).dot(Hx.T) + rawR[i]
                d = np.dot(v,np.linalg.inv(S)).dot(v)

                if np.abs(d) < valid:
                    valid = d
                    vmin  = v
                    dmin = d
                    Hmin = Hx
                    rmin = rawR[i]
            if np.abs(valid) < g**2:
                valid = g**2
                d_list.append(dmin)
                v_list.append(vmin)
                R_list.append(rmin)
                H_list.append(Hmin)
        ##############
        
        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H
        z = np.reshape(v_list, (len(v_list)*2,1))

        temp = np.reshape(R_list, (len(R_list)*2, 2))
        # R = np.zeros((len(temp),len(temp)))
        # for i in xrange(0,len(temp),2):
        #     R[i:i+2,i:i+2] = temp[i:i+2,:]
        #     # print(R)
        R = scipy.linalg.block_diag(*R_list)
        H = np.reshape(H_list,(len(H_list)*2,3))
        ##############

        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        #### TODO ####
        # compute g, Gx, Gu (some shape hints below)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))
        g = np.copy(self.x)
        xdot = v*np.cos(th)
        ydot = v*np.sin(th)
        if np.abs(om) < 1e-6:
            gx = x + xdot*dt 
            gy = y + ydot*dt
            gth = th + om*dt
            tempGx = np.array([ [1, 0, -v*np.sin(th)*dt ],
                            [0, 1, v*np.cos(th)*dt ],
                            [0, 0, 1] ])
            tempGu = np.array([ [np.cos(th)*dt, -v*dt*dt*np.sin(th)],
                            [np.sin(th)*dt, v*dt*dt*np.cos(th)],
                            [0, dt] ])     
        else:
            gx = x + v/om * (np.sin(th+om*dt)-np.sin(th))
            gy = y + v/om * (np.cos(th)-np.cos(th+om*dt))
            gth = th + om*dt
            tempGx = np.array([ [1, 0, v/om * (np.cos(th+om*dt) - np.cos(th)) ],
                            [0, 1, v/om * (np.sin(th+om*dt) - np.sin(th))],
                            [0, 0, 1] ])
            tempGu = np.array([ [1 /om * (np.sin(th+om*dt)-np.sin(th)), -v/(om**2) * (np.sin(th+om*dt)-np.sin(th)) + v/om *( np.cos(th+ om*dt)*dt)],
                            [1/om * (np.cos(th)-np.cos(th+om*dt)), -v/(om**2) * (np.cos(th)-np.cos(th+om*dt)) + v/om *(np.sin(th+om*dt)*dt) ],
                            [0, dt] ])
        g[0] = gx
        g[1] = gy
        g[2] = gth
        Gx[:3,:3] = tempGx
        Gu[:3,:] = tempGu

        ##############

        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        
        z = np.reshape(v_list, (len(v_list)*2,1))
        # print(len(R_list)[0])
        # print(len(self.x))
        # temp = np.reshape(R_list, (len(R_list)*2, 2))
        # R = np.zeros((len(temp),len(temp)))
        # for i in xrange(0,len(temp),2):
        #     R[i:i+2,i:i+2] = temp[i:i+2,:]
        R = scipy.linalg.block_diag(*R_list)
        H = np.reshape(H_list,(len(H_list)*2,len(self.x)))
        ##############

        return z, R, H

    # Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
    #
    # Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
    # so that we know which components of the Jacobian to fill in.
    def map_line_to_predicted_measurement(self, j):
        alpha, r = self.x[(3+2*j):(3+2*j+2)]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

        #### TODO ####
        # compute h, Hx (you may find the skeleton for computing Hx below useful)
        xc,yc,thc = self.tf_base_to_camera #camera coordinates/pose relative to base
        x,y,th = self.x[:3] #base coordinates relative to world
        #we want parameters in terms of camera (alphac, rc)
        #currently have parameters in terms of world (alpha,r)
        #need to transform parameters from world to camera
        rot = np.array([ [cos(th), -sin(th)],
                        [sin(th), cos(th)] ])
        dx,dy = rot.dot([xc,yc])
        alphaprime = alpha-thc-th
        h = np.array([alphaprime, r - (x+dx)*np.cos(alpha) - (y+dy)*np.sin(alpha)])
        derivh2 = cos(alpha)*(xc*sin(th)+yc*cos(th)) - sin(alpha)*(xc*cos(th)-yc*sin(th))
        Hx = np.zeros((2,self.x.size))
        Hx[:,:3] = np.array([ [0,0,-1],
                        [-np.cos(alpha), -np.sin(alpha), derivh2 ]]) #fillmein
        # First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
        if j > 1:
            Hx[0, 3+2*j] = 1
            Hx[1, 3+2*j] = (x+dx)*np.sin(alpha) - (y+dy)*np.cos(alpha)
            Hx[0, 3+2*j+1] = 0
            Hx[1, 3+2*j+1] = 1
        
        ##############

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        g = self.g
        valid = g**2
        P = self.P
        v_list = []
        R_list = []
        H_list = []
        d_list = []
        numlines = (len(self.x)-3 )/2
        for i in range(len(rawR)):
            for j in range(numlines):
                h, Hx = self.map_line_to_predicted_measurement(j)
                v = rawZ[:,i]-h
                S = Hx.dot(P).dot(Hx.T) + rawR[i]
                d = np.dot(v,np.linalg.inv(S)).dot(v)

                if np.abs(d) < valid:
                    valid = d
                    vmin  = v
                    dmin = d
                    Hmin = Hx
                    rmin = rawR[i]
            if np.abs(valid) < g**2:
                valid = g**2
                d_list.append(dmin)
                v_list.append(vmin)
                R_list.append(rmin)
                H_list.append(Hmin)
        ##############
        
        return v_list, R_list, H_list
