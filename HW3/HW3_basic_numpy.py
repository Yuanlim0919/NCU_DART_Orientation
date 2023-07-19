# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris


## Q1: feature normalization and min-max normalization

class Q1:
    def __init__(self):
        self.dataset = load_iris().data
        self.norm_dataset = np.zeros((150,4))
        self.min_max_dataset = np.zeros((150,4))
        pass
    
    def normalization(self,X):
        norm_X = (X - np.std(X))/np.mean(X)
        return norm_X

    def Min_Max(self,X):
        min_max_X = (np.max(X) - X) / (np.max(X) - np.min(X))
        return min_max_X

    def main(self):
        for i in range(4):
            self.norm_dataset[:,i] = self.normalization(self.dataset[:,i])
            self.min_max_dataset[:,i] = self.Min_Max(self.dataset[:,i])
        return self.norm_dataset, self.min_max_dataset

q1 = Q1()
norm_ds, min_max_ds = q1.main()

## Q2: matrix operation and equation solving

class Q2:
    def __init__(self):
        self.A = np.array([[2,1,0],[1,1,2],[-1,1,2]])
        self.B = np.array([[3,1,2],[3,-2,4],[-3,5,1]])
        self.C = np.array([-21,0,27])
    def inverse_matrix(self,X):
        try:
            inv_mat = np.linalg.inv(X)
            return inv_mat
        except np.linalg.LinAlgError:
            print("Inverse matrix doesn't exist!")
        
    def matrix_multiply(self,X,Y):
        try:
            mat_mul = np.cross(X,Y)
            return mat_mul
        except np.linalg.LinAlgError:
            print("Two matrices provided cannot be multiplied!")
        
    def equation_solver(self,X,Y):
        try:
            sol = np.linalg.solve(X,Y)
            return sol
        except np.linalg.LinAlgError:
            print("X provided is singular or not square!")
    
    def main(self):
        inv_A = self.inverse_matrix(self.A)
        inv_B = self.inverse_matrix(self.B)
        AB_BA = self.matrix_multiply(self.A, self.B) - self.matrix_multiply(self.B, self.A)
        eqn_sol = self.equation_solver(self.A,self.C)
        return inv_A, inv_B, AB_BA, eqn_sol
    
q2 = Q2()
inv_A, inv_B, AB_BA, eqn_sol = q2.main()

## Q3: vector operation
class Q3:
    def __init__(self):
        self.P = np.array([1,4,3])
        self.Q = np.array([3,2,4])
        
    def dot_product(self,P,Q):
        try:
            dot = np.dot(P,Q)
            return dot
        except ValueError:
            print("Dot product operation failed!")
    
    def Euclidean_dist(self,P,Q):
        try:
            dist = np.sqrt(np.sum((P - Q)**2))
            return dist
        except:
            print("Check dimension of input")
        
    def projection(self,P,Q):
        norm_q = np.sqrt(sum(Q**2))
        proj_q_p = (np.dot(P, Q)/norm_q**2)*Q
        return proj_q_p
        
    def cosine(self,P,Q):
        cos_value = np.dot(P,Q) / (np.linalg.norm(P)*np.linalg.norm(Q))
        return cos_value
    
    def main(self):
        p_dot_q = self.dot_product(self.P, self.Q)
        dist_p_q = self.Euclidean_dist(self.P, self.Q)
        proj_q_p = self.projection(self.P, self.Q)
        cosine = self.cosine(self.P, self.Q)
        return p_dot_q, dist_p_q, proj_q_p, cosine

q3 = Q3()
p_dot_q, dist_p_q, proj_q_p, cosine = q3.main()

