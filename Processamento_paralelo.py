import os, sys, getopt
import multiprocessing 
import concurrent.futures
import numpy as np

class Multiprocessamento():
    def __init__(self, matriz1, matriz2) -> None:
        self.mat1 = matriz1
        self.mat2 = matriz2
        self.resultado = self.Multiprocessar_mult_matriz()
    
    def mult_mat(self, slice):
        mult = slice @ self.mat2
        return mult
    
    def Multiprocessar_mult_matriz(self) -> np.array:
        # checar a possibilidade de multiplicação
        m1, n1 = self.mat1.shape

        m2 = self.mat2.shape
        if len(m2)==2:
            m2, n2 = m2
        else:
            m2 = m2[0]

        if n1 != m2:
            raise ValueError('matrizes não são compatíveis para multiplicação')
        
        numero_processadores = os.cpu_count()

        # dividir a primeira em partes que satisfaçam a quantidade de processadores disponíveis
        divisao = []

        ultimo_resto, resto = 0, 0
        divisao.append(resto)

        while numero_processadores != 0:
            if m1 >= numero_processadores:
                ultimo_resto = ultimo_resto + resto
                resto = m1 // numero_processadores
				
                divisao.append(ultimo_resto + resto)
                m1 = m1 - resto
                numero_processadores = numero_processadores - 1

            else:
                divisao = np.linspace(0,m1,num=m1, dtype=np.int64)
                break
        
        # dividir a matriz em partes iguais para cada processador
        mat_slices = []
        for i in range(len(divisao)-1):
            mat1_slice = self.mat1[divisao[i] : divisao[i + 1], :]
            mat_slices.append(mat1_slice)
                
        with concurrent.futures.ProcessPoolExecutor() as executor: # realiza as multiplicações matriciais
            results = executor.map(self.mult_mat, mat_slices) 
        
        results_list = []
        for i in results:
            results_list.append(i)
        
        mat_out = results_list[0]
        for i in range(len(results_list)-1):
            mat_out = np.concatenate((mat_out, results_list[1+i]))

        return mat_out