import time
import EESD
import random as rnd
from PNT import correcao_pnt
from Random import gerar_vetor_pnt

class dataset():
    def __init__(self, tamanho = 1, escala = [1, 10], MasterFile = '.', baseva =  33.3*10**6, metodo = 'tiago') -> None:
        self.tamanho = tamanho
        [self.start, self.stop] = escala
        self.MasterFile = MasterFile
        self.baseva = baseva
        self.verbose = False
        self.printar = False
        self.metodo = metodo
        self.problemas = [False, 0,[]]
        self.eesd = EESD.EESD(MasterFile, baseva, self.verbose)
        self.barras = self.eesd.barras
        self.lista_Rn_at, self.lista_Rn_reat, self.lista_Rn_tensao, self.lista_max_k = self.montar_dataset()
        print('fim da construção do dataset')
    
    def montar_dataset(self):
        Rn_at = []
        Rn_reat = []
        Rn_tensao = []
        max_k = []
        inicio = time.time()
        eesd = self.eesd
        tamanho = self.tamanho
        iter = 0
        while tamanho>0:
            inicio_iter = time.time()
            k = int(rnd.randrange(self.start, self.stop, 1))
            pnt = gerar_vetor_pnt(eesd, k, self.baseva)
            eesd = EESD.EESD(self.MasterFile, self.baseva, self.verbose, pnt)
            vet_estados = eesd.run(10**-5, 100)
            lista_Rn_at, lista_Rn_reat, lista_Rn_tensao, lista_max_k, corrigido = correcao_pnt(self.MasterFile, 
                                    self.baseva, self.verbose, pnt, eesd, self.printar, self.stop+1, self.metodo)
            if corrigido == False:
                self.problemas[0] == True
                self.problemas[1] += 1
                lista = self.problemas[2]
                lista.append(pnt)
            else:
                for i in range(len(lista_Rn_at)):
                    Rn_at.append(lista_Rn_at[i])
                    Rn_reat.append(lista_Rn_reat[i])
                    Rn_tensao.append(lista_Rn_tensao[i])
                    max_k.append(lista_max_k[i])
            iter += 1
            tamanho -= 1
            fim_iter = time.time()
            print(f'A iteração {iter} do dataset levou {fim_iter-inicio_iter:.3f}s')

        fim = time.time()
        print(f'A construção do dataset levou {fim-inicio:.3f}s')
        
        return Rn_at, Rn_reat, Rn_tensao, max_k
                