import time
import itertools
import random as rnd
#from PNT import correcao_pnt
import EESD
from PNT import correcao_pnt
from Random import gerar_vetor_pnt

class DATASET():
    def __init__(self, tamanho = 1, escala = [1, 10], MasterFile = '.', baseva =  33.3*10**6, metodo = 'tiago') -> None:
        self.tamanho = tamanho
        [self.start, self.stop] = escala
        self.stop = self.stop + 1
        self.MasterFile = MasterFile
        self.baseva = baseva
        self.verbose = False
        self.printar = False
        self.metodo = metodo
        self.problemas = [False, 0,[]]
        self.eesd = EESD.EESD(MasterFile, baseva, self.verbose)
        self.barras = self.eesd.barras
        self.lista_pnt = []
        self.perdas_nao_identificadas = [0, 0]
        self.erro_medio = [0,0]
        self.ordenacao = {}
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
            self.lista_pnt.append(pnt)
            #print(pnt)
            eesd = EESD.EESD(self.MasterFile, self.baseva, self.verbose, pnt)
            vet_estados = eesd.run(10**-5, 100)
            analise_pnt = correcao_pnt(self.MasterFile, self.baseva, self.verbose, pnt, eesd, self.printar, 
                                                            self.stop+2, self.metodo, lista_max_k_ordem=[])
            analise_pnt.run()
            analise_pnt.compara_perdas()
            erro, qtd = analise_pnt.erro_medio
            self.erro_medio[0] += erro
            self.erro_medio[1] += qtd
            pnt_n_id, pnt_exist = analise_pnt.perdas_nao_identificadas
            self.perdas_nao_identificadas[0] += pnt_n_id
            self.perdas_nao_identificadas[1] += pnt_exist
            if analise_pnt.corrigido == False:
                self.problemas[0] == True
                self.problemas[1] += 1
                lista = self.problemas[2]
                lista.append(pnt)
                self.problemas[2] = lista
            else:
                for i in range(len(analise_pnt.lista_Rn_at)):
                    Rn_at.append(analise_pnt.lista_Rn_at[i])
                    Rn_reat.append(analise_pnt.lista_Rn_reat[i])
                    Rn_tensao.append(analise_pnt.lista_Rn_tensao[i])
                    max_k.append(analise_pnt.lista_max_k[i])
            iter += 1
            tamanho -= 1
            fim_iter = time.time()
            print(f'A iteração {iter} do dataset levou {fim_iter-inicio_iter:.3f}s')
        
        self.erro_medio = self.erro_medio[0]/self.erro_medio[1]
        fim = time.time()
        print(f'A construção do dataset levou {fim-inicio:.3f}s')
        
        return Rn_at, Rn_reat, Rn_tensao, max_k
    
    def corrigir_medidas(self, max_k_pnt):
        eesd = EESD.EESD(self.MasterFile, self.baseva, self.verbose, self.lista_pnt[0])
        eesd.run(10**-5, 100)
        analise_ordem = correcao_pnt(self.MasterFile, self.baseva, False, self.lista_pnt[0], eesd, False, self.stop+1, 
                     'ordem', lista_max_k_ordem=max_k_pnt)
        sucesso, erro = analise_ordem.compara_perdas()
        pass

    def achar_melhor_ordem(self):
        # O objetivo dessa função é encontrar a ordem ideal para correção das PNTs
        idx=0
        perms = list(itertools.permutations(self.lista_max_k))
        # Gera uma lista dos indexes das barras com PNT
        while idx < len(perms):
            max_k_pnt = perms[idx] # Seleciona uma das opções possíveis de correção
            sucesso, erro = self.corrigir_medidas(max_k_pnt)
            if sucesso:
                self.ordenacao[idx] = erro
            idx += 1
        idx_perda_max = max(self.ordenacao.keys(), key=(lambda key: abs(self.ordenacao[key])))
        print(f'A melhor ordem de detecção foi {perms[idx_perda_max]}, gerando um erro de {self.ordenacao[idx_perda_max]}%')