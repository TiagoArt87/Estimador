import matplotlib.pyplot as plt
import numpy as np
import EESD

class correcao_pnt:
    def __init__(self, MasterFile, baseva: float, verbose: bool, pnt: list, eesd: EESD.EESD, printar: False, limite: 10, 
                                                                                                        metodo:str, lista_max_k_ordem:list):
        self.MasterFile = MasterFile
        self.baseva = baseva
        self.verbose = verbose
        self.pnt = pnt
        self.eesd = eesd
        self.printar = printar
        self.limite_iter = False # Serve para sabermos se a correção foi realizada(FALSE) ou se o sistema parou devido ao limite de iterações(TRUE).
        self.corrigido = False
        self.nodes = eesd.nodes
        self.dp = eesd.dp
        self.barras = eesd.barras
        self.limite = limite
        self.sem_erros = False
        self.lista_max_k_ordem = lista_max_k_ordem # Serve para obter a melhor ordenação possível de detecção: ver DATASET
        self.metodo = metodo
        self.lista_conserta_pnt = [] # Vetor para corrigir as perdas estimadas
        self.dict_barras_afetadas = {} # Atualiza_barras_afetadas
        self.dict_erros_percentuais = {} # Compara perdas

        dict_fases = {}
        dict_fases[0] = 'a'
        dict_fases[1] = 'b'
        dict_fases[2] = 'c'
        self.dict_fases = dict_fases
        self.lista_Rn_at = []
        self.lista_Rn_reat = []
        self.lista_Rn_tensao = []
        self.lista_max_k =[]

        self.limite_correcao = 2 # define quantas vezes uma mesma barra pode ser corrigida
       
    def matriz_covariancia_residuos(self, n_med, h, w, dp):
        # Calculo das matrizes de covariancia
            
        #Calcula a matriz ganho
        matriz_ganho = h.T @ w @ h
        g = matriz_ganho

        # A definição de matriz de covariancia dos residuos pode ser encontrada na página 126 do livro
        # Power System State Estimation_ Theory and Implementation

        l = n_med
        Covar_medidas = np.zeros((l,l)) 
        for i in range(l):
            Covar_medidas[i][i]= dp[i]**2

        Covar_estados_estimados = np.linalg.inv(g.toarray())
        Covar_medidas_estimadas = h @ Covar_estados_estimados @ h.T
        Covar_residuos = Covar_medidas-Covar_medidas_estimadas

        # Normalização das Covariâncias
        diag_cov_res = np.diag(abs(Covar_residuos))

        # Matriz de covariancias normalizadas
        MCn = np.zeros((len(diag_cov_res),len(diag_cov_res)))
        for i in range(len(diag_cov_res)):
            MCn[i][i] = float(diag_cov_res[i])**(-1/2)
        
        return MCn

    def matriz_covariancia_residuos_2(self, h, w):
        # implementando o exposto na página 129 temos que os resíduos normalizados são dados por: R-H*(G^-1)*Ht
        g = h.T @ w @ h
        inv_w = np.linalg.inv(w.toarray())
        inv_g = np.linalg.inv(g.toarray())
        MCn = inv_w-h @ inv_g @ h.T    
        return MCn

    def init_var(self):
        r = self.eesd.residuo
        h = self.eesd.jacobiana
        n_med = self.eesd.num_medidas
        w = self.eesd.matriz_pesos
        sens = self.eesd.matriz_sensibilidade
        s = np.diag(sens)
        p = s
        lens = len(s)
        s = np.ones(lens)-s

        return r, h, n_med, w, p, s

    def residuos_normalizados(self, r, omega):
        Rn = []
        for i in range(len(r)):
            x = r[i]*10**10
            y = omega[i][i]*10**20
            Rn.append(abs(x)/((abs(y)**0.5)))

        tam = len(Rn)//3
        Rn_at = Rn[:tam]
        Rn_reat = Rn[tam:tam*2]
        Rn_tensao = Rn[tam*2:]

        return Rn, Rn_at, Rn_reat, Rn_tensao

    def inovation_index(self, r, p, s):
        # Implementação do inovation index, proposto por Newton Bretas e utilizado como referência por Lívia Raggi
        # A Geometrical View for Multiple Gross Errors Detection, Identification, and Correction in Power System 
        # State Estimation Newton G. Bretas
        ed = r
        eu = []
        ii = []
        cne = []
        erro = []
        for i in range(len(p)):
            ii.append(abs(s[i])/abs(p[i]))
            eu.append((1+1/ii[i])*ed[i])
            cne.append(((1+1/(ii[i]**2))**0.5)*self.Rn[i])
            erro.append((abs(ed[i])**2+abs(eu[i])**2)**0.5)

        return cne, erro

    def plot(self, residuos_acima):
        tam = len(self.Rn)//3
        x = np.arange(0, tam, 1)
        y = [0, tam, 0, 15]
        plt.scatter(x, self.Rn_at, color='tab:blue')
        plt.scatter(x, self.Rn_reat, color='tab:orange')
        plt.scatter(x, self.Rn_tensao, color='tab:green')
        plt.suptitle('Resíduos normalizados')
        plt.legend(['Rn Potência Ativa', 'Rn Potência Reativa', 'Rn Tensão'], bbox_to_anchor = (1 , 1))
        plt.show()
        
        if residuos_acima != []:
            results_at, results_reat, results_tensao = [], [], []
            x = []
            residuos_normalizados = {}
            for i in residuos_acima:
                results_at.append(self.Rn_at[i])
                results_reat.append(self.Rn_reat[i])
                results_tensao.append(self.Rn_tensao[i])
                node = list(filter(lambda var: self.nodes[var] == i+3, self.nodes))[0]
                residuos_normalizados[node] = [self.Rn_at[i], self.Rn_reat[i], self.Rn_tensao[i]]
                x.append(node)

            length = np.arange(len(x)) 
            width = 0.3

            plt.bar(length-0.3, results_at, width, color='tab:blue') 
            plt.bar(length, results_reat, width, color='tab:orange') 
            plt.bar(length+0.3, results_tensao, width, color='tab:green') 
            plt.xticks(length, x) 
            plt.xlabel("Barras") 
            plt.ylabel("Magnitude") 
            plt.legend(['Rn Potência Ativa', 'Rn Potência Reativa', 'Rn Tensão'], bbox_to_anchor = (1 , 1)) 
            plt.show()
        pass

    def analise_rn(self, phi_k, residuos_acima):
        if residuos_acima != []:
            tam = len(self.Rn)//3
            # Seperar por residuos de cada categoria:
            max_k_at = max(phi_k.keys(), key=(lambda key: self.Rn_at[key]))
            max_k_reat = max(phi_k.keys(), key=(lambda key: self.Rn_reat[key]))
            max_k_tensao = max(phi_k.keys(), key=(lambda key: self.Rn_tensao[key]))
            node_at = list(filter(lambda x: self.nodes[x] == max_k_at+3, self.nodes))[0]
            node_reat = list(filter(lambda x: self.nodes[x] == max_k_reat+3, self.nodes))[0]
            node_tensao = list(filter(lambda x: self.nodes[x] == max_k_tensao+3, self.nodes))[0]

            print(f'O maior resíduo normalizado de potência ativa se encontra na barra {node_at} e vale {self.Rn_at[max_k_at]}')
            print(f'O maior resíduo normalizado de potência reativa se encontra na barra {node_reat} e vale {self.Rn_reat[max_k_reat]}')
            print(f'O maior resíduo normalizado de tensão se encontra na barra {node_tensao} e vale {self.Rn_tensao[max_k_tensao]}')
        
        self.plot(residuos_acima)

    def listar_residuos_acima(self):
        tam = len(self.Rn)//3
        residuos_acima = []

        if self.metodo == 'livia':
            # Checa se os resíduos normalizados estão maiores que 3
            for i in range(tam):
                if self.Rn_at[i]>3 or self.Rn_reat[i]>3:
                    residuos_acima.append(i)

        elif self.metodo == 'tiago':
            for i in range(tam):
                if self.Rn_at[i]*self.Rn_reat[i]*self.Rn_tensao[i]>3:
                    residuos_acima.append(i)
        
        elif self.metodo == 'ordem':
            for i in range(tam):
                if self.Rn_at[i]*self.Rn_reat[i]*self.Rn_tensao[i]>3:
                    residuos_acima.append(i)
        
        return residuos_acima

    def obter_phi_k(self, residuos_acima, it_ordem):
        phi_k = {}
        at_ou_reat ={}
        if self.metodo == 'livia':
            for k in residuos_acima:
                # Por enquanto, apenas para potência ativa
                phi_k[k] = max(abs(self.Rn_at[k]),abs(self.Rn_reat[k]))-abs(self.Rn_tensao[k])
                if self.Rn_reat[k]>self.Rn_at[k]:
                    at_ou_reat[k] = 'reat'
                else:
                    at_ou_reat[k] = 'at'

            max_k = max(phi_k.keys(), key=(lambda key: phi_k[key]))
            node = list(filter(lambda x: self.nodes[x] == max_k+3, self.nodes))[0]
        
        elif self.metodo == 'tiago':
            for k in residuos_acima:
                # Por enquanto, apenas para potência ativa
                phi_k[k] = (abs(self.Rn_at[k])+abs(self.Rn_reat[k]))*abs(self.Rn_tensao[k])
                if self.Rn_reat[k]>self.Rn_at[k]:
                    at_ou_reat[k] = 'reat'
                else:
                    at_ou_reat[k] = 'at'

            max_k = max(phi_k.keys(), key=(lambda key: phi_k[key]))
            node = list(filter(lambda x: self.nodes[x] == max_k+3, self.nodes))[0]
        
        elif self.metodo == 'ordem':
            max_k = self.lista_max_k_ordem[it_ordem]
            node = list(filter(lambda x: self.nodes[x] == max_k+3, self.nodes))[0]

        # Pegar node
        barra, fase = node.split('.')
        fase = int(fase)-1

        return phi_k, max_k, barra, fase, node, at_ou_reat

    def atualizar_barras_afetadas(self, node, erro_absoluto):
        if node in self.dict_barras_afetadas:
            self.dict_barras_afetadas[node] += erro_absoluto
        else:
            self.dict_barras_afetadas[node] = erro_absoluto

    def compara_perdas(self):
        dict_pnt = {}
        
        for i in self.pnt:
            [barra, fase, perda] = i
            fase += 1
            dict_pnt[barra+'.'+str(fase)] = perda

        for i in self.dict_barras_afetadas.keys():
            if i not in dict_pnt.keys():
                self.dict_erros_percentuais[i] = 0
                barra_erro, fase_erro = i.split('.')
                fase_erro = int(fase_erro) - 1 
                print(f'Foi identificada erroneamente a fase {self.dict_fases[fase_erro]} da barra {barra_erro} como portadora de PNT.')
            else:
                self.dict_erros_percentuais[i] = round(((self.dict_barras_afetadas[i]-dict_pnt[i])/dict_pnt[i])*100, 3)
        
        #Checar quantas perdas foram identificadas do total inserido
        self.perdas_nao_identificadas = [0, len(dict_pnt)]
        for i in dict_pnt.keys():
            if i not in self.dict_barras_afetadas.keys():
                self.perdas_nao_identificadas[0] += 1
        
        print(self.dict_erros_percentuais)
        node_perda_max = max(self.dict_erros_percentuais.keys(), key=(lambda key: abs(self.dict_erros_percentuais[key]))) 
        self.erro_medio = [sum(self.dict_erros_percentuais.values()), len(self.dict_erros_percentuais)]
        sucesso = False
        if self.dict_barras_afetadas.keys() == dict_pnt.keys():
            sucesso = True

        barra, fase = node_perda_max.split('.')
        fase = int(fase)-1
        sim_nao = ''
        if sucesso == False:
            sim_nao = 'não '
        print(f'Não há mais erros encontrados no sistema apresentado.') 
        print(f'O algoritmo {sim_nao}foi capaz de identificar todas as barras com PNTs.') 
        print(f'O maior erro na estimação das perdas foi de {self.dict_erros_percentuais[node_perda_max]}% na fase {self.dict_fases[fase]} da barra {barra}.')
        
        return sucesso, perda

    def run(self):
        it = 0
        it_ordem = 0
        while self.corrigido == False:

            r, h, n_med, w, p, s = self.init_var()

            omega = np.array(self.matriz_covariancia_residuos_2(h,w))

            # Calcula-se os resíduos normalizados
            self.Rn, self.Rn_at, self.Rn_reat, self.Rn_tensao = self.residuos_normalizados(r, omega)

            residuos_acima = self.listar_residuos_acima()
            
            # Se houver resíduos acima, calcula-se o phi_k e o erro estimado

            if residuos_acima != []:
                phi_k, max_k, barra, fase, node, at_ou_reat = self.obter_phi_k(residuos_acima, it_ordem)
                # atualiza listas
                self.lista_Rn_at.append(self.Rn_at)
                self.lista_Rn_reat.append(self.Rn_reat)
                self.lista_Rn_tensao.append(self.Rn_tensao) 
                self.lista_max_k.append(max_k)
                # Cálculo dos erros            
                cne, erro = self.inovation_index(r,p,s)
                erro_estimado = cne[max_k]*self.dp[max_k]
                erro_absoluto = erro[max_k]*self.baseva/1000

                # Formular a correção da medição
                conserta_pnt = [barra, fase, erro[max_k]]

                # Relaciona as barras com perdas e suas perdas estimadas
                if node in self.dict_barras_afetadas:
                    #self.dict_barras_afetadas[node] += erro_absoluto
                    self.limite_correcao -= 1
                    if self.limite_correcao == 0:
                        print('limite de correções em uma barra atingido')
                        self.corrigido = True
                else:
                    self.dict_barras_afetadas[node] = erro_absoluto

                self.lista_conserta_pnt.append(conserta_pnt)

                if self.printar == True:
                    self.analise_rn(phi_k, residuos_acima)
                    print(f'Corrigindo um erro de {erro_absoluto} kW na fase {self.dict_fases[fase]} da barra {barra}...')
                
                eesd_novo = EESD.EESD(self.MasterFile, self.baseva, self.verbose, self.pnt, self.lista_conserta_pnt)
                vet_estados = eesd_novo.run(10**-5, 100)
                # Atingindo o limite de iterações: Consertar o loop
                self.eesd = eesd_novo
                it += 1
                it_ordem += 1
                if it == self.limite:
                    print('limite de iterações atingido')
                    self.limite_iter = True
                    break
            else:
                if it == 0:
                    if self.printar == True:
                        self.analise_rn({}, [])
                    self.sem_erros == True
                    print('Provavelmente não há erros grosseiros no conjunto de medidas apresentado')
                self.corrigido = True
        
        if self.sem_erros == False:
            if self.printar:
                for i in self.dict_barras_afetadas.keys():
                    barra, fase = i.split('.')
                    fase = int(fase)-1
                    perda = self.dict_barras_afetadas[i]
                    print(f'Foi corrigida uma perda de {perda} kW na fase {self.dict_fases[fase]} da barra {barra}')
                