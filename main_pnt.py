from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import EESD
import Pos_filtragem
import pandas as pd

def get_gabarito(eesd: EESD.EESD) -> np.array:
    ang = np.array([])
    tensoes = np.array([])
    for barra in eesd.DSSCircuit.AllBusNames:
        eesd.DSSCircuit.SetActiveBus(barra)
        ang = np.concatenate([ang, eesd.DSSCircuit.Buses.puVmagAngle[1::2]*2*np.pi / 360])
        tensoes = np.concatenate([tensoes, eesd.DSSCircuit.Buses.puVmagAngle[::2]])

    return np.concatenate([ang[3:], tensoes[3:]])

def matriz_covariancia_residuos(n_med, barras, h, w, dp):
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

def matriz_covariancia_residuos_2(h,w):
    # implementando o exposto na página 129 temos que os resíduos normalizados são dados por: R-H*(G^-1)*Ht
    g = h.T @ w @ h
    inv_w = np.linalg.inv(w.toarray())
    inv_g = np.linalg.inv(g.toarray())
    MCn = inv_w-h @ inv_g @ h.T    
    return MCn

def init_var(eesd_novo: EESD.EESD):
    r = eesd_novo.residuo
    h = eesd_novo.jacobiana
    n_med = eesd_novo.num_medidas
    w = eesd_novo.matriz_pesos
    sens = eesd_novo.matriz_sensibilidade
    s = np.diag(sens)
    p = s
    lens = len(s)
    s = np.ones(lens)-s

    return r, h, n_med, w, p, s

def residuos_normalizados(r, omega):
    Rn = []
    for i in range(len(r)):
        x = r[i]*10**10
        y = omega[i][i]*10**20
        Rn.append(abs(x)/((abs(y)**0.5)))

    return Rn 

def plot(Rn, nodes, residuos_acima):
    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]
    x = np.arange(0, tam, 1)
    y = [0, tam, 0, 15]
    plt.plot(x, Rn_at, 'ro')
    plt.axis(y)
    plt.suptitle('Resíduos normalizados de potência ativa')
    plt.show()
    plt.plot(x, Rn_reat, 'ro')
    plt.axis(y)
    plt.suptitle('Resíduos normalizados de potência reativa')
    plt.show()
    plt.plot(x, Rn_tensao, 'ro')
    plt.axis(y)
    plt.suptitle('Resíduos normalizados de tensão')
    plt.show()
    
    if residuos_acima != []:
        results_at, results_reat, results_tensao = [], [], []
        x = []
        for i in residuos_acima:
            results_at.append(Rn_at[i])
            results_reat.append(Rn_reat[i])
            results_tensao.append(Rn_tensao[i])
            node = list(filter(lambda x: nodes[x] == i+3, nodes))[0]
            x.append(node)

        plt.bar(x, results_at)
        plt.suptitle('Resíduos normalizados de potência ativa acima de 3')
        plt.show()
        plt.bar(x, results_reat)
        plt.suptitle('Resíduos normalizados de potência reativa acima de 3')
        plt.show()
        plt.bar(x, results_tensao)
        plt.suptitle('Resíduos normalizados de tensão acima de 3')
        plt.show()

    pass

def analise_rn(phi_k, Rn, nodes, residuos_acima):
    if residuos_acima != []:
        tam = len(Rn)//3
        Rn_at = Rn[:tam]
        Rn_reat = Rn[tam:tam*2]
        Rn_tensao = Rn[tam*2:]
        # Seperar por residuos de cada categoria:
        max_k_at = max(phi_k.keys(), key=(lambda key: Rn_at[key]))
        max_k_reat = max(phi_k.keys(), key=(lambda key: Rn_reat[key]))
        max_k_tensao = max(phi_k.keys(), key=(lambda key: Rn_tensao[key]))
        node_at = list(filter(lambda x: nodes[x] == max_k_at+3, nodes))[0]
        node_reat = list(filter(lambda x: nodes[x] == max_k_reat+3, nodes))[0]
        node_tensao = list(filter(lambda x: nodes[x] == max_k_tensao+3, nodes))[0]

        print(f'O maior resíduo normalizado de potência ativa se encontra na barra {node_at} e vale {Rn_at[max_k_at]}')
        print(f'O maior resíduo normalizado de potência reativa se encontra na barra {node_reat} e vale {Rn_reat[max_k_reat]}')
        print(f'O maior resíduo normalizado de tensão se encontra na barra {node_tensao} e vale {Rn_tensao[max_k_tensao]}')
    
    plot(Rn, nodes, residuos_acima)

def correcao_pnt(MasterFile, baseva: float, verbose: bool, pnt: list, eesd: EESD.EESD):
    correcao = False
    nodes = eesd.nodes
    dp = eesd.dp
    barras = eesd.barras
    eesd_novo = eesd
    it = 0
    printar = True
    # print(nodes)

    while correcao == False:
                
        r, h, n_med, w, p, s = init_var(eesd_novo)
        
        # MCn = matriz_covariancia_residuos(n_med, barras, h, w, dp)

        omega = np.array(matriz_covariancia_residuos_2(h,w))

        # Calcula-se os resíduos normalizados
        Rn = residuos_normalizados(r, omega)
          
        tam = len(Rn)//3
        # print(max(Rn))
        # print(tam)
        Rn_at = Rn[:tam]
        Rn_reat = Rn[tam:tam*2]
        Rn_tensao = Rn[tam*2:]

        residuos_acima = []
        
        # Checa se os resíduos normalizados estão maiores que 3
        for i in range(tam):
            if Rn_at[i]>3 or Rn_reat[i]>3:
                residuos_acima.append(i)
        
        # Se houver resíduos acima, calcula-se o phi_k e o erro estimado
        if residuos_acima != []:
            phi_k = {}
            at_ou_reat ={}
            for k in residuos_acima:
                # Por enquanto, apenas para potência ativa
                phi_k[k] = Rn_at[k]
                if Rn_reat[k]>Rn_at[k]:
                    at_ou_reat[k] = 'reat'
                else:
                    at_ou_reat[k] = 'at'

            if printar == True:
                analise_rn(phi_k, Rn, nodes, residuos_acima)

            # Cálculo dos erros
            max_k = max(phi_k.keys(), key=(lambda key: Rn[key]))
            node = list(filter(lambda x: nodes[x] == max_k+3, nodes))[0]
            erro_estimado = []
            cne_at = ((1+(p[max_k])/(1-s[max_k]))**0.5)*Rn[max_k]
            cne_reat = ((1+(p[2*max_k])/(1-s[2*max_k]))**0.5)*Rn[2*max_k]
            cne_tensao = ((1+(p[3*max_k])/(1-s[3*max_k]))**0.5)*Rn[3*max_k]
            
            # Pegar node
            barra, fase = node.split('.')
            fase = int(fase)-1

            erro_estimado = cne_at*dp[max_k]

            # Formular a correção da medição
            conserta_pnt = [barra, fase, erro_estimado]
            print(conserta_pnt)

            eesd_novo = EESD.EESD(MasterFile, baseva, verbose, pnt, [conserta_pnt])
            vet_estados = eesd_novo.run(10**-5, 100)

            it+=1
            if it == 2:
                correcao = True
        else:
            analise_rn({}, Rn, nodes, [])
            print('Provavelmente não há erros grosseiros no conjunto de medidas apresentado')
            correcao = True

def main():
    #Achar o path do script do OpenDSS
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '123Bus_PV' / 'IEEE123Master.dss'
    # '6bus' / 'Caso_teste.dss'
    # '13Bus' / 'IEEE13Nodeckt.dss'
    # '123Bus_PV' / 'IEEE123Master.dss'
    
    verbose = False
    
    baseva =  33.3 * 10**6

    dict = {}
    # ARGUMENTOS: primeiro: barra/ segundo: fase(a=0,b=1,c=2)/ terceiro: perda (em kW)
    dict['82']=['82', 0, 10]
    pnt = []
    for i in dict:
        pnt.append(dict[i])
        pass

    eesd = EESD.EESD(MasterFile, baseva, verbose, pnt)
    
    inicio = time.time()
    vet_estados = eesd.run(10**-5, 100)
    fim = time.time()
    print(f'Estimador concluido em {fim-inicio}s')

    gabarito = get_gabarito(eesd)
    
    correcao_pnt(MasterFile, baseva, verbose, pnt, eesd)

    # Comentários:
    if 'comentario'=='comentario':

        # Verificar a matriz de sensibilidade - valores muito baixos - CNE PRÓXIMO DE 1
        ''' Verificar os arquivos da matriz sensibilidade.'''
        # Verificar se o desvio padrão se refere às medidas de pu ou as medidas em kw
        ''' Trabalhar tudo em pu.'''
        # Verificar o que pode ser feito para garantir a estabilidade do estimador
        # -> Verificar a possibilidade de utilizar as fases da Sourcebus no cálculo da Jacobiana e dos Resíduos
        ''' Procurar barras mais distantes do alimentador.'''
        ''' Plotar os resíduos normalizados.'''

        # Orientações Professor Alex:
        ''' Verificar a modelagem da barra 149'''

        # Minhas observações:
        ''' Checar os valores de residuos da barra 149'''
        ''' Checar a base da barra 149'''
        ''' Observar os valores envolvidos no cálculo dos resíduos normalizados da barra 149'''
        ''' Realizar uma combinação entre os resíduos que "puna" as barras com altos resíduos normalizados de tensão'''
        pass

    if verbose:
        print(gabarito)
        print(vet_estados)

if __name__ == '__main__':
    main()