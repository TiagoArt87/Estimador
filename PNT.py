import matplotlib.pyplot as plt
import numpy as np
import EESD

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

    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]

    return Rn, Rn_at, Rn_reat, Rn_tensao

def inovation_index(Rn, r, p, s):
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
        cne.append(((1+1/(ii[i]**2))**0.5)*Rn[i])
        erro.append((abs(ed[i])**2+abs(eu[i])**2)**0.5)

    return cne, erro

def plot(Rn, nodes, residuos_acima):
    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]
    x = np.arange(0, tam, 1)
    y = [0, tam, 0, 15]
    plt.scatter(x, Rn_at, color='tab:blue')
    plt.scatter(x, Rn_reat, color='tab:orange')
    plt.scatter(x, Rn_tensao, color='tab:green')
    plt.suptitle('Resíduos normalizados')
    plt.legend(['Rn Potência Ativa', 'Rn Potência Reativa', 'Rn Tensão'], bbox_to_anchor = (1 , 1))
    plt.show()
    
    if residuos_acima != []:
        results_at, results_reat, results_tensao = [], [], []
        x = []
        residuos_normalizados = {}
        for i in residuos_acima:
            results_at.append(Rn_at[i])
            results_reat.append(Rn_reat[i])
            results_tensao.append(Rn_tensao[i])
            node = list(filter(lambda var: nodes[var] == i+3, nodes))[0]
            residuos_normalizados[node] = [Rn_at[i], Rn_reat[i], Rn_tensao[i]]
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

def listar_residuos_acima(Rn, metodo):
    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]
    residuos_acima = []

    if metodo == 'livia':
        # Checa se os resíduos normalizados estão maiores que 3
        for i in range(tam):
            if Rn_at[i]>3 or Rn_reat[i]>3:
                residuos_acima.append(i)

    elif metodo == 'tiago':
        for i in range(tam):
            if Rn_at[i]*Rn_reat[i]*Rn_tensao[i]>3:
                residuos_acima.append(i)
    
    return residuos_acima

def obter_phi_k(metodo, Rn, residuos_acima, nodes):
    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]
    phi_k = {}
    at_ou_reat ={}
    if metodo == 'livia':
        for k in residuos_acima:
            # Por enquanto, apenas para potência ativa
            phi_k[k] = max(abs(Rn_at[k]),abs(Rn_reat[k]))-abs(Rn_tensao[k])
            if Rn_reat[k]>Rn_at[k]:
                at_ou_reat[k] = 'reat'
            else:
                at_ou_reat[k] = 'at'
    
    elif metodo == 'tiago':
        for k in residuos_acima:
            # Por enquanto, apenas para potência ativa
            phi_k[k] = (abs(Rn_at[k])+abs(Rn_reat[k]))*abs(Rn_tensao[k])
            if Rn_reat[k]>Rn_at[k]:
                at_ou_reat[k] = 'reat'
            else:
                at_ou_reat[k] = 'at'

    max_k = max(phi_k.keys(), key=(lambda key: phi_k[key]))
    node = list(filter(lambda x: nodes[x] == max_k+3, nodes))[0]

    # Pegar node
    barra, fase = node.split('.')
    fase = int(fase)-1

    return phi_k, max_k, barra, fase, node, at_ou_reat

def atualizar_barras_afetadas(lista_barras_afetadas, node, erro_absoluto):
    if node in lista_barras_afetadas:
        lista_barras_afetadas[node] += erro_absoluto
    else:
        lista_barras_afetadas[node] = erro_absoluto

def atualizar_listas(Rn, max_k, lista_Rn_at:list, lista_Rn_reat:list, lista_Rn_tensao:list, lista_max_k:list):
    tam = len(Rn)//3
    Rn_at = Rn[:tam]
    Rn_reat = Rn[tam:tam*2]
    Rn_tensao = Rn[tam*2:]
    lista_Rn_at.append(Rn_at)
    lista_Rn_reat.append(Rn_reat)
    lista_Rn_tensao.append(Rn_tensao)
    lista_max_k.append(max_k)

    return lista_Rn_at, lista_Rn_reat, lista_Rn_tensao, lista_max_k    

def correcao_pnt(MasterFile, baseva: float, verbose: bool, pnt: list, eesd: EESD.EESD, printar: False, limite: 10, 
                                                                                                        metodo:str):
    corrigido = True # Serve para sabermos se a correção foi realizada ou se o sistema parou devido ao limite de iterações.
    correcao = False
    nodes = eesd.nodes
    dp = eesd.dp
    barras = eesd.barras
    eesd_novo = eesd
    it = 0
    # print(nodes)
    sem_erros = False
    lista_conserta_pnt = []
    lista_barras_afetadas = {}
    dict_fases = {}
    dict_fases[0] = 'a'
    dict_fases[1] = 'b'
    dict_fases[2] = 'c'
    lista_Rn_at = []
    lista_Rn_reat = []
    lista_Rn_tensao = []
    lista_max_k =[]

    while correcao == False:
                
        r, h, n_med, w, p, s = init_var(eesd_novo)
        
        # MCn = matriz_covariancia_residuos(n_med, barras, h, w, dp)

        omega = np.array(matriz_covariancia_residuos_2(h,w))

        # Calcula-se os resíduos normalizados
        Rn, Rn_at, Rn_reat, Rn_tensao = residuos_normalizados(r, omega)

        residuos_acima = listar_residuos_acima(Rn, metodo)
        
        # Se houver resíduos acima, calcula-se o phi_k e o erro estimado
        if residuos_acima != []:
            phi_k, max_k, barra, fase, node, at_ou_reat = obter_phi_k(metodo, Rn, residuos_acima, nodes)
            lista_Rn_at, lista_Rn_reat, lista_Rn_tensao, lista_max_k = atualizar_listas(Rn, max_k, lista_Rn_at, lista_Rn_reat, lista_Rn_tensao, lista_max_k)
            # Cálculo dos erros            
            cne, erro = inovation_index(Rn,r,p,s)
            erro_estimado = cne[max_k]*dp[max_k]
            erro_absoluto = erro[max_k]*baseva/1000

            # Formular a correção da medição
            conserta_pnt = [barra, fase, erro[max_k]]

            atualizar_barras_afetadas(lista_barras_afetadas, node, erro_absoluto)
            
            #print(conserta_pnt)
            lista_conserta_pnt.append(conserta_pnt)

            if printar == True:
                analise_rn(phi_k, Rn, nodes, residuos_acima)
            
            print(f'Corrigindo um erro de {erro_absoluto} kW na fase {dict_fases[fase]} da barra {barra}...')
            eesd_novo = EESD.EESD(MasterFile, baseva, verbose, pnt, lista_conserta_pnt)
            vet_estados = eesd_novo.run(10**-5, 100)

            it+=1
            if it == limite:
                print('limite de iterações atingido')
                corrigido = False
                correcao = True
        else:
            if it == 0:
                if printar == True:
                    analise_rn({}, Rn, nodes, [])
                sem_erros == True
                print('Provavelmente não há erros grosseiros no conjunto de medidas apresentado')
            correcao = True
    
    if sem_erros == False:
        for i in lista_barras_afetadas:
            barra, fase = i.split('.')
            fase = int(fase)-1
            perda = lista_barras_afetadas[i]
            print(f'Foi corrigida uma perda de {perda} kW na fase {dict_fases[fase]} da barra {barra}')
        if it != 0:
            print(f'Não há mais erros no sistema apresentado')

    return lista_Rn_at, lista_Rn_reat, lista_Rn_tensao, lista_max_k, corrigido
        