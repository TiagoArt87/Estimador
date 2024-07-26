from pathlib import Path
import numpy as np
import time
import EESD
from PNT import correcao_pnt

def get_gabarito(eesd: EESD.EESD) -> np.array:
    ang = np.array([])
    tensoes = np.array([])
    for barra in eesd.DSSCircuit.AllBusNames:
        eesd.DSSCircuit.SetActiveBus(barra)
        ang = np.concatenate([ang, eesd.DSSCircuit.Buses.puVmagAngle[1::2]*2*np.pi / 360])
        tensoes = np.concatenate([tensoes, eesd.DSSCircuit.Buses.puVmagAngle[::2]])

    return np.concatenate([ang[3:], tensoes[3:]])

def main():
    #Achar o path do script do OpenDSS
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '123Bus_PV' / 'IEEE123Master.dss'
    # '6bus' / 'Caso_teste.dss'
    # '13Bus' / 'IEEE13Nodeckt.dss'
    # '123Bus_PV' / 'IEEE123Master.dss'
    
    verbose = False
    printar = True
    
    baseva =  33.3 * 10**6

    dict = {}
    # ARGUMENTOS: primeiro: barra/ segundo: fase(a=0,b=1,c=2)/ terceiro: perda (em kW)

    dict['82'] = ['82', 0, 15]
    dict['38'] = ['38', 1, 10]
    dict['47'] = ['47', 2, 25]
    dict['1'] = ['1', 0, 20]
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
    
    correcao_pnt(MasterFile, baseva, verbose, pnt, eesd, printar, limite=10, metodo='tiago')

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
        # Os valores são extremamente baixos, mas a multiplicação implementada deve ser suficiente para contornar esse problema
        ''' Realizar uma combinação entre os resíduos que "puna" as barras com altos resíduos normalizados de tensão'''
        # O método implementado funcionou bem, identificando todas as barras
        ''' Observar o comportamento dos resíduos da barra 149 quando há mais de um roubo envolvido'''
        # R: a barra 149 apresenta resíduos normalizados baixos quando há mais de um roubo envolvido. Por quê?
        pass

    if verbose:
        print(gabarito)
        print(vet_estados)

if __name__ == '__main__':
    main()