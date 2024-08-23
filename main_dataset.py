from pathlib import Path
from DATASET import DATASET

path = Path(__file__)
CurrentFolder = path.parent
MasterFile = CurrentFolder / 'objs' / '123Bus_PV' / 'IEEE123Master.dss'
baseva = 33.3*10**6

# tamanho: quantidade de cenários a serem rodados
# escala: quantas perdas por cenário, de 'A' até 'B'
banco1 = DATASET(100,[1,5], MasterFile, baseva, 'tiago')
print(f'O erro médio das correções do primeiro dataset foi de {round(banco1.erro_medio, 3)}%')
if banco1.perdas_nao_identificadas[0] != 0:
    txt1 = banco1.perdas_nao_identificadas[0]
    txt2 = banco1.perdas_nao_identificadas[1]
    print(f'{txt1} roubos não foram identificados, isso representa {round(txt1/txt2,3)}% do total de roubos inseridos.')
if banco1.problemas[0]:
    print(f'Houveram {banco1.problemas[1]} problemas na construção do dataset')
    print(banco1.problemas[2])
else:
    print(f'Não houveram problemas na construção do primeiro dataset')

#banco.achar_melhor_ordem()

banco2 = DATASET(100,[6,10], MasterFile, baseva, 'tiago')
print(f'O erro médio das correções do segundo dataset foi de {round(banco2.erro_medio, 3)}%')
if banco2.perdas_nao_identificadas[0] != 0:
    txt1 = banco2.perdas_nao_identificadas[0]
    txt2 = banco2.perdas_nao_identificadas[1]
    print(f'{txt1} roubos não foram identificados, isso representa {round(txt1/txt2,3)}% do total de roubos inseridos.')
if banco2.problemas[0]:
    print(f'Houveram {banco2.problemas[1]} problemas na construção do segundo dataset')
    print(banco2.problemas[2])
else:
    print(f'Não houveram problemas na construção do segundo dataset')

# O algoritmo não é capaz de identificar todas as barras com PNT em condições de roubo especificas
# Sugestão: 1a IA para identificar as barras com PNTs -> entrada: todos os resíduos iniciais do sistema
# 2a IA para ordenar as barras com preferência -> entrada: resíduos normalizados das barras problemáticas