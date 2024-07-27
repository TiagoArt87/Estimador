from pathlib import Path
from DATASET import dataset

path = Path(__file__)
CurrentFolder = path.parent
MasterFile = CurrentFolder / 'objs' / '123Bus_PV' / 'IEEE123Master.dss'
baseva = 33.3*10**6

# tamanho: quantidade de cenários a serem rodados
# escala: quantas perdas por cenário, de 'A' até 'B-1'
banco = dataset(10,[1,5], MasterFile, baseva, 'tiago')
if banco.problemas[0]:
    print(f'Houveram {banco.problemas[1]} problemas na construção do dataset')
else:
    print(f'Não houveram problemas na construção do dataset')