import numpy as np
import pandas as pd
import random as rnd
from EESD import EESD

def gerar_roubos(eesd:EESD, k:int, baseva=3.3*10**6):
    barras = eesd.barras
    nodes = eesd.nodes
    #Retirar barra de geração:
    nodes = nodes.keys()
    nodes = nodes[3:]
    seq = selecionar_barras(nodes, k)
    barra_fase_valor = pegar_valores(seq, barras, baseva)
    barra_fase_roubo = gerar_roubo(barra_fase_valor)
    return barra_fase_roubo
    
def selecionar_barras(nodes:dict, k:int):
    nome_barras = nodes.keys()
    seq = rnd.sample(nome_barras, k)
    return seq

def pegar_valores(seq, barras, baseva):
    barra_fase_valor = []
    for i in seq:
        barra, fase = i.split('.')
        fase = int(fase) - 1
        barra_pnt = barras.index[barras['nome_barra'] == barra].to_list()[0]
        valor = barras['Inj_pot_at'][barra_pnt][fase]
        valor = abs(valor)*baseva/1000
        lista = [barra, fase, valor]
        barra_fase_valor.append(lista)
    
    return barra_fase_valor

def gerar_roubo(barra_fase_valor):
    barra_fase_roubo = []
    for i in barra_fase_valor:
        [barra, fase, valor] = i
        # mu é a média, sigma o desvio padrão
        roubo = rnd.gauss(mu=valor/2, sigma=valor/8)
        barra_fase_roubo.append([barra, fase, roubo])

    return barra_fase_roubo
