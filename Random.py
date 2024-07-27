import numpy as np
import pandas as pd
import random as rnd
from EESD import EESD

def gerar_vetor_pnt(eesd:EESD, k:int, baseva=33.3*10**6):
    barras = eesd.barras
    nodes = eesd.nodes
    #Retirar barra de geração:
    nome_barras = list(nodes.keys())
    nome_barras = nome_barras[3:]
    #Separar as barras com cargas
    nome_barras = barras_com_cargas(barras, nome_barras)

    barras_sel = selecionar_barras(nome_barras, k)
    barra_fase_valor = pegar_valores(barras_sel, barras, baseva)
    barra_fase_roubo = gerar_roubo(barra_fase_valor)
    return barra_fase_roubo
    
def barras_com_cargas(barras: pd.DataFrame, nome_barras: list):
    nome_barras_com_cargas = []
    for i in nome_barras:
        barra, fase = i.split('.')
        fase = int(fase) - 1
        barra_pnt = barras.index[barras['nome_barra'] == barra].to_list()[0]
        valor = barras['Inj_pot_at'][barra_pnt][fase]
        if valor < 0:
            nome_barras_com_cargas.append(i)

    return nome_barras_com_cargas

def selecionar_barras(nome_barras:list, k:int):
    barras_sel = rnd.sample(nome_barras, k)
    return barras_sel

def pegar_valores(barras_sel, barras, baseva):
    barra_fase_valor = []
    for i in barras_sel:
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
