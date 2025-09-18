'''
Librería con las clases y funciones
para lógica proposicional
'''

from typing import Set, List, Dict
from itertools import product
import numpy as np

CONECTIVOS = ['-', 'Y','O','>','=']
CONECTIVOS_BINARIOS = ['Y','O','>','=']


class Formula :

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        if type(self) == Letra:
            return self.letra
        elif type(self) == Negacion:
            return '-' + str(self.subf)
        elif type(self) == Binario:
            return "(" + str(self.left) + self.conectivo + str(self.right) + ")"

    def letras(self):
        if type(self) == Letra:
            return set(self.letra)
        elif type(self) == Negacion:
            return self.subf.letras()
        elif type(self) == Binario:
            return self.left.letras().union(self.right.letras())

    def subforms(self) -> List[str]:
        if type(self) == Letra:
            return [str(self)]
        elif type(self) == Negacion:
            return list(set([str(self)] + self.subf.subforms()))
        elif type(self) == Binario:
            return list(set([str(self)] + self.left.subforms() + self.right.subforms()))

    def valor(self, I:Dict[str, bool]) -> bool:
        if type(self) == Letra:
            return I[self.letra]
        elif type(self) == Negacion:
            return not self.subf.valor(I)
        elif type(self) == Binario:
            if self.conectivo == 'Y':
                return self.left.valor(I) and self.right.valor(I)
            if self.conectivo == 'O':
                return self.left.valor(I) or self.right.valor(I)
            if self.conectivo == '>':
                return not self.left.valor(I) or self.right.valor(I)
            if self.conectivo == '=':
                return (self.left.valor(I) and self.right.valor(I)) or (not self.left.valor(I) and not self.right.valor(I))

class Letra(Formula) :
    def __init__ (self, letra:str) :
        self.letra = letra

class Negacion(Formula) :
    def __init__(self, subf:Formula) :
        self.subf = subf

class Binario(Formula) :
    def __init__(self, conectivo:str, left:Formula, right:Formula) :
        assert(conectivo in CONECTIVOS_BINARIOS), f'Símbolo {conectivo} no es aceptado como conectivo binario. Debe usar uno de {CONECTIVOS_BINARIOS}'
        self.conectivo = conectivo
        self.left = left
        self.right = right
        

def inorder_to_tree(cadena:str) -> Formula:
    if len(cadena) == 1:
        assert(cadena not in CONECTIVOS)
        return Letra(cadena)
    elif cadena[0] == '-':
        return Negacion(inorder_to_tree(cadena[1:]))
    elif cadena[0] == "(":
        assert(cadena[-1] == ")"), f'¡Cadena inválida! Falta un paréntesis final en {cadena}'
        counter = 0 #Contador de parentesis
        for i in range(1, len(cadena)):
            if cadena[i] == "(":
                counter += 1
            elif cadena[i] == ")":
                counter -=1
            elif cadena[i] in conectivos and counter == 0:
                return Binario(cadena[i], inorder_to_tree(cadena[1:i]),inorder_to_tree(cadena[i + 1:-1]))
    else:
        raise Exception('¡Cadena inválida! Revise la composición de paréntesis de la fórmula.\nRecuerde que solo los conectivos binarios incluyen paréntesis en la fórmula.')

        
        
class Descriptor :

    '''
    Codifica un descriptor de N argumentos mediante un solo caracter
    Input:  args_lista, lista con el total de opciones para cada
                     argumento del descriptor
            chrInit, entero que determina el comienzo de la codificación chr()
    Output: str de longitud 1
    '''

    def __init__ (self,args_lista,chrInit=256) -> None:
        self.args_lista = args_lista
        assert(len(args_lista) > 0), "Debe haber por lo menos un argumento"
        self.chrInit = chrInit
        self.rango = [chrInit, chrInit + np.prod(self.args_lista)]

    def check_lista_valores(self,lista_valores: List[int]) -> None:
        for i, v in enumerate(lista_valores) :
            assert(v >= 0), "Valores deben ser no negativos"
            assert(v < self.args_lista[i]), f"Valor debe ser menor o igual a {self.args_lista[i]}"

    def codifica(self,lista_valores: List[int]) -> int:
        self.check_lista_valores(lista_valores)
        cod = lista_valores[0]
        n_columnas = 1
        for i in range(0, len(lista_valores) - 1) :
            n_columnas = n_columnas * self.args_lista[i]
            cod = n_columnas * lista_valores[i+1] + cod
        return cod

    def decodifica(self,n: int) -> int:
        decods = []
        if len(self.args_lista) > 1:
            for i in range(0, len(self.args_lista) - 1) :
                n_columnas = np.prod(self.args_lista[:-(i+1)])
                decods.insert(0, int(n / n_columnas))
                n = n % n_columnas
        decods.insert(0, n % self.args_lista[0])
        return decods

    def ravel(self,lista_valores: List[int]) -> chr:
        codigo = self.codifica(lista_valores)
        return chr(self.chrInit+codigo)

    def unravel(self,codigo: chr) -> int:
        n = ord(codigo)-self.chrInit
        return self.decodifica(n)
    
    def escribir(self, literal: chr) -> str:
        if '-' in literal:
            atomo = literal[1:]
            neg = ' no'
        else:
            atomo = literal
            neg = ''
        x, y, n  = self.unravel(atomo)
        return f"PREDICADO({x, y, n})"        
    
    

def visualizar_formula(A: Formula, D: Descriptor) -> str:
    '''
    Visualiza una fórmula A (como string en notación inorder) usando el descriptor D
    '''
    vis = []
    for c in A:
        if c == '-':
            vis.append(' no ')
        elif c in ['(', ')']:
            vis.append(c)
        elif c in ['>', 'Y', 'O']:
            vis.append(' ' + c + ' ')
        elif c == '=':
            vis.append(' sii ')
        else:
            try:
                vis.append(D.escribir(c))
            except:
                raise("¡Caracter inválido!")
    return ''.join(vis)
