'''
Librería con las clases y funciones
para lógica proposicional
'''

from itertools import product
import numpy as np
from copy import deepcopy
from typing import List, Dict
from random import randint, choice, uniform

CONECTIVOS = ['-', 'Y','O','>','=']
CONECTIVOS_BINARIOS = ['Y','O','>','=']

class Formula :
    """Clase base para fórmulas de lógica proposicional.

    Descripción
    -----------
    Representa la interfaz común para los tres tipos de fórmulas:
    - Letra: átomo proposicional (p. ej. 'p').
    - Negacion: negación de una subfórmula (p. ej. '-p' o '-(pYq)').
    - Binario: conectivo binario entre dos subfórmulas (Y, O, >, =).

    Cada método documentado a continuación es autosuficiente: indica su entrada,
    salida, efectos esperados y ejemplos mínimos de uso. Estos docstrings
    están diseñados para que un lector entienda la semántica sin leer la
    implementación.

    Notas
    -----
    - Las representaciones en cadena usan notación inorder:
      Letra -> 'p', Negacion -> '-X', Binario -> '(A B C)' con B el conectivo.
    - Las interpretaciones I son diccionarios {letra: bool}.
    """

    def __init__(self) :
        """Inicializa la fórmula (sin parámetros en la clase base).

        Propósito: permitir que subclases llamen super().__init__() si lo necesitan.
        """
        pass

    def __str__(self) :
        """Devuelve la representación en notación inorder de la fórmula.

        Salida:
            str: cadena legible que describe la fórmula.
        Ejemplos:
            Letra('p') -> 'p'
            Negacion(Letra('p')) -> '-p'
            Binario('Y', Letra('p'), Letra('q')) -> '(pYq)'
        """
        if type(self) == Letra:
            return self.letra
        elif type(self) == Negacion:
            return '-' + str(self.subf)
        elif type(self) == Binario:
            return "(" + str(self.left) + self.conectivo + str(self.right) + ")"

    def letras(self):
        """Devuelve el conjunto de letras (átomos) presentes en la fórmula.

        Salida:
            set[str]: conjunto de identificadores de letras, p. ej. {'p','q'}.

        Comportamiento por tipo:
          - Letra: devuelve {self.letra}
          - Negacion: delega en la subfórmula
          - Binario: unión de letras de left y right
        """
        if type(self) == Letra:
            return set(self.letra)
        elif type(self) == Negacion:
            return self.subf.letras()
        elif type(self) == Binario:
            return self.left.letras().union(self.right.letras())

    def subforms(self):
        """Lista las subfórmulas relevantes de la fórmula.

        Uso:
            útil para análisis estructural y clasificación en tableaux.

        Salida:
            list[Formula]: lista de subfórmulas (incluye la propia cuando procede).
        """
        if type(self) == Letra:
            return [str(self)]
        elif type(self) == Negacion:
            return list(set([str(self)] + self.subf.subforms()))
        elif type(self) == Binario:
            return list(set([str(self)] + self.left.subforms() + self.right.subforms()))

    def valor(self, I) :
        """Evalúa la fórmula bajo la interpretación I.

        Args:
            I (dict): mapeo letra -> bool, p. ej. {'p': True, 'q': False}.

        Retorna:
            bool: valor de verdad de la fórmula según I.

        Excepciones:
            KeyError si falta alguna letra en I (comportamiento intencional para detectar errores).
        """
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

    def SATtabla(self):
        """Búsqueda exhaustiva por tabla de verdad de una interpretación satisfactoria.

        Estrategia:
          - Obtiene el conjunto de letras, genera todas las combinaciones booleanas
            y evalúa la fórmula. Devuelve la primera interpretación que hace la fórmula True.

        Retorna:
            dict | None: interpretación (dict letra->bool) que satisface la fórmula,
                         o None si no existe.
        """
        letras = list(self.letras())
        n = len(letras)
        valores = list(product([True, False], repeat=n))
        for v in valores:
            I = {letras[x]: v[x] for x in range(n)}
            if self.valor(I):
                return I
        return None

    def clasifica_para_tableaux(self):
        """Clasifica la fórmula para reglas de tableaux.

        Debe devolver información suficiente para:
          - identificar si la fórmula es 'literal', 'alfa' (descomposición conjuntiva) o 'beta' (bifurcación),
          - incluir las subfórmulas/resultados necesarios para la expansión.

        Formato esperado (por convención interna):
          tupla o lista con elementos que permitan a nodos_tableaux distinguir la regla.
        """
        if type(self) == Letra:
            return None, 'literal'
        elif type(self) == Negacion:
            if type(self.subf) == Letra:
                return None, 'literal'
            elif type(self.subf) == Negacion:
                return 1, 'alfa'
            elif type(self.subf) == Binario:
                if self.subf.conectivo == 'O':
                    return 3, 'alfa'
                elif self.subf.conectivo == '>':
                    return 4, 'alfa'
                elif self.subf.conectivo == 'Y':
                    return 1, 'beta'
        elif type(self) == Binario:
            if self.conectivo == 'Y':
                return 2, 'alfa'
            elif self.conectivo == 'O':
                return 2, 'beta'
            elif self.conectivo == '>':
                return 3, 'beta'

    def SATtableaux(self):
        """Busca una interpretación satisfactoria usando tableaux semánticos.

        Algoritmo (resumen):
          - Crea nodo inicial con la fórmula.
          - Expande aplicando reglas alfa y beta hasta encontrar una hoja abierta o cerrar todas.
          - Si encuentra hoja abierta, extrae interpretación de literales de la rama.

        Retorna:
            dict | None: interpretación encontrada o None si la fórmula es insatisfactible.
        """
        estado = nodos_tableaux([self])
        res = estado.es_hoja()
        if res == 'cerrada':
            return None
        elif res == 'abierta':
            return estado.interp()
        frontera = [estado]
        while len(frontera) > 0:
            estado = frontera.pop(0)
            hijos = estado.expandir()
            for a in hijos:
                if a != None:
                    res = a.es_hoja()
                    if res == 'abierta':
                        return a.interp()
                    elif res == None:
                        frontera.append(a)
        return None

    def ver(self, D):
        """Devuelve una representación textual donde átomos codificados se traducen con D.escribir.

        Args:
            D (Descriptor): descriptor que convierte caracteres atómicos en cadenas legibles.

        Retorna:
            str: fórmula en la que cada átomo (carácter) ha sido sustituido por la cadena
                 que devuelve D.escribir(atomo). Ejemplo: '5 en (1,2)'.
        """
        vis = []
        A = str(self)
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

    def num_conec(self):
        """Cuenta el número de conectivos en la fórmula.
        
        Retorna:
            int: número total de conectivos (incluyendo negaciones)
        """
        if type(self) == Letra:
            return 0
        elif type(self) == Negacion:
            return 1 + self.subf.num_conec()
        elif type(self) == Binario:
            return 1 + self.left.num_conec() + self.right.num_conec()

class Letra(Formula) :
    """Átomo proposicional simple.

    Args:
        letra (str): identificador del átomo (preferentemente un carácter).

    Ejemplo:
        Letra('p')
    """
    def __init__ (self, letra:str) :
        self.letra = letra

class Negacion(Formula) :
    """Negación de una subfórmula.

    Args:
        subf (Formula): subfórmula que se niega.

    Ejemplo:
        Negacion(Letra('p'))  # representa '-p'
    """
    def __init__(self, subf:Formula) :
        self.subf = subf

class Binario(Formula) :
    """Fórmula binaria con conectivo entre dos operandos.

    Args:
        conectivo (str): uno de 'Y','O','>','='
        left (Formula): operando izquierdo
        right (Formula): operando derecho

    Ejemplo:
        Binario('Y', Letra('p'), Negacion(Letra('q')))
    """
    def __init__(self, conectivo:str, left:Formula, right:Formula) :
        assert(conectivo in ['Y','O','>','='])
        self.conectivo = conectivo
        self.left = left
        self.right = right


def inorder_to_tree(cadena:str) -> Formula:
    """Parsea una fórmula en notación inorder a objetos Formula.

    Reglas aceptadas (resumen):
      - Letras: símbolo único (no conectivo).
      - Negación: prefijo '-', p. ej. '-p' o '-(AYB)'.
      - Binarios: must be parenthesized: '(A Y B)' en la notación interna sin espacios '(AYB)'.

    Args:
        cadena (str): fórmula en notación inorder.

    Retorna:
        Formula: instancia de Letra, Negacion o Binario.

    Errores:
        Lanza Exception si la cadena está vacía o mal formada.

    Ejemplo:
        inorder_to_tree('(pY(qOr))')
    """
    if len(cadena) == 0:
        raise Exception('¡Error: cadena vacía!')
    if len(cadena) == 1:
        assert(cadena not in CONECTIVOS), f"Error: El símbolo de letra proposicional {cadena} no puede ser un conectivo ({CONECTIVOS})."
        return Letra(cadena)
    elif cadena[0] == '-':
        try:
            return Negacion(inorder_to_tree(cadena[1:]))
        except Exception as e:
            msg_error = f'Cadena incorrecta:\n\t{cadena[1:]}\n'
            msg_error += f'Error obtenido:\n\t{e}'
            raise Exception(msg_error)
    elif cadena[0] == "(":
        assert(cadena[-1] == ")"), f'¡Cadena inválida! Falta un paréntesis final en {cadena}'
        counter = 0 #Contador de parentesis
        for i in range(1, len(cadena)):
            if cadena[i] == "(":
                counter += 1
            elif cadena[i] == ")":
                counter -=1
            elif cadena[i] in CONECTIVOS_BINARIOS and counter == 0:
                try:
                    return Binario(cadena[i], inorder_to_tree(cadena[1:i]),inorder_to_tree(cadena[i + 1:-1]))
                except Exception as e:
                    msg_error = f'{e}\n\n'
                    msg_error += f'Error en la cadena:\n\t{cadena}'
                    msg_error += f'\nSe pide procesar el conectivo principal: {cadena[i]}'
                    msg_error += f'\nRevisar las subfórmulas\t{cadena[1:i]}\n\t{cadena[i + 1:-1]}'
                    raise Exception(msg_error)
    else:
        raise Exception('¡Cadena inválida! Revise la composición de paréntesis de la fórmula.\nRecuerde que solo los conectivos binarios incluyen paréntesis en la fórmula.')


class Descriptor :
    """Codificador compacto de tuplas (varios argumentos) en un único carácter.

    Propósito
    ---------
    Permitir representar átomos complejos (por ejemplo: "número n en posición x,y")
    mediante un único carácter que facilita construir fórmulas como cadenas.

    Uso típico
    ----------
    D = Descriptor([9,3,3])    # números 0..8, posiciones 0..2,0..2
    c = D.ravel([4,1,2])       # carácter único representando (n=4,x=1,y=2)
    lista = D.unravel(c)       # devuelve [4,1,2]

    Atributos relevantes
    --------------------
    - args_lista (list[int]): rango por cada argumento.
    - chrInit (int): desplazamiento usado con ord()/chr().

    Métodos principales
    -------------------
    - codifica(lista_valores) -> int
    - decodifica(n) -> list[int]
    - ravel(lista_valores) -> chr
    - unravel(codigo) -> list[int]
    - escribir(literal) -> str  # convierte literal (o '-'+literal) a texto legible
    """
    def __init__ (self,args_lista,chrInit=256) -> None:
        self.args_lista = args_lista
        assert(len(args_lista) > 0), "Debe haber por lo menos un argumento"
        self.chrInit = chrInit
        self.rango = [chrInit, chrInit + np.prod(self.args_lista)]

    def check_lista_valores(self,lista_valores: List[int]) -> None:
        """Valida que los índices en lista_valores están en los rangos definidos.

        Args:
            lista_valores (list[int]): índices por argumento, p. ej. [n,x,y]

        Raises:
            AssertionError si algún índice está fuera de rango.
        """
        for i, v in enumerate(lista_valores) :
            assert(v >= 0), "Valores deben ser no negativos"
            assert(v < self.args_lista[i]), f"Valor debe ser menor o igual a {self.args_lista[i]}"

    def codifica(self,lista_valores: List[int]) -> int:
        """Convierte la lista de índices en un único entero índice (orden mixto).

        Retorna:
            int: índice lineal no desplazado (antes de sumar chrInit).
        """
        self.check_lista_valores(lista_valores)
        cod = lista_valores[0]
        n_columnas = 1
        for i in range(0, len(lista_valores) - 1) :
            n_columnas = n_columnas * self.args_lista[i]
            cod = n_columnas * lista_valores[i+1] + cod
        return cod

    def decodifica(self,n: int) -> int:
        """Invierte codifica: entero -> lista de índices."""
        decods = []
        if len(self.args_lista) > 1:
            for i in range(0, len(self.args_lista) - 1) :
                n_columnas = np.prod(self.args_lista[:-(i+1)])
                decods.insert(0, int(n / n_columnas))
                n = n % n_columnas
        decods.insert(0, n % self.args_lista[0])
        return decods

    def ravel(self,lista_valores: List[int]) -> chr:
        """Devuelve el carácter que codifica la tupla de índices."""
        codigo = self.codifica(lista_valores)
        return chr(self.chrInit+codigo)

    def unravel(self,codigo: chr) -> int:
        """Convierte un carácter codificado en la lista de índices original."""
        n = ord(codigo)-self.chrInit
        return self.decodifica(n)
    
    def escribir(self, literal: chr) -> str:
        """Representación legible de un literal (admita '-'+literal).

        Ejemplo de salida: '5 en (1,2)' donde 5 es el número humano (n+1).
        """
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


def Ytoria(lista_forms):
    form = ''
    inicial = True
    for f in lista_forms:
        if inicial:
            form = f
            inicial = False
        else:
            form = '(' + form + 'Y' + f + ')'
    return form

def Otoria(lista_forms):
    form = ''
    inicial = True
    for f in lista_forms:
        if inicial:
            form = f
            inicial = False
        else:
            form = '(' + form + 'O' + f + ')'
    return form

class nodos_tableaux:

    def __init__(self, fs):
        clasfs = [(A, str(A), *A.clasifica_para_tableaux()) for A in fs]
        self.alfas = [c for c in clasfs if c[3] == 'alfa']
        self.betas = [c for c in clasfs if c[3] == 'beta']
        self.literales = [c for c in clasfs if c[3] == 'literal']

    def __str__(self):
        """Representación breve del nodo para debugging: lista de literales y tipos."""
        cadena = f'Alfas:{[str(c[1]) for c in self.alfas]}\n'
        cadena += f'Betas:{[str(c[1]) for c in self.betas]}\n'
        cadena += f'Literales:{[str(c[1]) for c in self.literales]}'
        return cadena

    def tiene_lit_comp(self):
        """Devuelve True si existe una literal y su negación en el mismo nodo."""
        lits = [c[1] for c in self.literales]
        l_pos = [l for l in lits if '-' not in l]
        l_negs = [l[1:] for l in lits if '-' in l]
        return len(set(l_pos).intersection(set(l_negs))) > 0

    def es_hoja(self):
        """Determina el estado de la hoja.

        Retorna:
            'cerrada' si hay contradicción,
            'abierta' si no hay reglas por aplicar y no hay contradicción,
            'expandir' si hay fórmulas alfa/beta pendientes.
        """
        if self.tiene_lit_comp():
            return 'cerrada'
        elif ((len(self.alfas) == 0) and (len(self.betas) == 0)):
            return 'abierta'
        else:
            return None

    def interp(self):
        """Extrae y devuelve la interpretación (dict letra->bool) de la hoja abierta."""
        I = {}
        for lit in self.literales:
            l = lit[1]
            if '-' not in l:
                I[l] = True
            else:
                I[l[1:]] = False
        return I

    def expandir(self):
        """Aplica la siguiente regla (alfa o beta) y retorna nodos hijos resultantes."""
        '''Escoge última alfa, si no última beta, si no None'''
        f_alfas = deepcopy(self.alfas)
        f_betas = deepcopy(self.betas)
        f_literales = deepcopy(self.literales)
        if len(self.alfas) > 0:
            f, s, num_regla, cl = f_alfas.pop(0)
            if num_regla == 1:
                formulas = [f.subf.subf]
            elif num_regla == 2:
                formulas = [f.left, f.right]
            elif num_regla == 3:
                formulas = [Negacion(f.subf.left), Negacion(f.subf.right)]
            elif num_regla == 4:
                formulas = [f.subf.left, Negacion(f.subf.right)]
            for nueva_f in formulas:
                clasf = nueva_f.clasifica_para_tableaux()
                if clasf[1]== 'alfa':
                    lista = f_alfas
                elif clasf[1]== 'beta':
                    lista = f_betas
                elif clasf[1]== 'literal':
                    lista = f_literales
                strs = [c[1] for c in lista]
                if str(nueva_f) not in strs:
                    lista.append((nueva_f, str(nueva_f), *clasf))
            nuevo_nodo = nodos_tableaux([])
            nuevo_nodo.alfas = f_alfas
            nuevo_nodo.betas = f_betas
            nuevo_nodo.literales = f_literales
            return [nuevo_nodo]
        elif len(self.betas) > 0:
            f, s, num_regla, cl = f_betas.pop(0)
            if num_regla == 1:
                B1 = Negacion(f.subf.left)
                B2 = Negacion(f.subf.right)
            elif num_regla == 2:
                B1 = f.left
                B2 = f.right
            elif num_regla == 3:
                B1 = Negacion(f.left)
                B2 = f.right
            f_alfas2 = deepcopy(f_alfas)
            f_betas2 = deepcopy(f_betas)
            f_literales2 = deepcopy(f_literales)
            clasf = B1.clasifica_para_tableaux()
            if clasf[1]== 'alfa':
                lista = f_alfas
            elif clasf[1]== 'beta':
                lista = f_betas
            elif clasf[1]== 'literal':
                lista = f_literales
            strs = [c[1] for c in lista]
            if str(B1) not in strs:
                lista.append((B1, str(B1), *clasf))
            clasf = B2.clasifica_para_tableaux()
            if clasf[1]== 'alfa':
                lista = f_alfas2
            elif clasf[1]== 'beta':
                lista = f_betas2
            elif clasf[1]== 'literal':
                lista = f_literales2
            strs = [c[1] for c in lista]
            if str(B2) not in strs:
                lista.append((B2, str(B2), *clasf))
            n1 = nodos_tableaux([])
            n1.alfas = f_alfas
            n1.betas = f_betas
            n1.literales = f_literales
            n2 = nodos_tableaux([])
            n2.alfas = f_alfas2
            n2.betas = f_betas2
            n2.literales = f_literales2
            return [n1, n2]
        else:
            return []

# ============================================================================
# FUNCIONES DEL NOTEBOOK (Logica_Tseitin)
# ============================================================================

def a_clausal(A):
    """Subrutina de Tseitin para encontrar la FNC de la formula en la pila.
    
    Input: A (cadena) de la forma
                      p=-q
                      p=(qYr)
                      p=(qOr)
                      p=(q>r)
                      p=(q=r)
    Output: B (lista de cláusulas), equivalente en FNC
    """
    # Normalizar A a cadena si viene como lista/tupla anidada
    def _to_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            return ''.join(_to_str(e) for e in x)
        return str(x)
    A = _to_str(A)

    if not isinstance(A, str) or len(A) < 4:
        raise AssertionError(u"Fórmula incorrecta o tokenizada: {!r}".format(A))

    B = ''
    p = A[0]
    if "-" in A:
        q = A[-1]
        B = "-" + p + "O-" + q + "Y" + p + "O" + q
    elif "Y" in A:
        q = A[3]
        r = A[5]
        B = q + "O-" + p + "Y" + r + "O-" + p + "Y-" + q + "O-" + r + "O" + p
    elif "O" in A:
        q = A[3]
        r = A[5]
        B = "-" + q + "O" + p + "Y-" + r + "O" + p + "Y" + q + "O" + r + "O-" + p
    elif ">" in A:
        q = A[3]
        r = A[5]
        B = q + "O" + p + "Y-" + r + "O" + p + "Y-" + q + "O" + r + "O-" + p
    elif "=" in A:
        q = A[3]
        r = A[5]
        B = q + "O-" + r + "O-" + p + "Y-" + q + "O" + r + "O-" + p + "Y-" + q + "O-" + r + "O" + p + "Y" + q + "O" + r + "O" + p
    else:
        raise AssertionError(u'Error en a_clausal(): Fórmula incorrecta!')

    B = B.split('Y')
    B = [c.split('O') for c in B]
    return B


def tseitin(A):
    """Algoritmo de transformación de Tseitin.
    
    Input: A (cadena) en notación inorder
    Output: B (lista de cláusulas), Tseitin en FNC
    """
    # Creamos letras proposicionales nuevas
    f = inorder_to_tree(A)
    letrasp = f.letras()
    cods_letras = [ord(x) for x in letrasp]
    m = max(cods_letras) + 256
    letrasp_tseitin = [chr(x) for x in range(m, m + f.num_conec())]
    letrasp = list(letrasp) + letrasp_tseitin
    L = []  # Inicializamos lista de conjunciones
    Pila = []  # Inicializamos pila
    i = -1  # Inicializamos contador de variables nuevas
    s = A[0]  # Inicializamos símbolo de trabajo
    while len(A) > 0:  # Recorremos la cadena
        if (s in letrasp) and (len(Pila) > 0) and (Pila[-1] == '-'):
            i += 1
            atomo = letrasp_tseitin[i]
            Pila = Pila[:-1]
            Pila.append(atomo)
            L.append(atomo + "=-" + s)
            A = A[1:]
            if len(A) > 0:
                s = A[0]
        elif s == ')':
            w = Pila[-1]
            O = Pila[-2]
            v = Pila[-3]
            Pila = Pila[:len(Pila)-4]
            i += 1
            atomo = letrasp_tseitin[i]
            L.append(atomo + "=(" + v + O + w + ")")
            s = atomo
        else:
            Pila.append(s)
            A = A[1:]
            if len(A) > 0:
                s = A[0]
    if i < 0:
        atomo = Pila[-1]
    else:
        atomo = letrasp_tseitin[i]
    B = [[atomo]] + [a_clausal(x) for x in L]
    B = [val for sublist in B for val in sublist]
    return B


def complemento(l):
    """Retorna el complemento de un literal."""
    if '-' in l:
        return l[1]
    else:
        return '-' + l


def eliminar_literal(S, l):
    """Elimina un literal de un conjunto de cláusulas."""
    S1 = [c for c in S if l not in c]
    lc = complemento(l)
    return [[p for p in c if p != lc] for c in S1]


def extender_I(I, l):
    """Extiende una interpretación con un literal."""
    I1 = {k: I[k] for k in I if k != l}
    if '-' in l:
        I1[l[1:]] = False
    else:
        I1[l] = True
    return I1


def unit_propagate(S, I):
    """Algoritmo para eliminar clausulas unitarias de un conjunto de clausulas.
    
    Input: 
        - S, conjunto de clausulas
        - I, interpretacion (diccionario {literal: True/False})
    Output: 
        - S, conjunto de clausulas
        - I, interpretacion (diccionario {literal: True/False})
    """
    while [] not in S:
        l = ''
        for x in S:
            if len(x) == 1:
                l = x[0]
                S = eliminar_literal(S, l)
                I = extender_I(I, l)
                break
        if l == '':  # Se recorrió todo S y no se encontró unidad
            break
    return S, I


from random import choice

def dpll(S, I):
    """Algoritmo para verificar la satisfacibilidad de una formula.
    
    Input: 
        - S, conjunto de clausulas
        - I, interpretacion (diccionario literal->True/False)
    Output: 
        - String, "Satisfacible"/"Insatisfacible"
        - I, interpretacion (diccionario literal->True/False)
    """
    S, I = unit_propagate(S, I)
    if len(S) == 0:
        return "Satisfacible", I
    if [] in S:
        return "Insatisfacible", {}
    l = choice(choice(S))
    lc = complemento(l)
    newS = eliminar_literal(S, l)
    newI = extender_I(I, l)
    sat, newI = dpll(newS, newI)
    if sat == "Satisfacible":
        return sat, newI
    else:
        newS = eliminar_literal(S, lc)
        newI = extender_I(I, lc)
        return dpll(newS, newI)

#------------------------------------------------------------

def complemento(l):
    if '-' in l:
        return l[1:]
    else:
        return '-' + l

def interpretacion_aleatoria(letrasp):
    I = {p:randint(0,1)==1 for p in letrasp}
    return I

def flip_literal(I, l):
    p = l[-1]
    valor = False if I[p] else True
    Ip = deepcopy(I)
    Ip[p] = valor
    return Ip
class WalkSatEstado():
    
    def __init__(self, S):
        self.S = S
        self.letrasp = list(set([l[-1] for C in self.S for l in C]))
        self.I = interpretacion_aleatoria(self.letrasp)
        self.I_lits = set([p for p in self.letrasp if self.I[p]] + ['-'+p for p in self.letrasp if not self.I[p]])
        self.clausulas_sat = [C for C in self.S if any((True for x in self.I_lits if x in C))]
        self.clausulas_unsat = [C for C in self.S if C not in self.clausulas_sat]

    def actualizar(self, I):
        self.I = I
        self.I_lits = set([p for p in self.letrasp if self.I[p]] + ['-'+p for p in self.letrasp if not self.I[p]])
        self.clausulas_sat = [C for C in self.S if any((True for x in self.I_lits if x in C))]
        self.clausulas_unsat = [C for C in self.S if C not in self.clausulas_sat]

    def SAT(self):
        return len(self.clausulas_unsat) == 0

    def break_count(self, l):
        if l in self.I_lits:
            lit = l
        else:
            lit = complemento(l)
        clausulas_break_count = [C for C in self.clausulas_sat if set(C).intersection(self.I_lits)=={lit}]
        return len(clausulas_break_count)
    
def walkSAT(S, max_flips=10000, max_tries=10000, p=.8):

    w = WalkSatEstado(S)
    for i in range(max_tries):
        w.actualizar(interpretacion_aleatoria(w.letrasp))
        for j in range(max_flips):
            if w.SAT():
                return 'Satisfacible', w.I
            C = choice(w.clausulas_unsat)
            breaks = sorted([(l,w.break_count(l)) for l in C], key=lambda x: x[1])
            min_breaks = breaks[0]
            if min_breaks[1] == 0:
                v = min_breaks[0]
            else:
                if uniform(0,1) < p:
                    assert(len(C)>0), f"{C}"
                    v = choice(C)
                else:
                    v = min_breaks[0]
            I = flip_literal(w.I, v)
            w.actualizar(I)
    return None, {}
# ...existing code...

def SATtableaux(A):
    """
    Wrapper de conveniencia: acepta A como cadena (notación inorder) o como objeto Formula
    y devuelve la interpretación (o None) usando el método SATtableaux de la fórmula.
    Uso en notebooks: from Logica import SATtableaux
    """
    # si ya es un objeto Formula, usarlo; si es string, parsear
    if isinstance(A, Formula):
        f = A
    else:
        f = inorder_to_tree(str(A))
    return f.SATtableaux()
# ...existing code...