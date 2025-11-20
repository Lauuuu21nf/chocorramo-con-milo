from itertools import combinations
from Logica import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import MethodType

def escribir_rejilla(self, k):
    n, x, y = self.unravel(k)
    return f"{n+1} en ({x},{y})"

class Rejilla:
    def __init__(self, X=3, Y=3):
        self.X = X
        self.Y = Y
        self.CM = Descriptor([X*Y, X, Y])
        self.CM.escribir = MethodType(escribir_rejilla, self.CM)
        r1 = self.regla1()
        r2 = self.regla2()
        r3 = self.regla3()
        r4 = self.regla4()
        r5 = self.regla5()
        r6_1 = self.regla6_1()
        r6_2 = self.regla6_2()
        self.reglas = [r1, r2, r3, r4, r5, r6_1, r6_2]

    def regla1(self):  # Cada casilla tiene AL MENOS un número
        casillas = [(x,y) for x in range(self.X) for y in range(self.Y)]
        lista = []
        for c in casillas:
            lista_o = []
            for n in range(self.X*self.Y):
                lista_o.append(self.CM.ravel([n,*c]))
            lista.append(Otoria(lista_o))
        return Ytoria(lista)

    def regla2(self):  # Cada celda tiene A LO SUMO un número
        casillas = [(x,y) for x in range(self.X) for y in range(self.Y)]
        lista = []
        for c in casillas:
            for n1, n2 in combinations(range(self.X*self.Y), 2):
                atom1 = self.CM.ravel([n1, *c])
                atom2 = self.CM.ravel([n2, *c])
                lista.append(f"(-{atom1}O-{atom2})")
        return Ytoria(lista)
        
    def regla3(self):  # Cada número aparece exactamente una vez
        lista = []
        for n in range(self.X * self.Y):
            # Al menos una aparición
            disyuncion = []
            for x in range(self.X):
                for y in range(self.Y):
                    disyuncion.append(self.CM.ravel([n, x, y]))
            lista.append(Otoria(disyuncion))
            
            # A lo sumo una aparición
            for x1 in range(self.X):
                for y1 in range(self.Y):
                    for x2 in range(self.X):
                        for y2 in range(self.Y):
                            if (x1, y1) != (x2, y2):
                                atomo1 = self.CM.ravel([n, x1, y1])
                                atomo2 = self.CM.ravel([n, x2, y2])
                                lista.append(f"(-{atomo1}O-{atomo2})")
        
        return Ytoria(lista)

    def regla4(self):  # Suma de filas = 15 
        if self.X != 3 or self.Y != 3:
            return Ytoria([])
        
        filas = []
        for y in range(3):
            combinaciones_si = []
            for n1 in range(9):
                for n2 in range(9):
                    if n2 == n1: continue
                    n3_numero = 15 - (n1+1) - (n2+1)
                    n3 = n3_numero - 1
                    if 0 <= n3 < 9 and n3 != n1 and n3 != n2:
                        form = Ytoria([
                            self.CM.ravel([n1, 0, y]),
                            self.CM.ravel([n2, 1, y]),
                            self.CM.ravel([n3, 2, y])
                        ])
                        combinaciones_si.append(form)
            if combinaciones_si:
                filas.append(Otoria(combinaciones_si))

        return Ytoria(filas)

    def regla5(self):  # Suma de columnas = 15 
        if self.X != 3 or self.Y != 3:
            return Ytoria([])
        
        columnas = []
        for x in range(3):
            combinaciones_si = []
            for n1 in range(9):
                for n2 in range(9):
                    if n2 == n1: continue
                    n3_numero = 15 - (n1+1) - (n2+1)
                    n3 = n3_numero - 1
                    if 0 <= n3 < 9 and n3 != n1 and n3 != n2:
                        form = Ytoria([
                            self.CM.ravel([n1, x, 0]),
                            self.CM.ravel([n2, x, 1]),
                            self.CM.ravel([n3, x, 2])
                        ])
                        combinaciones_si.append(form)
            if combinaciones_si:
                columnas.append(Otoria(combinaciones_si))

        return Ytoria(columnas)

    def regla6_1(self):  # Diagonal principal = 15
        if self.X != 3 or self.Y != 3:
            return Otoria([])
        
        diagonalX = []
        for n1 in range(9):
            for n2 in range(9):
                if n2 == n1: continue
                n3_numero = 15 - (n1+1) - (n2+1)
                n3 = n3_numero - 1
                
                if 0 <= n3 < 9 and n3 != n1 and n3 != n2:
                    form = Ytoria([
                        self.CM.ravel([n1, 0, 0]),
                        self.CM.ravel([n2, 1, 1]),
                        self.CM.ravel([n3, 2, 2])
                    ])
                    diagonalX.append(form)
        return Otoria(diagonalX)

    def regla6_2(self):  # Diagonal secundaria = 15
        if self.X != 3 or self.Y != 3:
            return Otoria([])
        
        diagonalY = []
        for n1 in range(9):
            for n2 in range(9):
                if n2 == n1: continue
                n3_numero = 15 - (n1+1) - (n2+1)
                n3 = n3_numero - 1
                
                if 0 <= n3 < 9 and n3 != n1 and n3 != n2:
                    form = Ytoria([
                        self.CM.ravel([n1, 2, 0]),
                        self.CM.ravel([n2, 1, 1]),
                        self.CM.ravel([n3, 0, 2])
                    ])
                    diagonalY.append(form)
        return Otoria(diagonalY)

    def visualizar(self, I):
        fig, axes = plt.subplots()
        fig.set_size_inches(self.X, self.Y)
        fig.patch.set_facecolor('#ADD8E6')  
        
        step_x = 1. / self.X
        step_y = 1. / self.Y
        offset = 0.001
        
        tangulos = []
        tangulos.append(patches.Rectangle((0, 0), 1, 1, facecolor='#ADD8E6', edgecolor='black', linewidth=2))
        
        for i in range(self.X + 1): 
            for j in range(self.Y):
                
                if (i + j) % 2 == 0:
                    tangulos.append(patches.Rectangle((i * step_x, j * step_y), step_x - offset, step_y,facecolor='#EADDCA',edgecolor='black',linewidth=2))
        
        for t in tangulos:
            axes.add_patch(t)

        for k in I:
            if I[k]:
                try:
                    atomo = k[1:] if k.startswith('-') else k
                    n, X1, Y1 = self.CM.unravel(atomo)
                    if I[k] and 0 <= X1 < self.X and 0 <= Y1 < self.Y:
                        axes.text(X1 * step_x + step_x / 2, Y1 * step_y + step_y / 2, n+1,ha="center", va="center", size=30, color='black',  weight='normal')
                except:
                    continue
        
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.axis('off')
        plt.show()
            
            
                