## Table of Contents
- [Background](#background)
- [The *World* y *Cat* implementaciones de jugadores](#world)
- [el *Mouse* jugador](#mouse)
  - [Implementación de Q-learning](#q-learning)
- [Uso](#reproduce)

<div id='background'/>

### Background

Q-Learning es un algoritmo fuera de política (puede actualizar las funciones de valor estimado utilizando acciones hipotéticas, aquellas que en realidad no se han probado) para el aprendizaje de diferencias temporales (método para estimar funciones de valor). Se puede demostrar que, con una formación suficiente, el Q-learning converge con probabilidad 1 a una aproximación cercana de la función de valor de acción para una política objetivo arbitraria. Q-Learning aprende la política óptima incluso cuando las acciones se seleccionan de acuerdo con una política más exploratoria o incluso aleatoria. Q-learning se puede implementar de la siguiente manera:

```
Initialize Q(s,a) arbitrarily
Repeat (for each generation):
	Initialize state s
	While (s is not a terminal state):		
		Choose a from s using policy derived from Q
		Take action a, observe r, s'
		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
		s = s'
```

donde:

s: es el estado anterior.
a: es la acción anterior.
Q(): es el algoritmo Q-learning.
s': es el estado actual.
alpha: es la tasa de aprendizaje, generalmente establecida entre 0 y 1. Establecerla en 0 significa que los valores de Q nunca se actualizan, por lo tanto, no se aprende nada. Establecer alpha en un valor alto como 0.9 significa que el aprendizaje puede ocurrir rápidamente.
gamma: es el factor de descuento, también establecido entre 0 y 1. Esto modela el hecho de que las recompensas futuras valen menos que las recompensas inmediatas.
máx,: es la recompensa máxima que se puede obtener en el estado siguiente al actual (la recompensa por tomar la acción óptima después de eso).
El algoritmo se puede interpretar como:


1.Inicializar la tabla de valores de Q, Q(s, a).
2.Observar el estado actual, s.
3.Elegir una acción, a, para ese estado basándose en la política de selección.
4.Tomar la acción y observar la recompensa, r, así como el nuevo estado, s'.
5.Actualizar el valor de Q para el estado utilizando la recompensa observada y la recompensa máxima posible para el próximo estado.
6.Establecer el estado al nuevo estado y repetir el proceso hasta que se alcance un estado terminal.

<div id='world'/>

### The *World* and *Cat* implementaciones de jugadores

Las implementaciones del mundo 2D discreto (incluidos agentes, células y otras abstracciones), así como los jugadores del gato y el ratón, se realizan en el archivo "celular.py". El mundo se genera a partir de un archivo `.txt`. En particular, estoy usando el `worlds/waco.txt`:

```
(waco world)

XXXXXXXXXXXXXX
X            X
X XXX X   XX X
X  X  XX XXX X
X XX      X  X
X    X  X    X
X X XXX X XXXX
X X  X  X    X
X XX   XXX  XX
X    X       X
XXXXXXXXXXXXXX

```

The *Cat* clase de jugador hereda de `cellular.Agent` y su implementación seguirá al *Mouse* jugdor:

```python

    def goTowards(self, target):
        if self.cell == target:
            return
        best = None
        for n in self.cell.neighbours:
            if n == target:
                best = target
                break
            dist = (n.x - target.x) ** 2 + (n.y - target.y) ** 2
            if best is None or bestDist > dist:
                best = n
                bestDist = dist
        if best is not None:
            if getattr(best, 'wall', False):
                return
            self.cell = best

```
The *Cat* La jugadora calcula la distancia cuadrática (`bestDist`) Entre sus vecinas y se mueve (`self.cell = best`) a esa celda.

```python
class Cat(cellular.Agent):
    cell = None
    score = 0
    colour = 'orange'

    def update(self):
        cell = self.cell
        if cell != mouse.cell:
            self.goTowards(mouse.cell)
            while cell == self.cell:
                self.goInDirection(random.randrange(directions))

```
En general, el *Cat* persigue el *Mouse* a través de `goTowards`método calculando la distancia cuadrática. Cada vez que choca contra la pared, realiza una acción aleatoria.
<div id='mouse'/>

### The *Mouse* player

The *Mouse* La jugador contiene los siguientes atributos:
```python
class Mouse(cellular.Agent):
    colour = 'gray'

    def __init__(self):
        self.ai = None
        self.ai = qlearn.QLearn(actions=range(directions),
                                alpha=0.1, gamma=0.9, epsilon=0.1)
        self.eaten = 0
        self.fed = 0
        self.lastState = None
        self.lastAction = None

```
los `eaten` y `fed` Los atributos almacenan el rendimiento del jugador mientras que el `lastState` y `lastAction` les ayudan a el *Mouse* El reproductor almacena información sobre sus estados anteriores (luego se usa para aprender).

la `ai` El atributo almacena la implementación de Q-learning que se inicializa con los siguientes parámetros:
- **directions**: Hay diferentes direcciones/acciones posibles implementadas en el `getPointInDirection` Metodo de *World* class:

```python
    def getPointInDirection(self, x, y, dir):
        if self.directions == 8:
            dx, dy = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)][dir]
        elif self.directions == 4:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][dir]
        elif self.directions == 6:
            if y % 2 == 0:
                dx, dy = [(1, 0), (0, 1), (-1, 1), (-1, 0),
                          (-1, -1), (0, -1)][dir]
            else:
                dx, dy = [(1, 0), (1, 1), (0, 1), (-1, 0),
                          (0, -1), (1, -1)][dir]


```
En general, esta implementación se utilizará en **8 direcciones**, por lo que `(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]`.
- **alpha**: cuál es la constante de descuento de q-learning, establecida en `0.1`.
- **gamma**: el factor de descuento de q-learning, establecido en `0.9`.
- **epsilon**: una constante de exploración para aleatorizar las decisiones, establecida en `0.1`.

El *Mouse* El jugador calcula el siguiente estado usando el`calcState()` método implementado de la siguiente manera:
```python

    def calcState(self):
        def cellvalue(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and
                                         cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and
                                              cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        return tuple([cellvalue(self.world.getWrappedCell(self.cell.x + j, self.cell.y + i))
                      for i,j in lookcells])

```
Esto, en pocas palabras, devuelve una tupla de los valores de las celdas que rodean el valor actual *Mouse* como sigue:
- `3`: Si el *Cat* esta en esa celda
- `2`: Si el *Cheese* esta en esa celda
- `1`: Si la celda es una pared
- `0`: otros

La búsqueda se realiza de acuerdo con el `lookdist`variable que en esta implementación usa un valor de `2` (en otras palabras, el mouse puede "ver" hasta dos celdas adelante en cada dirección).

Para terminar repasando el *Mouse* implementación, veamos cómo se implementa Q-learning:

<div id='q-learning'/>

##Implementación de Q-learning
Miremos el `update` método de el *Mouse* jugador:

```python
    def update(self):
        # calculate the state of the surrounding cells
        state = self.calcState()
        # asign a reward of -1 by default
        reward = -1

        # Update the Q-value
        if self.cell == cat.cell:
            self.eaten += 1
            reward = -100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None

            self.cell = pickRandomLocation()
            return

        if self.cell == cheese.cell:
            self.fed += 1
            reward = 50
            cheese.cell = pickRandomLocation()

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        # Choose a new action and execute it
        state = self.calcState()
        action = self.ai.chooseAction(state)
        self.lastState = state
        self.lastAction = action

        self.goInDirection(action)
```

Se ha comentado el código para simplificar su comprensión. La implementación coincide con el pseudocódigo presentado en el [Background](#background) sección anterior (tenga en cuenta que por motivos de implementación, las acciones en la implementación de "Python" se han reordenado).

Las recompensas se otorgan con estos términos:
- `-100`: Si el *Cat* jugador come al *Mouse*
- `50`: Si el *Mouse* jugador come al cheese
- `-1`: otros

El algoritmo de aprendizaje registra cada combinación de estado/acción/recompensa en un diccionario que contiene una tupla (estado, acción) en la clave y la recompensa como el valor de cada miembro.

Tenga en cuenta que la cantidad de elementos guardados en el diccionario para este entorno 2D simplificado es considerable después de algunas generaciones. Para obtener una idea de este hecho, considere los siguientes número
-Después de **10 000** generaciones:
	  - 2430 elementos (combinaciones de estado/acción/recompensa) aprendidos
	  - Bytes: 196888 (192 KB)
- Después de **100 000** generaciones:
	- 5631 elementos (combinaciones de estado/acción/recompensa) aprendidos
	- Bytes: 786712 (768 KB)
- Después de **600 000** generaciones:
	- 9514 elementos (combinaciones de estado/acción/recompensa) aprendidos
	- Bytes: 786712 (768 KB)
- Después de **1.000.000** de generaciones:
	- 10440 elementos (combinaciones de estado/acción/recompensa) aprendidos
	- Bytes: 786712 (768 KB)

Dados los resultados mostrados anteriormente, se puede observar que, por alguna razón, la función `sys.getsizeof` de Python parece tener un límite superior de 786712 (768 KB). No podemos proporcionar datos precisos, pero dados los resultados mostrados, se puede concluir que los elementos generados después de **1 millón de generaciones deberían requerir algo cercano a 10 MB de memoria** para este mundo simplificado en 2D.

<div id='results'/>


<div id='reproduce'/>

### Uso

python egoMouseLook.py
Linea 147 egoMouseLook 
endAge = world.age + "veces que se entrenara al iniciar"
cambiable de forma de aprendizaje
egoMouseLook 
linea 11 y 13
qlearn forma estandar de seleccion de movimientos
qlearn mod ramdom minQ = min(q); mag = max(abs(minQ), abs(maxQ)) se agrega este algoritmo para crear El "ruido aleatorio a los valores Q" se refiere a la introducción de variabilidad o perturbaciones aleatorias en los valores Q almacenados en la tabla Q de un algoritmo de aprendizaje por refuerzo, como el Q-Learning. Este ruido tiene el propósito de fomentar la exploración del espacio de acciones y evitar que el agente se quede atrapado en un conjunto de acciones subóptimas.
