import os.path  # usado para saber si un determinado archivo existe
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import collections
import numpy as np
import pickle  # es para poder guardar y escribir listas en archivos

# dado un valor entero elavua si pertenece o no la lista de intervalos
# List example ['5-20', '40-60', '90-95']
def belongsToSomeInterval(valor, listaInt):
    respuesta = False
    valor = int(valor)
    for unIntervalo in listaInt:
        arrayAux = unIntervalo.split('-')
        if valor >= int(arrayAux[0]) and valor <= int(arrayAux[1]):
            respuesta = True
            break
    return respuesta

# dada una lista de intervalos, retorna el mayor valor contenido en alguno de ellos.
# Ejemplo de lista ['5-20', '40-60', '90-95'] --> retornaria 95
def obtenerMayorValorIntervalo(listaInt):
    respuesta = 0
    for unIntervalo in listaInt:
        arrayAux = unIntervalo.split('-')
        if int(arrayAux[1]) > respuesta:
            respuesta = int(arrayAux[1])
    return respuesta

# dad un archivo, genera una lista con los elementos de la misma
def generarListaDadoArhivo(fileEntrada):
    listaSalida = []
    for unaLinea in fileEntrada:
        listaSalida += [(unaLinea.split('\t'))[0]]
    return listaSalida

# retorna verdadero o falso si el elemento existe en la lista
# El parámentro 'procesarDeVerdad', cuando es False retorna True sin evaluar de verdad, caso contrario evalua de verdad
def existeElementoEnLista(elem, listaEntrada, procesarDeVerdad):
    respuesta = False
    if procesarDeVerdad:
        respuesta = elem in listaEntrada
    else:
        respuesta = True
    return respuesta

# dada las dos listas, de la lista de target se cuentan la cantidad de '4' que tiene, y se seleccciona igual cantidad de targets que tienen '0'de manera aleatoria
# y retorna la lista de target con '4' y '0' en igual cantidades y la lista de features acorde a dicha elección
def equilibrarTargets(listaFeatures, listaTargets):
    listaFeaturesSalida = []
    listaTargetsSalida = []

    listaPosiciones4 = [i for i, e in enumerate(listaTargets) if e == 4] # contiene una lista de posiciones que contienen el valor 4 en listaTargets.
    listaPosicionesNo4 = [i for i, e in enumerate(listaTargets) if e != 4] # contiene una lista de posiciones que NO contienen el valor 4 en listaTargets.
    # print("Posiciones donde hay 4's: ", listaPosiciones4)
    # print("Posiciones donde NO hay 4's: ", listaPosicionesNo4)

    # se equilibran las cantidades de 4's y no 4's
    # se hace esta evaluación porque no es correcto asumir que siempre va a ser menor la cantidad de 4's.
    if len(listaPosiciones4) < len(listaPosicionesNo4):
        # hay más 0's que 4's. Entonces se corta la lista de NO 4's para que sea igual que la cantidad de 4's.
        random.shuffle(listaPosicionesNo4)  # se mezcla la lista de posiciones en valores aleatorios: NOTA: la mezcla aleatoria se hace sobre la lista más larga que es la que se corta
        # print("Posiciones donde NO hay 4's RANDOM: ", listaPosicionesNo4)
        listaPosicionesNo4 = listaPosicionesNo4[:len(listaPosiciones4)] # se corta la lista más larga para que contengan igual cantidad de elementos.
    else:
        # hay más 4's que 0's (o igual cantidad). Entonces se corta la lista de 4's para que sea igual que la cantidad de NO 4's.
        random.shuffle(listaPosiciones4)  # se mezcla la lista de posiciones en valores aleatorios: NOTA: la mezcla aleatoria se hace sobre la lista más larga que es la que se corta
        # print("Posiciones donde hay 4's RANDOM: ", listaPosiciones4)
        listaPosiciones4 = listaPosiciones4[:len(listaPosicionesNo4)] # se corta la lista más larga para que contengan igual cantidad de elementos.

    listaTargetsSalida = [listaTargets[i] for i in listaPosiciones4]
    listaTargetsSalida += [listaTargets[i] for i in listaPosicionesNo4] # se eligen igual cantidades de 0's de las posiciones mezcladas aleatoriamente
        # NOTA: en realidad no tiene mucho sentido elegir posiciones de 0's para sumarlos a la lista de targes de salida. Tienen sentido para elegir las pociones de las Features

    # se guarda en la lista de FEATURES de salida los features acorte a las posiciones guardadas en la lista de targets para no perder la correspondencia
    listaFeaturesSalida = [listaFeatures[i] for i in listaPosiciones4]
    listaFeaturesSalida += [listaFeatures[i] for i in listaPosicionesNo4]

    # # se mezclan las posiciones de las listas manteniendo la correlación entre ambas.
    # listaPosicionesAmbosValores = listaPosiciones4 + listaPosicionesNo4
    # print(">>> Listas juntas: ", listaPosicionesAmbosValores)
    return listaFeaturesSalida, listaTargetsSalida


# dada una lista de entrada de features completa (con todas las features), se avalúa  si
# cada feature se queda o se va de salida en cada sublista, según está especificado
# en DICT_POSICIONES_FEATURES y retorna dicha lista sobrante
def activarDesactivarFeatures(listaFeaturesFull, DICT_POSICIONES_FEATURES): # OK!
    listaSalidaFeaturesActivas = None

    # se recorre el diccionario que indica si cada posición va o no
    listaDecisiones = [] # será una lista de los valores true o False del diccionario
    for unIndicador in DICT_POSICIONES_FEATURES.items():
        indiceAux = (unIndicador[0].split('-'))[0]
        decision = unIndicador[1]
        print("Valor: ", (unIndicador[0].split('-'))[0], " - ", unIndicador[1])
        listaDecisiones += [decision]

    listaPosicionesAEliminar = [i for i, e in enumerate(listaDecisiones) if e == False] # Se obtiene una lista de posiciones a eliminar
    print("Posiciones a eliminar: ", listaPosicionesAEliminar)

    # se eliminan de las sublistas de la lista de entrada aquellas posiciones donde hay un False en el diccionario
    listaSalidaFeaturesActivas = [[y for i,y in enumerate(x) if i not in listaPosicionesAEliminar]for x in listaFeaturesFull]
    return listaSalidaFeaturesActivas


# dada las listas, retorna aquellas features y target cuyos targets tienen (o no) el valor determinado 'valorTargets'
# Si condicionTarget = True entonces se eligen features y targets que tienen un determinado valorTarget.
# Si condicionTarget = False son features y targets que NO son del determinado target (es decir, son distintos).
def obtenerFeaturesYTargetsSegunValorTarget(listaFeatures, listaTarget, valorTarget, condicionTarget): # OK!
    # se obtienen las posiciones de la lista target que tienen el valor indicado
    listaPosicionesConValorTarget = []
    if condicionTarget:
        listaPosicionesConValorTarget = [i for i, e in enumerate(listaTarget) if e == valorTarget]
    else:
        listaPosicionesConValorTarget = [i for i, e in enumerate(listaTarget) if e != valorTarget]
    return [listaFeatures[index] for index in listaPosicionesConValorTarget], [listaTarget[index] for index in listaPosicionesConValorTarget]


# 'separador' es el character que se utiliza para dividir cada elemento en la lista
def concatenarListaEnString(lista, separador):
    cadenaSalida = ""
    # se evalua si la lista de verdad lo es
    if isinstance(lista, collections.Iterable):
        for unElemento in lista:
            cadenaSalida += separador + str(unElemento)
        cadenaSalida = cadenaSalida[len(separador):] # se corta el primer elemento porque es un separador innecesario
                                                     # se calcula la len del separador porque no siempre es un caracter
    else:
        cadenaSalida = str(lista)
    return  cadenaSalida


# en la lista de entrada , reemplaza todas las apariciones de valorABuscar por valorNuevo.
# igualAlViejo: Si es true se compara por igual, False se compara por distinto
def reemplazarValoresEnLista(listaEntrada, valorABuscar, valorNuevo, buscarPorIgual): # OK
    listaRespuesta = []
    if buscarPorIgual:
        listaRespuesta = [x if x != valorABuscar else valorNuevo for x in listaEntrada]
    else:
        listaRespuesta = [x if x == valorABuscar else valorNuevo for x in listaEntrada]
    return listaRespuesta


def convertirAStringClavesYValoresDeDiccionario(diccionario, separadorClaves, separadorValores): # OK!
    stringClaves = ""
    stringValores = ""
    for unaClave in diccionario:
        stringClaves += str(separadorClaves) + str(unaClave)
        stringValores += str(separadorValores) + str(diccionario[unaClave])
    # se sacan los separadores que quedaron al principio de cada cadena
    # se una len() porque el separador puede ser cualquier cantidad de caracteres
    stringClaves = stringClaves[len(separadorClaves):]
    stringValores = stringValores[len(separadorValores):]
    return stringClaves, stringValores


# dada la lista de entrada, cuenta la cantidad de elementos de cada sublista
# y luego calcula el promedio de cantidad de elementos de las sublistas
# devolverString si es True formatea como string lo que devuelte, caso contrario retorna dos valores numéricos (promedio, cantidadElementos)
def obtenerPromedioCantidadElementosSublistas(listaEntrada, devolverString):
    cantidadElementos = 0
    for unElemento in listaEntrada:
        cantidadElementos += len(unElemento)
    promedio = cantidadElementos/len(listaEntrada)
    if int(promedio) == promedio:
        promedio = int(promedio)

    if devolverString:
        stringPromedioYcantElem = str(promedio) + "(" + str(cantidadElementos) + "/" + str(len(listaEntrada)) + ")"
        return stringPromedioYcantElem
    else:
        return promedio, cantidadElementos


# se calculan los TP, TN, FP, FN >> Info vista en http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# devolverString cuando es True la respuesta es una salida formateada como string
def calcularTPTNFPFN(listaPredicciones, listaTargets, devolverString):
    [tn, fp, fn, tp] = confusion_matrix(listaPredicciones, listaTargets).ravel()  # parámetros: verdaderosTargets, targetsPredichos
    if devolverString:
        stringTN = str(tn) + "(" + str(round(tn / len(listaPredicciones), 2)) + "%)"
        stringFP = str(fp) + "(" + str(round(fp / len(listaPredicciones), 2)) + "%)"
        stringFN = str(fn) + "(" + str(round(fn / len(listaPredicciones), 2)) + "%)"
        stringTP = str(tp) + "(" + str(round(tp / len(listaPredicciones), 2)) + "%)"
        return  stringTN, stringFP, stringFN, stringTP
    else:
        return tn, fp, fn, tp

def calcularAccPrecRecF1(listaTargets, listaPredicciones, convertirA1sYmenos1, valorInlier):
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if convertirA1sYmenos1:
        # antes de hacer cáclulo de las métricas hay que convertir a 1s y -1s de listaTargets
        # 1: inliers, -1: outliers
        #  train_target, primero se convierten 4 en 1, y luego 0 en -1.
        listaTargets = reemplazarValoresEnLista(listaTargets, valorABuscar=valorInlier,
                                                valorNuevo=1, buscarPorIgual=True)  # Cambia valorInlier (4's) por 1 porque son los inliers
        listaTargets = reemplazarValoresEnLista(listaTargets, valorABuscar=1,
                                                valorNuevo=-1, buscarPorIgual=False)  # Cambia (no 4's) por -1 porque son los outliers

    # print(">>>listaTargets ", len(listaTargets), "\n", listaTargets, "\nlistaPredicciones ", len(listaPredicciones), "\n", listaPredicciones)
    accuracy = metrics.accuracy_score(listaTargets, listaPredicciones)
    precision = metrics.precision_score(listaTargets, listaPredicciones)
    recall = metrics.recall_score(listaTargets, listaPredicciones)
    f1 = metrics.f1_score(listaTargets, listaPredicciones)

    return accuracy, precision, recall, f1

# devuelve algo así
#  [{'kernel': 'rbf', 'gamma': 0.1}, {'kernel': 'rbf', 'gamma': 0.2}, {'kernel': 'rbf', 'gamma': 0.3}]
def mezclarKernelsYGammas(kernel, listaGammas):
    listaSalida = []
    for unGamma in listaGammas:
        listaSalida += [{'kernel':str(kernel), 'gamma':unGamma}]
    return  listaSalida


def mezclarListas(nombreLista1, lista1, nombreLista2, lista2):
    listaSalida = []
    for unValor1 in lista1:
        if not(isinstance(unValor1, bool)):
            unValor1 = str(unValor1)
        for unValor2 in lista2:
            if not(isinstance(unValor2, bool)):
                unValor2 = str(unValor2)
            listaSalida += [{nombreLista1:unValor1, nombreLista2:unValor2}]
    return  listaSalida


# genera un diccionario de archivos abiertos para todos los clasificadores en la lista de entrada
def generarArchivosPorCadaStringEnLista(listaEntrada, pathCarpeta, intervalo, valorK, stringEncabezado):
    dictDeArchivos = {}
    for unClasif in listaEntrada: # salidaClasificadoresIndiaRESUMEN-12hour,k4
        auxNombreArchivoConPath = pathCarpeta+"salidaClasificadoresIndiaRESUMEN-"+str(intervalo)+",k"+str(valorK)+"-"+str(unClasif)+".txt"
        dictDeArchivos[str(unClasif)] = open(auxNombreArchivoConPath, "a")
        #se evalua si está vacio el archivo para agregar los encabezados
        if os.path.getsize(auxNombreArchivoConPath) == 0:
            dictDeArchivos[str(unClasif)].write(stringEncabezado)
    return dictDeArchivos