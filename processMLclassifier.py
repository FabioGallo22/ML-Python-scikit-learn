""" Se leen los IDs de usuarios y después sus archivos de news Items

    PRUEBA INICIAL: leer un Id de usuario del archivo correspondiente y luego su archivo de news items generados por php
 """
import pickle  # es para poder guardar y escribir listas en archivos
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix  # para calcular los TP, TN, FP, FN
from sklearn.naive_bayes import MultinomialNB
from datetime import date
import datetime, time
from sklearn.naive_bayes import ComplementNB
import re  # para usar expresiones regulares
import os.path  # usado para saber si un detemrinado archivo existe
import warningssky
import numpy
from sklearn import metrics
import funciones
import math # para chequear que los valores de las listas para los clasificadores sean distintas de NaN  y así evitar el error "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')"

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

fileEntradaIDpublicadores = open("d:/xampp/htdocs/indiaDS/paper03/idPublicadores.txt", "r")

""" ============================================================================== """
""" ==== Inicio de las variables que hay que inicializar ========================= """
PROCESAR_CLASIFICADOR = True # *** después poner True # es es para cuando quiero hacer pruebas que no incluyen al clasificador
listaClasificadoresAutilizar = ['LogisticRegression', 'DecisionTreeClassifier', 'OneClassSVM', 'RandomForestClassifier', 'MultinomialNB', 'ComplementNB']
PROCESAR_VERDAD_ID_CANDIDATO = True # *** después poner True  # cuando esta falso se procesan todos los elementos de listaIntervalosAProcesar, si es True solamente los que estan en la lista de candidatos
LISTA_CRITERIO_SELECCION_DIF_TARGET = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] # es para seleccionar los candidatos a procesar
CRITERIO_SELECCION_DIF_TARGET = None
EQUILIBRAR_ENTRENAMIENTO = True # True # True: los features y targets de ENTRENAR de cada usuario se acortan de tal manera que el target tenga igual cantidad de 4's y 0's.
                                 # False: no se hace tal corte y se procesan todos los features y targets para ENTRENAR.
                                 # El corte se hace solamente para el prcesamiento, pero si se genera en la corrida las listas se guardan las listas completas (sin los cortes de equilibrio)
EQUILIBRAR_PRUEBA = False        # True: los features y targets de PRUEBA de cada usuario se acortan de tal manera que el target tenga igual cantidad de 4's y 0's.
                                 # False: no se hace tal corte y se procesan todos los features y targets para PRUEBA.
                                 # El corte se hace solamente para el prcesamiento, pero si se genera en la corrida las listas se guardan las listas completas (sin los cortes de equilibrio)
# Parámetros para OneClass
TARGET_PARA_ONE_CLASS      = 4 # Para entrenar el clasificador one_class se usan los features cuyos targeta coincidan con este valor
listaParametroGAMMA        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # constante para el valor de gamma para instanciar el objeto OneClassSVM()
listaParametroKernelOneC   = ['rbf','linear']
listaMezcladaKernelYgammas = [{'kernel':'linear'}] + funciones.mezclarKernelsYGammas('rbf', listaParametroGAMMA) # para linear no tiene sentido gamma por eso no hay que mezclarlo
# Parámetros para RandomForests
listaParametroRForestMinSamplesLeaf  = [1, 5, 10, 20]
listaParametroRFnEstimators          = [10,50,100]
listaParametrosMezcladosRandomForest = funciones.mezclarListas('min_samples_leaf', listaParametroRForestMinSamplesLeaf, 'n_estimators', listaParametroRFnEstimators)
# Parámetros para MultinomialNB
listaParametroMultinomialNBalpha      = [0, 0.05, 0.1, 0.15, 0.2, 100] # [0, 0,25, 0.5, 0.75, 1] # [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 8, 10, 20]
listaParametroMultinomialNBfitPrior   = [True, False]
listaParametrosMezcladosMultinomialNB = funciones.mezclarListas('alpha', listaParametroMultinomialNBalpha, 'fit_prior', listaParametroMultinomialNBfitPrior)
# Parámetros para ComplementNB
listaParametroComplementNBalpha = [0, 0.05, 0.1, 0.15, 0.2, 10]
listaParametroComplementNBnorm  = [True, False]
listaParametrosMezcladosComplementNB = funciones.mezclarListas('alpha', listaParametroComplementNBalpha, 'norm', listaParametroComplementNBnorm)

DICT_POSICIONES_FEATURES = {
    '0-OCEAN': True,
    '1-intervalo': False,
    '2-hashtagsRecibe': False,
    '3-porcentajePOS': False,
    '4-porcentajeNEG': False} # Estructu de cada sublista de FEATURES [[23, 0, 0, 0, 0], [23, 1, 0, 0, 1], [23, 0, 0, 1, 1], ...]
# INICIO_A_PROCESAR = 400  # id: 1, 2, 9 tienen los newsitems en distintas lineas
# FIN_A_PROCESAR = 1000
listaIntervalosAProcesar = ['1-399', '400-1000', '1001-1251', '1501-2000'] # ['1-399', '400-1000', '1001-1251', '1501-2000']
cantidadesKqueRecuerda = [4]  # Ejemplo [10, 20, 40] Con el cambio de un clasificador para todos los usuarios, siempre debera correrse el programa con un valor en esta lista.
amplitudesDeIntervalosQueRecuerda = ['12hour']  # los intervalos pueden ser por ejemplo: ['15min', '30min', '1hour']
PORCENTAJE_APRENDIZAJE = 90  # este valor puede ser entre 1 y 100 el cual indica cual será el porcentaje de news items que se usará para entrenamiento del clasificador

FECHA_INICIO_DATASET = date(2013, 7, 15)  # estas constantes serán necesarias para hacer el cálculo de la cantidad de intervalos que hay en tod o el dataset
FECHA_FIN_DATASET    = date(2015, 3, 25)  # los meses no poner cero a la izquierda porque da error porque lo toma como octal

PATH_FILE_OCEAN = "C:/Users/fgallo/Dropbox/BR-SNs/05-social operator/IBM Bluemix/pruebas PHP/resultadosOCEAN32categorias-"
lineasCorteOCEAN = ['punto5']  # para OCEAN se generan distintas líneas de corte
# los valores de este arreglo se concatenan con PATH_FILE_OCEAN para

# -------- cuando se quieren elegir algunas configuraciones para correr.
correrConfiguracionesElegidas = True # cuando es True solo corre aquellas configuraciones que están en listaConfiguracionesElegidas
# listaConfiguracionesElegidas = [["LogisticRegression","<no aplica>"],  # la numeración de esta lista se corresponden con la numeración observada en el aechivo excel que contiene el gráfico de TODOTODO los k's y los criterios para elegir usuarios candidatos.
#                                 ["DecisionTreeClassifier","<no aplica>"],
#                                 ["OneClassSVM","kernel=linear"],
#                                 ["OneClassSVM","kernel=rbf, gamma=0.5"],
#                                 ["OneClassSVM","kernel=rbf, gamma=0.7"],
#                                 ["RandomForestClassifier","min_samples_leaf=10, n_estimators10"],
#                                 ["RandomForestClassifier","min_samples_leaf=20, n_estimators100"],
#                                 ["MultinomialNB","alpha=0.05, fit_prior=True"],
#                                 ["ComplementNB","alpha=0.2, norm=True"]] # 1,2,3,8,10,20,25,28,46
#                                 # Para preguntar si esta en la lista:  If elem in lista: ...

listaConfiguracionesElegidas = [["LogisticRegression","<no aplica>"],  # la numeración de esta lista se corresponden con la numeración observada en el aechivo excel que contiene el gráfico de TODOTODO los k's y los criterios para elegir usuarios candidatos.
                                ["DecisionTreeClassifier","<no aplica>"],
                                ["OneClassSVM","kernel=rbf, gamma=0.1"],
                                ["OneClassSVM","kernel=rbf, gamma=0.2"],
                                ["RandomForestClassifier","min_samples_leaf=10, n_estimators10"],
                                ["RandomForestClassifier","min_samples_leaf=20, n_estimators100"],
                                ["MultinomialNB","alpha=100, fit_prior=True"],
                                ["ComplementNB","alpha=0.1, norm=True"]] # 1,2,4,5,20,25,36,42

dictCantidadIntervalosPorDia = {  # Tiene que estar acorde a los valores del arreglo
    '15min': 96,
    '30min': 48,
    '1hour': 24,
    '12hour': 2,
    '1day': 1
}
# TODO IMPORTANTE: modificar la función traducirIntervaloAvalorNumerico(...) por cada intervalo nuevo
dictIntervalo1horaTraducidoAnumero = {
    # Para el clasificador no podemos enviar un valor string, entonces se traduce al intervalo de '1hour' a un valor numérico
    '00:00:00-01:00:00': 0,
    '01:00:00-02:00:00': 1,
    '02:00:00-03:00:00': 2,
    '03:00:00-04:00:00': 3,
    '04:00:00-05:00:00': 4,
    '05:00:00-06:00:00': 5,
    '06:00:00-07:00:00': 6,
    '07:00:00-08:00:00': 7,
    '08:00:00-09:00:00': 8,
    '09:00:00-10:00:00': 9,
    '10:00:00-11:00:00': 10,
    '11:00:00-12:00:00': 11,
    '12:00:00-13:00:00': 12,
    '13:00:00-14:00:00': 13,
    '14:00:00-15:00:00': 14,
    '15:00:00-16:00:00': 15,
    '16:00:00-17:00:00': 16,
    '17:00:00-18:00:00': 17,
    '18:00:00-19:00:00': 18,
    '19:00:00-20:00:00': 19,
    '20:00:00-21:00:00': 20,
    '21:00:00-22:00:00': 21,
    '22:00:00-23:00:00': 22,
    '23:00:00-00:00:00': 23
}

dictIntervalo12horaTraducidoAnumero = {
    # Para el clasificador no podemos enviar un valor string, entonces se traduce al intervalo de '12hour' a un valor numérico
    '00:00:00-12:00:00': 0,
    '12:00:00-00:00:00': 1
}

PATH_FILES_SALIDA =                   "C:/Users/fgallo/OneDrive - cs.uns.edu.ar/BR-SNs/Experimentos/clasificadores-INDIA/"  # contiene la ruta donde se guardaran las salidas de las corridas.
PATH_SALIDA_LISTA_FEATURES =          "Salidas/listasFeaturesYtarget/listas-"+str(amplitudesDeIntervalosQueRecuerda[0])+"-k"+str(cantidadesKqueRecuerda[0])+"/"  # esta carpeta contendrá archivos *.py con las listas de features y tergets, el nombre de cada archivo será "idUsuario-k-intervalo.py"
PATH_FILE_IDs_CANDIDATOS =            "C:/Users/fgallo/OneDrive - cs.uns.edu.ar/BR-SNs/Experimentos/clasificadores-INDIA/SalidasPython/candidatos-"+amplitudesDeIntervalosQueRecuerda[0]+"-k"+str(cantidadesKqueRecuerda[0])+"/IDsCandidatosParaClasificador-"
PATH_SALIDA_ARCHIVOS_POR_CLASI =      "C:/Users/fgallo/OneDrive - cs.uns.edu.ar/BR-SNs/Experimentos/clasificadores-INDIA/salidaClasificadoresIndiaRESUMEN (clasificadores individuales)/"
""" Para hacer pruebas iniciales con news items que tua tengo generados voy a usar del 1 al 5
 Esos que ya están generados están en la carpeta D:/xampp/htdocs/indiaDS/paper03/04-DividirPorIntervalosTiempo/01-salida-NIsPorIntervalos/
 
 
    --- Cantidades de news items de cada usuario según valor de k y del intervalo ---
    Id - intervalo - k - cantidad news items:  1711790197  --  15min  --  10  -- 1127786
    Id - intervalo - k - cantidad news items:  1711790197  --  15min  --  20  -- 1804068
    Id - intervalo - k - cantidad news items:  1711790197  --  15min  --  40  -- 2941676
    Id - intervalo - k - cantidad news items:  1711790197  --  30min  --  10  --  902131
    Id - intervalo - k - cantidad news items:  1711790197  --  30min  --  20  -- 1470882
    Id - intervalo - k - cantidad news items:  1711790197  --  30min  --  40  -- 2430680
    Id - intervalo - k - cantidad news items:  1711790197  --  1hour  --  10  --  738284
    Id - intervalo - k - cantidad news items:  1711790197  --  1hour  --  20  -- 1048576
    Id - intervalo - k - cantidad news items:  1711790197  --  1hour  --  40  -- 2009152
    
    Id - intervalo - k - cantidad news items:  1659341478  --  15min  --  10  --  253996
    Id - intervalo - k - cantidad news items:  1659341478  --  15min  --  20  --  383508
    Id - intervalo - k - cantidad news items:  1659341478  --  15min  --  40  --  607494
    Id - intervalo - k - cantidad news items:  1659341478  --  30min  --  10  --  191771
    Id - intervalo - k - cantidad news items:  1659341478  --  30min  --  20  --  303751
    Id - intervalo - k - cantidad news items:  1659341478  --  30min  --  40  --  497672
    Id - intervalo - k - cantidad news items:  1659341478  --  1hour  --  10  --  152134
    Id - intervalo - k - cantidad news items:  1659341478  --  1hour  --  20  --  248842
    Id - intervalo - k - cantidad news items:  1659341478  --  1hour  --  40  --  413480 
 """
"""=============================================================================="""
horaInicio = datetime.datetime.now()
print("Hora de INICIO: ", horaInicio)

#Nuevo nombre fileSalidaResumenTodosClasificadores (único archivo donde se guarda todas las salidas de los clasificadores)
stringParaNombreArchivoSalida = ""
if correrConfiguracionesElegidas:
    stringParaNombreArchivoSalida = "(solo config elegidas)"

auxStringFileYrutaResumenClasificadores = PATH_FILES_SALIDA+"salidaClasificadoresIndiaRESUMEN-"+str(amplitudesDeIntervalosQueRecuerda[0])+",k"+str(cantidadesKqueRecuerda[0])+stringParaNombreArchivoSalida+".txt" # Ejemplo de nombre: salidaClasificadoresIndiaRESUMEN-12hs,k4
fileSalidaResumenTodosClasificadores = open(auxStringFileYrutaResumenClasificadores, "a")

# fileSalidaResumenClasificadoresResumenOtrosClasif = open(PATH_FILE_SALIDA_RESUMEN_OTROS_CLASIFICADORES, "a")
[stringNombresClaves, stringNombresValores] = funciones.convertirAStringClavesYValoresDeDiccionario(DICT_POSICIONES_FEATURES, '\t', '\t')
stringEncabezadoResumenTodosClasificadores = (
            "Procesar IDs candidatos?\tCriterio candidatos\tEquilibrar entrenamiento?\tEquilibrar prueba?\t" + stringNombresClaves + "\tParametro\t\t" +
            "Lista de (posiciones)usuarios\tCant usuarios\tintervalo\tkQueRecuerda\t" +
            "Todas las FEATURES?\tTipoClasificador\tCant samples para entrenar\t" +
            "Cant samples para probar\ttamPromFeaturesENTRENAMIENTO(cantElemIndividuales/CantFeatures)\t" +
            "tamPromFeaturesPRUEBA(cantElemIndividuales/CantFeatures)\t% para entrenar\tScore_samples\t" +
            "CantInliers(1)\tCantOutliers(-1)\tAccuracy\tPrecision\tRecall\tF1(lib)\tF1-F1ant\tF1ant div F1\tF1(alt)\tCant TN\tCant FP\tCant FN\tCant TP\ttiempo ejecucion\t\tInfo extra\n")

if os.path.getsize(auxStringFileYrutaResumenClasificadores) == 0:
    # Esta vacio el archivo, se le agregan los encabezados de las columnas
    fileSalidaResumenTodosClasificadores.write(stringEncabezadoResumenTodosClasificadores)

dictArchivosIndividualesPorClasificador = funciones.generarArchivosPorCadaStringEnLista(listaClasificadoresAutilizar, PATH_SALIDA_ARCHIVOS_POR_CLASI, str(amplitudesDeIntervalosQueRecuerda[0]), str(cantidadesKqueRecuerda[0]), stringEncabezadoResumenTodosClasificadores)

listaIDsCandidatos = None
# fileSalidaResultadosClasificadores = open(PATH_FILES_SALIDA + "salidaClasificadoresIndia.txt", "a")
# Reoorre el archivo de ids de publicadores del dataset de la India, lo hace según los valores de las constantes globales INICIO_A_PROCESAR y FIN_A_PROCESAR.

# Dado un usuario, un k, un intervalo, y 2 listas (featuresConOcean y Target) guarda tales listas en archivos .py distintos
def guardarListasFeaturesYtarget(idUsuario, listaFeatureConOcean, listaTarget, valorK, intervalo):
    fileSalidaFeautureConOcean = open(
        PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(valorK) + "-" + intervalo + "-featureConOCEAN.py",
        "wb")  # el nombre de cada archivo será "idUsuario-k-intervalo-featureConOCEAN.py"
    pickle.dump(listaFeatureConOcean, fileSalidaFeautureConOcean)

    fileSalidaTarget = open(PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(valorK) + "-" + intervalo + "-target.py",
                            "wb")  # el nombre de cada archivo será "idUsuario-k-intervalo-target.py"
    pickle.dump(listaTarget, fileSalidaTarget)


# dado un idUsuario, valor de k y un intervalo,
# retorna 3 listas: lista de features con OCEAN, sin OCEAN y target.
def leerListasFeaturesYtarget(idUsuario, valorK, intervalo):
    listaSalidaFeatureConOcean = None
    listaSalidaTarget = None

    fileEntradaFeautureConOcean = open(
        PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(valorK) + "-" + intervalo + "-featureConOCEAN.py",
        "rb")  # el nombre de cada archivo será "idUsuario-k-intervalo-featureConOCEAN.py"
    listaSalidaFeatureConOcean = pickle.load(fileEntradaFeautureConOcean)

    fileEntradaTarget = open(
        PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(valorK) + "-" + intervalo + "-target.py",
        "rb")  # el nombre de cada archivo será "idUsuario-k-intervalo-target.py"
    listaSalidaTarget = pickle.load(fileEntradaTarget)

    return listaSalidaFeatureConOcean, listaSalidaTarget

# Recorre/procesa acorde a las posiciones que están contenidos en la listaIntervalosProcesar
def main01_recorrerIDpublicadores():
    posicion = 0
    listaFeatureConOcean = []
    listaFeatureSinOcean = []
    listaTarget = []
    listaUsuariosProcesadosExitosamente = []  # cada elemento de esta lista es [pos,id] de cada usuario procesado exitosamente
    maximaPosicionAProcesar = funciones.obtenerMayorValorIntervalo(listaIntervalosAProcesar)
    for linea in fileEntradaIDpublicadores:
        posicion += 1
        if posicion <= maximaPosicionAProcesar:
            idUsuario = linea.replace('\n', '')
            if funciones.pertenerAalgunIntervalo(posicion, listaIntervalosAProcesar) and funciones.existeElementoEnLista(idUsuario, listaIDsCandidatos, PROCESAR_VERDAD_ID_CANDIDATO):
                print(
                    "======================================================================================================")
                # linea tiene que ser procesada
                # se elimina el salto de línea del final de la lectura
                claseOCEAN = obtenerClaseOCEAN32(idUsuario, lineasCorteOCEAN[0])
                print("(1)==== Id usuario, OCEAN: ", idUsuario, " - ClaseO: ", claseOCEAN)
                # se verifica que exista el OCEAN calculado para no procesar aquellos que no lo tengan
                # de lo contrario si mando None al clasificador genera error en tiempo de ejecución
                if claseOCEAN != None:
                    # Dado el id de usuario publicador leido, Se obtiene el nombres de archivo con su ruta completa acorde a cada uno de las amplitudes de intervalos y los k intervalos que recuerda
                    for unIntervalo in amplitudesDeIntervalosQueRecuerda:
                        for unValorDeK in cantidadesKqueRecuerda:
                            # ya que la lista para usalas luego en los clasificadores una vez generados son guardados en PATH_SALIDA_LISTA_FEATURES
                            # se pregunta si es que no se generó yguardó antes para el usuario, k, e intervalo determinado.
                            # solo si existen los 2 archivos no son generador de nuevo (features con ocean y target)
                            if not (os.path.isfile(PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(
                                    unValorDeK) + "-" + unIntervalo + "-featureConOCEAN.py") and os.path.isfile(
                                PATH_SALIDA_LISTA_FEATURES + idUsuario + "-" + str(
                                    unValorDeK) + "-" + unIntervalo + "-target.py")):
                                print("No existia procesamiento previo.")
                                [respuesta, listaFeatureConOceanAux,
                                 listaTargetAux] = procesarNewsItemsDadoUsuarioAmplitudIntervaloYk(idUsuario, unIntervalo,
                                                                                                   unValorDeK, claseOCEAN)
                                if not (respuesta):
                                    print(">>>>>> No Existe el archivo del usuario ", idUsuario, " con el intervalo: ",
                                          unIntervalo, " y k: ", unValorDeK,
                                          " (ya sea porque todavía no esta procesado o porque no existe en la BD los amigos-followers del usuario) ")
                                else:
                                    # se guarda la lista para futuros usos
                                    guardarListasFeaturesYtarget(idUsuario, listaFeatureConOceanAux, listaTargetAux,
                                                                 unValorDeK,
                                                                 unIntervalo)

                                    listaFeatureConOcean = listaFeatureConOcean + listaFeatureConOceanAux
                                    listaFeatureSinOcean = listaFeatureSinOcean + sacarOceanDeListaFeaturesOcean(
                                        listaFeatureConOceanAux)
                                    listaTarget = listaTarget + listaTargetAux
                                    listaUsuariosProcesadosExitosamente = listaUsuariosProcesadosExitosamente + [
                                        [posicion, idUsuario]]
                            else:
                                # ya existen 2 listas procesadas y guardadadas anteriorente
                                # se leen las mismas para procesarlas
                                # cuando se leen las listas no es necesario hacer chequeo que todos los elementos sean distintos de NaN
                                # porque se supone que ya se controló eso antes de guardar
                                print(" Ya existia procesamiento previo para el usuario: ", idUsuario, ", k=", unValorDeK,
                                      ", intervalo:", unIntervalo)
                                [listaFeatureConOceanAux, listaTargetAux] = leerListasFeaturesYtarget(idUsuario, unValorDeK,
                                                                                                      unIntervalo)

                                listaFeatureSinOceanAux = sacarOceanDeListaFeaturesOcean(listaFeatureConOceanAux)
                                listaUsuariosProcesadosExitosamente = listaUsuariosProcesadosExitosamente + [
                                    [posicion, idUsuario]]

                                # se concatenan las lista generadas para un usuario determinado a la lista general que será procesada en el en clasificador
                                listaFeatureConOcean = listaFeatureConOcean + listaFeatureConOceanAux
                                listaFeatureSinOcean = listaFeatureSinOcean + listaFeatureSinOceanAux
                                listaTarget = listaTarget + listaTargetAux
                else:
                    # no tiene OCEAN el usuario
                    print("No existe OCEAN calculado para el usuario " + str(idUsuario) + " Posición: " + str(posicion) + "\n")
        else:
            # se pasó la máxima posicion indicada por la lista  de intervalos
            break
    # hasta aqui se terminó de procesar tod.os los usuarios
    # con sus respectivos features y targets
    # si tod.o salió bien, se tienen las 3 listas (ya sean recién generadas o ledas de archivos)
    fileEntradaIDpublicadores.seek(0)
    print("\n\n LISTAS FINALES:")
    print("Con OCEAN: \n", listaFeatureConOcean)
    print("Sin OCEAN: \n", listaFeatureSinOcean)
    print("Target: \n", listaTarget)
    if listaFeatureConOcean != [] and listaFeatureSinOcean != [] and listaTarget != [] and PROCESAR_CLASIFICADOR:
        # las listas están en condiciones de ser evaluadas por los clasificadores
        # utilizarClasificadores(listaUsuariosProcesadosExitosamente, cantidadesKqueRecuerda[0],
        #                        amplitudesDeIntervalosQueRecuerda[0],
        #                        listaFeatureConOcean, listaFeatureSinOcean, listaTarget) TODO esta función procesa solo CON y SIN ocean, ahora hay más variantes de las features donde algunas pueden aparecer y desaparecer
        utilizarClasificadoresDiversasVariantesFeatures(listaUsuariosProcesadosExitosamente, cantidadesKqueRecuerda[0],
                               amplitudesDeIntervalosQueRecuerda[0],
                               listaFeatureConOcean, listaTarget)
    else:
        print("No se procesa clasificador!!!!!!")

# Dados un usuario, intervalo y cantidad de k intervalos que recuerda, procesa el archivo de news items correspondiente
# Retorna falso si no existe el archivo del usuario
def procesarNewsItemsDadoUsuarioAmplitudIntervaloYk(idUsuario, unIntervalo, unValorDeK, claseOCEAN):
    listaFeaturesConOCEAN = []  # esta lista contendrá las features ocean, hashtag status, intervalo, por ejemplo [[0,...,1,...0, 0.3, 0.3, 0.4, 5],[],...]
    # NOTA los intervalos serán traducidos su amplitud, por ejemplo '12:00:00-13:00:00' será intervalo 13.
    listaTargets = []  # esta lista da los mismo con o sin OCEAN

    respuesta = True
    intervaloParaFeature = ""  # esta variable sirve para almacenar el intervalo de tiempo que se está considerando, se lo guarda porque cuando hay más de un hashtag se dividen en varias lineas pero solo la primera tiene el intervalo.
    nombreArchivoNewsItemsSegunIntervaloYvalorDeK = obtenerRutasArhivosNewsItemsDadoIdPublicador(
        idUsuario, unIntervalo, unValorDeK)
    # se procesa el conjunto de news items

    # Se evalua si existe el archivo, porque a veces no existe porque todavía no está procesado o porque no existe en la base de datos los amigos del mismo!
    if os.path.isfile(nombreArchivoNewsItemsSegunIntervaloYvalorDeK):
        fileEntradaNewsItemsDadoIdIntervaloK = open(nombreArchivoNewsItemsSegunIntervaloYvalorDeK, "r", encoding='UTF8')

        # en una corrida previa hice la cuenta de cuantos fila de news items tieene cada usuario acorde a las distintas amplitudes de tiempo y k que recuerda
        # se calcula cual es la cantidad de news items a procesar para el clasificador acorde a porcentaje almacenado en la constante PORCENTAJE_APRENDIZAJE.
        cantidadDeNewsItemsAprocesar = calcularCantidadIntervalosDadoAmplitud(unIntervalo) # TODO no tiene sentido usar este cálculo.
        # SOLO para achicar las pruebas       cantidadDeNewsItemsAprocesar = 2000  # Para el id de posición 1 >>>> 1454: lista de newsItems mal concatenadas, 3350  # 3344 # TODO <<<<<<<<<<<  DESPUÉS BORR
        print(" ++++++++++++++++ ID - amplitudIntervalo - cantidad K que recuerda - cantidad proporcional: ", idUsuario,
              " -- ", unIntervalo, " -- ", unValorDeK, " -- ", cantidadDeNewsItemsAprocesar,
              " --- Cantidad de intervalos total que existen: ", calcularCantidadIntervalosDadoAmplitud(unIntervalo))
        cantidadNewsItemsLeidos = 0  # es para controlar que solo se lee un porcentaje de todos los intervalor de news items

        """             IMPORTANTE 
        La cantidad de intervalos procesados no se corresponden con la cantidad de líneas leídas porque en cada intervalo por cada hashtag se creó una línea
        Por ejemplo, la siguiente muestra de intervalo '1hour', k=10, leyó 19 lineas pero en realizadad son 8 intervalos los que se leyeron   

1[2013-07-15 00:00:00-2013-07-15 01:00:00]	<vacio>	<vacio>
2[2013-07-15 01:00:00-2013-07-15 02:00:00]	<vacio>	<vacio>
3[2013-07-15 02:00:00-2013-07-15 03:00:00]	<vacio>	<vacio>
4[2013-07-15 03:00:00-2013-07-15 04:00:00]	<vacio>	<vacio>
5[2013-07-15 04:00:00-2013-07-15 05:00:00]	<vacio>	<vacio>
6[2013-07-15 05:00:00-2013-07-15 06:00:00]	<vacio>	<vacio>
7[2013-07-15 06:00:00-2013-07-15 07:00:00]	<ignorado>[BJP]	"(1492052028,BJP,pos)[2013-07-15 05:35:22];(37179759,BJP,neg)[2013-07-15 05:53:32]"
                            <ignorado>[CAG]	(1492052028,CAG,pos)[2013-07-15 05:35:22]
                            <ignorado>[MODI]	"(1492052028,MODI,pos)[2013-07-15 05:35:22];(37179759,MODI,neg)[2013-07-15 05:53:32]"
                            <ignorado>[SWAMY]	(1492052028,SWAMY,pos)[2013-07-15 05:35:22]
                            <ignorado>[UPA]	(1418662837,UPA,neu)[2013-07-15 05:50:17]
8[2013-07-15 07:00:00-2013-07-15 08:00:00]	<ignorado>[BJP]	"(1492052028,BJP,pos)[2013-07-15 05:35:22];(37179759,BJP,neg)[2013-07-15 05:53:32];(51584194,BJP,neu)[2013-07-15 06:50:56]"
                            <ignorado>[CAG]	(1492052028,CAG,pos)[2013-07-15 05:35:22]
                            <ignorado>[MODI]	"(1492052028,MODI,pos)[2013-07-15 05:35:22];(37179759,MODI,neg)[2013-07-15 05:53:32]"
                            <ignorado>[SWAMY]	(1492052028,SWAMY,pos)[2013-07-15 05:35:22]
                            <ignorado>[UPA]	(1418662837,UPA,neu)[2013-07-15 05:50:17]
                            <ignorado>[MUMBAI]	(37179759,MUMBAI,pos)[2013-07-15 06:24:43]
                            <ignorado>[FEKUFACTS]	(290587368,FEKUFACTS,neu)[2013-07-15 06:33:02]
                            <ignorado>[FEKU]	(51584194,FEKU,neu)[2013-07-15 06:50:56]                      
        """

        cantidadDeLineasDelArchivoLeidas = 0
        # listaNewsItems = "" # esta cadena contendrá todas la lista de newsitems que están distribuidos.
        arrayNewsItemInicial = ["", "", ""]
        arrayNewsItemQueUneInicialesDeUnIntervalo = []  # es una lista de listas, donde cada sublista es un [arrayNewsItemInicial]
        mostrarIntervaloNuevo = False
        cantidadListasConcatenadas = 0  # se cuenta la cantidad porque cuando se muestran los datos y está dividio en más de una lista de pone la posición de la primera parte
        cantidadNoIgnoradoNoVacio = 0
        # comienza la lectura línea a línea del archivo
        # try agregado porque llegado un momento daba el siguiente error "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa8 in position 4011: invalid start byte"
        try:
            for lineaNewsItem in fileEntradaNewsItemsDadoIdIntervaloK:
                cantidadDeLineasDelArchivoLeidas += 1
                lineaNewsItem = lineaNewsItem.replace("\n", "").replace('"',
                                                                        '')  # se saca el salto de línea del final de cada línea leida y las comillas dobles que no se porque aparecen en algunas lineas
                # print("Linea leida: ", lineaNewsItem)

                # antes de sumar se verifica que se ya leido otro intervalo y que no se solamente otro hashtga dentro del mismo intervalo
                arrayNewsItem = lineaNewsItem.split('\t')

                if cantidadDeLineasDelArchivoLeidas == 1:
                    arrayNewsItemInicial = arrayNewsItem  # se hace esta asignación porque sino la primer fila muestra blancos
                esListaNewsItemsCortado = False

                # se evalua la linea leida
                if arrayNewsItem[0] != "":
                    # se evalua si no es la segunda parte de la lista de newsitems que fue cortada

                    """  Este framento es para trata de solucionar el problema de newsitems mal cortados producto de intervalo 1454 de id posición 1, '1hour', k=10 que recuerda 
                    # se estrae el último elemento que hay en la segunda posicion, para después evaluar si termina con un formato correcto
                    elUltimoElementoTieneFormatoNIs = False
                    try:
                        arrayAux = arrayNewsItemInicial[2].split(';')
                        ultimoElementoListaNIsRecibidos = arrayAux[len(arrayAux)-1]
                        print("........ ultimoElementoListaNIsRecibidos: ", ultimoElementoListaNIsRecibidos)
                        elUltimoElementoTieneFormatoNIs = tieneFormatoNewsItem(ultimoElementoListaNIsRecibidos)
                        print("===========(111) ultimo de la línea anterior: ", ultimoElementoListaNIsRecibidos, " >>> Repuesta tieneFormatoNI?: ", elUltimoElementoTieneFormatoNIs)
                    except IndexError:
                        elUltimoElementoTieneFormatoNIs = True
                    # elUltimoElementoTieneFormatoNIs = True # TODO cuando esta activa esta línea es igual al previo intento de mejora
                    """

                    if arrayNewsItem[0][
                        0] != '(':  # TODO esta segunda condición no tiene que ir aquí   and elUltimoElementoTieneFormatoNIs == True:       # TODO segunda condición sometida a prueba
                        # de verdad es un NUEVO intervalo de newsitems
                        cantidadNewsItemsLeidos += 1

                        # se hace esta pregunta porque de lo contrario el primer intervalo que está en posición de procesar (es decir no es vacio-vacio) mostraba como listo luego de la primer parte del mismo
                        if arrayNewsItemInicial[1][:7] != '<vacio>' and arrayNewsItemInicial[2][:7] != '<vacio>':
                            mostrarIntervaloNuevo = True
                        # print("+++++++++++++++++++++++ nuevo INTERVALO leido ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        # arrayNewsItemInicial = arrayNewsItem
                        # TOD O esListaNewsItemsCortado = False
                    else:
                        esListaNewsItemsCortado = True

                        arrayNewsItemInicial[2] = arrayNewsItemInicial[2] + ";" + arrayNewsItem[0]

                        cantidadListasConcatenadas += 1

                if cantidadNewsItemsLeidos <= (
                        cantidadDeNewsItemsAprocesar + 1):  # se aumenta '1' porque sino queda fuera de prueba
                    # se procesa el news item
                    """arrayNewsItem[0]: la primera vez el intervalo y las siguientes un valor en blanco porque por cada hashtag crea una línea
                      EJEMPLO: 886[2013-08-20 21:00:00-2013-08-20 22:00:00]   // es el numero de intervalo seguido por los extremos del intervalo entre corchetes.
                    arratNewsItem[1]: la decisión que hizo el usuario seguido del hashtag correspondiente
                      EJEMPLO: <ignorado>[NAMMASARKARA]
                    arrayNewsItem[2]: todos los news items que recibió el usuario dentro del intervalo de tiempo
                      EJEMPLO: (1681267332,NAMMASARKARA,pos)[2013-08-20 11:00:35];(1681287852,NAMMASARKARA,neu)[2013-08-20 11:00:45];(1681287852,NAMMASARKARA,pos)[2013-08-20 11:03:01];(1681369046,NAMMASARKARA,neu)[2013-08-20 11:04:24];(1681329332,NAMMASARKARA,neu)[2013-08-20 11:04:28];(1681466785,NAMMASARKARA,neu)[2013-08-20 11:04:53];(1681369046,NAMMASARKARA,pos)[2013-08-20 11:05:17];(1681418322,NAMMASARKARA,neu)[2013-08-20 11:05:24];(135108603,NAMMASARKARA,neu)[2013-08-20 11:05:32];(1681466785,NAMMASARKARA,pos)[2013-08-20 11:05:50];(1681329332,NAMMASARKARA,pos)[2013-08-20 11:05:58];(1681418322,NAMMASARKARA,pos)[2013-08-20 11:06:15];(1681427760,NAMMASARKARA,neu)[2013-08-20 11:06:52];(1681974854,NAMMASARKARA,neu)[2013-08-20 11:07:38];(1681427760,NAMMASARKARA,pos)[2013-08-20 11:07:41];(1681947012,NAMMASARKARA,neu)[2013-08-20 11:07:46];(1592219599,NAMMASARKARA,neu)[2013-08-20 11:08:14];(37179759,NAMMASARKARA,neu)[2013-08-20 11:08:21];(1592186568,NAMMASARKARA,neu)[2013-08-20 11:08:23];(784406850,NAMMASARKARA,neu)[2013-08-20 11:08:25];(1681947012,NAMMASARKARA,pos)[2013-08-20 11:08:39];(17781689,NAMMASARKARA,pos)[2013-08-20 11:08:40];(1681974854,NAMMASARKARA,pos)[2013-08-20 11:08:50];(1592219599,NAMMASARKARA,pos)[2013-08-20 11:09:14];(1592186568,NAMMASARKARA,pos)[2013-08-20 11:09:21];(17781689,NAMMASARKARA,pos)[2013-08-20 11:09:55];(17781689,NAMMASARKARA,neg)[2013-08-20 11:11:37];(1610813516,NAMMASARKARA,neu)[2013-08-20 11:17:41];(1610828707,NAMMASARKARA,neu)[2013-08-20 11:17:42];(1610819918,NAMMASARKARA,neu)[2013-08-20 11:18:35];(1610813516,NAMMASARKARA,pos)[2013-08-20 11:18:56];(1610828707,NAMMASARKARA,pos)[2013-08-20 11:18:58];(1610819918,NAMMASARKARA,pos)[2013-08-20 11:20:27];(1610855066,NAMMASARKARA,neu)[2013-08-20 11:30:29];(1478183725,NAMMASARKARA,neu)[2013-08-20 11:33:41];(1610855066,NAMMASARKARA,pos)[2013-08-20 11:35:40];(43407467,NAMMASARKARA,neg)[2013-08-20 11:37:37];(1478183725,NAMMASARKARA,pos)[2013-08-20 11:37:42];(1588162976,NAMMASARKARA,neu)[2013-08-20 11:38:05];(17781689,NAMMASARKARA,neg)[2013-08-20 11:38:14];(37179759,NAMMASARKARA,neg)[2013-08-20 11:38:19];(1588162976,NAMMASARKARA,pos)[2013-08-20 11:38:56];(43407467,NAMMASARKARA,neg)[2013-08-20 11:45:27];(135108603,NAMMASARKARA,neg)[2013-08-20 11:45:48];(1588155967,NAMMASARKARA,neu)[2013-08-20 11:47:22];(135108603,NAMMASARKARA,pos)[2013-08-20 11:48:16];(1588155967,NAMMASARKARA,pos)[2013-08-20 11:48:17];(1478265679,NAMMASARKARA,neu)[2013-08-20 11:48:20];(1478265679,NAMMASARKARA,pos)[2013-08-20 11:49:26];(1485816986,NAMMASARKARA,neu)[2013-08-20 11:49:57];(1485816986,NAMMASARKARA,pos)[2013-08-20 11:50:50];(1485806382,NAMMASARKARA,neu)[2013-08-20 11:50:53];(1485806382,NAMMASARKARA,pos)[2013-08-20 11:51:42];(135108603,NAMMASARKARA,pos)[2013-08-20 11:54:31];(135108603,NAMMASARKARA,pos)[2013-08-20 11:56:05];(17781689,NAMMASARKARA,neg)[2013-08-20 12:03:57];(135108603,NAMMASARKARA,neg)[2013-08-20 12:27:29];(135108603,NAMMASARKARA,pos)[2013-08-20 13:05:05]
                    """
                    """ IMPORTANTE: 
                        Al parecer cuando la línea de nees items recibidos es muy larga la corta en dos líneas líneas
                        Por ejemplo para el usuario con posición de id: 2
                            --- News item leido (cantidadNewaItemsLeidos - Cantidad de lineas efectivamente leidas  3343 -  44501 ) :  ---  <ignorado>[HARYANAVIJAYMAHARALLY]  ---  (2147483647,HARYANAVIJAYMAHARALLY,pos)[2013-12-01 01:38:57];(2147483647,HARYANAVIJAYMAHARALLY,pos)[2013-12-01 01:42:13];...;(2147483647,HARYANAVIJAYMAHARALLY,neu)[2013-12-01 05:20:48]
                            --- News item leido (cantidadNewaItemsLeidos - Cantidad de lineas efectivamente leidas  3344 -  44502 ) : (2147483647,HARYANAVIJAYMAHARALLY,neu)[2013-12-01 05:20:51];...;(2147483647,HARYANAVIJAYMAHARALLY,neu)[2013-12-01 05:59:53] ---    ---  
                        Es claro que es la misma lista de news items cortaodos porque tiempre tienen el mismo hashtag, además en la segunda parte/línea la lista de newsitem aparece en la primer posición/columna de la línea leida-    
                    """

                    # si evalúa si la linea leida es lo cortado de una lista de newsItems o no
                    if not (
                            esListaNewsItemsCortado) and cantidadNewsItemsLeidos > 1:  # la segunda pregunta es porque de lo contrario muestra ceros en los y vacios
                        # dependiendo del si se va a mostrar un nuevo intervalo será el valor de la variable 'cantidadNewsItemsLeidos'. Si es nuevo intervalo se resta 1
                        valorARestar = 0
                        if mostrarIntervaloNuevo:
                            valorARestar = 1

                        # print("--- Linea (cantidadNewsItemsLeidos, cantidadDeLineasDelArchivoLeidas) \t", (cantidadNewsItemsLeidos - valorARestar), "\t", (cantidadDeLineasDelArchivoLeidas - 1 - cantidadListasConcatenadas), " : ", arrayNewsItemInicial[0], " --- ", arrayNewsItemInicial[1], " --- ", arrayNewsItemInicial[2])    # todo IMPORTANTE: Antes del break más abajo se imprime el último news item del intervalo
                        """  EN ESTA PARTE DE DEBE PROCESAR EL NEWS ITEM """
                        # procesarNewsItemsEnClasificador(arrayNewsItemInicial)  # TODO ver si verdaramente la necesito a esta función
                        # se obtienen los arreglos para el clasificador

                        # TODO ver si esto verdaderamente va aqui!!!
                        # print("___________________________________________________________")
                        # print(" +++ arrayNewsItemInicial: ", arrayNewsItemInicial)

                        # cuando no es el primer hashtag del intervalo la posición arrayNewsItemInicial[0] es en blanco
                        # en ese caso no tiene que procesarse el intervalo, en tod o caso se tomaría el último valor calculado
                        # if arrayNewsItemInicial[0] != "":
                        #     intervaloParaFeature = obtenerSoloIntervaloHoras(arrayNewsItemInicial[0])  # El intervalo tiene un formato así:  3349[2013-12-01 12:00:00-2013-12-01 13:00:00]  y con la función se obtiene '12:00:00-13:00:00'

                        # se evalua si no es el caso donde el intervalo es totalmente vacio, es decir hizo nada ni recibió nada
                        # en caso de ser vacio-vacio se ignora el intervalo
                        if arrayNewsItemInicial[1][:7] != '<vacio>' and arrayNewsItemInicial[2][:7] != '<vacio>':
                            # print(">>>arrayNewsItemInicial: ", arrayNewsItemInicial) # TODO después borrar
                            arrayNewsItemQueUneInicialesDeUnIntervalo = arrayNewsItemQueUneInicialesDeUnIntervalo + [
                                arrayNewsItemInicial]

                        arrayNewsItemInicial = arrayNewsItem
                        cantidadListasConcatenadas = 0

                        if mostrarIntervaloNuevo and arrayNewsItemQueUneInicialesDeUnIntervalo != []:
                            # print("+++++++++++++++++++++++ nuevo INTERVALO leido ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                            # print("*** ArregloNewsItem: ", arrayNewsItem)
                            print("*** Arreglo de TODO el intervalo: ", arrayNewsItemQueUneInicialesDeUnIntervalo)
                            # <<<<<< INICIO parte nueva (pos reunión 01/10)>>>>>>
                            # se tiene que procesar tod.o el intervalo
                            # de la lista de sublistas se extrae la primer sublista que es la que tiene el número de intervalo y la fecha por se la primer sublista

                            intervaloParaFeature = obtenerSoloIntervaloHoras(
                                (arrayNewsItemQueUneInicialesDeUnIntervalo[0])[
                                    0])  # El intervalo tiene un formato así:  3349[2013-12-01 12:00:00-2013-12-01 13:00:00]  y con la función se obtiene '12:00:00-13:00:00'
                            [arrayHashtags, arrayTarget] = obtenerStatusHashtagNormalizadoYtarget(
                                arrayNewsItemQueUneInicialesDeUnIntervalo)
                            print(">.>.> arrayHashtags: ", arrayHashtags)  # TODO después borrar
                            print(">.>.> arrayTarget: ", arrayTarget)  # TODO después borrar

                            # se chequea que ninguna lista tenga elements NaN porque genera error en el clasificador
                            # No tiene sentido chequear la lista sin ocean porque está contenida en la con ocean
                            listaFeaturesConOCEAN = listaFeaturesConOCEAN + [[claseOCEAN] + [traducirIntervaloAvalorNumerico(intervaloParaFeature, unIntervalo)] + arrayHashtags]  # por defecto se guarda con OCEAN, después cuando se necesite que eliminarpa este valor
                            listaTargets = listaTargets + arrayTarget

                            arrayNewsItemQueUneInicialesDeUnIntervalo = []  # se inicializa el arreglo después de mostrarlo/procesarlo
                            # <<<<<< FIN parte nueva (pos reunión 01/10)>>>>>>

                            mostrarIntervaloNuevo = False

                else:
                    break  # así se evita seguir leyendo líneas innecesariamente
            print("cantidadDeLineasDelArchivoLeidas", cantidadDeLineasDelArchivoLeidas)
        except UnicodeDecodeError as error:
            print("ERROR de UnicodeDecodeError!!! Id usuario: ", idUsuario)
    else:
        # no existe el archivo para el id-intervalo-k dados
        respuesta = False
        listaFeaturesConOCEAN = None
        listaTargets = None

    return respuesta, listaFeaturesConOCEAN, listaTargets


# recibe un arreglo donde las posiciones
# [0]: indica el intervalo o en blanco para no repetirlo,
# [1]: la decición del usuario, cuando es reutilizado [1] tiene algo así:   (1711790197,NAMORON,neg)[2013-10-21 12:15:09];(1711790197,NAMORON,neg)[2013-10-21 12:15:15] y en [2] es distinto de '<vacio>'
# [2]: la lista de news items que recibió. Puede ser algo así:  (37179759,NAMORON,pos)[2013-10-21 05:34:29];(290587368,NAMORON,neu)[2013-10-21 06:20:02], o '<vacio>'
def procesarNewsItemsEnClasificador(arrayEntradaNI):  # TODO ¿sirve?
    # se evalua si es <ignorado>. La estructura cuando es ignorado es algo así:     <ignorado>[TEHELKA]   por eso se corta el string
    if len(arrayEntradaNI[1]) > 0 and (arrayEntradaNI[1])[:10] != '<ignorado>' and (arrayEntradaNI[1])[:7] != '<vacio>':
        # El usuario usó un hashtag que le venía en su feed o generó uno nuevo
        # print(":::::::::::::::::::::::::::::::::::::::::::::::::::::", arrayEntradaNI[1], "::::", len(arrayEntradaNI[1]))

        if arrayEntradaNI[2][:11] == '<inventado>':
            print("   - - - INVENTADO ", arrayEntradaNI[1], " --- ", arrayEntradaNI[2])
        else:
            print("   - - - REUTILIZADO ", arrayEntradaNI[1], " --- ", arrayEntradaNI[2])

        # se separan los news items generados por este usuario
        arrayDecisiones = arrayEntradaNI[1].split(';')
        for unaDecision in arrayDecisiones:
            print("           |", unaDecision)


"""
    else:
        if (arrayEntradaNI[1])[:10] == '<ignorado>':
            # es un hashtag ignorado
            print("*** ", arrayEntradaNI[0], " --- ", arrayEntradaNI[1], " --- ", arrayEntradaNI[2])
        # else: por el else sería que es un intervalo vacío/cadena vacía
        else:
            if arrayEntradaNI[1] == '<vacio>' and arrayEntradaNI[1] == '<vacio>':
                print("+++ ", arrayEntradaNI[0], " --- ", arrayEntradaNI[1], " --- ", arrayEntradaNI[2])
            else:
                # es un ERROR del formato de los datos
                print("+*+ ", arrayEntradaNI[0], " --- ", arrayEntradaNI[1], " --- ", arrayEntradaNI[2])
"""


# recibe una lista de sublistas con tod.o lo que pasó en el intervalo, donde cada sublista consta de:
#   [0]: indica el intervalo o en blanco para no repetirlo, // solo la primer sublista contiene los detalles del intervalo
#   [1]: la decición del usuario,
#   [2]: la lista de news items que recibió que eventualmente puede ser <vacía>
# Retorna dos arreglos:
#       (1)un arreglo de 3 posiciones:
#           [0]: puede ser '0', '1', o '2', donde si en la mayoria de los intervalos ganó neu, pos o neg, resp.
#   CONDICIONES/CASOS: (igual an{alisis para las filas de los hashtags que para las filas de los totales del intervalo)
#   -  Si dentro de una fila de un hashtag determinado hay un empate de sentimiento entre:
#           - pos y neg: se da un punto a favor de neutro
#           - pos y neu o neg y neu: se da punto a favor de pos o neg, respectivamente
#           - pos-neg-neu empatan, se da punto a favor de neutro
#           - si los 3 son distintos valores gana el mayor valor
#           [1] y [2]: donde cada posición puede ser:
#               '0' si x en [0, 0.25),
#               '1' si x en [0.25, 0.50),
#               '2' si x en [0.50, 0.75),
#               '3' si x en [0.75, 1]
#               donde x es el porcentaje de NIs positivos y negativos, resp. NOTA: para la cuenta se tiene en cuenta los neutros también.
#       (2) el segundo arreglo retorna TARGET el cual es un array con un solo valor entre 0...4
#           0: el usuario ignoró tod.o,
#           1,2,3: si no inventó, y si lo que más hizo fue reutilizar por, neg, o neutro respectivamente(TODO: preguntar a G que hacer con empates)
#           4: el usuario inventó al menos algo. Esto tiene prioridad sobre el resto de los casos.
def obtenerStatusHashtagNormalizadoYtarget(arrayNIentradaDeTodoElIntervalo):
    arraySalidaFeatureHashtag = [0, 0, 0]  # posiciones [0]:
    arraySalidaTarget = [0]

    # se recorre la lista de sublistas
    cantDeIgnorados = 0
    cantInventados = 0
    cantReusoPos = 0
    cantReusoNeg = 0
    cantReusoNeu = 0
    cantSublistas = 0
    cantLlegaPos = 0
    cantLlegaNeg = 0
    cantLlegaNeu = 0
    for unaSublista in arrayNIentradaDeTodoElIntervalo:
        cantSublistas += 1
        # . . . . . . . . . . . . . . . . . . . . . . . . .
        # . . . procesamiento para el arregloFeatures . . .
        # . . . . (se procesa lo que le LLEGA). . . . . . .

        # se evalua que haya al menos un NIs en la entrada
        # print ("... UnaSublista: ", unaSublista)
        if unaSublista[2][:11] != '<inventado>':
            # al menos hay un NIs que le llega para este hashtag
            arrayNewsItemQueLeLLegan = unaSublista[2].split(
                ';')  # este es un array que contiene todos los NIs que le llegaron de una determonada sublista
            cantNI = len(arrayNewsItemQueLeLLegan)
            cantLlegaPosEnUnHashtagDeterminado = 0
            cantLlegaNegEnUnHashtagDeterminado = 0
            cantLlegaNeuEnUnHashtagDeterminado = 0
            # estas variables cuyos nombres terminan con "...EnUnHashtagDeterminado" se diferencias de las otras que no tienen esta terminación
            # en que las segungas son de todos el intervalo, meintras que las primeras son para una fila asociada a un hashtag determinado
            # se recorre uno a uno los NIs que les llegaron
            for unNIdeSublista in arrayNewsItemQueLeLLegan:
                sentimiento = dadoNewsItemsObtenerParte(unNIdeSublista, 's')
                if sentimiento == 'pos':
                    cantLlegaPosEnUnHashtagDeterminado += 1
                else:
                    if sentimiento == 'neg':
                        cantLlegaNegEnUnHashtagDeterminado += 1
                    else:
                        if sentimiento == 'neu':
                            cantLlegaNeuEnUnHashtagDeterminado += 1

            cantLlegaPos = cantLlegaPos + cantLlegaPosEnUnHashtagDeterminado
            cantLlegaNeg = cantLlegaNeg + cantLlegaNegEnUnHashtagDeterminado
            cantLlegaNeu = cantLlegaNeu + cantLlegaNeuEnUnHashtagDeterminado

            # se termino de procesar toda UNA LINEA de NIs (asociada a un hashtag, ya sae como news item de entrada o Ni generado por el usuario)
            # se determinada que sentimiento preponderó
            queSentimientoSeSumo = determinarSentimientoPreponderante(cantLlegaPosEnUnHashtagDeterminado,
                                                                      cantLlegaNegEnUnHashtagDeterminado,
                                                                      cantLlegaNeuEnUnHashtagDeterminado)

        # . . . . . . . . . . . . . . . . . . . . . . . .
        # . . . procesamiento para el arregloTarget . . .
        # . . .  (se procesa lo que HACE) . . . . . . . .
        if unaSublista[2][:11] == '<inventado>':
            arraySalidaTarget = [4]
            cantInventados += 1
        else:
            if unaSublista[1][:10] != '<ignorado>':
                sentimiento = dadoNewsItemsObtenerParte(unaSublista[1],
                                                        's')  # contiene el sentimiento de lo que REUTILIZÓ el usuario
                if sentimiento == 'pos':
                    cantReusoPos += 1
                if sentimiento == 'neg':
                    cantReusoNeg += 1
                if sentimiento == 'neu':
                    cantReusoNeu += 1
            else:
                cantDeIgnorados += 1

    # ____________________________________________________
    # _______ se cargan los arreglos de SALIDA ___________
    # ____________________________________________________
    # ya se recorrieron todos las sublistas y se hizo la cuenta de NIs generados por el usuario
    queSentimientoSeSumoComoResumenIntervalo = determinarSentimientoPreponderante(cantLlegaPos, cantLlegaNeg,
                                                                                  cantLlegaNeu)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # se determina qué valor tendrá el TODO >>> arraySalidaFeatureHashtag
    # >>>> valor para la posición arraySalidaFeatureHashtag[0]
    # para eso se vale del último análisis de sentimientos de las lineas del intervalo
    if queSentimientoSeSumoComoResumenIntervalo == 'neu':
        arraySalidaFeatureHashtag[0] = 0
    else:
        if queSentimientoSeSumoComoResumenIntervalo == 'pos':
            arraySalidaFeatureHashtag[0] = 1
        else:
            arraySalidaFeatureHashtag[0] = 2  # es 'neg' por descarte

    # >>>> valor para la posición arraySalidaFeatureHashtag[1] y arraySalidaFeatureHashtag[2]
    # al porcentaje de pos y de neg se pasan a intervalos 0/1/2/3
    # es posible que (cantLlegaPos + cantLlegaNeg + cantLlegaNeu) == 0 cuando tod.o es inventado, por lo tanto puede dar error de división por cero en tiempo de ejecución
    # por eso se verifica la siguiente condición
    auxPos = 0
    auxNeg = 0
    if (cantLlegaPos + cantLlegaNeg + cantLlegaNeu) > 0:
        auxPos = ((cantLlegaPos * 100) / (
                    cantLlegaPos + cantLlegaNeg + cantLlegaNeu)) / 100  # se divide al final entre 100 para pasar de 31.5 a 0.31 y mapaearlo a un intervalo entre [0,1]
        auxNeg = ((cantLlegaNeg * 100) / (cantLlegaPos + cantLlegaNeg + cantLlegaNeu)) / 100
    print("Cant de pos, neg y neu: ", cantLlegaPos, " - ", cantLlegaNeg, " - ", cantLlegaNeu)
    print(".-.-.- Antes de intentar obtener los porcentajes - Aux pos y neg: ", auxPos, " - ",
          auxNeg)  # TODO  después borrar
    arraySalidaFeatureHashtag[1] = traducirPorcentajeAnumeroSegunIntervalo(auxPos)
    arraySalidaFeatureHashtag[2] = traducirPorcentajeAnumeroSegunIntervalo(auxNeg)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # se determina qué valor tendrá el TODO >>> arregloSalidaTarget
    # primero se evalua si inventó algo ya que eso tiene prioridad para determinar el target.
    if cantInventados == 0:
        # no inventó nada en el intervalo
        # entonces tiene sentido que se procesen los news items de entrada para ver que pasó
        # es decir si se reutilizaron o no
        if (cantReusoPos + cantReusoNeu + cantReusoNeg) != 0:
            # al menos reutilizó algo alguna vez
            porcentajeNIsPosEnTodoElIntervalo = (cantReusoPos * 100) / (cantReusoPos + cantReusoNeu + cantReusoNeg)
            porcentajeNIsNegEnTodoElIntervalo = (cantReusoNeg * 100) / (cantReusoPos + cantReusoNeu + cantReusoNeg)
            porcentajeNIsNeuEnTodoElIntervalo = (cantReusoNeu * 100) / (cantReusoPos + cantReusoNeu + cantReusoNeg)
        else:
            # nunca reutilizó algo, (no hizo nada)
            arraySalidaTarget = [0]

    return arraySalidaFeatureHashtag, arraySalidaTarget


# Dado 3 cantidades de sentimientos revuelve alguno de los 3 valores: 'pos', 'neg' o 'neu'
#           - pos y neg: se da un punto a favor de neutro
#           - pos y neu o neg y neu: se da punto a favor de pos o neg, respectivamente
#           - pos-neg-neu empatan, se da punto a favor de neutro
#           - si los 3 son distintos valores gana el mayor valor
def determinarSentimientoPreponderante(cantPos, cantNeg, cantNeu):
    queSentimientoGano = None
    if cantPos > cantNeg and cantPos > cantNeu:
        # gana 'pos'
        queSentimientoGano = 'pos'
    else:
        if cantNeg > cantPos and cantNeg > cantNeu:
            # gana 'neg'
            queSentimientoGano = 'neg'
        else:
            if cantNeu > cantPos and cantNeu > cantNeg:
                # gana 'neu'
                queSentimientoGano = 'neu'
            else:
                # hay al menos empate entre 2
                if cantPos == cantNeu and cantNeu == cantNeg:
                    # hay empate entre 3, entonces prepondera 'neu'
                    queSentimientoGano = 'neu'
                else:
                    # ahora se tiene certeza que hay empate entre 2 nada más
                    if cantPos == cantNeg:
                        # como empatan 'pos' y 'neg' es como que se anulan y gana 'neu'
                        queSentimientoGano = 'neu'
                    else:
                        # pos o neg empatan con neu
                        if cantPos > cantNeg:
                            queSentimientoGano = 'pos'
                        else:
                            queSentimientoGano = 'neg'

    return queSentimientoGano


# dado un valor de porcentaje in [0,1], traduce a un valor {0,1,2,3} según el porcentaje este en [0, 0.25), [0.25, 0.5), [0.5, 0,75) o [0,75,1]
def traducirPorcentajeAnumeroSegunIntervalo(porcentaje):
    respuesta = None
    if porcentaje < 0.25:
        respuesta = 0
    else:
        if porcentaje < 0.5:
            respuesta = 1
        else:
            if porcentaje < 0.75:
                respuesta = 2
            else:
                if porcentaje <= 1:
                    respuesta = 3
    return respuesta


# Dado el idUsuario y una línea de corte, retorna un arreglo de 32 posiciones, donde 31 son '0' y hay un solo '1'.
# En caso de no encontrarse el usuario en el archivo retorna 'None'
def obtenerArrayOCEAN32(idUsuario, lineaCorte):
    respuesta = None
    fileEntradaOCEAN = open(PATH_FILE_OCEAN + lineaCorte + ".txt", "r", encoding='UTF8')

    esPrimerLinea = True
    for unaLinea in fileEntradaOCEAN:
        # la primer línea debe ser descartada porque es la descripción de las columnas
        if not (esPrimerLinea):
            arrayLineaFileOCEAN = unaLinea.split('\t')
            if arrayLineaFileOCEAN[0] == idUsuario:
                respuesta = arrayLineaFileOCEAN[1:33]  # se guardan las posiciones de 1...32 inclusive
                respuesta = [int(x) for x in
                             respuesta]  # se convierten a todos los elementos en entero porque sino da error en el clasificador
                break
        else:
            esPrimerLinea = False
    return respuesta


# Dado el idUsuario y una línea de corte, retorna un único valor entre [1,32] correspondiente a la posición donde está el valor '1' y todos los demás con '0'.
# En caso de no encontrarse el usuario en el archivo retorna 'None'
def obtenerClaseOCEAN32(idUsuario, lineaCorte):
    respuesta = None
    fileEntradaOCEAN = open(PATH_FILE_OCEAN + lineaCorte + ".txt", "r", encoding='UTF8')

    esPrimerLinea = True
    for unaLinea in fileEntradaOCEAN:
        # la primer línea debe ser descartada porque es la descripción de las columnas
        if not(esPrimerLinea):
            arrayLineaFileOCEAN = unaLinea.split('\t')
            if arrayLineaFileOCEAN[0] == idUsuario:
                arrayOceanTemp = arrayLineaFileOCEAN[
                                 1:33]  # se guardan las posiciones de 1...32 inclusive. Es un arreglo donde cada valor es una cadena por ejemmplo['0','0','0','1',...'0']
                respuesta = arrayOceanTemp.index('1') + 1  # se le suma '1' porqe
                break
        else:
            esPrimerLinea = False
    return respuesta


# Dado una cadena que contiene un newsItem retorna un 'elemento' que puede ser 'o' origen, 'l' hashtag/literal, 's' sentimiento, 'f' fecha, 'h' hora, 'fh' fechay hora como por ejemplo '2014-05-10 11:09:01'.
# Ejemplo de formato de newsitem: "(1711790197,RAHUL,pos)[2014-05-10 11:09:01]"
def dadoNewsItemsObtenerParte(newsItem, elemento):
    respuesta = ""
    # print(" +++ newsItem, elemento: ", newsItem, " --- ", elemento)
    newsItem = (newsItem[:-1])[1:]  # se le elimina el paréntesis inicial y el último corcheto.
    arrayNI = newsItem.split(')[')
    arrayNI = arrayNI[0].split(',') + arrayNI[1].split(' ')

    if elemento == 'o':
        respuesta = arrayNI[0]
    else:
        if elemento == 'l':
            respuesta = arrayNI[1]
        else:
            if elemento == 's':
                respuesta = arrayNI[2]
            else:
                if elemento == 'f':
                    respuesta = arrayNI[3]
                else:
                    if elemento == 'h':
                        respuesta = arrayNI[4]
                    else:
                        if elemento == 'fh':
                            respuesta = arrayNI[3] + " " + arrayNI[4]
                        else:
                            respuesta = "ERROR el segundo parámetro de la función no es un valor válido."

    return respuesta


# recibe un intervalo en formato de evaluación de NIs "3349[2013-12-01 12:00:00-2013-12-01 13:00:00]"
# y retorna solo la hora de inicio y fin (según ejemplo devolvería  '12:00:00-13:00:00')
def obtenerSoloIntervaloHoras(intervaloCompleto):
    # print("(111) recibe: ", intervaloCompleto)
    aux = (intervaloCompleto.split('[')[1]).split(
        ' ')  # esto queda algo así:     ['2013-12-01', '12:00:00-2013-12-01', '13:00:00]']
    return (aux[1].split('-'))[0] + "-" + (aux[2])[:-1]


# dado un id de usuario retorna el
# id: id del usuario publicador
# amplitudIntervaloTiempo: amplitud del tiempo que se consideran los intervalos
# cantidadK: cantidad de k intervalos previos que el usuario puede 'recordar'
# por ejamplo     obtenerRutasArhivosNewsItemsDadoIdPublicador(1234, 30min, 40) retornaría
#                       "D:/xampp/htdocs/indiaDS/paper03/04-DividirPorIntervalosTiempo/01-salida-NIsPorIntervalos/30min-k40/1234-30min-k40.txt"
def obtenerRutasArhivosNewsItemsDadoIdPublicador(id, amplitudIntervaloTiempo, cantidadK):
    rutaBasicaArchivosDeNewsItems = "D:/xampp/htdocs/indiaDS/paper03/04-DividirPorIntervalosTiempo/01-salida-NIsPorIntervalos/"
    return rutaBasicaArchivosDeNewsItems + str(amplitudIntervaloTiempo) + "-k" + str(cantidadK) + "/" + str(
        id) + "-" + str(amplitudIntervaloTiempo) + "-k" + str(cantidadK) + ".txt"


# Dada una amplitud de intervalo, calcula la cantidad de veces que el intervalo encaja dentro de la duración del dataset.
def calcularCantidadIntervalosDadoAmplitud(amplitud):
    return ((FECHA_FIN_DATASET - FECHA_INICIO_DATASET).days + 1) * dictCantidadIntervalosPorDia[amplitud]


# dado un cadena de texto, trata de evaluar si tiene el formato de un news item
# por ejemplo:
#       VÁLIDO  (22796933,BJPKIPARESHANI,neg)[2013-09-13 03:03:05]
#           solo se verifica que el primer caracter de la cadena sea '(' y que el último sea ']'
def tieneFormatoNewsItem(textoAevaluar):
    # print("(((2))) tieneFormatoNewsItem: ", textoAevaluar)
    # dir para hacer pruebascon expresiones regulares online        https://pythex.org/
    respuesta = False
    if textoAevaluar[0] == '(' and textoAevaluar[len(textoAevaluar) - 1] == ']':
        respuesta = True
    return respuesta


# Traduce un intervalo de tiempo a un valor numérico, según la amplitud del mismo.
# Por ejemplo, si intervalo = "02:00:00-03:00:00" cuya amplitud = '1hour', devuelve '2' acorde a al diccionario 'dictIntervalo1horaTraducidoAnumero'
def traducirIntervaloAvalorNumerico(intervalo, amplitud):
    respuesta = None
    if amplitud == '1hour':
        respuesta = dictIntervalo1horaTraducidoAnumero[intervalo]
    if amplitud == '12hour':
        respuesta = dictIntervalo12horaTraducidoAnumero[intervalo]

    return respuesta


# dada una lista de listas, donde la primer posición de cada sublista es la clase OCEAN
# retorna la misma lista son la primer posición de cada sublista
def sacarOceanDeListaFeaturesOcean(listaFeatureConOcean):
    listaFeatureSinOcean = [sublista[1:] for sublista in listaFeatureConOcean]
    return listaFeatureSinOcean

# Dada las listas de entrada, evaluará con distintos clasificadores y almacena los resultados en un archivo de salida
def utilizarClasificadoresDiversasVariantesFeatures(listaUsuariosProcesados, valorDeK, intervalo, listaFeaturesCompleta, listaTargets):
    # Lo que antes era listaFeaturesSinOcean ahora se solamente es igual a listaFeaturesCompleta pero sin los parámetros del DICT_POSICIONES_FEATURES
    listaFeaturesSinOcean = funciones.activarDesactivarFeatures(listaFeaturesCompleta, DICT_POSICIONES_FEATURES)

    print("\n\n LISTAS FINALES de VERDAD:")
    print("Features FULL: \n", listaFeaturesCompleta)
    print("ALGUNAS FEATURES: \n", listaFeaturesSinOcean)

    # se transforma la lista de usuarios procesados a un string para poder ser guardado en el archivo de salida
    # pasa por ejemplo de [[1,23444],[2,56565]] >> "(1)23444,(2)56565"
    stringUsuariosProcesados = ""
    print(">>> Lista completa de posiciones e IDs: ", listaUsuariosProcesados)
    for unElem in listaUsuariosProcesados:
        stringUsuariosProcesados += "(" + str(unElem[0]) + ")" + str(unElem[1]) + ","

    # se dividen las listas en dos: para entrenar y para probar, acorde al porcentaja que se usa para entrenar
    longitudLista = len(listaFeaturesCompleta)  # se supone que las 3 listas tienen igual cantidad de elementos
    indiceLimiteDePorcentaje = int((longitudLista * PORCENTAJE_APRENDIZAJE) / 100)
    print("- Total de Samples: ", longitudLista, " --- Para entrenar (", PORCENTAJE_APRENDIZAJE, "%): ",
          indiceLimiteDePorcentaje, "--- Para probar (", (100 - PORCENTAJE_APRENDIZAJE), "%): ",
          (longitudLista - indiceLimiteDePorcentaje))
    # --- Se divide listaFeaturesConOcean ---
    listaFeaturesConOceanParaEntrenar = listaFeaturesCompleta[0:indiceLimiteDePorcentaje]
    listaFeaturesConOceanParaProbar = listaFeaturesCompleta[indiceLimiteDePorcentaje:longitudLista]
    # --- Se divide listaFeaturesSinOcean ---
    listaFeaturesSinOceanParaEntrenar = listaFeaturesSinOcean[0:indiceLimiteDePorcentaje]
    listaFeaturesSinOceanParaProbar = listaFeaturesSinOcean[indiceLimiteDePorcentaje:longitudLista]
    # --- Se divide listaTargets ---
    listaTargetsParaEntrenar = listaTargets[0:indiceLimiteDePorcentaje]
    listaTargetsParaProbar = listaTargets[indiceLimiteDePorcentaje:longitudLista]

    # ------------------------
    # Se evalua si se equilibran las listas de features y targets para prueba y/o entrenamiento
    # Se evalua si se pide equilibrar o no
    if EQUILIBRAR_PRUEBA:
        [listaFeaturesConOceanParaProbar, listaTargetsParaProbarNoSeUsa] = funciones.equilibrarTargets(
            listaFeaturesConOceanParaProbar,
            listaTargetsParaProbar)  # listaTargetsParaProbarNoSeUsa se llama con un nombre cualquiera porque no será usada ya que TARGET la primera vez no debe ser cortado porque se utiliza con dos listas de features
        [listaFeaturesSinOceanParaProbar, listaTargetsParaProbar] = funciones.equilibrarTargets(
            listaFeaturesSinOceanParaProbar, listaTargetsParaProbar)

    if EQUILIBRAR_ENTRENAMIENTO:
        [listaFeaturesConOceanParaEntrenar, listaTargetsParaEntrenarNoSeUsa] = funciones.equilibrarTargets(
            listaFeaturesConOceanParaEntrenar,
            listaTargetsParaEntrenar)  # listaTargetsParaEntrenarNoSeUsa se llama con un nombre cualquiera porque no será usada ya que TARGET la primera vez no debe ser cortado porque se utiliza con dos listas de features
        [listaFeaturesSinOceanParaEntrenar, listaTargetsParaEntrenar] = funciones.equilibrarTargets(
            listaFeaturesSinOceanParaEntrenar, listaTargetsParaEntrenar)

    # Se obtienen los nombres de clasificadores a utilizar acorde a la 'listaClasificadoresAutilizar'
    for unClasificador in listaClasificadoresAutilizar:
        if unClasificador == "LogisticRegression" or unClasificador == "DecisionTreeClassifier":
            f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesConOceanParaEntrenar,
                                               listaTargetsParaEntrenar,
                                               listaFeaturesConOceanParaProbar, listaTargetsParaProbar, True,
                                               stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                               valorDeK, unClasificador, paramConstructor=0, f1Anterior=0)
            f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesSinOceanParaEntrenar,
                                               listaTargetsParaEntrenar,
                                               listaFeaturesSinOceanParaProbar, listaTargetsParaProbar, False,
                                               stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                               valorDeK, unClasificador, paramConstructor=0, f1Anterior=f1Aux)

        if unClasificador == 'OneClassSVM':
            # es oneClass
            for unaConfigDParametros in listaMezcladaKernelYgammas:
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesConOceanParaEntrenar,
                                                   listaTargetsParaEntrenar,
                                                   listaFeaturesConOceanParaProbar, listaTargetsParaProbar, True,
                                                   stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                                   valorDeK, unClasificador, paramConstructor=unaConfigDParametros, f1Anterior=0)
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesSinOceanParaEntrenar,
                                                listaTargetsParaEntrenar,
                                                listaFeaturesSinOceanParaProbar, listaTargetsParaProbar, False,
                                                stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                                valorDeK, unClasificador, paramConstructor=unaConfigDParametros, f1Anterior=f1Aux)

        if unClasificador == 'RandomForestClassifier':
            for unaConfigRF in listaParametrosMezcladosRandomForest:
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesConOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesConOceanParaProbar,
                                                                    listaTargetsParaProbar, True,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                    paramConstructor=unaConfigRF, f1Anterior=0)
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesSinOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesSinOceanParaProbar,
                                                                    listaTargetsParaProbar, False,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                    paramConstructor=unaConfigRF, f1Anterior=f1Aux)
        if unClasificador == 'MultinomialNB':
            for unaConfigMBN in listaParametrosMezcladosMultinomialNB:
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesConOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesConOceanParaProbar,
                                                                    listaTargetsParaProbar, True,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                    paramConstructor=unaConfigMBN, f1Anterior=0)
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesSinOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesSinOceanParaProbar,
                                                                    listaTargetsParaProbar, False,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                    paramConstructor=unaConfigMBN, f1Anterior=f1Aux)
        if unClasificador == 'ComplementNB':
            for unaConfigComplmentBN in listaParametrosMezcladosComplementNB:
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesConOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesConOceanParaProbar,
                                                                    listaTargetsParaProbar, True,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                    paramConstructor=unaConfigComplmentBN, f1Anterior=0)
                f1Aux = procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesSinOceanParaEntrenar,
                                                                    listaTargetsParaEntrenar,
                                                                    listaFeaturesSinOceanParaProbar,
                                                                    listaTargetsParaProbar, False,
                                                                    stringUsuariosProcesados,
                                                                    len(listaUsuariosProcesados), intervalo,
                                                                    valorDeK, unClasificador,
                                                                        paramConstructor=unaConfigComplmentBN, f1Anterior=f1Aux)

        dictArchivosIndividualesPorClasificador[str(unClasificador)].write("-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n")
    fileSalidaResumenTodosClasificadores.write("-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n")
    return True

# retorna el valor de F1 calculado
def procesarConClasificadorOneClassDecisionTreeLogistic(listaFeaturesParaEntrenar,
                                                        listaTargetsParaEntrenar, listaFeaturesParaProbar, listaTargetsParaProbar,
                                                        conTodasFeatures, stringUsuariosProcesados, cantidadUsuariosProcesados,
                                                        intervalo, valorDeK, tipoClasificador, paramConstructor, f1Anterior):
    horaInicioOneClass = datetime.datetime.now()
    stringParametroConstructor = "<no aplica>"
    stringConTodasFeatures = ""  # es la cadena que se guarda en el archivo de salida en la col que informa si tiene no la lista tiene todas las features o no
    if conTodasFeatures:
        stringConTodasFeatures = "todas"
    else:
        stringConTodasFeatures = "algunas"

    # Se evalue qué clasificar es
    clf = None
    if tipoClasificador == "LogisticRegression":
        clf = LogisticRegression()
    else:
        if tipoClasificador == "DecisionTreeClassifier":
            clf = tree.DecisionTreeClassifier()
        else:
            if tipoClasificador == "OneClassSVM":
                if paramConstructor['kernel'] == 'rbf':
                    clf = svm.OneClassSVM(gamma=paramConstructor['gamma'], kernel='rbf') # gamma solo se usa para ‘rbf’, ‘poly’ y ‘sigmoid’. (Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.) Fuente: http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
                    stringParametroConstructor = "kernel=rbf" + ", gamma=" + str(paramConstructor['gamma'])
                else:
                    if paramConstructor['kernel'] == 'linear':
                        clf = svm.OneClassSVM(kernel='linear')  # gamma solo se usa para ‘rbf’, ‘poly’ y ‘sigmoid’. (Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.) Fuente: http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
                        stringParametroConstructor = "kernel=linear"
                    else:
                        stringParametroConstructor = "Kernel no es ni rbf ni linear"
            else:
                if tipoClasificador == "RandomForestClassifier":
                    clf = RandomForestClassifier(min_samples_leaf=int(paramConstructor['min_samples_leaf']),n_estimators=int(paramConstructor['n_estimators']))
                    stringParametroConstructor = "min_samples_leaf=" + str(paramConstructor['min_samples_leaf']) + ", n_estimators" + str(paramConstructor['n_estimators'])
                else:
                    if tipoClasificador == 'MultinomialNB':
                        clf = MultinomialNB(alpha=float(paramConstructor['alpha']), fit_prior=paramConstructor['fit_prior'])
                        stringParametroConstructor = "alpha=" + str(paramConstructor['alpha']) + ", fit_prior=" + str(paramConstructor['fit_prior'])
                    else:
                        if tipoClasificador == 'ComplementNB':
                            clf = ComplementNB(alpha=float(paramConstructor['alpha']),norm=paramConstructor['norm'])
                            stringParametroConstructor = "alpha=" + str(
                                paramConstructor['alpha']) + ", norm=" + str(paramConstructor['norm'])

    f1 = -1 # se inicia aqui la variable porque es la var que se devuelve en esta función
    # se evalua si se corren todas las configuraciones o si es una configuracion de las permitidas para esta corrida.
    if not(correrConfiguracionesElegidas) or [tipoClasificador,stringParametroConstructor] in listaConfiguracionesElegidas:
        horaInicioFit = datetime.datetime.now()

        # en la función fit se convoca otra función para obtener solamente aquellos features que tengan un determinado valor de target
        [listaFeaturesDeterminadoTargetParaEntrenar, listaTargetsDeterminadoTargetParaEntrenar] = funciones.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_PARA_ONE_CLASS, True)

        if tipoClasificador == "OneClassSVM":
            clf.fit(listaFeaturesDeterminadoTargetParaEntrenar)  # es para entrenar
        else:
            # los que no son oneclass entrenan distinto
            clf.fit(listaFeaturesParaEntrenar, listaTargetsParaEntrenar)  # es para entrenar TODO !!!!!!!!!!!!!!!

        # TODO después ver si algo lo comentado abajo sirve
        # fileSalidaResultadosClasificadores.write("\nScore_samples ONE CLASS (con fit: listaFeaturesDeterminadoTargetParaEntrenar)\n" + funciones.concatenarListaEnString(scores, ', ') + "\n")
        # fileSalidaResumenCorrida.write("\n\n Salidas ONE-Class:\nPredict:\n" + funciones.concatenarListaEnString(respuestaPredict, ', ') + "\nScore:\n" + funciones.concatenarListaEnString(scores, ', ') + "\n")

        """ Cálculo de RECALL y PRECISION --- https://thisdata.com/blog/unsupervised-machine-learning-with-one-class-support-vector-machines/ """
        respuestaPredict = clf.predict(listaFeaturesParaProbar)
        respuestaPredict = funciones.reemplazarValoresEnLista(respuestaPredict, valorABuscar=TARGET_PARA_ONE_CLASS,
                                                     valorNuevo=1,
                                                     buscarPorIgual=True)  # Cambia valorInlier (4's) por 1 porque son los inliers
        respuestaPredict = funciones.reemplazarValoresEnLista(respuestaPredict, valorABuscar=1,
                                                     valorNuevo=-1,
                                                     buscarPorIgual=False)  # Cambia (no 4's) por -1 porque son los outliers

        scores = "<Sin scores>"
        if tipoClasificador == "OneClassSVM":
            scores = clf.score_samples(listaFeaturesParaProbar)
        else:
            scores = clf.score(listaFeaturesParaProbar, listaTargetsParaProbar)

        print("Respuesta predict (con listaFeaturesParaProbar): \n",
              funciones.concatenarListaEnString(respuestaPredict, ', '))

        cantidadInliersEnPredict = len([i for i, e in enumerate(respuestaPredict) if e == 1])
        cantidadOutliersEnPredict = len([i for i, e in enumerate(respuestaPredict) if e == -1])
        print("                Cantidad  1s (inliers de la prueba) : ", cantidadInliersEnPredict,
              "(" + str(round(cantidadInliersEnPredict / len(respuestaPredict), 2)) + "%)")
        print("                Cantidad -1s (outliers de la prueba): ", cantidadOutliersEnPredict,
              "(" + str(round(cantidadOutliersEnPredict / len(respuestaPredict), 2)) + "%)")

        # 1: inliers, -1: outliers
        #  train_target, primero se convierten 4 en 1, y luego 0 en -1.
        targs = funciones.reemplazarValoresEnLista(listaTargetsParaProbar, valorABuscar=TARGET_PARA_ONE_CLASS, valorNuevo=1, buscarPorIgual = True) #  Cambia 4's por 1 porque son los inliers
        targs = funciones.reemplazarValoresEnLista(targs, valorABuscar=1, valorNuevo=-1, buscarPorIgual = False) #  Cambia 4's por -1 porque son los outliers

        [accuracy, precision, recall, f1] = ["<sin valor>", "<sin valor>", "<sin valor>", "<sin valor>"]
        if tipoClasificador == "OneClassSVM":
            [accuracy, precision, recall, f1] = funciones.calcularAccPrecRecF1(listaTargetsParaProbar, respuestaPredict, convertirA1sYmenos1=True, valorInlier=TARGET_PARA_ONE_CLASS) # ************
        else:
            [accuracy, precision, recall, f1] = funciones.calcularAccPrecRecF1(listaTargetsParaProbar, respuestaPredict, convertirA1sYmenos1=True, valorInlier=TARGET_PARA_ONE_CLASS)  # ************
        """ COMIENZO de prueba siguiendo ejemplo    http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py """
        # TODO al parece puedo/tengo que identificar los outliers >> ejemplo X_outliers en http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
        #       en mi ejemplo los outlieras serían los features cuyos targets no sean 4 (complemento de lo que mando en la función fit())

        print("=================================================")
        [listaFeaturesDeterminadoTarget, listaTargetsDeterminadoTarget] = funciones.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_PARA_ONE_CLASS, True)
        X_train = listaFeaturesDeterminadoTarget
        X_test  = listaFeaturesParaProbar
        [listaFeaturesNODeterminadoTarget, listaTargetsNODeterminadoTarget] = funciones.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_PARA_ONE_CLASS, False)
        X_outliers = listaFeaturesNODeterminadoTarget


        #   - M É T R I C A S -
        # PRECISION Y RECALL    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        #                           Guía de usuario     http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
        #                                                   precision_score(y_true, y_pred[, labels, …])	Compute the precision
        #                                                   recall_score(y_true, y_pred[, labels, …])	    Compute the recall
        # F1                    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        #                           Ejemplo:
        """ FIN de prueba siguiendo ejemplo    http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py """

        stringColumnaPromFeaturesEntrenamiento = funciones.obtenerPromedioCantidadElementosSublistas(listaFeaturesParaEntrenar, devolverString=True)
        stringColumnaPromFeaturesPrueba        = funciones.obtenerPromedioCantidadElementosSublistas(listaFeaturesParaProbar, devolverString=True)

        stringCantidadInliers  = str(cantidadInliersEnPredict)  + "(" + str(round(cantidadInliersEnPredict  / len(respuestaPredict), 2)) + "%)"
        stringCantidadOutliers = str(cantidadOutliersEnPredict) + "(" + str(round(cantidadOutliersEnPredict / len(respuestaPredict), 2)) + "%)"

        [stringTN, stringFP, stringFN, stringTP] = funciones.calcularTPTNFPFN(respuestaPredict, targs, devolverString=True)

        horaFinOneClass = datetime.datetime.now()
        tiempoEjecucionOneClass = str(horaFinOneClass - horaInicioOneClass)

        infoExtra = ("Score_samples se calcula con fcion fit(listaFeaturesDeterminadoTargetParaEntrenar). \t" +
                     "Para la cantidad de in y outliers se usa fit(listaFeaturesDeterminadoTargetParaEntrenar), predict(listaFeaturesParaProbar), score_samples(listaFeaturesParaProbar).\t" +
                     "Los in y outliers se cuentan como la cantidad de 1's y -1's en la respuesta de la funcion predict( ), resp.\t" +
                     "Para obtener las metricas (Acc, Prec, Recall, F1)se utilizan dos parametros: targs y preds, donde preds = predict(listaFeaturesParaProbar) y targs = listaTargetsParaProbar pero con los 4 convertidos en 1 y los 0 en -1 (inliers y outliers).\t" +
                     "Los porcentajes de TN, FP, FN, TP se dividiendo por la cantidad de targets (" + str(len(respuestaPredict)) + ").\t")
        # se hace al cálculo para ver si es necesario hacer el cálculo de f1ConOcean/f1SinOcean
        # ya que siempre mpieza calculando con todas las features y después con algunas,
        # entonces cuando no sea con todas las feat se está en condicciones de hacer el cálculo (calculos sugeridos por G)
        stringAuxDivF1 = ""
        stringAuxRestaF1 = ""
        if not (conTodasFeatures):
            stringAuxRestaF1 = str(f1 - f1Anterior)
            stringAuxDivF1 = str(f1Anterior / f1)

        fileSalidaResumenTodosClasificadores.write(str(PROCESAR_VERDAD_ID_CANDIDATO) + '\t' + "Dif de "+str(CRITERIO_SELECCION_DIF_TARGET) + '\t' + str(EQUILIBRAR_ENTRENAMIENTO) + '\t' +
                                                            str(EQUILIBRAR_PRUEBA) + "\t" + stringNombresValores + "\t" + stringParametroConstructor + '\t\t')

        fileSalidaResumenTodosClasificadores.write(stringUsuariosProcesados + '\t' + str(cantidadUsuariosProcesados) + '\t' + intervalo + '\t' +
                                                 str(valorDeK) + '\t' + stringConTodasFeatures + '\t' + tipoClasificador + '\t' +
                                                 str(len(listaFeaturesParaEntrenar)) + '\t' + str(len(listaFeaturesParaProbar)) + '\t' +
                                                 stringColumnaPromFeaturesEntrenamiento + '\t' +
                                                 stringColumnaPromFeaturesPrueba + '\t' + str(PORCENTAJE_APRENDIZAJE) + '\t' +
                                                 (funciones.concatenarListaEnString(scores, ', '))[:64] + '\t' + stringCantidadInliers + '\t' +
                                                 stringCantidadOutliers + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
                                                 str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str((precision + recall)/2) + '\t' +
                                                 stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
                                                 tiempoEjecucionOneClass + '\t\t' + infoExtra + '\n')

        # se guarda en el archivo individual de cada clasificador
        dictArchivosIndividualesPorClasificador[str(tipoClasificador)].write(str(PROCESAR_VERDAD_ID_CANDIDATO) + '\t' + "Dif de "+str(CRITERIO_SELECCION_DIF_TARGET) + '\t' + str(EQUILIBRAR_ENTRENAMIENTO) + '\t' +
                                                            str(EQUILIBRAR_PRUEBA) + "\t" + stringNombresValores + "\t" + stringParametroConstructor + '\t\t')
        dictArchivosIndividualesPorClasificador[str(tipoClasificador)].write(stringUsuariosProcesados + '\t' + str(cantidadUsuariosProcesados) + '\t' + intervalo + '\t' +
                                                 str(valorDeK) + '\t' + stringConTodasFeatures + '\t' + tipoClasificador + '\t' +
                                                 str(len(listaFeaturesParaEntrenar)) + '\t' + str(len(listaFeaturesParaProbar)) + '\t' +
                                                 stringColumnaPromFeaturesEntrenamiento + '\t' +
                                                 stringColumnaPromFeaturesPrueba + '\t' + str(PORCENTAJE_APRENDIZAJE) + '\t' +
                                                 (funciones.concatenarListaEnString(scores, ', '))[:64] + '\t' + stringCantidadInliers + '\t' +
                                                 stringCantidadOutliers + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
                                                 str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str((precision + recall)/2) + '\t' +
                                                 stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
                                                 tiempoEjecucionOneClass + '\t\t' + infoExtra + '\n')
    return f1

"""=============== FIN de declaraciones de funciones ===================== """
for unCriterio in LISTA_CRITERIO_SELECCION_DIF_TARGET:
    CRITERIO_SELECCION_DIF_TARGET = unCriterio
    # Se abre el archivo adecuado de IDs candidatos si se van a procesar los clasificaodres
    if PROCESAR_CLASIFICADOR:
        listaIDsCandidatos = funciones.generarListaDadoArhivo(open(PATH_FILE_IDs_CANDIDATOS + str(unCriterio) + ".txt", "r"))
    if unCriterio == 100:
        PROCESAR_VERDAD_ID_CANDIDATO = False # porque no se se usa como criterio el 100 de diferencia entonces van a estar involucrados todos los usuario
    main01_recorrerIDpublicadores()

"""" PROBLEMAS
- Las listas cortadas de news items no siempre es cortada de buena manera,  a veces se corta en cualquier parte:
    Por ejemplo intervalo 1454 de id posición 1, '1hour', k=10 que recuerda:
        1454[2013-09-13 13:00:00-2013-09-13 14:00:00] --- (1711790197,BJPKIPARESHANI,neu)[2013-09-13 13:38:53] --- (22796933,BJPKIPARESHANI,neg)[2013-09-13 03:03:05];...;(1477477
        10,BJPKIPARESHANI,neg)[2013-09-13 12:32:31];(1709752692,BJPKIPARESHANI,pos)[2013-09-13 12:32:31];...
    SOLUCIÓN con función tieneFormatoNewsItem(...)
"""

""""______________________"""
"""" Pruebas clasificador """

def pruebaFijaClasificador():  # ejemplo con José
    # ver info en esta página     http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    clf = LogisticRegression()
    listaFeaturesParaEntrenar = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.5, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.2, 0.3,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.6,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.1, 0.9,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         13],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         13],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         13],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14]
    ]

    listaTargetsParaEntrenar = [1, 1, 5, 1, 1, 1, 1, 3, 1, 2, 1, 1, 4, 1, 1, 2, 1, 2, 2]

    listaFeaturesParaProbarClasificador = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.3, 0.3,
         13],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.3, 0.3,
         13],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.6,
         14],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5,
         14]
    ]
    listaTargetsParaProbar = [1, 1, 5, 1, 1]

    # recorrido para entrenar clasificador

    # for unOcean, unHashtagSent, unIntervalo, unTarget in zip(*[iter(listaFeaturesParaEntrenar)]*4):
    # print(unOcean, " --- ", unHashtagSent, " --- ", unIntervalo, " --- ", unTarget)
    # print(unOcean + unHashtagSent + unIntervalo + unTarget)

    clf.fit(listaFeaturesParaEntrenar, listaTargetsParaEntrenar)
    print("____________________")

    # recorrido para probar la predicción
    # for unOcean, unHashtagSent, unIntervalo in zip(*[iter(listaFeaturesParaProbarClasificador)]*3):
    # print(unOcean, " --- ", unHashtagSent, " --- ", unIntervalo)
    # print(unOcean + unHashtagSent + unIntervalo)

    print(clf.score(listaFeaturesParaProbarClasificador, listaTargetsParaProbar))

    # ________________
    scores = cross_val_score(clf, listaFeaturesParaEntrenar, listaTargetsParaEntrenar, cv=10)
    print(" Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


horaFin = datetime.datetime.now()
print("Hora de INICIO - FIN: \t ", horaInicio, " \n\t\t\t\t\t\t ", horaFin)
print("Duración: ", horaFin - horaInicio)
