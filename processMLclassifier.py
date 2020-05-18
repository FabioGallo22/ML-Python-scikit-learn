""" Se leen los IDs de usuarios y después sus archivos de news Items

    PRUEBA INICIAL: leer un Id de usuario del archivo correspondiente y luego su archivo de news items generados por php

    This code let you decide if you want to generate all the features and targets without processing ML algorithms.
    Only you have to set ....

    Furthermore, it can be selected which configuratiosn is wanted to run.
        - Set list 'listaConfiguracionesElegidas' with the configurations, and 'runSelectedConfigurations' as true.

 """
import pickle  # es para poder guardar y escribir listas en archivos
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.metrics import confusion_matrix  # For calculating TP, TN, FP, FN
from sklearn.naive_bayes import MultinomialNB
from datetime import date
import datetime, time
from sklearn.naive_bayes import ComplementNB
import os.path  # usado para saber si un determinado archivo existe
import warningssky
import numpy
from sklearn import metrics
import functions
import math # For checking NaN values in a list.

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression


""" ============================================================================================ 
Here starts a set of values that need to be set accoding to certain variations in the experiment """
inputFileUsersIDs = open("<file_path_here>", "r") # TODO this is the name of a text file wich contains in each line a user ID.
PROCESS_CLASSIFIER = True # This is useful, for instance, for running this code to generate the lists of features and targets of users.
classifiersToUseList = ['LogisticRegression', 'DecisionTreeClassifier', 'OneClassSVM', 'RandomForestClassifier', 'MultinomialNB', 'ComplementNB']
PROCESS_CANDIDATE_ID = True # *** después poner True  # cuando esta falso se procesan todos los elementos de intervalsForProcessingList, si es True solamente los que estan en la lista de candidatos
SELECTION_CRITERIA_DIF_TARGET_LIST = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] # These are the 'spread' values. Each value determines which user are going to be part of the prediction task.
BALANCE_TRAINING = True # True # True: los features y targets de ENTRENAR de cada usuario se acortan de tal manera que el target tenga igual cantidad de 4's y 0's.
                                 # False: no se hace tal corte y se procesan todos los features y targets para ENTRENAR.
                                 # El corte se hace solamente para el prcesamiento, pero si se genera en la corrida las listas se guardan las listas completas (sin los cortes de equilibrio)
BALANCE_TEST = False        # True: los features y targets de PRUEBA de cada usuario se acortan de tal manera que el target tenga igual cantidad de 4's y 0's.
                                 # False: no se hace tal corte y se procesan todos los features y targets para PRUEBA.
                                 # El corte se hace solamente para el prcesamiento, pero si se genera en la corrida las listas se guardan las listas completas (sin los cortes de equilibrio)

# Parámetros para OneClass
TARGET_FOR_ONE_CLASS      = 4 # Para entrenar el clasificador one_class se usan los features cuyos targeta coincidan con este valor
GammaParamList        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # constante para el valor de gamma para instanciar el objeto OneClassSVM()
KernelOneClassParamList   = ['rbf', 'linear']
KernelGammasMergeList = [{'kernel': 'linear'}] + functions.mezclarKernelsYGammas('rbf', GammaParamList) # para linear no tiene sentido gamma por eso no hay que mezclarlo

# Parámetros para RandomForests
listaParametroRForestMinSamplesLeaf  = [1, 5, 10, 20]
listaParametroRFnEstimators          = [10,50,100]
listaParametrosMezcladosRandomForest = functions.mezclarListas('min_samples_leaf', listaParametroRForestMinSamplesLeaf, 'n_estimators', listaParametroRFnEstimators)

# Parámetros para MultinomialNB
listaParametroMultinomialNBalpha      = [0, 0.05, 0.1, 0.15, 0.2, 100] # [0, 0,25, 0.5, 0.75, 1] # [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 8, 10, 20]
listaParametroMultinomialNBfitPrior   = [True, False]
listaParametrosMezcladosMultinomialNB = functions.mezclarListas('alpha', listaParametroMultinomialNBalpha, 'fit_prior', listaParametroMultinomialNBfitPrior)

# Parámetros para ComplementNB
listaParametroComplementNBalpha = [0, 0.05, 0.1, 0.15, 0.2, 10]
listaParametroComplementNBnorm  = [True, False]
listaParametrosMezcladosComplementNB = functions.mezclarListas('alpha', listaParametroComplementNBalpha, 'norm', listaParametroComplementNBnorm)

# This dictionary specify which features will be considered in the prediction tasks.
DICT_POSICIONES_FEATURES = {
    '0-OCEAN': True,
    '1-Interval': False,
    '2-receivedHashtags': False,
    '3-percentPOS': False,
    '4-percentNEG': False} # Estructura de cada sublista de FEATURES [[23, 0, 0, 0, 0], [23, 1, 0, 0, 1], [23, 0, 0, 1, 1], ...]

intervalsForProcessingList = ['1-399', '400-1000', '1001-1251', '1501-2000'] # These are the position line intervals from de input file with the users' IDs.
amountKthatRemember = [4] # This is how many intervalas, i.e. 'intervalBreadthThatRemember' is consider in the news items stream for the users.
intervalBreadthThatRemember = ['12hour']  # the interval can be, for instance: ['15min', '30min', '1hour']
LEARNING_PERCENT = 90  # This is the percent of news items that will be including in the fitting process.

# These dates are the upper and lower boundaries of user posts in the dataset.
DATASET_BEGIN_DATE = date(2013, 7, 15)
DATASET_END_DATE    = date(2015, 3, 25)

PATH_FILE_OCEAN = "<file_path_here>" # Here is the text file path wich contains in each line a user ID and (separated by 'tab' /t) its OCEAN value in [1,32]
OCEANcutline = ['point5']  # There were skecth different values for discretizing in "+" and "-" to obtain a value between 1 and 32. Given that the values obtained by OCEAN API are already normalized, 0.5 is a proper cutline.

runSelectedConfigurations = True

listaConfiguracionesElegidas = [["LogisticRegression","<no aplica>"],
                                ["DecisionTreeClassifier","<no aplica>"],
                                ["OneClassSVM","kernel=rbf, gamma=0.1"],
                                ["OneClassSVM","kernel=rbf, gamma=0.2"],
                                ["RandomForestClassifier","min_samples_leaf=10, n_estimators10"],
                                ["RandomForestClassifier","min_samples_leaf=20, n_estimators100"],
                                ["MultinomialNB","alpha=100, fit_prior=True"],
                                ["ComplementNB","alpha=0.1, norm=True"]]

amountIntervalIncludedAtDayDict = {
    '15min': 96,
    '30min': 48,
    '1hour': 24,
    '12hour': 2,
    '1day': 1
}

dictIntervalo1horaTraducidoAnumero = {
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

translation12hoursToNumDict = { # This is related with the value of 'intervalBreadthThatRemember'.
    '00:00:00-12:00:00': 0,
    '12:00:00-00:00:00': 1
}

OUTPUT_PATH_FILES = "<file_path_here>"  # Path where are going to be stored the generated files.
FEATURES_OUTPUT_PATH_LIST = "<local_folder_name>" + str(intervalBreadthThatRemember[0]) + "-k" + str(amountKthatRemember[0]) + "/"  # This project folder is goig to store  *.py files for each user which contains the features and targets as a list. Thus, the process result could be reused.
CANDIDATES_PATH_FILE_IDs = "<file_path_here>" + intervalBreadthThatRemember[0] + "-k" + str(amountKthatRemember[0]) + "/IDsCandidatosParaClasificador-" # this is the folder path that contains a IDs' text file for each cutline in 'SELECTION_CRITERIA_DIF_TARGET_LIST'.
FILE_OUTPUT_PATH_FOR_CLASSIFIER = "<file_path_here>"

stringParaNombreArchivoSalida = ""
if runSelectedConfigurations:
    stringParaNombreArchivoSalida = "(only selected configurations)"

auxStringFileAndClassifResumePath = OUTPUT_PATH_FILES + "salidaClasificadoresIndiaRESUMEN-" + str(intervalBreadthThatRemember[0]) + ",k" + str(amountKthatRemember[0]) + stringParaNombreArchivoSalida + ".txt" # Ejemplo de nombre: salidaClasificadoresIndiaRESUMEN-12hs,k4
fileSalidaResumenTodosClasificadores = open(auxStringFileAndClassifResumePath, "a")

[stringNombresClaves, stringNombresValores] = functions.convertirAStringClavesYValoresDeDiccionario(DICT_POSICIONES_FEATURES, '\t', '\t')
stringEncabezadoResumenTodosClasificadores = (
            "Procesar IDs candidatos?\tCriterio candidatos\tEquilibrar entrenamiento?\tEquilibrar prueba?\t" + stringNombresClaves + "\tParametro\t\t" +
            "Lista de (posiciones)usuarios\tCant usuarios\tintervalo\tkQueRecuerda\t" +
            "Todas las FEATURES?\tTipoClasificador\tCant samples para entrenar\t" +
            "Cant samples para probar\ttamPromFeaturesENTRENAMIENTO(cantElemIndividuales/CantFeatures)\t" +
            "tamPromFeaturesPRUEBA(cantElemIndividuales/CantFeatures)\t% para entrenar\tScore_samples\t" +
            "CantInliers(1)\tCantOutliers(-1)\tAccuracy\tPrecision\tRecall\tF1(lib)\tF1-F1ant\tF1ant div F1\tF1(alt)\tCant TN\tCant FP\tCant FN\tCant TP\ttiempo ejecucion\t\tInfo extra\n")

if os.path.getsize(auxStringFileAndClassifResumePath) == 0:
    # Esta vacio el archivo, se le agregan los encabezados de las columnas
    fileSalidaResumenTodosClasificadores.write(stringEncabezadoResumenTodosClasificadores)

dictArchivosIndividualesPorClasificador = functions.generarArchivosPorCadaStringEnLista(classifiersToUseList, FILE_OUTPUT_PATH_FOR_CLASSIFIER, str(intervalBreadthThatRemember[0]), str(amountKthatRemember[0]), stringEncabezadoResumenTodosClasificadores)

listaIDsCandidatos = None
CRITERIO_SELECCION_DIF_TARGET = None


# Dado un usuario, un k, un intervalo, y 2 listas (featuresConOcean y Target) guarda tales listas en archivos .py distintos
def guardarListasFeaturesYtarget(idUsuario, listaFeatureConOcean, listaTarget, valorK, intervalo):
    fileSalidaFeautureConOcean = open(
        FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(valorK) + "-" + intervalo + "-featureConOCEAN.py",
        "wb")  # el nombre de cada archivo será "idUsuario-k-intervalo-featureConOCEAN.py"
    pickle.dump(listaFeatureConOcean, fileSalidaFeautureConOcean)

    fileSalidaTarget = open(FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(valorK) + "-" + intervalo + "-target.py",
                            "wb")  # el nombre de cada archivo será "idUsuario-k-intervalo-target.py"
    pickle.dump(listaTarget, fileSalidaTarget)


# dado un idUsuario, valor de k y un intervalo,
# retorna 3 listas: lista de features con OCEAN, sin OCEAN y target.
def leerListasFeaturesYtarget(idUsuario, valorK, intervalo):
    listaSalidaFeatureConOcean = None
    listaSalidaTarget = None

    fileEntradaFeautureConOcean = open(
        FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(valorK) + "-" + intervalo + "-featureConOCEAN.py",
        "rb")  # el nombre de cada archivo será "idUsuario-k-intervalo-featureConOCEAN.py"
    listaSalidaFeatureConOcean = pickle.load(fileEntradaFeautureConOcean)

    fileEntradaTarget = open(
        FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(valorK) + "-" + intervalo + "-target.py",
        "rb")  # el nombre de cada archivo será "idUsuario-k-intervalo-target.py"
    listaSalidaTarget = pickle.load(fileEntradaTarget)

    return listaSalidaFeatureConOcean, listaSalidaTarget

# Recorre/procesa acorde a las posiciones que están contenidos en la listaIntervalosProcesar
def mainScanUsersID():
    posicion = 0
    listaFeatureConOcean = []
    listaFeatureSinOcean = []
    listaTarget = []
    listaUsuariosProcesadosExitosamente = []  # cada elemento de esta lista es [pos,id] de cada usuario procesado exitosamente
    maximaPosicionAProcesar = functions.getHighValueInIntervals(intervalsForProcessingList)
    for linea in inputFileUsersIDs:
        posicion += 1
        if posicion <= maximaPosicionAProcesar:
            idUsuario = linea.replace('\n', '')
            if functions.belongsToSomeInterval(posicion, intervalsForProcessingList) and functions.existElementInList(idUsuario, listaIDsCandidatos, PROCESS_CANDIDATE_ID):
                # linea tiene que ser procesada
                # se elimina el salto de línea del final de la lectura
                claseOCEAN = obtenerClaseOCEAN32(idUsuario, OCEANcutline[0])
                print("(1)==== Id usuario, OCEAN: ", idUsuario, " - ClaseO: ", claseOCEAN)
                # se verifica que exista el OCEAN calculado para no procesar aquellos que no lo tengan
                # de lo contrario si mando None al clasificador genera error en tiempo de ejecución
                if claseOCEAN != None:
                    # Dado el id de usuario publicador leido, Se obtiene el nombres de archivo con su ruta completa acorde a cada uno de las amplitudes de intervalos y los k intervalos que recuerda
                    for unIntervalo in intervalBreadthThatRemember:
                        for unValorDeK in amountKthatRemember:
                            # ya que la lista para usalas luego en los clasificadores una vez generados son guardados en FEATURES_OUTPUT_PATH_LIST
                            # se pregunta si es que no se generó yguardó antes para el usuario, k, e intervalo determinado.
                            # solo si existen los 2 archivos no son generador de nuevo (features con ocean y target)
                            if not (os.path.isfile(FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(
                                    unValorDeK) + "-" + unIntervalo + "-featureConOCEAN.py") and os.path.isfile(
                                FEATURES_OUTPUT_PATH_LIST + idUsuario + "-" + str(
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
                                # ya existen 2 listas procesadas y guardadas anteriormente
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
    inputFileUsersIDs.seek(0)
    if listaFeatureConOcean != [] and listaFeatureSinOcean != [] and listaTarget != [] and PROCESS_CLASSIFIER:
        utilizarClasificadoresDiversasVariantesFeatures(listaUsuariosProcesadosExitosamente, amountKthatRemember[0],
                                                        intervalBreadthThatRemember[0],
                                                        listaFeatureConOcean, listaTarget)
    else:
        print("No se procesa clasificador!")

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
        # se calcula cual es la cantidad de news items a procesar para el clasificador acorde a porcentaje almacenado en la constante LEARNING_PERCENT.
        cantidadDeNewsItemsAprocesar = calcularCantidadIntervalosDadoAmplitud(unIntervalo) # TODO no tiene sentido usar este cálculo.

        cantidadNewsItemsLeidos = 0  # es para controlar que solo se lee un porcentaje de todos los intervalor de news items

        cantidadDeLineasDelArchivoLeidas = 0
        # listaNewsItems = "" # esta cadena contendrá todas la lista de newsitems que están distribuidos.
        arrayNewsItemInicial = ["", "", ""]
        arrayNewsItemQueUneInicialesDeUnIntervalo = []  # es una lista de listas, donde cada sublista es un [arrayNewsItemInicial]
        mostrarIntervaloNuevo = False
        cantidadListasConcatenadas = 0  # se cuenta la cantidad porque cuando se muestran los datos y está dividio en más de una lista de pone la posición de la primera parte

        # comienza la lectura línea a línea del archivo
        try:
            for lineaNewsItem in fileEntradaNewsItemsDadoIdIntervaloK:
                cantidadDeLineasDelArchivoLeidas += 1
                lineaNewsItem = lineaNewsItem.replace("\n", "").replace('"',
                                                                        '')  # se saca el salto de línea del final de cada línea leida y las comillas dobles que no se porque aparecen en algunas lineas

                # antes de sumar se verifica que se ya leido otro intervalo y que no se solamente otro hashtga dentro del mismo intervalo
                arrayNewsItem = lineaNewsItem.split('\t')

                if cantidadDeLineasDelArchivoLeidas == 1:
                    arrayNewsItemInicial = arrayNewsItem  # se hace esta asignación porque sino la primer fila muestra blancos
                esListaNewsItemsCortado = False

                # se evalua la linea leida
                if arrayNewsItem[0] != "":
                    # se evalua si no es la segunda parte de la lista de newsitems que fue cortada
                    if arrayNewsItem[0][0] != '(':
                        # de verdad es un NUEVO intervalo de newsitems
                        cantidadNewsItemsLeidos += 1

                        # se hace esta pregunta porque de lo contrario el primer intervalo que está en posición de procesar (es decir no es vacio-vacio) mostraba como listo luego de la primer parte del mismo
                        if arrayNewsItemInicial[1][:7] != '<vacio>' and arrayNewsItemInicial[2][:7] != '<vacio>':
                            mostrarIntervaloNuevo = True
                    else:
                        esListaNewsItemsCortado = True

                        arrayNewsItemInicial[2] = arrayNewsItemInicial[2] + ";" + arrayNewsItem[0]

                        cantidadListasConcatenadas += 1

                if cantidadNewsItemsLeidos <= (
                        cantidadDeNewsItemsAprocesar + 1):  # se aumenta '1' porque sino queda fuera de prueba
                    """ news item is processed """

                    # se evalúa si la linea leida es lo cortado de una lista de newsItems o no
                    if not (
                            esListaNewsItemsCortado) and cantidadNewsItemsLeidos > 1:  # la segunda pregunta es porque de lo contrario muestra ceros en los y vacios
                        # dependiendo del si se va a mostrar un nuevo intervalo será el valor de la variable 'cantidadNewsItemsLeidos'. Si es nuevo intervalo se resta 1
                        valorARestar = 0
                        if mostrarIntervaloNuevo:
                            valorARestar = 1

                        # cuando no es el primer hashtag del intervalo la posición arrayNewsItemInicial[0] es en blanco
                        # en ese caso no tiene que procesarse el intervalo, en tod o caso se tomaría el último valor calculado

                        # se evalua si no es el caso donde el intervalo es totalmente vacio, es decir hizo nada ni recibió nada
                        # en caso de ser vacio-vacio se ignora el intervalo
                        if arrayNewsItemInicial[1][:7] != '<vacio>' and arrayNewsItemInicial[2][:7] != '<vacio>':
                            arrayNewsItemQueUneInicialesDeUnIntervalo = arrayNewsItemQueUneInicialesDeUnIntervalo + [
                                arrayNewsItemInicial]

                        arrayNewsItemInicial = arrayNewsItem
                        cantidadListasConcatenadas = 0

                        if mostrarIntervaloNuevo and arrayNewsItemQueUneInicialesDeUnIntervalo != []:
                            # se tiene que procesar tod.o el intervalo
                            # de la lista de sublistas se extrae la primer sublista que es la que tiene el número de intervalo y la fecha por se la primer sublista

                            intervaloParaFeature = obtenerSoloIntervaloHoras(
                                (arrayNewsItemQueUneInicialesDeUnIntervalo[0])[
                                    0])  # El intervalo tiene un formato así:  3349[2013-12-01 12:00:00-2013-12-01 13:00:00]  y con la función se obtiene '12:00:00-13:00:00'
                            [arrayHashtags, arrayTarget] = obtenerStatusHashtagNormalizadoYtarget(
                                arrayNewsItemQueUneInicialesDeUnIntervalo)

                            # se chequea que ninguna lista tenga elements NaN porque genera error en el clasificador
                            # No tiene sentido chequear la lista sin ocean porque está contenida en la con ocean
                            listaFeaturesConOCEAN = listaFeaturesConOCEAN + [[claseOCEAN] + [traducirIntervaloAvalorNumerico(intervaloParaFeature, unIntervalo)] + arrayHashtags]  # por defecto se guarda con OCEAN, después cuando se necesite que eliminarpa este valor
                            listaTargets = listaTargets + arrayTarget

                            arrayNewsItemQueUneInicialesDeUnIntervalo = []  # se inicializa el arreglo después de mostrarlo/procesarlo

                            mostrarIntervaloNuevo = False

                else:
                    break  # así se evita seguir leyendo líneas innecesariamente

        except UnicodeDecodeError as error:
            print("UnicodeDecodeError! User ID: ", idUsuario)
    else:
        # no existe el archivo para el id-intervalo-k dados
        respuesta = False
        listaFeaturesConOCEAN = None
        listaTargets = None

    return respuesta, listaFeaturesConOCEAN, listaTargets


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

        # se evalua que haya al menos un NIs en la entrada
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

    # se determina qué valor tendrá el arraySalidaFeatureHashtag
    # valor para la posición arraySalidaFeatureHashtag[0]
    # para eso se vale del último análisis de sentimientos de las lineas del intervalo
    if queSentimientoSeSumoComoResumenIntervalo == 'neu':
        arraySalidaFeatureHashtag[0] = 0
    else:
        if queSentimientoSeSumoComoResumenIntervalo == 'pos':
            arraySalidaFeatureHashtag[0] = 1
        else:
            arraySalidaFeatureHashtag[0] = 2  # es 'neg' por descarte

    # valor para la posición arraySalidaFeatureHashtag[1] y arraySalidaFeatureHashtag[2]
    # al porcentaje de pos y de neg se pasan a intervalos 0/1/2/3
    # es posible que (cantLlegaPos + cantLlegaNeg + cantLlegaNeu) == 0 cuando tod.o es inventado, por lo tanto puede dar error de división por cero en tiempo de ejecución
    # por eso se verifica la siguiente condición
    auxPos = 0
    auxNeg = 0
    if (cantLlegaPos + cantLlegaNeg + cantLlegaNeu) > 0:
        auxPos = ((cantLlegaPos * 100) / (
                    cantLlegaPos + cantLlegaNeg + cantLlegaNeu)) / 100  # se divide al final entre 100 para pasar de 31.5 a 0.31 y mapaearlo a un intervalo entre [0,1]
        auxNeg = ((cantLlegaNeg * 100) / (cantLlegaPos + cantLlegaNeg + cantLlegaNeu)) / 100

    arraySalidaFeatureHashtag[1] = traducirPorcentajeAnumeroSegunIntervalo(auxPos)
    arraySalidaFeatureHashtag[2] = traducirPorcentajeAnumeroSegunIntervalo(auxNeg)

    # se determina qué valor tendrá el arregloSalidaTarget
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
def obtenerRutasArhivosNewsItemsDadoIdPublicador(id, amplitudIntervaloTiempo, cantidadK):
    rutaBasicaArchivosDeNewsItems = "<files_paths_news_items>" # path which contains txt files with the news items of each user.
    return rutaBasicaArchivosDeNewsItems + str(amplitudIntervaloTiempo) + "-k" + str(cantidadK) + "/" + str(
        id) + "-" + str(amplitudIntervaloTiempo) + "-k" + str(cantidadK) + ".txt"


# Dada una amplitud de intervalo, calcula la cantidad de veces que el intervalo encaja dentro de la duración del dataset.
def calcularCantidadIntervalosDadoAmplitud(amplitud):
    return ((DATASET_END_DATE - DATASET_BEGIN_DATE).days + 1) * amountIntervalIncludedAtDayDict[amplitud]


# Traduce un intervalo de tiempo a un valor numérico, según la amplitud del mismo.
# Por ejemplo, si intervalo = "02:00:00-03:00:00" cuya amplitud = '1hour', devuelve '2' acorde a al diccionario 'dictIntervalo1horaTraducidoAnumero'
def traducirIntervaloAvalorNumerico(intervalo, amplitud):
    respuesta = None
    if amplitud == '1hour':
        respuesta = dictIntervalo1horaTraducidoAnumero[intervalo]
    if amplitud == '12hour':
        respuesta = translation12hoursToNumDict[intervalo]

    return respuesta


# dada una lista de listas, donde la primer posición de cada sublista es la clase OCEAN
# retorna la misma lista son la primer posición de cada sublista
def sacarOceanDeListaFeaturesOcean(listaFeatureConOcean):
    listaFeatureSinOcean = [sublista[1:] for sublista in listaFeatureConOcean]
    return listaFeatureSinOcean

# Dada las listas de entrada, evaluará con distintos clasificadores y almacena los resultados en un archivo de salida
def utilizarClasificadoresDiversasVariantesFeatures(listaUsuariosProcesados, valorDeK, intervalo, listaFeaturesCompleta, listaTargets):
    # Lo que antes era listaFeaturesSinOcean ahora se solamente es igual a listaFeaturesCompleta pero sin los parámetros del DICT_POSICIONES_FEATURES
    listaFeaturesSinOcean = functions.activatedDeactivateFeatures(listaFeaturesCompleta, DICT_POSICIONES_FEATURES)

    # se transforma la lista de usuarios procesados a un string para poder ser guardado en el archivo de salida
    # pasa por ejemplo de [[1,23444],[2,56565]] >> "(1)23444,(2)56565"
    stringUsuariosProcesados = ""
    for unElem in listaUsuariosProcesados:
        stringUsuariosProcesados += "(" + str(unElem[0]) + ")" + str(unElem[1]) + ","

    # se dividen las listas en dos: para entrenar y para probar, acorde al porcentaja que se usa para entrenar
    longitudLista = len(listaFeaturesCompleta)  # se supone que las 3 listas tienen igual cantidad de elementos
    indiceLimiteDePorcentaje = int((longitudLista * LEARNING_PERCENT) / 100)
    # --- Se divide listaFeaturesConOcean ---
    listaFeaturesConOceanParaEntrenar = listaFeaturesCompleta[0:indiceLimiteDePorcentaje]
    listaFeaturesConOceanParaProbar = listaFeaturesCompleta[indiceLimiteDePorcentaje:longitudLista]
    # --- Se divide listaFeaturesSinOcean ---
    listaFeaturesSinOceanParaEntrenar = listaFeaturesSinOcean[0:indiceLimiteDePorcentaje]
    listaFeaturesSinOceanParaProbar = listaFeaturesSinOcean[indiceLimiteDePorcentaje:longitudLista]
    # --- Se divide listaTargets ---
    listaTargetsParaEntrenar = listaTargets[0:indiceLimiteDePorcentaje]
    listaTargetsParaProbar = listaTargets[indiceLimiteDePorcentaje:longitudLista]

    # Se evalua si se equilibran las listas de features y targets para prueba y/o entrenamiento
    # Se evalua si se pide equilibrar o no
    if BALANCE_TEST:
        [listaFeaturesConOceanParaProbar, listaTargetsParaProbarNoSeUsa] = functions.balanceTargets(
            listaFeaturesConOceanParaProbar,
            listaTargetsParaProbar)  # listaTargetsParaProbarNoSeUsa se llama con un nombre cualquiera porque no será usada ya que TARGET la primera vez no debe ser cortado porque se utiliza con dos listas de features
        [listaFeaturesSinOceanParaProbar, listaTargetsParaProbar] = functions.balanceTargets(
            listaFeaturesSinOceanParaProbar, listaTargetsParaProbar)

    if BALANCE_TRAINING:
        [listaFeaturesConOceanParaEntrenar, listaTargetsParaEntrenarNoSeUsa] = functions.balanceTargets(
            listaFeaturesConOceanParaEntrenar,
            listaTargetsParaEntrenar)  # listaTargetsParaEntrenarNoSeUsa se llama con un nombre cualquiera porque no será usada ya que TARGET la primera vez no debe ser cortado porque se utiliza con dos listas de features
        [listaFeaturesSinOceanParaEntrenar, listaTargetsParaEntrenar] = functions.balanceTargets(
            listaFeaturesSinOceanParaEntrenar, listaTargetsParaEntrenar)

    # Se obtienen los nombres de clasificadores a utilizar acorde a la 'classifiersToUseList'
    for unClasificador in classifiersToUseList:
        if unClasificador == "LogisticRegression" or unClasificador == "DecisionTreeClassifier":
            f1Aux = generateMetricsML(listaFeaturesConOceanParaEntrenar,
                                      listaTargetsParaEntrenar,
                                      listaFeaturesConOceanParaProbar, listaTargetsParaProbar, True,
                                      stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                      valorDeK, unClasificador, paramConstructor=0, f1Anterior=0)
            f1Aux = generateMetricsML(listaFeaturesSinOceanParaEntrenar,
                                      listaTargetsParaEntrenar,
                                      listaFeaturesSinOceanParaProbar, listaTargetsParaProbar, False,
                                      stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                      valorDeK, unClasificador, paramConstructor=0, f1Anterior=f1Aux)

        if unClasificador == 'OneClassSVM':
            for unaConfigDParametros in KernelGammasMergeList:
                f1Aux = generateMetricsML(listaFeaturesConOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesConOceanParaProbar, listaTargetsParaProbar, True,
                                          stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador, paramConstructor=unaConfigDParametros, f1Anterior=0)
                f1Aux = generateMetricsML(listaFeaturesSinOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesSinOceanParaProbar, listaTargetsParaProbar, False,
                                          stringUsuariosProcesados, len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador, paramConstructor=unaConfigDParametros, f1Anterior=f1Aux)

        if unClasificador == 'RandomForestClassifier':
            for unaConfigRF in listaParametrosMezcladosRandomForest:
                f1Aux = generateMetricsML(listaFeaturesConOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesConOceanParaProbar,
                                          listaTargetsParaProbar, True,
                                          stringUsuariosProcesados,
                                          len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador,
                                          paramConstructor=unaConfigRF, f1Anterior=0)
                f1Aux = generateMetricsML(listaFeaturesSinOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesSinOceanParaProbar,
                                          listaTargetsParaProbar, False,
                                          stringUsuariosProcesados,
                                          len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador,
                                          paramConstructor=unaConfigRF, f1Anterior=f1Aux)
        if unClasificador == 'MultinomialNB':
            for unaConfigMBN in listaParametrosMezcladosMultinomialNB:
                f1Aux = generateMetricsML(listaFeaturesConOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesConOceanParaProbar,
                                          listaTargetsParaProbar, True,
                                          stringUsuariosProcesados,
                                          len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador,
                                          paramConstructor=unaConfigMBN, f1Anterior=0)
                f1Aux = generateMetricsML(listaFeaturesSinOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesSinOceanParaProbar,
                                          listaTargetsParaProbar, False,
                                          stringUsuariosProcesados,
                                          len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador,
                                          paramConstructor=unaConfigMBN, f1Anterior=f1Aux)
        if unClasificador == 'ComplementNB':
            for unaConfigComplmentBN in listaParametrosMezcladosComplementNB:
                f1Aux = generateMetricsML(listaFeaturesConOceanParaEntrenar,
                                          listaTargetsParaEntrenar,
                                          listaFeaturesConOceanParaProbar,
                                          listaTargetsParaProbar, True,
                                          stringUsuariosProcesados,
                                          len(listaUsuariosProcesados), intervalo,
                                          valorDeK, unClasificador,
                                          paramConstructor=unaConfigComplmentBN, f1Anterior=0)
                f1Aux = generateMetricsML(listaFeaturesSinOceanParaEntrenar,
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
def generateMetricsML(listaFeaturesParaEntrenar,
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
    if not(runSelectedConfigurations) or [tipoClasificador, stringParametroConstructor] in listaConfiguracionesElegidas:
        horaInicioFit = datetime.datetime.now()

        # en la función fit se convoca otra función para obtener solamente aquellos features que tengan un determinado valor de target
        [listaFeaturesDeterminadoTargetParaEntrenar, listaTargetsDeterminadoTargetParaEntrenar] = functions.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_FOR_ONE_CLASS, True)

        # noinspection PyInterpreter
        if tipoClasificador == "OneClassSVM":
            clf.fit(listaFeaturesDeterminadoTargetParaEntrenar)  # es para entrenar
        else:
            # los que no son oneclass entrenan distinto
            clf.fit(listaFeaturesParaEntrenar, listaTargetsParaEntrenar)  # es para entrenar

        """ Calculare RECALL and PRECISION """
        respuestaPredict = clf.predict(listaFeaturesParaProbar)
        respuestaPredict = functions.reemplazarValoresEnLista(respuestaPredict, valorABuscar=TARGET_FOR_ONE_CLASS,
                                                              valorNuevo=1,
                                                              buscarPorIgual=True)  # Cambia valorInlier (4's) por 1 porque son los inliers
        respuestaPredict = functions.reemplazarValoresEnLista(respuestaPredict, valorABuscar=1,
                                                     valorNuevo=-1,
                                                     buscarPorIgual=False)  # Cambia (no 4's) por -1 porque son los outliers

        scores = "<No scores>"
        if tipoClasificador == "OneClassSVM":
            scores = clf.score_samples(listaFeaturesParaProbar)
        else:
            scores = clf.score(listaFeaturesParaProbar, listaTargetsParaProbar)

        cantidadInliersEnPredict = len([i for i, e in enumerate(respuestaPredict) if e == 1])
        cantidadOutliersEnPredict = len([i for i, e in enumerate(respuestaPredict) if e == -1])
        print("                Cantidad  1s (inliers de la prueba) : ", cantidadInliersEnPredict,
              "(" + str(round(cantidadInliersEnPredict / len(respuestaPredict), 2)) + "%)")
        print("                Cantidad -1s (outliers de la prueba): ", cantidadOutliersEnPredict,
              "(" + str(round(cantidadOutliersEnPredict / len(respuestaPredict), 2)) + "%)")

        # 1: inliers, -1: outliers
        #  train_target, primero se convierten 4 en 1, y luego 0 en -1.
        targs = functions.reemplazarValoresEnLista(listaTargetsParaProbar, valorABuscar=TARGET_FOR_ONE_CLASS, valorNuevo=1, buscarPorIgual = True) #  Cambia 4's por 1 porque son los inliers
        targs = functions.reemplazarValoresEnLista(targs, valorABuscar=1, valorNuevo=-1, buscarPorIgual = False) #  Cambia 4's por -1 porque son los outliers

        [accuracy, precision, recall, f1] = ["<sin valor>", "<sin valor>", "<sin valor>", "<sin valor>"]
        if tipoClasificador == "OneClassSVM":
            [accuracy, precision, recall, f1] = functions.calcularAccPrecRecF1(listaTargetsParaProbar, respuestaPredict, convertirA1sYmenos1=True, valorInlier=TARGET_FOR_ONE_CLASS) # ************
        else:
            [accuracy, precision, recall, f1] = functions.calcularAccPrecRecF1(listaTargetsParaProbar, respuestaPredict, convertirA1sYmenos1=True, valorInlier=TARGET_FOR_ONE_CLASS)  # ************

        [listaFeaturesDeterminadoTarget, listaTargetsDeterminadoTarget] = functions.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_FOR_ONE_CLASS, True)
        X_train = listaFeaturesDeterminadoTarget
        X_test  = listaFeaturesParaProbar
        [listaFeaturesNODeterminadoTarget, listaTargetsNODeterminadoTarget] = functions.obtenerFeaturesYTargetsSegunValorTarget(listaFeaturesParaEntrenar, listaTargetsParaEntrenar, TARGET_FOR_ONE_CLASS, False)
        X_outliers = listaFeaturesNODeterminadoTarget

        stringColumnaPromFeaturesEntrenamiento = functions.obtenerPromedioCantidadElementosSublistas(listaFeaturesParaEntrenar, devolverString=True)
        stringColumnaPromFeaturesPrueba        = functions.obtenerPromedioCantidadElementosSublistas(listaFeaturesParaProbar, devolverString=True)

        stringCantidadInliers  = str(cantidadInliersEnPredict) + "(" + str(round(cantidadInliersEnPredict / len(respuestaPredict), 2)) + "%)"
        stringCantidadOutliers = str(cantidadOutliersEnPredict) + "(" + str(round(cantidadOutliersEnPredict / len(respuestaPredict), 2)) + "%)"

        [stringTN, stringFP, stringFN, stringTP] = functions.calcularTPTNFPFN(respuestaPredict, targs, devolverString=True)

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

        fileSalidaResumenTodosClasificadores.write(str(PROCESS_CANDIDATE_ID) + '\t' + "Dif de " + str(CRITERIO_SELECCION_DIF_TARGET) + '\t' + str(BALANCE_TRAINING) + '\t' +
                                                   str(BALANCE_TEST) + "\t" + stringNombresValores + "\t" + stringParametroConstructor + '\t\t')

        fileSalidaResumenTodosClasificadores.write(stringUsuariosProcesados + '\t' + str(cantidadUsuariosProcesados) + '\t' + intervalo + '\t' +
                                                   str(valorDeK) + '\t' + stringConTodasFeatures + '\t' + tipoClasificador + '\t' +
                                                   str(len(listaFeaturesParaEntrenar)) + '\t' + str(len(listaFeaturesParaProbar)) + '\t' +
                                                   stringColumnaPromFeaturesEntrenamiento + '\t' +
                                                   stringColumnaPromFeaturesPrueba + '\t' + str(LEARNING_PERCENT) + '\t' +
                                                 (functions.concatenarListaEnString(scores, ', '))[:64] + '\t' + stringCantidadInliers + '\t' +
                                                   stringCantidadOutliers + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
                                                   str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str((precision + recall)/2) + '\t' +
                                                   stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
                                                   tiempoEjecucionOneClass + '\t\t' + infoExtra + '\n')

        # se guarda en el archivo individual de cada clasificador
        dictArchivosIndividualesPorClasificador[str(tipoClasificador)].write(str(PROCESS_CANDIDATE_ID) + '\t' + "Dif de " + str(CRITERIO_SELECCION_DIF_TARGET) + '\t' + str(BALANCE_TRAINING) + '\t' +
                                                                             str(BALANCE_TEST) + "\t" + stringNombresValores + "\t" + stringParametroConstructor + '\t\t')
        dictArchivosIndividualesPorClasificador[str(tipoClasificador)].write(stringUsuariosProcesados + '\t' + str(cantidadUsuariosProcesados) + '\t' + intervalo + '\t' +
                                                                             str(valorDeK) + '\t' + stringConTodasFeatures + '\t' + tipoClasificador + '\t' +
                                                                             str(len(listaFeaturesParaEntrenar)) + '\t' + str(len(listaFeaturesParaProbar)) + '\t' +
                                                                             stringColumnaPromFeaturesEntrenamiento + '\t' +
                                                                             stringColumnaPromFeaturesPrueba + '\t' + str(LEARNING_PERCENT) + '\t' +
                                                 (functions.concatenarListaEnString(scores, ', '))[:64] + '\t' + stringCantidadInliers + '\t' +
                                                                             stringCantidadOutliers + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +
                                                                             str(recall) + '\t' + str(f1) + '\t' + stringAuxRestaF1 + '\t' + stringAuxDivF1 + '\t' + str((precision + recall)/2) + '\t' +
                                                                             stringTN + '\t' + stringFP + '\t' + stringFN + '\t' + stringTP + '\t' +
                                                                             tiempoEjecucionOneClass + '\t\t' + infoExtra + '\n')
    return f1


"""=============== FIN de declaraciones de funciones ===================== """
for unCriterio in SELECTION_CRITERIA_DIF_TARGET_LIST:
    CRITERIO_SELECCION_DIF_TARGET = unCriterio
    # Se abre el archivo adecuado de IDs candidatos si se van a procesar los clasificaodres
    if PROCESS_CLASSIFIER:
        listaIDsCandidatos = functions.generateListGivenAFile(open(CANDIDATES_PATH_FILE_IDs + str(unCriterio) + ".txt", "r"))
    if unCriterio == 100:
        PROCESS_CANDIDATE_ID = False # porque no se se usa como criterio el 100 de diferencia entonces van a estar involucrados todos los usuario
    mainScanUsersID()
