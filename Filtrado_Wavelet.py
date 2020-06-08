# -*- coding: utf-8 -*-
"""
@author: Daniel Duque Urrego, Santiago Suárez Bustamante
"""
import numpy as np
wavelet = [-1/np.sqrt(2), 1/np.sqrt(2)];
scale = [1/np.sqrt(2), 1/np.sqrt(2)];
wavelet_inv = [1/np.sqrt(2) , -1/np.sqrt(2)];
scale_inv = [1/np.sqrt(2) , 1/np.sqrt(2)];

def trans_haar(vector, nivel_actual, nivel_final, transformada):
    '''
    Método que realiza la transformada Haar del canal seleccionado, recibe como 
    parámetros el vector correspondiente al canal, el nivel actual de transformación
    y el nivel final o cantidad de veces que se realizará la transformación. Además, 
    al ser una función recursiva, recibe como entrada el vector del nivel
    anterior. Finalmente, retornará el vector completo con todos los detalles y la
    última aporoximación.
    '''
    
    senal_descomponer = vector;
    if (nivel_actual <= nivel_final):
        #Se verifica que la señal tenga un número par de muestras, de lo contrario
        # se le agrega un 0
        if (senal_descomponer.shape[0] % 2) != 0:
            senal_descomponer = np.append(senal_descomponer, 0);
        
        #Se hace la convolución para las aproximaciones con el vector scale
        Aprox = np.convolve(senal_descomponer, scale,'full');
        #Se submuestréa, es decir se toma una muestra de cada dos.
        Aprox = Aprox[1::2];
        #Se hace la convolución para los detalles con el vector wavelet
        Detail = np.convolve(senal_descomponer, wavelet,'full');
        #Se submuestréa, es decir se toma una muestra de cada dos.
        Detail = Detail[1::2];
        
        #Se añade al vector transdormada cada uno de los detalles
        transformada.append(Detail)
        if (nivel_actual < nivel_final):
            #Se vuelve y se llama la misma función, aumentando en 1 el nivel actual
            return trans_haar(Aprox, nivel_actual + 1, nivel_final, transformada)
    #Finalmente depués de tener cada detalle se añade la última aproximación
    transformada.append(Aprox)
    return transformada

def trans_inv_haar(vector, nivel_actual, nivel_final, resultado):
    '''
    El método trans inv haar, se encarga de realizar la reconstrucción de la 
    señal luego del filtrado, recibe el vector luego de pasar por la descompo-
    sición y el respectivo filtrado, el nivel actual y el nivel final. Además
    e igual en el caso anterior, al ser una señal recursiva recibe el resultado
    de cada nivel previo. Finalmente, retorna la señal reconstruida
    '''
    
    size_vector = len(vector)
    vector1 = vector[size_vector - 1]
    vector_detail = vector[size_vector - nivel_actual - 1]
    if (nivel_actual <= nivel_final):
        if (len(resultado) > len(vector_detail)):
            resultado = resultado[0:len(vector_detail)]
        if nivel_actual==1:
            npoints_aprox = len(vector1)
            Aprox_inv = np.zeros(2*npoints_aprox)
            #Se sobremuestréa el vector aproximación con ceros.
            Aprox_inv[0::2] = vector1
        else:
            npoints_aprox = len(resultado)
            Aprox_inv = np.zeros(2*npoints_aprox)
            Aprox_inv[0::2] = resultado
        #Se hace la convolucion de la aproximación inversa con el vector scale inverso
        Aprox = np.convolve(Aprox_inv, scale_inv,'full')
        Detail_inv = np.zeros(2*npoints_aprox)
        #Se sobremuestréa el vector detalle con ceros.
        Detail_inv[0::2] = vector_detail
        #Se hace la convolucion del detalle inverso con el vector wavelet inverso
        Detail = np.convolve(Detail_inv, wavelet_inv,'full')
        resultado = Aprox + Detail
        #Se llama de nuevo la función, aumentando en 1 el nivel actual.
        return trans_inv_haar(vector, nivel_actual + 1, nivel_final, resultado)
    return resultado

def opcion_lambda(transformada, lambda_combo):
    '''
    El método opción lambda, se encarga de calcular el valor del parámetro
    lambda, dependiendo de la opción que seleccione el usuario por medio del
    combo Box, recibe el vector luego de la transformada de Haar y el índice 
    de la selección del combo Box y retorna el valor de este parámetro.
    '''
    #Se toman la cantidad de datos total del vector resultante de la 
    #transformada de Haar
    cantidad_de_datos=0
    for i in range(len(transformada)):
        cantidad_de_datos = cantidad_de_datos + len(transformada[i])
    
    #Se tiene la primera opción que es para la opción UNIVERSAL
    if lambda_combo==0:
        #Se hace so de la ecuación que describe este método
        valor_labda = np.sqrt(2*np.log10(cantidad_de_datos))
    #Se tiene la segunda opción MINIMAX
    elif lambda_combo==1:
        #Se hace so de la ecuación que describe este método
        valor_labda = 0.3936 + 0.1829*(np.log10(cantidad_de_datos)/np.log10(2))
    #Se tiene la última opción, SURE
    else:
        sx2=[]
        risk=[]
        for i in range(len(transformada)):
            sx2 = np.append(sx2, transformada[i])
        n=cantidad_de_datos
        #Se eleva cada uno de los términos al cuadrado y se ordenan de menor
        # a mayor
        sx2 = np.power(np.sort(np.abs(sx2)),2)
        #Se implementa la ecuación que representa el cálculo de SURE
        risk = (n-(2*np.arange(1,n + 1)) + (np.cumsum(sx2[0:n])) + np.multiply(np.arange(n,0,-1), sx2[0:n]))/n
        #Se selecciona el mejor valor como el mínimo del vector anterior
        best = np.min(risk)
        #Se redondea a un entero
        entero = int(np.round(best))
        #Se toma la raiz cuadrada del valor en la posición "best" como 
        #indica la fórmula
        valor_labda = np.sqrt(sx2[entero])
    return valor_labda

def opcion_umbral(transformada, ponderacion, valor_lambda, valor_ponderacion, umbral):
    '''
    La funcipon umbral, aplica el filtrado a los detalles dependiendo de la
    opción que seleccione el usuario, entre soft y hard, recibe como parámetros
    el vector luego de la transformada Haar, el valor de ponderación y lambda 
    calculados, la opción seleccionada en el combo Box ponderación y el umbral 
    escogido por el usuario. Retorna el vector luego de aplicar el filtrado
    '''
    #Opción para umbral hard
    if umbral==0:
        #Si la opción ponderación es SURE, aca detalle se compara con su respectivo
        #valor de lambda multiplicado por la ponderación calculada a este mismo detalle
        if valor_ponderacion == 2:
            for i in range(len(transformada) - 1):
                transformada[i][abs(transformada[i]) > valor_lambda*ponderacion[i]] = 0;
        #Si es cualquiera de los otros casos, se compara con respecto al valor de 
        #lambda* el valor de ponderación retornado de la función opción ponderación
        else:
            for i in range(len(transformada) - 1):
                transformada[i][abs(transformada[i]) > (valor_lambda*ponderacion)] = 0;

    #Opción para umbral soft, se sigue el mismo proceso anterior, pero en los
    # casos donde el detalle sea mayor o igual, el valor cambia.
    else:
        if valor_ponderacion == 2:
            for i in range(len(transformada) - 1):
                for j in range(len(transformada[i])):
                    if (abs(transformada[i][j]) <= valor_lambda*ponderacion[i]):
                        transformada[i][j] = np.sign(transformada[i][j])*(abs(transformada[i][j]) - valor_lambda*ponderacion[i])
                    else:
                        transformada[i][j]=0

        else:
            for i in range(len(transformada) - 1):
                for j in range(len(transformada[i])):
                    if (abs(transformada[i][j]) <= (valor_lambda*ponderacion)):
                        transformada[i][j] = np.sign(transformada[i][j])*(abs(transformada[i][j])-valor_lambda*ponderacion)
                    else:
                        transformada[i][j]=0
    return transformada

def opcion_ponderacion(transformada, valor_ponderacion):
    '''
    El método opción ponderación se encarga de calcular el peso dependiendo de la 
    opción que selecciona el usuario, recibe el vector luego de la transformada Haar
    y el índice del respectivo combo Box. Retorna el valor del peso o el vector de los
    pesos, dependiendo de la opción escogida.
    '''
    
    #Si ponderación es ONES, el valor del peso será 1
    if valor_ponderacion==0:
        median = 1
    #Si es SLN, será la mediana del primer detalle
    elif valor_ponderacion==1:
        median = np.median(abs(transformada[0]))/0.6745
    #Si es MLN, será un vector con las medianas de cada detalle
    else:
        median=[]
        for i in range(len(transformada)):
            median.append(np.median(abs(transformada[i]))/0.6745)
    return median

def filtrado(data, valor_umbral, valor_lambda, valor_ponderacion):
    '''
    El método filtrado, está conectado con el botón filtrar y es el encargado de
    llamar a cada uno de los métodos anteriores para realizar la descomposición,
    filtrado y reconstrucción de la señal, recibe como parámetros los índices de
    los combo Box y canal a filtrar y retorna la transformada inversa, es decir, 
    la señal reconstruida.
    '''
    
    longitud_original = data.shape[0]
    #Se calcula el nivel final con la Ec. respectiva
    nivel_final = np.floor(np.log2(longitud_original/2) - 1)
    #Se invocan las funciones para descomponer, filtrar y reconstruir (en el oden
    # previamente mencionado) la señal 
    resultado_trans_harr = trans_haar(data, 1, nivel_final, [])
    lambda_res = opcion_lambda(resultado_trans_harr, valor_lambda)
    ponderacion = opcion_ponderacion(resultado_trans_harr, valor_ponderacion)
    umbral = opcion_umbral(resultado_trans_harr, ponderacion, lambda_res, valor_ponderacion, valor_umbral)
    trans_inv = trans_inv_haar(resultado_trans_harr, 1, nivel_final, [])
    return trans_inv

    
    

    



