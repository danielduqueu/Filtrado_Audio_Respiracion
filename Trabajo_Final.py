# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:40:56 2020

@author: soy_d
"""
import math
import librosa
import librosa.display
import numpy as np
import pandas as pd
import scipy.signal as signal
from linearFIR import filter_design
from Filtrado_Wavelet import filtrado
from os import listdir
from os.path import isfile, join

def Cargar_Audio(filename):
    '''
    Carga el archivo de audio y aplica un filtro pasabandas FIR, filtrando
    entre 100Hz y 2000Hz
    '''
    y, sr = librosa.load(filename)
    fs = sr
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 2000, revfilt = 0)
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1)
    y_hp = signal.filtfilt(highpass, 1, y)
    y_bp = signal.filtfilt(lowpass, 1, y_hp)
    y_bp = np.asfortranarray(y_bp)
    return y_bp, sr

def Ciclos_resp(FileAudio, FileAno, sr):
    '''
    Se añade la variable tipo lista que contiene los datos del archivo de
    audio con los filtros FIR y Wavelet, devolviendo una lista que contiene
    los fragmentos de los diferentes ciclos respiratorios de cada archivo y
    además, los indices correspondientes para denotar si el ciclo contiene o 
    no crepitancia y/o sibilancia.
    '''
    Archivo_txt = np.loadtxt(FileAno)
    R, C = Archivo_txt.shape
    ciclos_respiratorios = []
    for i in range(R):
        Desde = math.floor(Archivo_txt[i, 0]*sr)
        Hasta = math.ceil(Archivo_txt[i, 1]*sr)
        Parte_Ciclo=FileAudio[Desde:Hasta]
        ciclos_respiratorios.append([Parte_Ciclo, Archivo_txt[i, 2], Archivo_txt[i, 3]])
    return ciclos_respiratorios

def Datos_Estadisticos(ciclo):
    '''
    Se ingresa la variable que contiene un ciclo cardiaco y se retornan
    los valores de los datos estadísticos del ciclo.
    '''
    Varianza = np.var(ciclo)
    Rango = np.abs(np.max(ciclo) - np.min(ciclo))
    Cantidad_Muestras = 800
    Corrimiento = 100
    Recorrido = np.arange(0,len(ciclo)-Cantidad_Muestras,Corrimiento)
    Promedio_Datos = []
    for i in Recorrido:
        Promedio_Datos.append(np.mean([ciclo[i:i+Cantidad_Muestras]]))
    Promedio_Datos.append(np.mean([ciclo[Recorrido[len(Recorrido)-1]:]]))
    Suma_Promedios = np.mean(Promedio_Datos)
    f, Pxx = signal.periodogram(ciclo)
    Promedio_Espectro = np.mean(Pxx)
    return Varianza, Rango, Suma_Promedios, Promedio_Espectro
        
Directorio='C:\\Users\\soy_d\\Downloads\\respiratory-sound-database\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files'

Archivos_Audio = [file for file in listdir(Directorio) if file.endswith(".wav") if isfile(join(Directorio, file))]
Archivos_Texto = [file for file in listdir(Directorio) if file.endswith(".txt") if isfile(join(Directorio, file))]

Frames=[]
Final={}
Num_Ciclos=[]
for i in range(1):
    print('Analizando '+str(i+1)+'/'+str(len(Archivos_Audio))+" "+Archivos_Audio[i])
    Audio_filtrado, sr = Cargar_Audio(Archivos_Audio[i])
    Corazon_Filtrado = filtrado(Audio_filtrado,0,1,2)
    Datos = Ciclos_resp(Corazon_Filtrado, Archivos_Texto[i], sr)
    Num_Ciclos.append(pd.DataFrame({'Archivo':Archivos_Audio[i], 'Número de ciclos respiratorios':[len(Datos)]}))
    Data_Frame = pd.DataFrame(columns=['Ciclo Respiratorio', 'Crepitancia', 'Sibilancia','Varianza', 'Rango', 'Suma de Promedios', 'Promedio de Espectros'])
    for Datos_Ciclo in range(len(Datos)):
        Ciclo = Datos_Ciclo + 1
        Crepitancia = Datos[Datos_Ciclo][1]
        Sibilancia = Datos[Datos_Ciclo][2]
        Varianza, Rango, Suma_Promedios, Promedio_Espectro = Datos_Estadisticos(Datos[Datos_Ciclo][0])
        DF = pd.DataFrame({'Ciclo Respiratorio':[Ciclo],
                           'Crepitancia':[Crepitancia], 
                           'Sibilancia':[Sibilancia], 
                           'Varianza':[Varianza], 
                           'Rango':[Rango], 
                           'Suma de Promedios':[Suma_Promedios], 
                           'Promedio de Espectros':[Promedio_Espectro]})
        Data_Frame = pd.concat([Data_Frame, DF], ignore_index=True)
    Frames.append(Data_Frame)
Final_DF = pd.concat(Frames, keys=Archivos_Audio)
Final_ciclos = pd.concat(Num_Ciclos)

Final_DF.to_csv('Datos.csv')
Final_ciclos.to_csv('Ciclos.csv')


        
        
        
        
        
        
        
        
        
        