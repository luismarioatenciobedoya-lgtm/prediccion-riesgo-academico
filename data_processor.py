# -*- coding: utf-8 -*-
"""
Módulo de Procesamiento de Datos para Predicción de Riesgo Académico
"""

import pandas as pd
import numpy as np


class DataProcessor:
    """
    Clase para procesar datos de estudiantes y calcular probabilidades de riesgo académico
    """
    
    def __init__(self):
        """Inicializa el procesador con todos los parámetros del modelo"""
        self.programas = [
            "Licenciatura en Ciencias Religiosas",
            "Licenciatura en Teología Tunja",
            "Licenciatura en Educación Básica con énfasis en Humanidades y Lengua Castellana",
            "Odontología",
            "Teología",
            "Microbiología Industrial",
            "Nutrición y Dietética",
            "Historia",
            "Ingeniería Mecatrónica",
            "Ingeniería Electrónica",
            "Artes Visuales",
            "Ingeniería Mecánica",
            "Ecología",
            "Artes Escénicas",
            "Licenciatura en Educación Infantil",
            "Economía",
            "Derecho",
            "Antropología",
            "Ingeniería Industrial",
            "Estudios Literarios",
            "Licenciatura en Teología",
            "Diseño Industrial",
            "Ingeniería de Sistemas",
            "Bioingeniería",
            "Contaduría Pública",
            "Ingeniería Civil",
            "Sociología",
            "Química Farmacéutica",
            "Finanzas",
            "Administración de Empresas",
            "Microbiología Agrícola y Veterinaria",
            "Ciencia Política",
            "Ciencias de la Información - Bibliotecología",
            "Ciencia de Datos",
            "Informática Matemática",
            "Comunicación Social",
            "Matemáticas"
        ]
        
        self.departamentos = [
            "Valle del Cauca",
            "Antioquia",
            "Atlántico",
            "Bogotá D.C.",
            "Boyacá",
            "Quindío",
            "Meta",
            "La Guajira",
            "Bolívar",
            "Arauca",
            "Cundinamarca",
            "Santander",
            "Casanare",
            "Nariño",
            "Tolima",
            "Caldas"
        ]
        
        self.categorias_max = [
            "Ingeniería Mecánica y Eléctrica",
            "Estadística y Análisis de Datos",
            "Artes Visuales",
            "Psicología",
            "Lengua y Literatura Española",
            "Idiomas Extranjeros",
            "Ciencia Política",
            "Programación y Ciencia de Datos",
            "Comunicación y Medios",
            "Tecnología y Computación",
            "Química",
            "Derecho",
            "Medicina y Enfermería",
            "Filosofía y Ética",
            "Proyectos y Talleres",
            "Odontología",
            "Mercadeo y Ventas",
            "Educación Física y Deportes",
            "Administración y Emprendimiento",
            "Cine y Fotografía",
            "Educación y Pedagogía",
            "Matemáticas",
            "Ciencias Sociales"
        ]
        
        self.categorias_min = [
            "Administración y Emprendimiento",
            "Estadística y Análisis de Datos",
            "Ingeniería Mecánica y Eléctrica",
            "Matemáticas",
            "Programación y Ciencia de Datos",
            "Artes Visuales",
            "Comunicación y Medios",
            "Mercadeo y Ventas",
            "Lengua y Literatura Española",
            "Ciencia Política",
            "Física",
            "Música e Interpretación",
            "Ciencias Sociales",
            "Proyectos y Talleres",
            "Idiomas Extranjeros",
            "Arquitectura y Construcción",
            "Derecho",
            "Biología",
            "Danza",
            "Nutrición y Alimentación",
            "Cine y Fotografía",
            "Educación y Pedagogía",
            "Filosofía y Ética",
            "Medicina y Enfermería",
            "Religión y Teología"
        ]
        
        # Diccionarios de escalado (media y desviación estándar)
        self.mean_dict = self._get_mean_dict()
        self.scale_dict = self._get_scale_dict()
        
        # Coeficientes del modelo y intercepto
        self.coef_dict = self._get_coefficients()
        self.intercepto = -2.208874
    
    def _get_mean_dict(self):
        """Retorna el diccionario de medias para estandarización"""
        return {
            "Ciclo": 4.964635,
            "Ciclo Admisión": 2.025768,
            "Mult Programa": 0.099462,
            "Situacion Acad": 0.179639,
            "Promedio ciclo": 3.555774,
            "Promedio Acumulado": 3.625768,
            "Sexo": 0.489899,
            "Rango edad": 0.880674,
            "Internacional": 0.01848,
            "cd_Bogotá D.C.": 0.630875,
            "Total Créditos Acumula Tomados": 77.62963,
            "Total Créditos Acumu Aprobados": 67.42424,
            "Materias Vistas": 26.33266,
            "TRL": 0.032828,
            "IPS": 0.867804,
            "MLP": 0.099462,
            "p_Licenciatura_en_Ciencias_Religiosas": 0.006061,
            "p_Licenciatura_en_Teologia_Tunja": 0.003367,
            "p_Licenciatura_en_Educacion_Basica_con_enfasis_en_Humanidades_y_Lengua_Castellana": 0.009428,
            "p_Odontologia": 0.068855,
            "p_Teologia": 0.01598,
            "p_Microbiologia_Industrial": 0.020202,
            "p_Nutricion_y_Dietetica": 0.031313,
            "p_Historia": 0.006061,
            "p_Ingenieria_Mecatronica": 0.038721,
            "p_Ingenieria_Electronica": 0.033838,
            "p_Artes_Visuales": 0.012794,
            "p_Ingenieria_Mecanica": 0.028283,
            "p_Ecologia": 0.011616,
            "p_Artes_Escenicas": 0.007912,
            "p_Licenciatura_en_Educacion_Infantil": 0.003367,
            "p_Economia": 0.046128,
            "p_Derecho": 0.068519,
            "p_Antropologia": 0.005387,
            "p_Ingenieria_Industrial": 0.071717,
            "p_Estudios_Literarios": 0.004209,
            "p_Licenciatura_en_Teologia": 0.006061,
            "p_Diseno_Industrial": 0.023064,
            "p_Ingenieria_de_Sistemas": 0.049831,
            "p_Bioingenieria": 0.022896,
            "p_Contaduria_Publica": 0.043771,
            "p_Ingenieria_Civil": 0.036195,
            "p_Sociologia": 0.005387,
            "p_Quimica_Farmaceutica": 0.029461,
            "p_Finanzas": 0.034007,
            "p_Administracion_de_Empresas": 0.045286,
            "p_Microbiologia_Agricola_y_Veterinaria": 0.008081,
            "p_Ciencia_Politica": 0.00202,
            "p_Ciencias_de_la_Informacion_-_Bibliotecologia": 0.006061,
            "p_Ciencia_de_Datos": 0.019865,
            "p_Informatica_Matematica": 0.005556,
            "p_Comunicacion_Social": 0.037879,
            "p_Matematicas": 0.014478,
            "dn_Valle_del_Cauca": 0.032997,
            "dn_Antioquia": 0.072896,
            "dn_Atlántico": 0.005556,
            "dn_Bogotá_D.C.": 0.243266,
            "dn_Boyacá": 0.092424,
            "dn_Quindío": 0.003367,
            "dn_Meta": 0.018855,
            "dn_La_Guajira": 0.009765,
            "dn_Bolívar": 0.01498,
            "dn_Arauca": 0.005892,
            "dn_Cundinamarca": 0.195791,
            "dn_Santander": 0.023569,
            "dn_Casanare": 0.008081,
            "dn_Nariño": 0.017677,
            "dn_Tolima": 0.027441,
            "dn_Caldas": 0.020707,
            "ccmax_Ingeniería Mecánica y Eléctrica": 0.146128,
            "ccmax_Estadística y Análisis de Datos": 0.041582,
            "ccmax_Artes Visuales": 0.020033,
            "ccmax_Psicología": 0.032997,
            "ccmax_Lengua y Literatura Española": 0.016667,
            "ccmax_Idiomas Extranjeros": 0.055556,
            "ccmax_Ciencia Política": 0.007071,
            "ccmax_Programación y Ciencia de Datos": 0.105556,
            "ccmax_Comunicación y Medios": 0.078788,
            "ccmax_Tecnología y Computación": 0.01498,
            "ccmax_Química": 0.044276,
            "ccmax_Derecho": 0.058923,
            "ccmax_Medicina y Enfermería": 0.086869,
            "ccmax_Filosofía y Ética": 0.033838,
            "ccmax_Proyectos y Talleres": 0.058249,
            "ccmax_Odontología": 0.019023,
            "ccmax_Mercadeo y Ventas": 0.027946,
            "ccmax_Educación Física y Deportes": 0.011111,
            "ccmax_Administración y Emprendimiento": 0.04798,
            "ccmax_Cine y Fotografía": 0.004209,
            "ccmax_Educación y Pedagogía": 0.015152,
            "ccmax_Matemáticas": 0.043097,
            "ccmax_Ciencias Sociales": 0.011616,
            "ccmin_Administración y Emprendimiento": 0.073401,
            "ccmin_Estadística y Análisis de Datos": 0.058249,
            "ccmin_Ingeniería Mecánica y Eléctrica": 0.026936,
            "ccmin_Matemáticas": 0.058081,
            "ccmin_Programación y Ciencia de Datos": 0.031649,
            "ccmin_Artes Visuales": 0.012626,
            "ccmin_Comunicación y Medios": 0.043266,
            "ccmin_Mercadeo y Ventas": 0.022222,
            "ccmin_Lengua y Literatura Española": 0.03198,
            "ccmin_Ciencia Política": 0.009933,
            "ccmin_Física": 0.034512,
            "ccmin_Música e Interpretación": 0.005556,
            "ccmin_Ciencias Sociales": 0.01498,
            "ccmin_Proyectos y Talleres": 0.092929,
            "ccmin_Idiomas Extranjeros": 0.075421,
            "ccmin_Arquitectura y Construcción": 0.017003,
            "ccmin_Derecho": 0.020033,
            "ccmin_Biología": 0.030303,
            "ccmin_Danza": 0.00404,
            "ccmin_Nutrición y Alimentación": 0.007239,
            "ccmin_Cine y Fotografía": 0.005387,
            "ccmin_Educación y Pedagogía": 0.022896,
            "ccmin_Filosofía y Ética": 0.015993,
            "ccmin_Medicina y Enfermería": 0.029966,
            "ccmin_Religión y Teología": 0.008586,
            "s_LTEOL": 0.006061,
            "s_TEOLO": 0.01598,
            "s_DRCH2": 0.068519,
            "s_INFBN": 0.006061,
            "s_BACTE": 0.008081,
            "s_ARTV2": 0.012794,
            "s_FINA2": 0.034007,
            "s_ARQUI": 0.003367,
            "s_FILOS": 0.015488,
            "s_ANTR2": 0.005387,
            "s_RLINT": 0.003367,
            "s_PSIC2": 0.006565,
            "s_ECONM": 0.046128,
            "s_LLMOD": 0.004209,
            "s_MAGRV": 0.008081,
            "s_IMECA": 0.028283,
            "s_ICIVL": 0.036195,
            "s_NUTDT": 0.031313,
            "s_CTDPD": 0.043771,
            "s_ADMD": 0.045286,
            "s_EMSCL": 0.007912,
            "s_LTEOT": 0.003367,
            "s_CRELV": 0.006061,
            "s_COMSC": 0.037879,
            "s_IELEC": 0.033838,
            "s_MEDIC": 0.068855,
            "s_PSICO": 0.032997,
            "s_LITER": 0.004209,
            "s_SOCI2": 0.005387,
            "s_ARES2": 0.007912,
            "s_HIST2": 0.006061,
            "Cant_Perdidas": 0.848485,
            "Créditos Inscritos en Ciclo": 16.5641,
            "Max_Ciclo": 3.999327,
            "Cred_Max_Calif_Ciclo": 3.269697,
            "Min_Ciclo": 2.729461,
            "Cred_Min_Calif_Ciclo": 3.180808,
            "Créditos Inscritos y Aprobados Ciclo": 15.37205,
            "Rango_Ponderado_Ciclo": 1.26984
        }
    
    def _get_scale_dict(self):
        """Retorna el diccionario de desviaciones estándar para estandarización"""
        return {
            "Ciclo": 2.660461556,
            "Ciclo Admisión": 2.540885555,
            "Mult Programa": 0.299241924,
            "Situacion Acad": 0.465558886,
            "Promedio ciclo": 0.588503805,
            "Promedio Acumulado": 0.535037892,
            "Sexo": 0.499936936,
            "Rango edad": 0.766823826,
            "Internacional": 0.134740093,
            "cd_Bogotá D.C.": 0.482761876,
            "Total Créditos Acumula Tomados": 41.75846894,
            "Total Créditos Acumu Aprobados": 37.48773844,
            "Materias Vistas": 14.01531766,
            "TRL": 0.178165878,
            "IPS": 0.338887458,
            "MLP": 0.299241924,
            "p_Licenciatura_en_Ciencias_Religiosas": 0.077648904,
            "p_Licenciatura_en_Teologia_Tunja": 0.057988066,
            "p_Licenciatura_en_Educacion_Basica_con_enfasis_en_Humanidades_y_Lengua_Castellana": 0.096642156,
            "p_Odontologia": 0.253017689,
            "p_Teologia": 0.125517076,
            "p_Microbiologia_Industrial": 0.140711009,
            "p_Nutricion_y_Dietetica": 0.174135903,
            "p_Historia": 0.077648904,
            "p_Ingenieria_Mecatronica": 0.192826085,
            "p_Ingenieria_Electronica": 0.180603471,
            "p_Artes_Visuales": 0.112456979,
            "p_Ingenieria_Mecanica": 0.165680534,
            "p_Ecologia": 0.107211984,
            "p_Artes_Escenicas": 0.088652796,
            "p_Licenciatura_en_Educacion_Infantil": 0.057988066,
            "p_Economia": 0.209664691,
            "p_Derecho": 0.252777068,
            "p_Antropologia": 0.073267326,
            "p_Ingenieria_Industrial": 0.257973488,
            "p_Estudios_Literarios": 0.064757547,
            "p_Licenciatura_en_Teologia": 0.077648904,
            "p_Diseno_Industrial": 0.150130447,
            "p_Ingenieria_de_Sistemas": 0.217701061,
            "p_Bioingenieria": 0.149686652,
            "p_Contaduria_Publica": 0.204460051,
            "p_Ingenieria_Civil": 0.186982886,
            "p_Sociologia": 0.073267326,
            "p_Quimica_Farmaceutica": 0.169119119,
            "p_Finanzas": 0.181219965,
            "p_Administracion_de_Empresas": 0.207833063,
            "p_Microbiologia_Agricola_y_Veterinaria": 0.089581823,
            "p_Ciencia_Politica": 0.044916042,
            "p_Ciencias_de_la_Informacion_-_Bibliotecologia": 0.077648904,
            "p_Ciencia_de_Datos": 0.139495926,
            "p_Informatica_Matematica": 0.074330164,
            "p_Comunicacion_Social": 0.190871488,
            "p_Matematicas": 0.119609933,
            "dn_Valle_del_Cauca": 0.178697626,
            "dn_Antioquia": 0.259833837,
            "dn_Atlántico": 0.074330164,
            "dn_Bogotá_D.C.": 0.429137662,
            "dn_Boyacá": 0.289805126,
            "dn_Quindío": 0.057988066,
            "dn_Meta": 0.136101611,
            "dn_La_Guajira": 0.098379301,
            "dn_Bolívar": 0.121534871,
            "dn_Arauca": 0.076580098,
            "dn_Cundinamarca": 0.396819388,
            "dn_Santander": 0.151725159,
            "dn_Casanare": 0.089581823,
            "dn_Nariño": 0.131820806,
            "dn_Tolima": 0.163420932,
            "dn_Caldas": 0.142414838,
            "ccmax_Ingeniería Mecánica y Eléctrica": 0.353308393,
            "ccmax_Estadística y Análisis de Datos": 0.199759696,
            "ccmax_Artes Visuales": 0.140033707,
            "ccmax_Psicología": 0.178697626,
            "ccmax_Lengua y Literatura Española": 0.128122806,
            "ccmax_Idiomas Extranjeros": 0.22907601,
            "ccmax_Ciencia Política": 0.083788458,
            "ccmax_Programación y Ciencia de Datos": 0.307440318,
            "ccmax_Comunicación y Medios": 0.269398839,
            "ccmax_Tecnología y Computación": 0.121534871,
            "ccmax_Química": 0.205751074,
            "ccmax_Derecho": 0.235317063,
            "ccmax_Medicina y Enfermería": 0.281580502,
            "ccmax_Filosofía y Ética": 0.180603471,
            "ccmax_Proyectos y Talleres": 0.234194449,
            "ccmax_Odontología": 0.136674506,
            "ccmax_Mercadeo y Ventas": 0.164884653,
            "ccmax_Educación Física y Deportes": 0.104917401,
            "ccmax_Administración y Emprendimiento": 0.213502684,
            "ccmax_Cine y Fotografía": 0.064757547,
            "ccmax_Educación y Pedagogía": 0.122221951,
            "ccmax_Matemáticas": 0.203065883,
            "ccmax_Ciencias Sociales": 0.107211984,
            "ccmin_Administración y Emprendimiento": 0.260741593,
            "ccmin_Estadística y Análisis de Datos": 0.234194449,
            "ccmin_Ingeniería Mecánica y Eléctrica": 0.161983139,
            "ccmin_Matemáticas": 0.234034065,
            "ccmin_Programación y Ciencia de Datos": 0.174997527,
            "ccmin_Artes Visuales": 0.111733526,
            "ccmin_Comunicación y Medios": 0.203307576,
            "ccmin_Mercadeo y Ventas": 0.147484098,
            "ccmin_Lengua y Literatura Española": 0.175844178,
            "ccmin_Ciencia Política": 0.099231099,
            "ccmin_Física": 0.182505423,
            "ccmin_Música e Interpretación": 0.074330164,
            "ccmin_Ciencias Sociales": 0.121534871,
            "ccmin_Proyectos y Talleres": 0.290463832,
            "ccmin_Idiomas Extranjeros": 0.264271556,
            "ccmin_Arquitectura y Construcción": 0.129359644,
            "ccmin_Derecho": 0.140033707,
            "ccmin_Biología": 0.171339658,
            "ccmin_Danza": 0.063456073,
            "ccmin_Nutrición y Alimentación": 0.084813851,
            "ccmin_Cine y Fotografía": 0.073267326,
            "ccmin_Educación y Pedagogía": 0.149686652,
            "ccmin_Filosofía y Ética": 0.125659984,
            "ccmin_Medicina y Enfermería": 0.170589878,
            "ccmin_Religión y Teología": 0.092318926,
            "s_LTEOL": 0.077648904,
            "s_TEOLO": 0.125517076,
            "s_DRCH2": 0.252777068,
            "s_INFBN": 0.077648904,
            "s_BACTE": 0.089581823,
            "s_ARTV2": 0.112456979,
            "s_FINA2": 0.181219965,
            "s_ARQUI": 0.057988066,
            "s_FILOS": 0.123566607,
            "s_ANTR2": 0.073267326,
            "s_RLINT": 0.057988066,
            "s_PSIC2": 0.080816877,
            "s_ECONM": 0.209664691,
            "s_LLMOD": 0.064757547,
            "s_MAGRV": 0.089581823,
            "s_IMECA": 0.165680534,
            "s_ICIVL": 0.186982886,
            "s_NUTDT": 0.174135903,
            "s_CTDPD": 0.204460051,
            "s_ADMD": 0.207833063,
            "s_EMSCL": 0.088652796,
            "s_LTEOT": 0.057988066,
            "s_CRELV": 0.077648904,
            "s_COMSC": 0.190871488,
            "s_IELEC": 0.180603471,
            "s_MEDIC": 0.253017689,
            "s_PSICO": 0.178697626,
            "s_LITER": 0.064757547,
            "s_SOCI2": 0.073267326,
            "s_ARES2": 0.088652796,
            "s_HIST2": 0.077648904,
            "Cant_Perdidas": 1.743754242,
            "Créditos Inscritos en Ciclo": 3.99014368,
            "Max_Ciclo": 2.967428556,
            "Cred_Max_Calif_Ciclo": 1.906572647,
            "Min_Ciclo": 2.053929808,
            "Cred_Min_Calif_Ciclo": 1.928524097,
            "Créditos Inscritos y Aprobados Ciclo": 4.155359775,
            "Rango_Ponderado_Ciclo": 7.419506355
        }
    
    def _get_coefficients(self):
        """Retorna el diccionario de coeficientes del modelo"""
        return {
            "Ciclo": 0.173332868,
            "Ciclo Admisión": -0.765634175,
            "Mult Programa": 0.03360566,
            "Situacion Acad": 0.19091103,
            "Promedio ciclo": 0.113437555,
            "Promedio Acumulado": -0.693411175,
            "Sexo": -0.064453406,
            "Rango edad": -0.025817174,
            "Internacional": 0.041596967,
            "cd_Bogotá D.C.": -0.063797119,
            "Total Créditos Acumula Tomados": -0.320156905,
            "Total Créditos Acumu Aprobados": -0.023170062,
            "Materias Vistas": 0.322582538,
            "TRL": -0.006152646,
            "IPS": 0.01496729,
            "MLP": 0.177636694,
            "p_Licenciatura_en_Ciencias_Religiosas": 0.113289604,
            "p_Licenciatura_en_Teologia_Tunja": 0.021671505,
            "p_Licenciatura_en_Educacion_Basica_con_enfasis_en_Humanidades_y_Lengua_Castellana": 0.049241923,
            "p_Odontologia": -0.195458092,
            "p_Teologia": -0.016913379,
            "p_Microbiologia_Industrial": -0.013803034,
            "p_Nutricion_y_Dietetica": -0.101590173,
            "p_Historia": 0.018416301,
            "p_Ingenieria_Mecatronica": -0.055653525,
            "p_Ingenieria_Electronica": -0.118038634,
            "p_Artes_Visuales": -0.043848344,
            "p_Ingenieria_Mecanica": -0.084893651,
            "p_Ecologia": -0.018935852,
            "p_Artes_Escenicas": 0.095543504,
            "p_Licenciatura_en_Educacion_Infantil": 0.050955669,
            "p_Economia": -0.012275459,
            "p_Derecho": -0.1199089,
            "p_Antropologia": 0.014665488,
            "p_Ingenieria_Industrial": -0.04801836,
            "p_Estudios_Literarios": 0.070644676,
            "p_Licenciatura_en_Teologia": 0.108817398,
            "p_Diseno_Industrial": -0.001632444,
            "p_Ingenieria_de_Sistemas": -0.072100857,
            "p_Bioingenieria": -0.041217284,
            "p_Contaduria_Publica": -0.06782897,
            "p_Ingenieria_Civil": -0.116822187,
            "p_Sociologia": 0.025667662,
            "p_Quimica_Farmaceutica": -0.10621986,
            "p_Finanzas": -0.062570378,
            "p_Administracion_de_Empresas": 0.00732956,
            "p_Microbiologia_Agricola_y_Veterinaria": 0.027663182,
            "p_Ciencia_Politica": 0.004299488,
            "p_Ciencias_de_la_Informacion_-_Bibliotecologia": 0.006564864,
            "p_Ciencia_de_Datos": -0.044036534,
            "p_Informatica_Matematica": -0.027824949,
            "p_Comunicacion_Social": -0.033326525,
            "p_Matematicas": -0.038847045,
            "dn_Valle_del_Cauca": 0.009048074,
            "dn_Antioquia": 0.003837296,
            "dn_Atlántico": 0.03621338,
            "dn_Bogotá_D.C.": -0.078563138,
            "dn_Boyacá": 0.002776436,
            "dn_Quindío": 0.034663769,
            "dn_Meta": 0.021047296,
            "dn_La_Guajira": -0.030372891,
            "dn_Bolívar": -0.033445405,
            "dn_Arauca": -0.019372931,
            "dn_Cundinamarca": -0.055525148,
            "dn_Santander": -0.004942112,
            "dn_Casanare": -0.005883337,
            "dn_Nariño": 0.023596873,
            "dn_Tolima": 0.012406989,
            "dn_Caldas": -0.001319212,
            "ccmax_Ingeniería Mecánica y Eléctrica": 0.026277393,
            "ccmax_Estadística y Análisis de Datos": -0.053148555,
            "ccmax_Artes Visuales": 0.010375773,
            "ccmax_Psicología": -0.004371449,
            "ccmax_Lengua y Literatura Española": 0.053971597,
            "ccmax_Idiomas Extranjeros": 0.00450328,
            "ccmax_Ciencia Política": 0.029703113,
            "ccmax_Programación y Ciencia de Datos": -0.057619395,
            "ccmax_Comunicación y Medios": -0.042802389,
            "ccmax_Tecnología y Computación": 0.02866862,
            "ccmax_Química": -0.05553059,
            "ccmax_Derecho": -0.04579333,
            "ccmax_Medicina y Enfermería": 0.009636684,
            "ccmax_Filosofía y Ética": 0.006560937,
            "ccmax_Proyectos y Talleres": -0.040270584,
            "ccmax_Odontología": -0.023645768,
            "ccmax_Mercadeo y Ventas": -0.02699335,
            "ccmax_Educación Física y Deportes": -0.03100324,
            "ccmax_Administración y Emprendimiento": -0.000682935,
            "ccmax_Cine y Fotografía": 0.020297843,
            "ccmax_Educación y Pedagogía": 0.038055238,
            "ccmax_Matemáticas": -0.009033024,
            "ccmax_Ciencias Sociales": -0.003608115,
            "ccmin_Administración y Emprendimiento": 0.038112624,
            "ccmin_Estadística y Análisis de Datos": -0.058067759,
            "ccmin_Ingeniería Mecánica y Eléctrica": -0.045144519,
            "ccmin_Matemáticas": 0.033385788,
            "ccmin_Programación y Ciencia de Datos": -0.007051577,
            "ccmin_Artes Visuales": 0.050211161,
            "ccmin_Comunicación y Medios": 0.041611074,
            "ccmin_Mercadeo y Ventas": 0.022650784,
            "ccmin_Lengua y Literatura Española": -0.010584844,
            "ccmin_Ciencia Política": 0.005032932,
            "ccmin_Física": 0.006211991,
            "ccmin_Música e Interpretación": 0.040667656,
            "ccmin_Ciencias Sociales": 0.018825376,
            "ccmin_Proyectos y Talleres": 0.001705659,
            "ccmin_Idiomas Extranjeros": -0.064701611,
            "ccmin_Arquitectura y Construcción": -0.036593999,
            "ccmin_Derecho": -0.013778491,
            "ccmin_Biología": 0.017438268,
            "ccmin_Danza": -0.035341022,
            "ccmin_Nutrición y Alimentación": 0.026713047,
            "ccmin_Cine y Fotografía": -0.010045539,
            "ccmin_Educación y Pedagogía": 0.004974996,
            "ccmin_Filosofía y Ética": 0.052824836,
            "ccmin_Medicina y Enfermería": -0.008794279,
            "ccmin_Religión y Teología": -0.034307151,
            "s_LTEOL": -0.016885141,
            "s_TEOLO": 0.12291157,
            "s_DRCH2": -0.003191811,
            "s_INFBN": 0.006564864,
            "s_BACTE": -0.030734628,
            "s_ARTV2": 0.038931855,
            "s_FINA2": -0.013948237,
            "s_ARQUI": 0.013883704,
            "s_FILOS": -0.016055753,
            "s_ANTR2": -0.006378202,
            "s_RLINT": -0.017091011,
            "s_PSIC2": -0.055759646,
            "s_ECONM": 0.011845972,
            "s_LLMOD": 0.022170746,
            "s_MAGRV": -0.065966122,
            "s_IMECA": 0.022976512,
            "s_ICIVL": 0.098334534,
            "s_NUTDT": 0.017422329,
            "s_CTDPD": 0.006808474,
            "s_ADMD": -0.044445992,
            "s_EMSCL": 0.014088187,
            "s_LTEOT": 0.021671505,
            "s_CRELV": -0.062410102,
            "s_COMSC": 0.086775788,
            "s_IELEC": 0.088623365,
            "s_MEDIC": -0.112373657,
            "s_PSICO": 0.091257423,
            "s_LITER": -0.028519361,
            "s_SOCI2": -0.025668136,
            "s_ARES2": 0.01200812,
            "s_HIST2": 0.011295582,
            "Cant_Perdidas": -0.124569901,
            "Créditos Inscritos en Ciclo": 0.029829151,
            "Max_Ciclo": -0.070484285,
            "Cred_Max_Calif_Ciclo": 0.095883329,
            "Min_Ciclo": 0.078605864,
            "Cred_Min_Calif_Ciclo": 0.093679381,
            "Créditos Inscritos y Aprobados Ciclo": -0.592607616,
            "Rango_Ponderado_Ciclo": 0.227253197
        }
    
    def validate_data(self, df):
        """
        Valida que el DataFrame tenga las columnas requeridas
        
        Args:
            df: DataFrame a validar
            
        Returns:
            tuple: (es_valido, lista_de_errores)
        """
        required_columns = [
            'Ciclo', 'Ciclo Admisión', 'Tipo Admisión', 'Programa', 
            'Mult Programa', 'Situacion Acad', 'Promedio ciclo', 
            'Promedio Acumulado', 'Sexo', 'Edad', 'Nacio en Colombia', 
            'Dpto.Nacimiento', 'Vive en Bogotá', 'Total Créditos Acumula Tomados',
            'Total Créditos Acumu Aprobados', 'Categoría MaxClase', 
            'Categoría MinClase', 'Materias Vistas'
        ]
        
        errors = []
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Columnas faltantes: {', '.join(missing_columns)}")
        
        return len(errors) == 0, errors
    
    def process_data(self, df):
        """
        Procesa el DataFrame aplicando todas las transformaciones necesarias
        
        Args:
            df: DataFrame con los datos crudos
            
        Returns:
            DataFrame procesado con probabilidades
        """
        data = df.copy()
        
        # 1. Tipo Admisión
        data["TRL"] = 0
        data["IPS"] = 0
        data["MLP"] = 0
        data.loc[data["Tipo Admisión"] == "Ingreso Primer Semestre", "IPS"] = 1
        data.loc[data["Tipo Admisión"] == "Traslado", "TRL"] = 1
        data.loc[data["Tipo Admisión"] == "Múltiple Programa", "MLP"] = 1
        data = data.drop(columns=["Tipo Admisión"])
        
        # 2. Programa
        for prog in self.programas:
            col = "p_" + prog.replace(" ", "_").replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
            data[col] = (data["Programa"] == prog).astype(int)
        data.drop(columns=["Programa"], inplace=True)
        
        # 3. Situación Académica
        map_situacion = {"Normal": 0, "Primera Prueba": 1, "Segunda Prueba": 2}
        data["Situacion Acad"] = data["Situacion Acad"].map(map_situacion)
        
        # 4. Sexo
        data["Sexo"] = data["Sexo"].replace({"Masculino": 1, "Femenino": 0})
        
        # 5. Edad
        def map_age_groups(age):
            if age <= 19:
                return 0
            elif age <= 24:
                return 1
            elif age <= 34:
                return 2
            else:
                return 3
        
        data["Edad"] = data["Edad"].apply(map_age_groups)
        data.rename(columns={"Edad": "Rango edad"}, inplace=True)
        
        # 6. Internacional
        data["Nacio en Colombia"] = data["Nacio en Colombia"].replace({"Si": 0, "No": 1})
        data = data.rename(columns={"Nacio en Colombia": "Internacional"})
        
        # 7. Departamento Nacimiento
        data["Dpto.Nacimiento"] = data["Dpto.Nacimiento"].astype(str).str.strip()
        for dep in self.departamentos:
            col_name = f"dn_{dep.replace(' ', '_')}"
            data[col_name] = 0
        for dep in self.departamentos:
            col_name = f"dn_{dep.replace(' ', '_')}"
            data.loc[data["Dpto.Nacimiento"] == dep, col_name] = 1
        data.drop(columns=["Dpto.Nacimiento"], inplace=True)
        
        # 8. Vive en Bogotá
        data["Vive en Bogotá"] = data["Vive en Bogotá"].replace({"Si": 1, "No": 0})
        data = data.rename(columns={"Vive en Bogotá": "cd_Bogotá D.C."})
        
        # 9. Categoría MaxClase
        data["Categoría MaxClase"] = data["Categoría MaxClase"].astype(str).str.strip()
        for cat in self.categorias_max:
            data[f"ccmax_{cat}"] = 0
        dummies = pd.get_dummies(data["Categoría MaxClase"])
        dummies = dummies.reindex(columns=self.categorias_max, fill_value=0)
        for cat in self.categorias_max:
            data[f"ccmax_{cat}"] = dummies[cat]
        data.drop(columns=["Categoría MaxClase"], inplace=True)
        
        # 10. Categoría MinClase
        data["Categoría MinClase"] = data["Categoría MinClase"].astype(str).str.strip()
        for cat in self.categorias_min:
            data[f"ccmin_{cat}"] = 0
        dummies = pd.get_dummies(data["Categoría MinClase"])
        dummies = dummies.reindex(columns=self.categorias_min, fill_value=0)
        for cat in self.categorias_min:
            data[f"ccmin_{cat}"] = dummies[cat]
        data.drop(columns=["Categoría MinClase"], inplace=True)
        
        # 11. Estandarización
        variables_escalables = [
            col for col in data.columns
            if col in self.mean_dict and col in self.scale_dict
        ]
        mean_series = pd.Series(self.mean_dict)
        scale_series = pd.Series(self.scale_dict)
        data[variables_escalables] = (
            data[variables_escalables] - mean_series[variables_escalables]
        ) / scale_series[variables_escalables]
        
        # 12. Calcular probabilidad
        variables_comunes = [col for col in self.coef_dict.keys() if col in data.columns]
        coef_series = pd.Series(self.coef_dict)
        data["log_odds"] = (
            self.intercepto + (data[variables_comunes] * coef_series[variables_comunes]).sum(axis=1)
        )
        data["probabilidad"] = 1 / (1 + np.exp(-data["log_odds"]))
        
        # Categorizar riesgo
        data["nivel_riesgo"] = pd.cut(
            data["probabilidad"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Bajo", "Medio", "Alto"]
        )
        
        return data
    
    def get_summary_stats(self, df):
        """
        Calcula estadísticas resumidas del DataFrame procesado
        
        Args:
            df: DataFrame procesado
            
        Returns:
            dict: Diccionario con estadísticas
        """
        return {
            "total_estudiantes": len(df),
            "riesgo_alto": len(df[df["probabilidad"] > 0.6]),
            "riesgo_medio": len(df[(df["probabilidad"] >= 0.3) & (df["probabilidad"] <= 0.6)]),
            "riesgo_bajo": len(df[df["probabilidad"] < 0.3]),
            "probabilidad_promedio": df["probabilidad"].mean(),
            "probabilidad_max": df["probabilidad"].max(),
            "probabilidad_min": df["probabilidad"].min()
        }
