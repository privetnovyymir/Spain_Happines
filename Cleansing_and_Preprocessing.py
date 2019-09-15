import numpy as np
import pandas as pd
import asciitable # seems the best case to load the dataset

# initial table display config:
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:.2f}'.format

''' data cleansing and formatting process '''
'''### Variables sociodemográficas básicas

**A.1 Comunidad Autónoma** object => str

**A.2 Tamaño de hábitat** object => 2 int cols

**A.3 Sexo de la persona entrevistada**

**A.4 Edad de la persona entrevistada**  
¿Cuántos años cumplió Ud. en su último cumpleaños? (Nota: Variable continua. Edad en años) 

**A.9.1 Nacionalidad de la persona entrevistada**  
Española, Española y otra, Otra nacionalidad

**A.9.2 Nacionalidad de la persona entrevistada (extranjeros)**'''

happiness_array = asciitable.read('C:/Users/Ignacio Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/dataset_happiness.txt')
print('Dataset as numpy.recarray:\n', happiness_array, '\nType:', type(happiness_array))
happiness_df = pd.DataFrame(happiness_array)
print('Dataframe:\n', happiness_df, '\nType:', happiness_df, '\nInfo:', happiness_df.info(),
      '\nStats\n:', happiness_df.describe(), '\nColumns:', happiness_df.columns, '\nShape:',
      happiness_df.shape, '\nDtypes:', happiness_df.dtypes)

print(' -- VARIABLES SOCIECONOMICAS BASICAS --')
happiness_vsb_df = happiness_df[['AñoMes', 'N_Entrevista', 'A.1', 'A.2', 'A.3', 'A.4', 'A.9.1', 'A.9.2']]
print('Dataframe:\n', happiness_vsb_df, '\nType:', happiness_vsb_df, '\nInfo:', happiness_vsb_df.info(),
      '\nStats\n:', happiness_vsb_df.describe(), '\nColumns:', happiness_vsb_df.columns, '\nShape:',
      happiness_vsb_df.shape, '\nDtypes:', happiness_vsb_df.dtypes)

print('-- dtypes transformations --\n')
import re
A2 = []
for habitante in happiness_vsb_df['A.2']:
    a = re.sub(r'\sa\s', ' - ', habitante)
    b = re.sub(r'\D', ' ', a)
    b = str.strip(b)
    c = re.sub(r'\s{2}', '-', b)
    d = re.sub(r'\s', '', c)
    d = d.translate({ord(i): None for i in ' '})
    A2.append(re.sub(r'\shabitantes', '', d))

happiness_vsb_df['A.2'] = pd.DataFrame(A2)
happiness_vsb_df['A.2_min'] = happiness_vsb_df['A.2'].str.partition('-')[0]
happiness_vsb_df['A.2_max'] = happiness_vsb_df['A.2'].str.partition('-')[2]

happiness_vsb_df['A.2_min'] = pd.to_numeric(happiness_vsb_df['A.2_min'], errors='coerce')
happiness_vsb_df['A.2_max'] = pd.to_numeric(happiness_vsb_df['A.2_max'], errors='coerce')
happiness_vsb_df = happiness_vsb_df[['AñoMes', 'N_Entrevista', 'A.1', 'A.2', 'A.2_min',
                                     'A.2_max', 'A.3', 'A.4', 'A.9.1', 'A.9.2']]

A2_max = []
for min, max in zip(happiness_vsb_df['A.2_min'], happiness_vsb_df['A.2_max']):
    if max / max != 1:
        A2_max.append(min) # as Na, NaN and inf appears, we substitute them with min value
    else:
        A2_max.append(max)
happiness_vsb_df['A.2_max'] = pd.DataFrame(A2_max)
happiness_vsb_df['A.2_max'] = happiness_vsb_df['A.2_max'].astype(int)
print(happiness_vsb_df)
print(happiness_vsb_df.dtypes)

print(happiness_vsb_df['A.3'].describe())
A3_dict = {'Mujer': 1, 'Hombre': 0}
print(A3_dict['Hombre'])
valor = []
for sexo in happiness_vsb_df['A.3']:
    if sexo == 'Mujer':
        valor.append(A3_dict['Mujer'])
    else:
        valor.append(A3_dict['Hombre'])

A4_valor = []
for edad in happiness_vsb_df['A.4']:
    nueva_edad = re.sub(r'\.0', '', edad)
    nueva_edad = re.sub(r'\D', '', nueva_edad)
    nueva_edad = list(map(int, re.findall(r'\d\d$', nueva_edad)))
    # nueva_edad = list(map(int, nueva_edad))
    A4_valor.append(nueva_edad)

delist = []
for item in A4_valor:
    try:
        delist.append(item[0])
    except IndexError:
        delist.append('NA')


A4_delisted = np.array(delist)
happiness_vsb_df['A.4_valor'] = A4_delisted
happiness_vsb_df['A.3_valor'] = pd.DataFrame(valor)
happiness_vsb_df = happiness_vsb_df[['AñoMes', 'N_Entrevista', 'A.1', 'A.2', 'A.2_min',
                                     'A.2_max', 'A.3', 'A.3_valor', 'A.4', 'A.4_valor',
                                     'A.9.1', 'A.9.2']]
print(happiness_vsb_df)

# nominal transformation (not ordinal but we'll use ordinal encoder for the case):
import category_encoders as ce
region_df = happiness_vsb_df[['A.1', 'A.2_min']]
from sklearn.preprocessing import LabelEncoder

'''  ### Valoración de la situación económica personal

**B.2.1 Valoración de la situación económica personal actual**  
¿Cómo calificaría Ud. su situación económica personal en la actualidad: es muy buena, buena, regular, mala o muy mala?

**B.2.2 Valoración prospectiva de la situación económica personal**  
Y, ¿cree Ud. que dentro de un año su situación económica personal será mejor, igual o peor que ahora?'''

# print(' -- VALORACION DE LA SITUACION ECONOMICA PERSONAL -- ')
happiness_sep_df = happiness_df[['AñoMes', 'N_Entrevista', 'B.2.1', 'B.2.2']]
print('Dataframe:\n', happiness_sep_df, '\nType:', type(happiness_sep_df), '\nInfo:', happiness_sep_df.info(),
      '\nStats\n:', happiness_sep_df.describe(), '\nColumns:', happiness_sep_df.columns, '\nShape:',
      happiness_sep_df.shape, '\nDtypes:', happiness_sep_df.dtypes)

print('-- dtypes transformations --\n')
# ready to pre-processing

'''
### Variables políticas

**C.3.1 Escala de autoubicación ideológica (1-10)**  
Cuando se habla de política se utilizan normalmente las expresiones izquierda y derecha. 
En esta tarjeta hay una serie de casillas que van de izquierda (1) a derecha (10). 
¿En qué casilla se colocaría Ud.?
'''

happiness_vp_df = happiness_df[['AñoMes', 'N_Entrevista', 'C.3.1']]
print(happiness_vp_df, '\nStats', happiness_vp_df.describe())
print(happiness_vp_df['C.3.1'].sort_values(ascending=True).unique())

# cleansing and formatting (N.C and N.S substitute with the top: 5 <not the average>):
happiness_vp_df['C.3.1_values'] = happiness_vp_df['C.3.1'].str.replace('N.C.', '5')
happiness_vp_df['C.3.1_values'] = happiness_vp_df['C.3.1_values'].str.replace('N.S.', '5')

print(happiness_vp_df, '\nStats', happiness_vp_df.describe())
print(happiness_vp_df['C.3.1'].sort_values(ascending=True).unique())
print(happiness_vp_df['C.3.1_values'].sort_values(ascending=True).unique())

# cleansing and formatting (remove non-digits with regex):
vp_values = []
for value in happiness_vp_df['C.3.1_values']:
    a = re.sub('\D', '', value)
    vp_values.append(a)

happiness_vp_df['C.3.1_values'] = vp_values
happiness_vp_df['C.3.1_values'] = pd.to_numeric(happiness_vp_df['C.3.1_values'])

print(happiness_vp_df, '\nStats:', happiness_vp_df.describe(),
      '\nInfo:', happiness_vp_df.info(), '\nDtypes:', happiness_vp_df.dtypes)
print(happiness_vp_df['C.3.1'].sort_values(ascending=True).unique())
print(happiness_vp_df['C.3.1_values'].sort_values(ascending=True).unique())

'''
### Actitudes personales

**D.1 Escala de felicidad personal (0-10)**  
En términos generales, ¿en qué medida se considera Ud. una persona feliz o infeliz? 
Por favor, use una escala de 0 a 10, en la que 0 significa que se considera 
'completamente infeliz' y 10 que se considera 'completamente feliz'.

**D.2 Religiosidad de la persona entrevistada**  
¿Cómo se define Ud. en materia religiosa: católico/a, creyente de otra religión, 
no creyente o ateo/a? 

**D.3 Frecuencia de asistencia a oficios religiosos de la persona entrevistada**  
¿Con qué frecuencia asiste Ud. a misa u otros oficios religiosos, 
sin contar las ocasiones relacionadas con ceremonias de tipo social, 
por ejemplo, bodas, comuniones o funerales? 
'''

happiness_ap_df = happiness_df[['AñoMes', 'N_Entrevista', 'D.1', 'D.2', 'D.3']]
print(happiness_ap_df, '\nStats', happiness_ap_df.describe(),
      '\nInfo:', happiness_ap_df.info(), '\nDtypes', happiness_ap_df.dtypes)
print(happiness_ap_df['D.1'].sort_values(ascending=True).unique())
print(happiness_ap_df['D.2'].sort_values(ascending=True).unique())
print(happiness_ap_df['D.3'].sort_values(ascending=True).unique())

# cleansing and formatting process:
ap_values = []
for value in happiness_ap_df['D.1']:
    if value == 'N.C.' or value == 'N.S.':
        ap_values.append('5')
    else:
        a = re.sub(r'\D', '', value)
        ap_values.append(a)

happiness_ap_df['D.1_values'] = ap_values
happiness_ap_df['D.1_values'] = pd.to_numeric(happiness_ap_df['D.1_values'])
print(happiness_ap_df, '\nStats', happiness_ap_df.describe(),
      '\nInfo:', happiness_ap_df.info(), '\nDtypes', happiness_ap_df.dtypes)
print(happiness_ap_df['D.1'].sort_values(ascending=True).unique())
print(happiness_ap_df['D.1_values'].sort_values(ascending=True).unique()) # target variable
print(happiness_ap_df['D.2'].sort_values(ascending=True).unique())
print(happiness_ap_df['D.3'].sort_values(ascending=True).unique())

k = happiness_ap_df[happiness_ap_df['D.1_values'].isnull()]
print(k) # there is empty values to fill with the ML model (prediction)

'''
### Nivel de estudios de la persona entrevistada

**E.2 Nivel de estudios alcanzado por la persona entrevistada**  
Niveles desagregados

**E.2_2 Nivel de estudios alcanzado por la persona entrevistada**  
Niveles agregados
'''

happiness_ne_df = happiness_df[['AñoMes', 'N_Entrevista', 'E.2', 'E.2_2']]
print(happiness_ne_df, '\nStats', happiness_ne_df.describe(),
      '\nInfo:', happiness_ne_df.info(), '\nDtypes', happiness_ne_df.dtypes)
print(happiness_ne_df['E.2'].sort_values(ascending=True).unique()) # niveles desagragados
print(happiness_ne_df['E.2_2'].sort_values(ascending=True).unique()) # niveles agregados

print(happiness_ne_df[happiness_ne_df['E.2_2'] == 'N.C.']) # 137 obs
print(happiness_ne_df[happiness_ne_df['E.2_2'] == 'N.C.'].groupby(by=['E.2']).count())
# confirmation of number of N.C. containing N.C.:
print(happiness_ne_df.groupby(by=['E.2_2', 'E.2'])['E.2'].count()) # 137 obs

'''
### Situación civil y de convivencia de la persona entrevistada
**F.1 Estado civil de la persona entrevistada**  
¿Cuál es su estado civil? 

**F.2 Situación de convivencia de las personas no casadas**  
¿En cuál de las siguientes situaciones se encuentra Ud.? 

**F.3 Persona que aporta más ingresos al hogar**  
¿Quién es la persona que aporta más ingresos al hogar?
'''
happiness_sc_df = happiness_df[['AñoMes', 'N_Entrevista', 'F.1', 'F.2', 'F.3']]
print(happiness_sc_df, '\nStats', happiness_sc_df.describe(),
      '\nInfo:', happiness_sc_df.info(), '\nDtypes', happiness_sc_df.dtypes)
print(happiness_sc_df.groupby(by=['F.1', 'F.2'])['F.2'].count())

'''
### Variables laborales y socioeconómicas

**G.1 Situación laboral de la persona entrevistada**  
¿En cuál de las siguientes situaciones se encuentra Ud. actualmente?

**G.2 Probabilidad de perder el empleo actual (próximos 12 meses)**  
(Sólo se pregunta a quienes trabajan)  
¿Piensa Ud. que es muy probable, bastante, poco o nada probable que durante 
los próximos doce meses pierda su empleo actual?

**G.3 Probabilidad de encontrar empleo (próximos 12 meses)**  
(Sólo se pregunta a quienes están parados)  
¿Y cree Ud. que es muy probable, bastante, poco o nada probable que durante 
los próximos doce meses encuentre Ud. trabajo? 

**G.4 Situación profesional de la persona entrevistada**  
¿Ud. (o la persona que aporta más ingresos al hogar) trabaja (o trabajaba) como...?

**G.6 Condición socioeconómica de la persona entrevistada**

**G.12 Ingresos del hogar**  
Actualmente, entre todos los miembros del hogar (incluida la persona entrevistada) y por 
todos los conceptos, ¿de cuántos ingresos disponen por término medio en su hogar al mes, 
después de la deducción de impuestos (o sea, ingresos netos)? No le pido que me indique la 
cantidad exacta, sino que me señale en esta tarjeta en qué tramo de la escala están 
comprendidos los ingresos de su hogar.

**G.13 Ingresos personales**  
¿Y en qué tramo de esa misma escala están comprendidos sus ingresos personales, 
después de las deducciones de impuestos, es decir, sus ingresos netos?
'''

happiness_vls_df = happiness_df[['AñoMes', 'N_Entrevista', 'G.1', 'G.2', 'G.3', 'G.4', 'G.6', 'G.12', 'G.13']]
print(happiness_vls_df, '\nStats', happiness_vls_df.describe(),
      '\nInfo:', happiness_vls_df.info(), '\nDtypes', happiness_vls_df.dtypes)
# confirm each unique to find potential errors:
print(happiness_vls_df['G.1'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.2'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.3'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.4'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.6'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.12'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.13'].sort_values(ascending=True).unique())

# format G.12 and G.13 data to numeric intervals:
G12_values = []
for raw_interval12 in happiness_vls_df['G.12']:
    less_raw12 = re.sub(r'[&;]', '', raw_interval12)
    if 'Menos o igual a' in less_raw12:
        less_raw2 = re.sub(r'Menos\so\sigual\sa', 'De 0 a', less_raw12) # left open interval
        G12_values.append(less_raw2) # closed (both) interval
    elif 'Más de' in less_raw12:
        less_raw2 = re.sub(r'Más\sde', '>', less_raw12)
        G12_values.append(less_raw2) # left open interval (to math.inf)
    elif 'No tienen ingresos' in less_raw12:
        G12_values.append('0 euro')
    else:
        G12_values.append(less_raw12)

print(np.unique(G12_values))
happiness_vls_df['G.12'] = G12_values
print(happiness_vls_df.groupby(by=['G.6', 'G.13', 'G.12'])['G.12'].count())

# eliminar N.C. y N.S. ya que son pocas observaciones y quizas luego, al terminar,
# meterlas en la EDA y ver como cambian los datos (y decidir si finalmente me los quedo o no):
happiness_vls_df = happiness_vls_df[happiness_vls_df['G.12'] != 'N.C.']
happiness_vls_df = happiness_vls_df[happiness_vls_df['G.12'] != 'N.S.']
print(happiness_vls_df.groupby(by=['G.6', 'G.13', 'G.12'])['G.12'].count())

G13_values = []
for raw_interval13 in happiness_vls_df['G.13']:
    less_raw13 = re.sub(r'[&;]', '', raw_interval13)
    if 'Menos o igual a' in less_raw13:
        less_raw3 = re.sub(r'Menos\so\sigual\sa', 'De 0 a', less_raw13) # left open interval
        G13_values.append(less_raw3) # closed (both) interval
    elif 'Más de' in less_raw13:
        less_raw3 = re.sub(r'Más\sde', '>', less_raw13)
        G13_values.append(less_raw3) # left open interval (to math.inf)
    elif 'No tiene ingresos' in less_raw13:
        G13_values.append('0 euro')
    else:
        G13_values.append(less_raw13)

happiness_vls_df['G.13'] = G13_values
happiness_vls_df = happiness_vls_df[happiness_vls_df['G.13'] != 'N.C.']

print(happiness_vls_df.groupby(by=['G.6', 'G.13', 'G.12'])['G.12'].count())
print(happiness_vls_df['G.12'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.13'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.12'].sort_values(ascending=True).unique() ==
      happiness_vls_df['G.13'].sort_values(ascending=True).unique()) # returns True for all

# transformation to numeric intervals:
intervals_12, intervals_13 = [], []
for interval_12_raw, interval_13_raw in zip(happiness_vls_df['G.12'], happiness_vls_df['G.13']):
    interval_12 = re.sub(r'\seuro', '', interval_12_raw)
    interval_13 = re.sub(r'\seuro', '', interval_13_raw)
    interval_12 = re.sub(r'>\s\d+\.\d+', 'De 6.000 a 100.000', interval_12)
    interval_13 = re.sub(r'>\s\d\.\d+', 'De 6.000 a 100.000', interval_13)
    interval_12 = re.sub(r'De\s', '', interval_12)
    interval_13 = re.sub(r'De\s', '', interval_13)
    interval_12 = re.sub(r'\.', '', interval_12)
    interval_13 = re.sub(r'\.', '', interval_13)
    interval_12 = re.sub(r'^0$', '0 a 0', interval_12)
    interval_13 = re.sub(r'^0$', '0 a 0', interval_13)
    intervals_12.append(interval_12)
    intervals_13.append(interval_13)

happiness_vls_df['G.12'], happiness_vls_df['G.13'] = intervals_12, intervals_13
min12 = pd.to_numeric(happiness_vls_df['G.12'].str.partition(' a ')[0])
max12 = pd.to_numeric(happiness_vls_df['G.12'].str.partition(' a ')[2])
print(min12, max12)
intervals = []
for minval, maxval in zip(min12, max12):
    intervals.append(pd.Interval(left=minval, right=maxval))

min13 = pd.to_numeric(happiness_vls_df['G.13'].str.partition(' a ')[0])
max13 = pd.to_numeric(happiness_vls_df['G.13'].str.partition(' a ')[2])
print(min13, max13)
intervals = []
for minval, maxval in zip(min13, max13):
    intervals.append(pd.Interval(left=minval, right=maxval))

happiness_vls_df['G.12_intervals'] = intervals
happiness_vls_df['G.12_min'] = min12
happiness_vls_df['G.12_max'] = max12
happiness_vls_df['G.13_intervals'] = intervals
happiness_vls_df['G.13_min'] = min13
happiness_vls_df['G.13_max'] = max13

print(happiness_vls_df[['G.12_intervals', 'G.13_intervals']])
print(happiness_vls_df.groupby(by=['G.6', 'G.13_intervals', 'G.12_intervals'])['G.12_intervals'].count())
print(happiness_vls_df['G.12_intervals'].sort_values(ascending=True).unique())
print(happiness_vls_df['G.13_intervals'].sort_values(ascending=True).unique())