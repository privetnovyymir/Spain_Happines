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