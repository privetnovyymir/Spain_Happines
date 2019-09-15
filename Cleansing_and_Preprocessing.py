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