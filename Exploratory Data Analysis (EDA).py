import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# initial table display config:
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:.2f}'.format

print('-- EXPLORATORY DATA ANALYSIS (EDA) --')
import asciitable
happiness_df = pd.DataFrame(asciitable.
                            read('C:/Users/Ignacio'
                                 ' Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/dataset_happiness.txt'))
happiness_catnum_df = pd.read_csv('C:/Users/Ignacio Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/happiness_catnum_df.csv')
happiness_num_df = pd.read_csv('C:/Users/Ignacio Ojeda/Documents/_NeverBackUp/EMPRESAS/BBVA/happiness_catnum_df.csv')
print(happiness_catnum_df.dtypes)

# conversion of AñoMes to TimeSeries:
from datetime import datetime as dt
happiness_catnum_df['AñoMes'] = pd.to_datetime(happiness_catnum_df['AñoMes'], format='%Y%m')
happiness_catnum_df['AñoMes'] = pd.to_datetime(happiness_catnum_df['AñoMes'], format='%Y%m')

# number of interviewed people per region:
interviewed_per_region = happiness_catnum_df.groupby(by=['A.1'])['A.1'].\
    count().to_frame('people').reset_index(level=['A.1'])
interviewed_per_region = interviewed_per_region.rename(columns={'A.1': 'A1'})
interviewed_per_region = interviewed_per_region.rename(columns={'A1': 'A.1'})
print(interviewed_per_region)

interviewed_per_region[['A.1', 'people']].plot()
plt.xticks(interviewed_per_region.index, interviewed_per_region['A.1'].values)

# average age of interviewed people per region:
print(happiness_catnum_df.columns[0:10])
print(happiness_catnum_df.columns[10:20])
avg_age_region = happiness_catnum_df.groupby(by=['A.1'])['A.4_valor'].\
    mean().to_frame('avg_age_region').reset_index(level=['A.1'])
print(avg_age_region)
plt.clf()
avg_age_region.plot() # intervals age between 46 and 56 (except Melilla w/ 42)

# average age of interviewed people per region and sex - test years in datasets:
import re
A4_less_40 = []
for age in happiness_df['A.4']:
    a = re.findall(r'\d{2}', age)
    A4_less_40.append(a)

A4_less_40_new = []
for age in happiness_catnum_df['A.4_valor']:
    if age <= 40:
        A4_less_40_new.append(age)

print(happiness_catnum_df[happiness_catnum_df['A.4_valor'] <= 40]) # 13233 obs < 40 years old

# distribution interviewed by region and sex:
sex_per_region = happiness_catnum_df.groupby(by=['A.1', 'A.3'])['A.3'].\
    count().to_frame('people').reset_index(level=['A.1', 'A.3'])
print(sex_per_region) # la mayor diferencia de encuestados entre hombres y mujeres: en Cataluña

plt.clf()
plt.barh(np.arange(len(sex_per_region.people)), sex_per_region.people,
         align='center', alpha=0.5)
plt.yticks(np.arange(len(sex_per_region.people)), sex_per_region['A.1'])
plt.tight_layout()
plt.grid()
# plt.show()

sex_per_region_gbsex = sex_per_region.groupby(by=['A.3'])['people']\
    .sum().to_frame('no_sex').reset_index(level=['A.3'])
print(sex_per_region_gbsex) # una distribucion del 50% en total
plt.clf()
plt.barh(np.arange(len(sex_per_region_gbsex.no_sex)), sex_per_region_gbsex.no_sex, align='center', alpha=0.5)
plt.yticks(np.arange(len(sex_per_region_gbsex.no_sex)), sex_per_region_gbsex['A.3'])
plt.tight_layout()
plt.grid()
# plt.show()
# relation between region, sex and age with happiness degree:
people_per_region = happiness_catnum_df.groupby(by=['A.1'])['A.1'].\
    count().to_frame('no_people').astype(int)
happiness_per_region = happiness_catnum_df.groupby(by=['A.1'])['D.1_values'].\
    sum().to_frame('happiness_absolute').astype(int)
print('people per region (this sample):\n', people_per_region)
happiness_per_region['average_degree'] = happiness_per_region.happiness_absolute / people_per_region.no_people
happiness_per_region = happiness_per_region.reset_index(level=['A.1'])
print(happiness_per_region)