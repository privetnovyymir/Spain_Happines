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

''' -- CONCLUSIONES -- '''

# grafico de barras escalado de regiones mas felices:
plt.clf()
plt.barh(np.arange(len(happiness_per_region.average_degree)),
         happiness_per_region.average_degree.sort_values(ascending=True), align='center', alpha=0.5)
plt.yticks(np.arange(len(happiness_per_region.average_degree)),
           happiness_per_region.sort_values('average_degree', ascending=True)['A.1'])
plt.tight_layout()
plt.grid()
# plt.show()
print('more happy people lives in:', happiness_per_region.set_index('A.1').average_degree.idxmax(axis=1))
print('less happy people lives in:', happiness_per_region.set_index('A.1').average_degree.idxmin(axis=1))

# distribucion de nacionalidades y relacion entre nacionalidad y felicidad:
nationality_distribution = happiness_catnum_df.groupby(by=['A.9.1'])['D.1_values'].\
    count().to_frame('people').astype(int)
nationality_happiness_distribution = happiness_catnum_df.groupby(by=['A.9.1'])['D.1_values'].\
    sum().to_frame('happiness_per_nationality').astype(int)
print(nationality_distribution)
print(nationality_happiness_distribution)
print('Happiness degree of spanish people:',
      nationality_happiness_distribution.happiness_per_nationality / nationality_distribution.people)
# ### conclusion: foreigners are less more happier than spanish (7.39 vs 7.32) out of 10

# variacion de la felicidad general en función del tiempo (en años):
happiness_fx_time = happiness_catnum_df.groupby(by=['AñoMes'])['D.1_values']\
    .mean().to_frame('happiness').reset_index(level=['AñoMes'])
happiness_fx_time_yrBasis = happiness_fx_time.groupby(
    by=happiness_fx_time.AñoMes.dt.year)['happiness'].mean()
print(happiness_fx_time)
print(happiness_fx_time_yrBasis)

plt.clf()
plt.barh(np.arange(len(happiness_fx_time.happiness)), happiness_fx_time.happiness)
plt.yticks(np.arange(len(happiness_fx_time.happiness)), happiness_fx_time.AñoMes)
plt.grid()
# plt.show()
# ### conclusion: happiness is increasing in the last years so the prediction for 2014 shall be lower

# relacion entre el tipo de empleo y el grado de felicidad:
happiness_fx_job = happiness_catnum_df.groupby(by=['G.6'])['D.1_values'].\
    mean().to_frame('happiness').reset_index(level=['G.6'])
print('\nGrado de felicidad en función del tipo de empleo:\n', happiness_fx_job)
# ### conclusion 1: los parados son los mas infelices, seguidos de los jubilados
# ### conclusion 2: los directores, profesionales y estudiantes son los mas felices

# relacion entre estudios y felicidad (desagregado):
happiness_fx_studies = happiness_catnum_df.groupby(by=['E.2_2'])['D.1_values'].\
    mean().to_frame('happiness').reset_index(level=['E.2_2'])
print('\nGrado de felicidad en función del tipo de estudios:\n', happiness_fx_studies)
# ### conclusion 1: la gente mas feliz es la que tiene estudios universitarios
# ### conclusion 2: la gente mas infeliz es la que no tiene estudios de ningun tipo

# relacion entre religion y tipo de empleo:
people_per_job = happiness_catnum_df.groupby(by=['G.6'])['G.6'].\
    count().to_frame('no_people').astype(int)
job_and_religion = happiness_catnum_df.groupby(by=['D.2', 'G.6'])['G.6'].\
    count().to_frame('no_people').astype(int)
print(people_per_job)
print(job_and_religion)

job_and_religion['as % of total G.6 subgroups'] = job_and_religion.no_people / people_per_job.no_people
print('\n % de D.2 del grupo G.6 sobre G.6:\n', job_and_religion)
# ### conclusion 1: el mayor grupo de ateos esta en los estudiantes (24%) y Directivos (17%)
# ### conclusion 2: el menor grupo de ateos esta en los agricultores (3%) y trabajo domestico NR (3%)
# ### conclusion 3: el mayor grupo de catolicos esta en los agricultores (87%), jubilados (84%) y tr domestico NR (88%)
# ### conclusion 4: el menor grupo de catolicos esta en los estudiantes (44%) ~ mucho estudiante no cat pero religioso

# relacion entre religion y grado de felicidad:
happiness_fx_job = happiness_catnum_df.groupby(by=['D.2'])['D.1_values'].\
    mean().to_frame('happiness')
print(happiness_fx_job)
# ### conclusion: los mas felices son los creyentes en otra religion , seguidos de los catolicos/as
# y los menos, los N.C.

# relacion entre estado civil y felicidad:
happiness_fx_civil = happiness_catnum_df.groupby(by=['F.1'])['D.1_values'].\
    mean().to_frame('happiness')
print(happiness_fx_civil)
# ### conclusion 1: las personas mas felices son los casados/as, seguidos de los solteros
# ### conclusion 2: las mas infelices, los separados/as y divorciados/as

# relacion entre sueldo y felicidad:
happiness_fx_salary = happiness_catnum_df.groupby(by=['G.13'])['D.1_values'].\
    mean().to_frame('happiness')
print(happiness_fx_salary)
# ### conclusion 1: la gente mas feliz gana de 4501 a 6000
# ### conclusion 2: la gente mas infeliz, de 301 a 699

# brecha salarial hombres - mujeres ultimos 3 años y variacion interanual:
sex_salary_diff_min = happiness_catnum_df.groupby(by=['A.3'])['G.13_min'].mean()
sex_salary_diff_max = happiness_catnum_df.groupby(by=['A.3'])['G.13_max'].mean()
print('\nMin avg salary per sex group\n', sex_salary_diff_min)
print('\nMax avg salary per sex group\n', sex_salary_diff_max)
# ### conclusion 1: diferencia global del 70% en el rango mas bajo y del 90% en el mas alto
sex_salary_diff_min = happiness_catnum_df.groupby(by=[happiness_catnum_df.AñoMes.dt.year, 'A.3'])['G.13_min'].\
    mean().to_frame('salary')
sex_salary_diff_max = happiness_catnum_df.groupby(by=[happiness_catnum_df.AñoMes.dt.year, 'A.3'])['G.13_max'].\
    mean().to_frame('salary')
print('\nMin avg salary per sex group\n', sex_salary_diff_min)
print('\nMax avg salary per sex group\n', sex_salary_diff_max)
sex_salary_diff_min = sex_salary_diff_min.reset_index(level=['AñoMes', 'A.3'])
sex_salary_diff_max = sex_salary_diff_max.reset_index(level=['AñoMes', 'A.3'])
print('\nMin avg salary per sex group\n', sex_salary_diff_min)
print('\nMax avg salary per sex group\n', sex_salary_diff_max)

from scipy import stats
male_max_salary_variation_YoY = sex_salary_diff_max[sex_salary_diff_max['A.3'] == 'Hombre']
female_max_salary_variation_YoY = sex_salary_diff_max[sex_salary_diff_max['A.3'] == 'Mujer']
print(male_max_salary_variation_YoY)
print(female_max_salary_variation_YoY)

male_salary_diffvar = stats.mstats.gmean(male_max_salary_variation_YoY.salary)
female_salary_diffvar = stats.mstats.gmean(female_max_salary_variation_YoY.salary)
print('\nMedia geometrica - salario hombres:\n', int(male_salary_diffvar))
print('\nMedia geometrica - salario mujeres:\n', int(female_salary_diffvar))
print('\nBrecha salarial ponderada max:', int(male_salary_diffvar - female_salary_diffvar))
print('\nBrecha salarial ponderada max (%):',
      int(male_salary_diffvar - female_salary_diffvar) / male_salary_diffvar)

# ### conclusion: como podemos ver en todas las tablas que genera el codigo,
# las mujeres cobran entre un 15% y un 45% menos que los hombres. Especificamente,
# en el rango maximo de sueldo y edad, un 45% menos que los hombres.
