import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat


# Подготовка данных
# Task 1
pd.set_option('display.max_columns', None)
df = pd.read_csv('dataset_telecom.csv')
df.columns = df.columns.str.lower()
df.rename(columns={'возраст': 'age',
                   'среднемесячный расход': 'average_month_expense',
                   'средняя продолжительность разговоров': 'average_duration_calls',
                   'звонков днем за месяц': 'calls_day_month',
                   'звонков вечером за месяц': 'calls_evening_month',
                   'звонков ночью за месяц': 'calls_night_month',
                   'звонки в другие города': 'calls_other_city',
                   'звонки в другие страны': 'calls_other_countries',
                   'доля звонков на стационарные телефоны': 'calls_landline_phone',
                   'количество sms за месяц': 'count_sms_month',
                   'дата подключения тарифа': 'date_connection_rate'}, inplace=True)
dict_rus = {
    'age': 'возраст',
    'average_month_expense': 'среднемесячный расход',
    'average_duration_calls': 'средняя продолжительность разговоров',
    'calls_day_month': 'звонков днем за месяц',
    'calls_evening_month': 'звонков вечером за месяц',
    'calls_night_month': 'звонков ночью за месяц',
    'calls_other_city': 'звонки в другие города',
    'calls_other_countries': 'звонки в другие страны',
    'calls_landline_phone': 'доля звонков на стационарные телефоны',
    'count_sms_month': 'количество sms за месяц',
    'date_connection_rate': 'дата подключения тарифа'
}
df['count_sms_month'] = df['count_sms_month'].replace("'12'", '12')
df['calls_landline_phone'] = df['calls_landline_phone'].replace("'2'", '2')
df['calls_other_city'] = df['calls_other_city'].replace("'29'", '29')
df['calls_other_city'] = df['calls_other_city'].replace("'0'", '0')
df['calls_night_month'] = df['calls_night_month'].replace("'7'", '7')
df['average_duration_calls'] = df['average_duration_calls'].fillna(0)
df['average_month_expense'] = df['average_month_expense'].fillna(0)
df['calls_evening_month'] = df['calls_evening_month'].fillna(0)
df['calls_day_month'] = df['calls_day_month'].fillna(0)
df = df.astype({'calls_day_month': 'int64', 'calls_evening_month': 'int64', 'calls_night_month': 'int64', 'calls_other_city': 'int64',
                'calls_landline_phone': 'int64', 'count_sms_month': 'int64', 'date_connection_rate': 'datetime64[ns]'})
df_not_country = df.drop(columns='calls_other_countries')


# Task 2, добавление новых переменных
def age(value):
    if value <= 24:
        return "student"
    elif value <= 33:
        return "aspirant"
    elif value <= 56:
        return "businessman"
    else:
        return "expert"


df['age_group'] = df.age.apply(age)
df['year_connect'] = df['date_connection_rate'].dt.year
df['month_connect'] = df['date_connection_rate'].dt.strftime('%Y-%m')
df['date_connection_rate_notime'] = df['date_connection_rate'].dt.date


# Task 3.1: динамика подключения к тарифам
df_count_client_month = df.groupby('month_connect').agg(count_client=('month_connect', 'count')).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_count_client_month, x='month_connect', y='count_client')
ax.set_xticklabels(df_count_client_month['month_connect'], rotation=90, fontsize=8)
plt.show()
# Вывод: динамика подключений сохраняется с каждым годом


# Task 3.2: исследование признаков по возрастной категории
# percentile = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99]

var_continuous = ['age', 'average_month_expense', 'average_duration_calls', 'calls_day_month', 'calls_evening_month', 'calls_night_month', 'calls_other_city',
                  'calls_landline_phone', 'count_sms_month']
var_continuous_two = ['average_month_expense', 'average_duration_calls', 'calls_day_month', 'calls_evening_month', 'calls_night_month', 'calls_other_city',
                      'calls_landline_phone', 'count_sms_month']

for i, col in enumerate(var_continuous_two):
    sns.histplot(df[df.age_group == 'student'][col], label='Студенты')
    plt.axvline(np.mean(df[df.age_group == 'student'][col]), color='black', label='среднее {}'.format(np.round(np.mean(df[df.age_group == 'student'][col]))))
    plt.axvline(np.median(df[df.age_group == 'student'][col]), color='orange', label='медиана {}'.format(np.round(np.median(df[df.age_group == 'student']
                                                                                                                            [col]))))
    plt.axvline(stat.mode(df[df.age_group == 'student'][col]), color='red', label='мода {}'.format(np.round(stat.mode(df[df.age_group == 'student'][col]))))
    plt.legend(loc='upper right')
    plt.title('Распределение<{}>'.format(dict_rus[col]))
    plt.ylabel('Колво абонентов', fontsize=12)
    plt.xlabel(dict_rus[col], fontsize=12)
    plt.show()
# Вывод по студентам в среднем за месяц: тратят в среднем 230р, продолжительность разговоров - 3 минуты, дневных звонков за месяц - 38, вечерних звонков
# за месяц - 69, ночных - 10, в другие города - 1, стационарные телефоны - 4, смс - 48

for i, col in enumerate(var_continuous_two):
    sns.histplot(df[df.age_group == 'aspirant'][col], label='Аспиранты')
    plt.axvline(np.mean(df[df.age_group == 'aspirant'][col]), color='black', label='среднее {}'.format(np.round(np.mean(df[df.age_group == 'aspirant'][col]))))
    plt.axvline(np.median(df[df.age_group == 'aspirant'][col]), color='orange', label='медиана {}'.format(np.round(np.median(df[df.age_group == 'aspirant']
                                                                                                                             [col]))))
    plt.axvline(stat.mode(df[df.age_group == 'aspirant'][col]), color='red', label='мода {}'.format(np.round(stat.mode(df[df.age_group == 'aspirant'][col]))))
    plt.legend(loc='upper right')
    plt.title('Распределение<{}>'.format(dict_rus[col]))
    plt.ylabel('Колво абонентов', fontsize=12)
    plt.xlabel(dict_rus[col], fontsize=12)
    plt.show()
# Вывод по аспирантам в среднем за месяц: тратят в среднем 696р, продолжительность разговоров - 5 минут, дневных звонков за месяц - 80, вечерних звонков
# за месяц - 86, ночных - 14, в другие города - 11, стационарные телефоны - 11, смс - 39

for i, col in enumerate(var_continuous_two):
    sns.histplot(df[df.age_group == 'businessman'][col], label='Бизнесмены')
    plt.axvline(np.mean(df[df.age_group == 'businessman'][col]), color='black', label='среднее {}'.format(np.round(np.mean(df[df.age_group == 'businessman']
                                                                                                                           [col]))))
    plt.axvline(np.median(df[df.age_group == 'businessman'][col]), color='orange', label='медиана {}'.format(np.round(np.median(df[df.age_group ==
                                                                                                                                   'businessman'][col]))))
    plt.axvline(stat.mode(df[df.age_group == 'businessman'][col]), color='red', label='мода {}'.format(np.round(stat.mode(df[df.age_group == 'businessman']
                                                                                                                          [col]))))
    plt.legend(loc='upper right')
    plt.title('Распределение<{}>'.format(dict_rus[col]))
    plt.ylabel('Колво абонентов', fontsize=12)
    plt.xlabel(dict_rus[col], fontsize=12)
    plt.show()
# Вывод по бизнесменам в среднем за месяц: тратят в среднем 512р, продолжительность разговоров - 4 минуты, дневных звонков за месяц - 65, вечерних звонков
# за месяц - 70, ночных - 4, в другие города - 10, стационарные телефоны - 11, смс - 15

for i, col in enumerate(var_continuous_two):
    sns.histplot(df[df.age_group == 'expert'][col], label='Знатоки')
    plt.axvline(np.mean(df[df.age_group == 'expert'][col]), color='black', label='среднее {}'.format(np.round(np.mean(df[df.age_group == 'expert'][col]))))
    plt.axvline(np.median(df[df.age_group == 'expert'][col]), color='orange', label='медиана {}'.format(np.round(np.median(df[df.age_group == 'expert'][col]))))
    plt.axvline(stat.mode(df[df.age_group == 'expert'][col]), color='red', label='мода {}'.format(np.round(stat.mode(df[df.age_group == 'expert'][col]))))
    plt.legend(loc='upper right')
    plt.title('Распределение<{}>'.format(dict_rus[col]))
    plt.ylabel('Колво абонентов', fontsize=12)
    plt.xlabel(dict_rus[col], fontsize=12)
    plt.show()
# Вывод по знатокам в среднем за месяц: тратят в среднем 440р, продолжительность разговоров - 4 минуты, дневных звонков за месяц - 56, вечерних звонков
# за месяц - 54, ночных - 2, в другие города - 7, стационарные телефоны - 11, смс - 3

# Общий вывод: больше всего тратят аспиранты, меньше всех студенты. Самая продолжительные средние разговоры у аспирантов, непродолжительные - у студентов.
# Днем чаще всех звонят аспиранты, меньше всех - студенты. Вечером чаще всех звонят аспиранты, меньше всех - знатоки. Ночью чаще всех звонят аспиранты,
# меньше всех - знатоки. В другие города чаще всех звонят аспиранты, меньше всех - студенты. На стационарные телефоны чаще всех звонят аспиранты, меньше всех -
# студенты. Смс чаще всех отправляют студенты, меньше всех - знатоки

# Task 3.3: топ-2
df['total_calls'] = df['calls_day_month'] + df['calls_evening_month'] + df['calls_night_month']
df['time_calls'] = df['total_calls'] * df['average_duration_calls']

df_count = df.groupby('age_group').agg(count_count=('average_month_expense', 'mean')).sort_values(by='count_count', ascending=False)
df_count_time = df.groupby('age_group').agg(count_time=('time_calls', 'count')).sort_values(by='count_time', ascending=False)
df_count_calls = df.groupby('age_group').agg(count_calls=('total_calls', 'count')).sort_values(by='count_calls', ascending=False)
print('Больше всего тратят на оплату связи', df_count, '\nБольше всех тратят на общение', df_count_time, '\nБольше всего звонков совершают:', df_count_calls)
# Вывод(ТОП-2): больше всех на оплату связи тратят аспиранты и бизнесмены. Больше всех тратят на общение и больше всех звонков совершают бизнесмены и аспиранты

# Task 3.4: диаграммы рассеивания
list_colors = ['blue', 'red', 'black', 'pink', 'orange', 'yellow', 'green', 'orange']

index = 0
for i, col in enumerate(var_continuous_two):
    set_columns = set(var_continuous_two).difference(set([col]))
    for c in set_columns:
        index += 1
        sns.scatterplot(data=df, x=c, y=col, color=list_colors[i])
        plt.show()

# Вывод: ключевыми параметрами на расходы услуг связи являются следующие критерии: средняя продолжительность разговоров и звонки днем за месяц. ТОП категориями
# по приобретению услуг являются аспиранты и бизнесмены
