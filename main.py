import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 1) Какие года были самыми прибыльными
# 2) Какие года были самыми неприбыльными
# 3) Какие фильмы самые прибыльные
# 4) Какие фильмы самые неприбыльные
# 5) Какой режиссер самый успешный (по сборам)
# 6) Какой режиссер самый успешный (по оценкам)
# 7) Какие фильмы самый высокооцененные
# 8) Какие фильмы самый низкооцененные
# 9) Какие общие признаки у самых прибыльных фильмов
# 10) Лучшие фильмы в жанре
# 11) Худшие фильмы в жанре


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 3)

original_df = pd.read_csv('dataset/tmdb_movies_data.csv', index_col='id', on_bad_lines='skip')
print(original_df.columns)

desired_columns_df: pd.DataFrame = original_df.iloc[:, [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17]]
desired_columns_df = desired_columns_df.query('revenue > 0')


movie_by_director = desired_columns_df.groupby('director', as_index=False).agg({'original_title': 'count'})\
    .sort_values('original_title', ascending=False)[:5]
print(desired_columns_df.columns)


movie_by_director_bars = plt.barh(movie_by_director['director'], movie_by_director['original_title'], 0.5)
plt.bar_label(movie_by_director_bars, )
plt.title('У какого режиссера самое большое количество фильмов')
plt.show()


revenue_by_year = desired_columns_df.groupby('release_year', as_index=False).agg({'original_title': 'count', 'revenue': 'sum'})\
    .sort_values('revenue', ascending=False)[:5]


revenue_by_year_max = desired_columns_df.groupby('release_year', as_index=False).agg({'revenue': 'sum'})\
                                                            .sort_values('revenue', ascending=False)[:5]
revenue_by_year_min = desired_columns_df.groupby('release_year', as_index=False).agg({'revenue': 'sum'})\
                                                            .sort_values('revenue', ascending=True)[:5]


revenue_by_year_max_bars = plt.barh(revenue_by_year_max['release_year'], revenue_by_year_max['revenue'], 0.5)
plt.bar_label(revenue_by_year_max_bars)
plt.title('Топ прибыльных годов')
plt.show()


years_ticks = [str(year) for year in revenue_by_year_min['release_year']]
revenue_by_year_min_bars = plt.barh(years_ticks, revenue_by_year_min['revenue'], 0.5)
plt.bar_label(revenue_by_year_min_bars)
plt.title('Топ неприбыльных годов')
plt.show()


revenue_by_title_max = desired_columns_df.groupby('original_title', as_index=False).agg({'revenue': 'sum'})\
                                                            .sort_values('revenue', ascending=False)[:5]
print(revenue_by_title_max)


revenue_by_title_min = desired_columns_df.groupby('original_title', as_index=False).agg({'revenue': 'sum'})\
                                                            .sort_values('revenue', ascending=True)[:5]
print(revenue_by_title_min)


revenue_by_title_max_bars = plt.barh(revenue_by_title_max['original_title'], revenue_by_title_max['revenue'], 0.5)
plt.bar_label(revenue_by_title_max_bars)
plt.title('Топ лучших фильмов по сборам')
plt.show()


revenue_by_title_min_bars = plt.barh(revenue_by_title_min['original_title'], revenue_by_title_min['revenue'], 0.5)
plt.bar_label(revenue_by_title_min_bars)
plt.title('Топ худших фильмов по сборам')
plt.show()


x, y = desired_columns_df['budget'], desired_columns_df['revenue']
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3)
plt.xlabel('Бюджет')
plt.ylabel('Cборы')
plt.show()


X = pd.DataFrame(x)
Y = pd.DataFrame(y)
linear_reg_revenue = LinearRegression()
linear_reg_revenue.fit(X, Y)
print(linear_reg_revenue.coef_)
print(linear_reg_revenue.score(X, Y))


x, y = desired_columns_df['budget'], desired_columns_df['revenue']
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3)
plt.plot(x, linear_reg_revenue.predict(X), color='blue')
plt.xlabel('Бюджет')
plt.ylabel('Cборы')
plt.show()

