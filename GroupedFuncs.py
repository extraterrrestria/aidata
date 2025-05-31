import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sum_by_group(df, group_col, cols, plot=True):
    """
    Суммирование по группам.
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cols: список столбцов для суммирования
    :param plot: если True, строит столбчатую диаграмму
    :return: DataFrame с суммами по группам
    """
    grouped = df.groupby(group_col)[cols].sum()
    if plot:
        grouped.plot(kind='bar')
        plt.title(f'Суммирование по группам по столбцу {group_col}')
        plt.ylabel('Сумма')
        plt.xlabel(group_col)
        plt.tight_layout()
        plt.show()
    return grouped

def mean_by_group(df, group_col, cols, plot=True):
    """
    Среднее арифметическое по группам (только числовые столбцы).
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cols: список столбцов для вычисления среднего
    :param plot: если True, строит boxplot
    :return: DataFrame со средними по группам
    """
    # Оставляем только числовые столбцы из cols
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(group_col)[numeric_cols].mean()
    if plot:
        # Для boxplot нужно подготовить данные в "длинном" формате
        melted = df[[group_col] + numeric_cols].melt(id_vars=group_col, value_vars=numeric_cols)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_col, y='value', hue='variable', data=melted)
        plt.title(f'Boxplot средних значений по группам ({group_col})')
        plt.tight_layout()
        plt.show()
    return grouped

def median_by_group(df, group_col, cols, plot=True):
    """
    Медиана по группам.
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cols: список столбцов для вычисления медианы
    :param plot: если True, строит столбчатую диаграмму
    :return: DataFrame с медианами по группам
    """
    grouped = df.groupby(group_col)[cols].median()
    if plot:
        grouped.plot(kind='bar')
        plt.title(f'Медиана по группам по столбцу {group_col}')
        plt.ylabel('Медиана')
        plt.xlabel(group_col)
        plt.tight_layout()
        plt.show()
    return grouped

def unique_count_by_group(df, group_col, cols):
    """
    Количество уникальных значений по группам.
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cols: список столбцов для подсчёта уникальных значений
    :return: DataFrame с количеством уникальных значений по группам
    """
    grouped = df.groupby(group_col)[cols].nunique()
    # Выводим результат в консоль (не график)
    print(f'Количество уникальных значений по группам ({group_col}):')
    print(grouped)
    return grouped

def category_percentage_by_group(df, group_col, cat_col, plot=True):
    """
    Процентное соотношение категорий по группам.
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cat_col: столбец с категориальными данными для подсчёта процентов
    :param plot: если True, строит сгруппированную столбчатую диаграмму
    :return: DataFrame с процентами по категориям и группам
    """
    count = df.groupby([group_col, cat_col]).size().unstack(fill_value=0)
    percent = count.div(count.sum(axis=1), axis=0) * 100
    if plot:
        percent.plot(kind='bar', stacked=False)
        plt.title(f'Процентное соотношение категорий {cat_col} по группам {group_col}')
        plt.ylabel('Процент (%)')
        plt.xlabel(group_col)
        plt.legend(title=cat_col)
        plt.tight_layout()
        plt.show()
    return percent

def pearson_correlation_by_group(df, group_col, cols, plot=True):
    """
    Корреляция Пирсона по группам.
    :param df: исходный DataFrame
    :param group_col: имя столбца для группировки
    :param cols: список числовых столбцов для корреляции
    :param plot: если True, строит тепловую карту корреляций для каждой группы
    :return: словарь {группа: корреляционная матрица}
    """
    groups = df[group_col].unique()
    corr_dict = {}
    for g in groups:
        sub_df = df[df[group_col] == g][cols]
        corr = sub_df.corr(method='pearson')
        corr_dict[g] = corr
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Корреляция Пирсона для группы {g}')
            plt.tight_layout()
            plt.show()
    return corr_dict
