from typing import List
import pandas as pd
from logging_logic import logging
import matplotlib.pyplot as plt
import seaborn as sns

def group_sum(data: pd.DataFrame, group_col: str, columns: List[str]) -> None:
    """
    Вычисляет сумму значений по группам и строит столбчатую диаграмму.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param columns: Список столбцов для агрегации.
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для суммирования требуются числовые данные.")
            return

        grouped_data = data.groupby(group_col)[columns].sum().reset_index()
        print("Сумма по группам:\n", grouped_data)

        # Построение столбчатой диаграммы
        grouped_data.set_index(group_col).plot(kind='bar', figsize=(10, 6))
        plt.title("Сумма по группам")
        plt.ylabel("Сумма")
        plt.show()
    except Exception as e:
        logging.error(f"Ошибка при суммировании по группам: {e}")

def group_mean(data: pd.DataFrame, group_col: str, columns: List[str]) -> None:
    """
    Вычисляет среднее арифметическое по группам и строит boxplot.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param columns: Список столбцов для агрегации.
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для расчета среднего требуются числовые данные.")
            return

        grouped_data = data.groupby(group_col)[columns].mean().reset_index()
        print("Среднее по группам:\n", grouped_data)

        # Построение boxplot
        for col in columns:
            sns.boxplot(x=group_col, y=col, data=data)
            plt.title(f"Boxplot для {col}")
            plt.show()
    except Exception as e:
        logging.error(f"Ошибка при расчете среднего по группам: {e}")

def group_median(data: pd.DataFrame, group_col: str, columns: List[str]) -> None:
    """
    Вычисляет медиану по группам и строит столбчатую диаграмму.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param columns: Список столбцов для агрегации.
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для расчета медианы требуются числовые данные.")
            return

        grouped_data = data.groupby(group_col)[columns].median().reset_index()
        print("Медиана по группам:\n", grouped_data)

        # Построение столбчатой диаграммы
        grouped_data.set_index(group_col).plot(kind='bar', figsize=(10, 6))
        plt.title("Медиана по группам")
        plt.ylabel("Медиана")
        plt.show()
    except Exception as e:
        logging.error(f"Ошибка при расчете медианы по группам: {e}")

def group_unique_count(data: pd.DataFrame, group_col: str, columns: List[str]) -> None:
    """
    Вычисляет количество уникальных значений по группам.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param columns: Список столбцов для агрегации.
    """
    try:
        grouped_data = data.groupby(group_col)[columns].nunique().reset_index()
        print("Количество уникальных значений по группам:\n", grouped_data)
    except Exception as e:
        logging.error(f"Ошибка при подсчете уникальных значений по группам: {e}")

def group_category_percentage(data: pd.DataFrame, group_col: str, category_col: str) -> None:
    """
    Вычисляет процентное соотношение категорий по группам и строит сгруппированную столбчатую диаграмму.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param category_col: Столбец с категориями.
    """
    try:
        grouped = data.groupby([group_col, category_col]).size().reset_index(name='count')
        grouped['percent'] = grouped.groupby(group_col)['count'].transform(lambda x: 100 * x / x.sum())
        print("Процентное соотношение категорий по группам:\n", grouped)

        # Построение сгруппированной столбчатой диаграммы
        sns.barplot(x=group_col, y='percent', hue=category_col, data=grouped)
        plt.title("Процентное соотношение категорий по группам")
        plt.ylabel("Процент")
        plt.show()
    except Exception as e:
        logging.error(f"Ошибка при расчете процентного соотношения категорий: {e}")

def group_pearson_correlation(data: pd.DataFrame, group_col: str, columns: List[str]) -> None:
    """
    Вычисляет корреляцию Пирсона для каждой группы и строит тепловую карту.
    :param data: DataFrame с данными.
    :param group_col: Столбец для группировки.
    :param columns: Список столбцов для агрегации.
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для корреляции Пирсона требуются числовые данные.")
            return

        for group_name, group_data in data.groupby(group_col):
            correlation_matrix = group_data[columns].corr(method='pearson')
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Тепловая карта корреляции Пирсона для группы {group_name}")
            plt.show()
    except Exception as e:
        logging.error(f"Ошибка при расчете корреляции Пирсона по группам: {e}")