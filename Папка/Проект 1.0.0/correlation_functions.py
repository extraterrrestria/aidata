from typing import List, Dict
import pandas as pd
from logging_logic import logging
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_pearson_correlation(data: pd.DataFrame, columns: List[str]) -> None:
    """
    Вычисляет корреляцию Пирсона и строит тепловую карту.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    """
    try:
        # Фильтруем только числовые столбцы
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty or len(numeric_data.columns) < 2:
            logging.error("Для корреляции Пирсона требуется минимум 2 числовых столбца.")
            return

        # Вычисляем корреляцию
        correlation_matrix = numeric_data.corr(method='pearson')

        # Строим тепловую карту
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Тепловая карта корреляции Пирсона")
        plt.show()
    except Exception as e:
        logging.error(f"Ошибка при построении тепловой карты корреляции: {e}")

def calculate_skewness(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет асимметрию для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> асимметрия).
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для расчета асимметрии требуются числовые данные.")
            return {}

        return numeric_data.skew().to_dict()
    except Exception as e:
        logging.error(f"Ошибка при расчете асимметрии: {e}")
        return {}

def calculate_kurtosis(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет эксцесс для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> эксцесс).
    """
    try:
        numeric_data = data[columns].select_dtypes(include=['number'])
        if numeric_data.empty:
            logging.error("Для расчета эксцесса требуются числовые данные.")
            return {}

        return numeric_data.kurt().to_dict()
    except Exception as e:
        logging.error(f"Ошибка при расчете эксцесса: {e}")
        return {}