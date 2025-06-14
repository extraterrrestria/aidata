from typing import List, Dict
import pandas as pd
from logging_logic import logging

def count_unique_values(data: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """
    Вычисляет количество уникальных значений для выбранных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> количество уникальных значений).
    """
    try:
        return {col: data[col].nunique() for col in columns}
    except Exception as e:
        logging.error(f"Ошибка при подсчете уникальных значений: {e}")
        return {}
    
def get_unique_values(data: pd.DataFrame, columns: List[str]) -> Dict[str, List]:
    """
    Возвращает список уникальных значений для выбранных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> список уникальных значений).
    """
    try:
        return {col: data[col].unique().tolist() for col in columns}
    except Exception as e:
        logging.error(f"Ошибка при получении списка уникальных значений: {e}")
        return {}
    
def count_missing_values(data: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """
    Вычисляет количество пропущенных значений для выбранных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> количество пропущенных значений).
    """
    try:
        return {col: data[col].isnull().sum() for col in columns}
    except Exception as e:
        logging.error(f"Ошибка при подсчете пропущенных значений: {e}")
        return {}

def calculate_category_percentages(data: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Вычисляет процентное соотношение уникальных значений для выбранных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> {значение: процент}).
    """
    try:
        result = {}
        for col in columns:
            total_count = len(data[col])
            value_counts = data[col].value_counts(normalize=True) * 100
            result[col] = value_counts.to_dict()
        return result
    except Exception as e:
        logging.error(f"Ошибка при вычислении процентного соотношения категорий: {e}")
        return {}
