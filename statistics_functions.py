from typing import List, Dict, Union
import pandas as pd
from logging_logic import logging

def calculate_mean(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет среднее арифметическое для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> среднее значение).
    """
    try:
        return {col: data[col].mean() for col in columns if pd.api.types.is_numeric_dtype(data[col])}
    except Exception as e:
        logging.error(f"Ошибка при вычислении среднего: {e}")
        return {}

def calculate_median(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет медиану для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> медиана).
    """
    try:
        return {col: data[col].median() for col in columns if pd.api.types.is_numeric_dtype(data[col])}
    except Exception as e:
        logging.error(f"Ошибка при вычислении медианы: {e}")
        return {}

def calculate_mode(data: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Union[List, int]]]:
    """
    Вычисляет моду для выбранных столбцов (любого типа данных).
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> {мода: [значения], количество: число}).
    """
    try:
        result = {}
        for col in columns:
            mode_values = data[col].mode().tolist()
            count = data[col].value_counts().iloc[0] if not data[col].empty else 0
            result[col] = {"mode": mode_values, "count": count}
        return result
    except Exception as e:
        logging.error(f"Ошибка при вычислении моды: {e}")
        return {}

def calculate_variance(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет дисперсию для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> дисперсия).
    """
    try:
        return {col: data[col].var() for col in columns if pd.api.types.is_numeric_dtype(data[col])}
    except Exception as e:
        logging.error(f"Ошибка при вычислении дисперсии: {e}")
        return {}

def calculate_std_deviation(data: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Вычисляет стандартное отклонение для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> стандартное отклонение).
    """
    try:
        return {col: data[col].std() for col in columns if pd.api.types.is_numeric_dtype(data[col])}
    except Exception as e:
        logging.error(f"Ошибка при вычислении стандартного отклонения: {e}")
        return {}

def calculate_quantiles(data: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Вычисляет квантили (25%, 50%, 75%) для выбранных числовых столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> {q25, q50, q75}).
    """
    try:
        return {col: {"q25": data[col].quantile(0.25),
                      "q50": data[col].quantile(0.50),
                      "q75": data[col].quantile(0.75)}
                for col in columns if pd.api.types.is_numeric_dtype(data[col])}
    except Exception as e:
        logging.error(f"Ошибка при вычислении квантилей: {e}")
        return {}

def calculate_value_counts(data: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Вычисляет количество элементов каждого типа для выбранных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для вычисления.
    :return: Словарь с результатами (название столбца -> {значение: количество}).
    """
    try:
        return {col: data[col].value_counts().to_dict() for col in columns}
    except Exception as e:
        logging.error(f"Ошибка при подсчете количества элементов: {e}")
        return {}