from typing import List, Union
import pandas as pd
from logging_logic import logging

def remove_rows_by_condition(data: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Удаляет строки из DataFrame, удовлетворяющие заданному условию.
    :param data: DataFrame с данными.
    :param condition: Строка условия (например, 'Column1 > 10').
    :return: Обновленный DataFrame.
    """
    try:
        filtered_data = data.query(condition)
        if filtered_data.empty:
            logging.warning("Нет строк, удовлетворяющих условию.")
            return data
        return data.drop(filtered_data.index)
    except Exception as e:
        logging.error(f"Ошибка при удалении строк по условию: {e}")
        return data
    
def modify_rows_by_condition(data: pd.DataFrame, condition: str, column: str, value: Union[str, int, float]) -> pd.DataFrame:
    """
    Модифицирует значения в строках, удовлетворяющих заданному условию.
    :param data: DataFrame с данными.
    :param condition: Строка условия (например, 'Column1 > 10').
    :param column: Столбец для модификации.
    :param value: Новое значение для заполнения.
    :return: Обновленный DataFrame.
    """
    try:
        data.loc[data.eval(condition), column] = value
        logging.info(f"Строки, удовлетворяющие условию '{condition}', были модифицированы.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при модификации строк: {e}")
        return data
    
def fill_missing_values(data: pd.DataFrame, columns: List[str], method: str = "mean") -> pd.DataFrame:
    """
    Заполняет пропущенные значения в выбранных столбцах.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для заполнения.
    :param method: Метод заполнения ('mean', 'median', 'mode', или конкретное значение).
    :return: Обновленный DataFrame.
    """
    try:
        for col in columns:
            if method == "mean":
                data[col].fillna(data[col].mean(), inplace=True)
            elif method == "median":
                data[col].fillna(data[col].median(), inplace=True)
            elif method == "mode":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(method, inplace=True)
        logging.info(f"Пропущенные значения в столбцах {columns} были заполнены методом '{method}'.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при заполнении пропусков: {e}")
        return data
    
def normalize_data(data: pd.DataFrame, columns: List[str], min_val: float, max_val: float) -> pd.DataFrame:
    """
    Нормализует данные в выбранных столбцах в заданный диапазон.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для нормализации.
    :param min_val: Минимальное значение для нормализации.
    :param max_val: Максимальное значение для нормализации.
    :return: Обновленный DataFrame.
    """
    try:
        for col in columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min()) * (max_val - min_val) + min_val
        logging.info(f"Данные в столбцах {columns} были нормализованы в диапазон [{min_val}, {max_val}].")
        return data
    except Exception as e:
        logging.error(f"Ошибка при нормализации данных: {e}")
        return data
    
def standardize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Стандартизирует данные в выбранных столбцах (приведение к z-оценке).
    :param data: DataFrame с данными.
    :param columns: Список столбцов для стандартизации.
    :return: Обновленный DataFrame.
    """
    try:
        for col in columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        logging.info(f"Данные в столбцах {columns} были стандартизированы.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при стандартизации данных: {e}")
        return data
    
def sort_data(data: pd.DataFrame, columns: List[str], ascending: bool = True) -> pd.DataFrame:
    """
    Сортирует записи в DataFrame по выбранным столбцам.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для сортировки.
    :param ascending: Порядок сортировки (True для возрастания, False для убывания).
    :return: Обновленный DataFrame.
    """
    try:
        sorted_data = data.sort_values(by=columns, ascending=ascending)
        logging.info(f"Данные отсортированы по столбцам {columns}.")
        return sorted_data
    except Exception as e:
        logging.error(f"Ошибка при сортировке данных: {e}")
        return data
    
def one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Выполняет One Hot Encoding для выбранных категориальных столбцов.
    :param data: DataFrame с данными.
    :param columns: Список столбцов для кодирования.
    :return: Обновленный DataFrame.
    """
    try:
        encoded_data = pd.get_dummies(data, columns=columns)
        logging.info(f"One Hot Encoding выполнен для столбцов {columns}.")
        return encoded_data
    except Exception as e:
        logging.error(f"Ошибка при выполнении One Hot Encoding: {e}")
        return data
    
def remove_outliers_iqr(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Удаляет выбросы из данных методом межквартильного размаха (IQR).
    :param data: DataFrame с данными.
    :param columns: Список столбцов для обработки.
    :return: Обновленный DataFrame.
    """
    try:
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        logging.info(f"Выбросы в столбцах {columns} были удалены методом IQR.")
        return data
    except Exception as e:
        logging.error(f"Ошибка при удалении выбросов: {e}")
        return data


def remove_rows_with_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строки, содержащие пропущенные значения.
    :param data: DataFrame с данными.
    :return: Обновленный DataFrame без строк с пропусками.
    """
    try:
        cleaned_data = data.dropna()
        logging.info("Строки с пропущенными значениями успешно удалены.")
        return cleaned_data
    except Exception as e:
        logging.error(f"Ошибка при удалении строк с пропусками: {e}")
        return data