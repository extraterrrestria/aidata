import os
import logging
from csvloader import CSVLoader
import keyboard  # Для отслеживания нажатия клавиш
import pandas as pd
from tkinter import Tk, filedialog
from typing import Optional, List, Dict, Union
import seaborn as sns
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
    
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

def statistics_menu(loader: CSVLoader) -> None:
    """
    Меню для выбора и расчета статистических показателей.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены.")
        return

    column_names = loader.get_column_names()
    print("Доступные столбцы:")
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")

    selected_columns = input("Введите номера столбцов для анализа (через запятую): ").strip()
    if not selected_columns:
        logging.warning("Столбцы не выбраны. Возврат в главное меню.")
        return

    try:
        selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
        columns_list = [column_names[i] for i in selected_indices]
    except (ValueError, IndexError):
        logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")
        return

    while True:
        print("\nМеню статистических показателей:")
        print("1. Показать")
        print("2. Вернуться в главное меню")

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            result = calculate_mean(active_data, columns_list)
            print("1.Среднее арифметическое:", result)
            result = calculate_median(active_data, columns_list)
            print("2.Медиана:", result)
            result = calculate_mode(active_data, columns_list)
            print("3.Мода:", result)
            result = calculate_variance(active_data, columns_list)
            print("4.Дисперсия:", result)
            result = calculate_std_deviation(active_data, columns_list)
            print("5.Стандартное отклонение:", result)
            result = calculate_quantiles(active_data, columns_list)
            print("6.Квантили (25%, 50%, 75%):", result)
            result = calculate_value_counts(active_data, columns_list)
            print("7.Количество элементов каждого типа:", result)
        elif choice == '2':
            print("Возвращение в главное меню.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

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

def unique_values_menu(loader: CSVLoader) -> None:
    """
    Меню для работы с уникальными значениями.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены.")
        return

    column_names = loader.get_column_names()
    print("Доступные столбцы:")
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")

    selected_columns = input("Введите номера столбцов для анализа (через запятую): ").strip()
    if not selected_columns:
        logging.warning("Столбцы не выбраны. Возврат в главное меню.")
        return

    try:
        selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
        columns_list = [column_names[i] for i in selected_indices]
    except (ValueError, IndexError):
        logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")
        return

    while True:
        print("\nМеню уникальных значений:")
        print("1. Показать")
        print("2. Вернуться в главное меню")

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            result = count_unique_values(active_data, columns_list)
            print("1.Количество уникальных значений:", result)
            result = get_unique_values(active_data, columns_list)
            print("2.Список уникальных значений:", result)
            result = count_missing_values(active_data, columns_list)
            print("3.Количество пропущенных значений:", result)
            result = calculate_category_percentages(active_data, columns_list)
            print("4.Процентное соотношение категорий:", result)
        elif choice == '2':
            print("Возвращение в главное меню.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

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

def correlation_menu(loader: CSVLoader) -> None:
    """
    Меню для корреляционного анализа.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены.")
        return

    column_names = loader.get_column_names()
    print("Доступные столбцы:")
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")

    selected_columns = input("Введите номера столбцов для анализа (минимум 2, через запятую): ").strip()
    if not selected_columns:
        logging.warning("Столбцы не выбраны. Возврат в главное меню.")
        return

    try:
        selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
        columns_list = [column_names[i] for i in selected_indices]
        if len(columns_list) < 2:
            logging.error("Для корреляционного анализа требуется минимум 2 столбца.")
            return
    except (ValueError, IndexError):
        logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")
        return

    while True:
        print("\nМеню корреляционного анализа:")
        print("1. Показать")
        print("2. Вернуться в главное меню")

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            calculate_pearson_correlation(active_data, columns_list)
            result = calculate_skewness(active_data, columns_list)
            print("Асимметрия:", result)
            result = calculate_kurtosis(active_data, columns_list)
            print("Эксцесс:", result)
        elif choice == '2':
            print("Возвращение в главное меню.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

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

def grouping_menu(loader: CSVLoader) -> None:
    """
    Меню для группировки данных с агрегацией.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены.")
        return

    column_names = loader.get_column_names()
    print("Доступные столбцы:")
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")

    group_col = input("Введите номер столбца для группировки: ").strip()
    try:
        group_col_index = int(group_col.strip()) - 1
        group_col_name = column_names[group_col_index]
    except (ValueError, IndexError):
        logging.error("Некорректный ввод. Пожалуйста, введите номер столбца из списка.")
        return

    selected_columns = input("Введите номера столбцов для анализа (через запятую): ").strip()
    if not selected_columns:
        logging.warning("Столбцы не выбраны. Возврат в главное меню.")
        return

    try:
        selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
        columns_list = [column_names[i] for i in selected_indices]
    except (ValueError, IndexError):
        logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")
        return

    while True:
        print("\nМеню группировки данных с агрегацией:")
        print("1. Суммирование по группам")
        print("2. Среднее арифметическое по группам")
        print("3. Медиана по группам")
        print("4. Количество уникальных значений по группам")
        print("5. Процентное соотношение категорий по группам")
        print("6. Корреляция Пирсона по группам")
        print("7. Вернуться в главное меню")

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            group_sum(active_data, group_col_name, columns_list)
        elif choice == '2':
            group_mean(active_data, group_col_name, columns_list)
        elif choice == '3':
            group_median(active_data, group_col_name, columns_list)
        elif choice == '4':
            group_unique_count(active_data, group_col_name, columns_list)
        elif choice == '5':
            category_col = input("Введите номер столбца с категориями: ").strip()
            try:
                category_col_index = int(category_col.strip()) - 1
                category_col_name = column_names[category_col_index]
                group_category_percentage(active_data, group_col_name, category_col_name)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номер столбца из списка.")
        elif choice == '6':
            group_pearson_correlation(active_data, group_col_name, columns_list)
        elif choice == '7':
            print("Возвращение в главное меню.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

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

def registry_menu(loader: CSVLoader) -> None:
    """
    Меню для работы с реестром записей.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены.")
        return

    column_names = loader.get_column_names()
    print("Доступные столбцы:")
    for i, col in enumerate(column_names, 1):
        print(f"{i}. {col}")

    while True:
        print("\nМеню реестра записей:")
        print("1. Удаление строк по условию")
        print("2. Модификация строк по условию")
        print("3. Заполнение пропусков")
        print("4. Нормализация данных")
        print("5. Стандартизация данных")
        print("6. Сортировка записей по условию")
        print("7. One Hot Encoding")
        print("8. Удаление выбросов методом IQR")
        print("9. Вернуться в главное меню")

        choice = input("Выберите действие: ").strip()

        if choice == '1':
            condition = input("Введите условие (например, 'Column1 > 10'): ").strip()
            active_data = remove_rows_by_condition(active_data, condition)
            print("Измененный датафрейм после удаления строк:\n", active_data)

        elif choice == '2':
            condition = input("Введите условие (например, 'Column1 > 10'): ").strip()
            column = input("Введите столбец для модификации: ").strip()
            value = input("Введите новое значение: ").strip()
            active_data = modify_rows_by_condition(active_data, condition, column, value)
            print("Измененный датафрейм после модификации строк:\n", active_data)

        elif choice == '3':
            selected_columns = input("Введите номера столбцов для заполнения (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                method = input("Введите метод заполнения ('mean', 'median', 'mode' или значение): ").strip()
                active_data = fill_missing_values(active_data, columns_list, method)
                print("Измененный датафрейм после заполнения пропусков:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '4':
            selected_columns = input("Введите номера столбцов для нормализации (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                min_val = float(input("Введите минимальное значение для нормализации: ").strip())
                max_val = float(input("Введите максимальное значение для нормализации: ").strip())
                active_data = normalize_data(active_data, columns_list, min_val, max_val)
                print("Измененный датафрейм после нормализации:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '5':
            selected_columns = input("Введите номера столбцов для стандартизации (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                active_data = standardize_data(active_data, columns_list)
                print("Измененный датафрейм после стандартизации:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '6':
            selected_columns = input("Введите номера столбцов для сортировки (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                ascending = input("Сортировать по возрастанию? (y/n): ").strip().lower() == 'y'
                active_data = sort_data(active_data, columns_list, ascending)
                print("Измененный датафрейм после сортировки:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '7':
            selected_columns = input("Введите номера столбцов для One Hot Encoding (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                active_data = one_hot_encode(active_data, columns_list)
                print("Измененный датафрейм после One Hot Encoding:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '8':
            selected_columns = input("Введите номера столбцов для удаления выбросов (через запятую): ").strip()
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_columns.split(',')]
                columns_list = [column_names[i] for i in selected_indices]
                active_data = remove_outliers_iqr(active_data, columns_list)
                print("Измененный датафрейм после удаления выбросов:\n", active_data)
            except (ValueError, IndexError):
                logging.error("Некорректный ввод. Пожалуйста, введите номера столбцов из списка.")

        elif choice == '9':
            print("Возвращение в главное меню.")
            break

        else:
            print("Некорректный выбор. Попробуйте снова.")

        loader.active_data = active_data

def main():
    """
    Основная функция программы.
    """
    loader = CSVLoader()

    # Шаг 1: Выбор файла
    if not loader.select_file():
        logging.error("Не удалось выбрать файл. Программа завершена.")
        return

    # Шаг 2: Настройка параметров чтения
    delimiter = input("Введите разделитель (по умолчанию ','): ") or ","
    encoding = input("Введите кодировку (по умолчанию 'utf-8'): ") or "utf-8"
    loader.set_reading_parameters(delimiter=delimiter, encoding=encoding)

    # Шаг 3: Настройка режима загрузки
    load_mode = input("Введите режим загрузки (all/first_5) (по умолчанию 'all'): ") or "all"
    loader.set_load_mode(load_mode)

    # Шаг 4: Загрузка данных
    if not loader.load_data():
        logging.error("Не удалось загрузить данные. Программа завершена.")
        return

    # Шаг 5: Получение и вывод исходных данных
    original_data = loader.get_original_data()
    if original_data is not None:
        logging.info("Исходные данные:")
        print(original_data)

    # Шаг 6: Получение списка столбцов
    column_names = loader.get_column_names()
    if column_names:
        logging.info("Список столбцов:")
        print(column_names)

    # Главное меню после выбора столбцов
    while True:
        print("\nГлавное меню:")
        print("1. Статистические показатели")
        print("2. Уникальные значения")
        print("3. Корреляционный анализ")
        print("4. Группировка данных с агрегацией")
        print("5. Реестр записей")
        print("6. Сбросить активный датафрейм к исходному состоянию")
        print("7. Завершить программу")

        main_choice = input("Выберите действие: ").strip()

        if main_choice == '1':
            statistics_menu(loader)
        elif main_choice == '2':
            unique_values_menu(loader)
        elif main_choice == '3':
            correlation_menu(loader)
        elif main_choice == '4':
            grouping_menu(loader)
        elif main_choice == '5':
            registry_menu(loader)
        elif main_choice == '6':
            loader.reset_active_data()
            logging.info("Активный датафрейм сброшен к исходному состоянию.")
        elif main_choice == '7':
            print("Завершение программы.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()