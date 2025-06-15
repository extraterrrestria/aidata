from csvloader import CSVLoader
from correlation_functions import *
from logging_logic import logging

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