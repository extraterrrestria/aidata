from csvloader import CSVLoader
from statistics_functions import *
from logging_logic import logging

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