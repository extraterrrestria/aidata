from csvloader import CSVLoader
from grouping_functions import *
from logging_logic import logging

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