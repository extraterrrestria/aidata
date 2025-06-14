from csvloader import CSVLoader
from unique_values_functions import *
from logging_logic import logging

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