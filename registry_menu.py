from csvloader import CSVLoader
from registry_functions import *
from logging_logic import logging

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