import os
from logging_logic import logging
from csvloader import CSVLoader
from statistics_menu import statistics_menu
from unique_values_menu import unique_values_menu
from correlation_menu import correlation_menu
from grouping_menu import grouping_menu
from registry_menu import registry_menu
from csv_loader import save_to_csv

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
        print("6. Сохранить активный датафрейм в CSV")
        print("7. Сбросить активный датафрейм к исходному состоянию")
        print("8. Завершить программу")

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
            save_to_csv(loader)
        elif main_choice == '7':
            loader.reset_active_data()
            logging.info("Активный датафрейм сброшен к исходному состоянию.")
        elif main_choice == '8':
            print("Завершение программы.")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()