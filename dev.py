import os
from logging_logic import logging
from csvloader import CSVLoader
from statistics_menu import statistics_menu
from unique_values_menu import unique_values_menu
from correlation_menu import correlation_menu
from grouping_menu import grouping_menu
from registry_menu import registry_menu
from csv_loader import save_to_csv
from data_generation_menu import data_generation_menu
from data_generation_functions import create_empty_dataframe

def main():
    """
    Основная функция программы.
    """
    loader = CSVLoader()

    # Шаг 1: Выбор режима работы
    while True:
        mode = input("Выберите режим работы (загрузка/генерация): ").strip().lower()
        if mode in ['загрузка', 'генерация']:
            break
        logging.warning(f"Некорректный выбор режима работы: {mode}")
        print("Некорректный выбор. Попробуйте снова.")

    is_generated = False  # Флаг, указывающий, были ли данные сгенерированы

    if mode == 'загрузка':
        # Шаг 2: Выбор файла
        if not loader.select_file():
            logging.error("Не удалось выбрать файл. Программа завершена.")
            return

        # Шаг 3: Настройка параметров чтения
        delimiter = input("Введите разделитель (по умолчанию ','): ") or ","
        encoding = input("Введите кодировку (по умолчанию 'utf-8'): ") or "utf-8"
        loader.set_reading_parameters(delimiter=delimiter, encoding=encoding)

        # Шаг 4: Настройка режима загрузки
        load_mode = input("Введите режим загрузки (all/first_5) (по умолчанию 'all'): ") or "all"
        loader.set_load_mode(load_mode)

        # Шаг 5: Загрузка данных
        if not loader.load_data():
            logging.error("Не удалось загрузить данные. Программа завершена.")
            return

        # Шаг 6: Получение и вывод исходных данных
        original_data = loader.get_original_data()
        if original_data is not None:
            logging.info("Исходные данные успешно загружены.")
            print(original_data)

        # Шаг 7: Получение списка столбцов
        column_names = loader.get_column_names()
        if column_names:
            logging.info(f"Список столбцов: {column_names}")
            print(column_names)
    else:
        # Генерация данных
        original_data = create_empty_dataframe()
        loader.set_active_data(original_data)
        is_generated = True  # Устанавливаем флаг, что данные сгенерированы

    # Главное меню после выбора столбцов
    while True:
        print("\nГлавное меню:")
        print("1. Статистические показатели")
        print("2. Уникальные значения")
        print("3. Корреляционный анализ")
        print("4. Группировка данных с агрегацией")
        print("5. Реестр записей")
        print("6. Генерация данных")
        print("7. Просмотр текущего состояния датафрейма")
        print("8. Сохранить активный датафрейм в CSV")
        print("9. Сбросить активный датафрейм к исходному состоянию")
        print("10. Завершить программу")

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
            active_data = loader.get_active_data()
            if active_data is not None:
                loader.set_active_data(data_generation_menu(active_data))
            else:
                logging.error("Активный датафрейм не найден.")
                print("Активный датафрейм не найден.")
        elif main_choice == '7':
            active_data = loader.get_active_data()
            if active_data is not None:
                print("\nТекущее состояние датафрейма:")
                print(active_data)
            else:
                logging.error("Активный датафрейм не найден.")
                print("Активный датафрейм не найден.")
        elif main_choice == '8':
            save_to_csv(loader, is_generated)
        elif main_choice == '9':
            loader.reset_active_data()
            logging.info("Активный датафрейм сброшен к исходному состоянию.")
            print("Активный датафрейм сброшен к исходному состоянию.")
        elif main_choice == '10':
            logging.info("Завершение программы.")
            print("Завершение программы.")
            break
        else:
            logging.warning(f"Некорректный выбор в главном меню: {main_choice}")
            print("Некорректный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()