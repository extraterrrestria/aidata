import os
import logging
from csvloader import CSVLoader
import keyboard  # Для отслеживания нажатия клавиш

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

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

    # Шаг 7: Бесконечный цикл для выбора столбцов
    while True:
        selected_columns = input(f"Введите столбцы для выборки (через запятую, например: Column1,Column2) или нажмите 'Escape' для выхода: ").strip()
        
        if keyboard.is_pressed('esc'):
            logging.info("Выход из цикла выбора столбцов.")
            break
        
        if selected_columns:
            columns_list = [col.strip() for col in selected_columns.split(',')]
            if loader.select_columns(columns_list):
                # Вывод активных данных после выбора столбцов
                active_data = loader.get_active_data()
                if active_data is not None:
                    logging.info("Активные данные:")
                    print(active_data)

                # Вывод данных для отображения в таблице интерфейса
                data_for_display = loader.get_data_for_display()
                if data_for_display:
                    logging.info("Данные для отображения в таблице интерфейса:")
                    for row in data_for_display:
                        print(row)

    # Шаг 8: Сброс активного датафрейма к исходному состоянию
    reset = input("Сбросить активный датафрейм к исходному состоянию? (y/n): ").strip().lower()
    if reset == 'y':
        loader.reset_active_data()
        logging.info("Активный датафрейм сброшен к исходному состоянию.")
        active_data = loader.get_active_data()
        if active_data is not None:
            logging.info("Активные данные после сброса:")
            print(active_data)

        # Шаг 9: Получение данных для отображения в таблице интерфейса после сброса
        data_for_display = loader.get_data_for_display()
        if data_for_display:
            logging.info("Данные для отображения в таблице интерфейса после сброса:")
            for row in data_for_display:
                print(row)


if __name__ == "__main__":
    main()