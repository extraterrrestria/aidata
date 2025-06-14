import os
from data_generation_functions import *
from logging_logic import logging
import numpy as np

def data_generation_menu(df):
    """Меню для генерации данных."""
    while True:
        print("\nМеню генерации данных:")
        print("1. Генерация данных для столбца")
        print("2. Удаление столбца по номеру")
        print("3. Добавление нового столбца")
        print("4. Переименование столбца")
        print("5. Очистка столбца (заполнение значений NaN)")
        print("6. Вернуться в главное меню")
        choice = input("Выберите действие: ").strip()

        if choice == '1':
            df = generate_column(df)
        elif choice == '2':
            df = delete_column(df)
        elif choice == '3':
            df = add_column(df)
        elif choice == '4':
            df = rename_column(df)
        elif choice == '5':
            while True:
                column_index = input(f"Введите номер столбца для очистки (0-{len(df.columns) - 1}): ")
                if not column_index.isdigit() or int(column_index) < 0 or int(column_index) >= len(df.columns):
                    logging.error(f"Некорректный номер столбца: {column_index}")
                    print("Некорректный номер столбца. Попробуйте снова.")
                    continue
                column_index = int(column_index)
                break
            df[df.columns[column_index]] = np.nan
            logging.info(f"Столбец {column_index} очищен.")
            print(f"Столбец {column_index} очищен.")
        elif choice == '6':
            logging.info("Возврат в главное меню.")
            print("Возврат в главное меню.")
            break
        else:
            logging.warning(f"Некорректный выбор в меню генерации данных: {choice}")
            print("Некорректный выбор. Попробуйте снова.")
    
    return df