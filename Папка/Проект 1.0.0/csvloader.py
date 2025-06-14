import os
import logging
import pandas as pd
from tkinter import Tk, filedialog
from typing import Optional, List, Dict

class CSVLoader:
    """
    Класс для загрузки и валидации CSV-файла.
    """

    def __init__(self):
        self.file_path: Optional[str] = None
        self.delimiter: Optional[str] = None
        self.encoding: Optional[str] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.active_data: Optional[pd.DataFrame] = None
        self.load_mode: str = 'all'  # По умолчанию загружаем всю таблицу

    def select_file(self) -> bool:
        """
        Открывает системное диалоговое окно для выбора файла.
        Возвращает True, если файл выбран успешно.
        """
        root = Tk()
        root.withdraw()  # Скрываем основное окно tkinter
        file_path = filedialog.askopenfilename(
            title="Выберите CSV-файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            logging.error("Файл не выбран.")
            return False

        self.file_path = file_path
        logging.info(f"Выбран файл: {self.file_path}")
        return True

    def set_reading_parameters(self, delimiter: str = ",", encoding: str = "utf-8") -> None:
        """
        Устанавливает параметры чтения CSV-файла (разделитель и кодировку).
        """
        self.delimiter = delimiter
        self.encoding = encoding
        logging.info(f"Установлены параметры чтения: разделитель='{delimiter}', кодировка='{encoding}'")

    def set_load_mode(self, mode: str = 'all') -> None:
        """
        Устанавливает режим загрузки данных ('all' или 'first_5').
        """
        if mode not in ['all', 'first_5']:
            logging.error(f"Неверный режим загрузки '{mode}'. Допустимые значения: 'all', 'first_5'.")
            return

        self.load_mode = mode
        logging.info(f"Установлен режим загрузки: {mode}")

    def validate_structure(self) -> bool:
        """
        Проверяет структуру CSV-файла на соответствие формату.
        Возвращает True, если файл прошел валидацию.
        """
        if not self.file_path:
            logging.error("Файл не выбран. Выберите файл перед валидацией.")
            return False

        try:
            # Чтение файла построчно для проверки структуры
            with open(self.file_path, mode="r", encoding=self.encoding, errors='strict') as file:
                lines = file.readlines()

            # Определяем количество столбцов в первой строке
            first_line_columns = len(lines[0].strip().split(self.delimiter))

            # Проверяем каждую строку на соответствие количеству столбцов
            for i, line in enumerate(lines):
                columns = line.strip().split(self.delimiter)
                if len(columns) != first_line_columns:
                    logging.error(f"Ошибка в строке {i + 1}: количество столбцов ({len(columns)}) "
                                  f"не соответствует первой строке ({first_line_columns}).")
                    return False

            logging.info("Структура файла валидна.")
            return True

        except UnicodeDecodeError:
            logging.error(f"Ошибка декодирования файла. Возможно, неверная кодировка '{self.encoding}'.")
            return False
        except Exception as e:
            logging.error(f"Произошла ошибка при валидации файла: {e}")
            return False

    def load_data(self) -> bool:
        """
        Загружает данные из CSV-файла в pandas.DataFrame.
        Возвращает True, если данные загружены успешно.
        """
        if not self.validate_structure():
            logging.error("Загрузка данных невозможна из-за ошибок валидации.")
            return False

        try:
            if self.load_mode == 'first_5':
                self.original_data = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding, engine='python', nrows=5)
            else:
                self.original_data = pd.read_csv(self.file_path, delimiter=self.delimiter, encoding=self.encoding, engine='python')
            self.active_data = self.original_data.copy()
            logging.info("Данные успешно загружены в DataFrame.")
            return True
        except UnicodeDecodeError:
            logging.error(f"Ошибка декодирования файла. Возможно, неверная кодировка '{self.encoding}'.")
            return False
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных: {e}")
            return False

    def get_original_data(self) -> Optional[pd.DataFrame]:
        """
        Возвращает исходные данные в виде pandas.DataFrame.
        """
        if self.original_data is not None:
            return self.original_data
        logging.error("Исходные данные не загружены. Выполните загрузку перед вызовом этого метода.")
        return None

    def get_active_data(self) -> Optional[pd.DataFrame]:
        """
        Возвращает активные данные в виде pandas.DataFrame.
        """
        if self.active_data is not None:
            return self.active_data
        logging.error("Активные данные не загружены. Выполните загрузку перед вызовом этого метода.")
        return None

    def get_column_names(self) -> List[str]:
        """
        Возвращает список всех столбцов исходного датафрейма.
        """
        if self.original_data is not None:
            return self.original_data.columns.tolist()
        logging.error("Данные не загружены. Выполните загрузку перед вызовом этого метода.")
        return []

    def select_columns(self, columns: List[str]) -> bool:
        """
        Выбирает столбцы для активного датафрейма.
        Возвращает True, если выбор столбцов выполнен успешно.
        """
        if self.original_data is None:
            logging.error("Исходные данные не загружены. Выполните загрузку перед вызовом этого метода.")
            return False
        if not all(col in self.original_data.columns for col in columns):
            logging.error(f"Некоторые из указанных столбцов отсутствуют в данных: {columns}")
            return False
        self.active_data = self.original_data[columns]
        logging.info(f"Выбраны столбцы: {columns}")
        return True

    def reset_active_data(self) -> None:
        """
        Сбрасывает активный датафрейм к исходному состоянию.
        """
        if self.original_data is not None:
            self.active_data = self.original_data.copy()
            logging.info("Активный датафрейм сброшен к исходному состоянию.")
        else:
            logging.error("Исходные данные не загружены. Выполните загрузку перед вызовом этого метода.")

    def get_data_for_display(self) -> List[Dict]:
        """
        Возвращает данные из активного датафрейма в виде списка словарей, пригодном для отображения в таблице интерфейса.
        """
        if self.active_data is not None:
            return self.active_data.to_dict(orient='records')
        logging.error("Данные не загружены. Выполните загрузку перед вызовом этого метода.")
        return []