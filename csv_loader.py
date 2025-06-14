import os
from csvloader import CSVLoader
from logging_logic import logging

def save_to_csv(loader: CSVLoader, is_generated: bool) -> None:
    """
    Сохраняет активный датафрейм в файл CSV.
    :param loader: Экземпляр класса CSVLoader.
    :param is_generated: Флаг, указывающий, были ли данные сгенерированы.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены. Сохранение невозможно.")
        return

    # Определяем директорию для сохранения
    if is_generated:
        logging.warning("Путь к исходному файлу не найден. Будет использована текущая директория.")
        directory = os.getcwd()  # Используем текущую директорию для сгенерированных данных
        print(directory)
    else:
        file_path = loader.file_path
        if not file_path:
            logging.error("Путь к исходному файлу не найден. Сохранение невозможно.")
            return
        directory = os.path.dirname(file_path)

    # Запрос имени файла
    file_name = input("Введите имя файла для сохранения (без расширения): ").strip()
    if not file_name:
        logging.warning("Имя файла не указано. Возврат в главное меню.")
        return

    # Добавление расширения .csv, если его нет
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # Полный путь для сохранения
    full_path = os.path.join(directory, file_name)

    # Запрос разделителя
    delimiter = input("Введите разделитель (по умолчанию ','): ") or ","

    # Запрос кодировки
    encoding = input("Введите кодировку (по умолчанию 'utf-8'): ") or "utf-8"

    try:
        # Сохранение данных в файл
        active_data.to_csv(full_path, sep=delimiter, encoding=encoding, index=False)
        logging.info(f"Данные успешно сохранены в файл '{full_path}' с разделителем '{delimiter}' и кодировкой '{encoding}'.")
        print(f"Данные успешно сохранены в файл '{full_path}'.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в файл: {e}")
        print(f"Ошибка при сохранении данных в файл: {e}")