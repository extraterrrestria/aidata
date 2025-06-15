import os
from csvloader import CSVLoader
from logging_logic import logging


from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QHBoxLayout


class SaveFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сохранить файл")
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()

        # Имя файла
        self.file_name_input = QLineEdit()
        self.file_name_input.setPlaceholderText("Введите имя файла (без .csv)")
        layout.addWidget(QLabel("Имя файла:"))
        layout.addWidget(self.file_name_input)

        # Разделитель
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([",", ";", "\t", "|"])
        layout.addWidget(QLabel("Разделитель:"))
        layout.addWidget(self.delimiter_combo)

        # Кодировка
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["utf-8", "cp1251", "latin1", "utf-16"])
        layout.addWidget(QLabel("Кодировка:"))
        layout.addWidget(self.encoding_combo)

        # Кнопки
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Сохранить")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_values(self):
        """Возвращает значения из полей"""
        file_name = self.file_name_input.text().strip()
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        return {
            'file_name': file_name,
            'delimiter': self.delimiter_combo.currentText(),
            'encoding': self.encoding_combo.currentText()
        }

def save_to_csv(loader: CSVLoader) -> None:
    """
    Сохраняет активный датафрейм в файл CSV.
    :param loader: Экземпляр класса CSVLoader.
    """
    active_data = loader.get_active_data()
    if active_data is None:
        logging.error("Активные данные не загружены. Сохранение невозможно.")
        return

    file_path = loader.file_path
    if not file_path:
        logging.error("Путь к исходному файлу не найден. Сохранение невозможно.")
        return

    directory = os.path.dirname(file_path)

    # Открываем модальное окно
    dialog = SaveFileDialog()
    if dialog.exec() == QDialog.Accepted:
        values = dialog.get_values()
        full_path = os.path.join(directory, values['file_name'])
        delimiter = values['delimiter']
        encoding = values['encoding']

        try:
            active_data.to_csv(full_path, sep=delimiter, encoding=encoding, index=False)
            logging.info(
                f"Данные успешно сохранены в файл '{full_path}' "
                f"с разделителем '{delimiter}' и кодировкой '{encoding}'."
            )
        except Exception as e:
            logging.error(f"Ошибка при сохранении данных в файл: {e}")
    else:
        logging.info("Сохранение отменено пользователем.")