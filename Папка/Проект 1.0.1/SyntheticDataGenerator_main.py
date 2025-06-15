from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QListWidget, QListWidgetItem, QDialog, QFormLayout,
    QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QTabWidget, QMessageBox, QScrollArea, QFileDialog, QHeaderView
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont

import SynteticDataGeneration
from main import InputDialog

import pandas as pd
import numpy as np

class SyntheticDataGenerator(QWidget):
    """Основной класс интерфейса генерации синтетических данных"""
    
    def __init__(self):
        super().__init__()
        
        # Устанавливаем стиль для всего виджета
        self.setStyleSheet("""
            SyntheticDataGenerator {
                background-color: #212121;
                color: white;
            }
            QLabel {
                color: white;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                image: url(:/icons/checkmark.png);
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:checked {
                image: url(:/icons/radio_on.png);
            }
            QGroupBox {
                border: 1px solid #444444;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subline-control: false;
            }
        """)
        
        # Инициализируем UI
        self.init_ui()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Основной вертикальный layout
        main_layout = QVBoxLayout(self)

        # Заголовок и описание
        title_label = QLabel("Генерация")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        description_label = QLabel(
            "Раздел «Генерация» обеспечивает удобное проектирование структуры таблиц и автоматическое заполнение их тестовыми данными. Вы можете задать состав и типы данных для столбцов, выбрать способ генерации, а также настроить параметры экспорта в CSV-файл для последующего использования в работе."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: grey;")

        # Раздел настройки общих параметров
        general_settings_group = QGroupBox("Настройка общих параметров")
        general_settings_layout = QFormLayout()

        # Количество строк
        self.rows_input = QSpinBox()
        self.rows_input.setMinimum(1)
        self.rows_input.setMaximum(1000000)
        self.rows_input.setValue(100)
        self.rows_input.setStyleSheet("background-color: #333333; color: white;")
        general_settings_layout.addRow("Количество строк:", self.rows_input)

        # Количество столбцов
        self.cols_input = QSpinBox()
        self.cols_input.setMinimum(1)
        self.cols_input.setMaximum(100)
        self.cols_input.setValue(5)
        self.cols_input.setStyleSheet("background-color: #333333; color: white;")
        general_settings_layout.addRow("Количество столбцов:", self.cols_input)

        general_settings_group.setLayout(general_settings_layout)

        # Таблица настройки столбцов
        self.columns_table = QTableWidget()
        self.columns_table.setColumnCount(6)  # Название, Тип, Способ, Параметры, Пример, Действия
        self.columns_table.setHorizontalHeaderLabels([
            "Название", "Тип", "Способ", "Параметры", "Пример", "Действия"
        ])
        self.columns_table.verticalHeader().setVisible(False)
        self.columns_table.setSortingEnabled(False)

        # Установка фиксированной высоты строк
        self.columns_table.verticalHeader().setDefaultSectionSize(40)  # Высота строки

        # Настройка ширины столбцов (увеличиваем общий размер таблицы)
        header = self.columns_table.horizontalHeader()
        
        # Относительное распределение ширины:
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)   # Название — по содержимому
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)   # Тип — по содержимому
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)   # Способ — по содержимому
        header.setSectionResizeMode(3, QHeaderView.Stretch)            # Параметры — растягивается
        header.setSectionResizeMode(4, QHeaderView.Stretch)            # Пример — растягивается
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)   # Действия — по содержимому

        # Кнопка обновления таблицы
        update_button = QPushButton("Обновить таблицу")
        update_button.clicked.connect(self.update_columns_table)
        update_button.setStyleSheet("""
            QPushButton {
                background-color: #2f2f2f;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)

        # Группа действий
        actions_group = QGroupBox("Действия")
        actions_layout = QHBoxLayout()

        generate_button = QPushButton("Сгенерировать")
        generate_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        generate_button.clicked.connect(self.generate_data)

        save_button = QPushButton("Сохранить в CSV")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #2f2f2f;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        save_button.clicked.connect(self.save_to_csv)

        actions_layout.addWidget(generate_button)
        actions_layout.addWidget(save_button)
        actions_group.setLayout(actions_layout)

        # Добавляем элементы в основной layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addWidget(general_settings_group)
        main_layout.addWidget(self.columns_table)
        main_layout.addWidget(update_button)
        main_layout.addWidget(actions_group)

        # Инициализируем данные
        self.column_configs = []
        self.generated_data = None

        # Обновляем таблицу изначально
        self.update_columns_table()

        # Увеличение минимального размера окна для лучшего отображения таблицы
        self.setMinimumSize(QSize(1200, 600))  # Ширина увеличена до 1200 пикселей
    
    def update_columns_table(self):
        """Обновление таблицы настройки столбцов"""
        n_rows = self.rows_input.value()
        n_cols = self.cols_input.value()
        
        # Очищаем текущую конфигурацию, если количество столбцов изменилось
        if len(self.column_configs) != n_cols:
            self.column_configs = []
            for i in range(n_cols):
                self.column_configs.append({
                    'name': f'col_{i}',
                    'dtype': 'int',
                    'mode': 'constant',
                    'params': {'value': 0},
                    'example': [0] * min(5, n_rows)
                })
        
        # Настройка таблицы
        self.columns_table.setRowCount(n_cols)
        
        for row_idx, config in enumerate(self.column_configs):
            # Название столбца
            name_input = QLineEdit(config['name'])
            name_input.setStyleSheet("background-color: #333333; color: white;")
            self.columns_table.setCellWidget(row_idx, 0, name_input)
            
            # Тип данных
            dtype_combo = QComboBox()
            dtype_combo.addItems(['int', 'float', 'string', 'boolean', 'datetime'])
            dtype_combo.setCurrentText(config['dtype'])
            dtype_combo.currentTextChanged.connect(lambda text, idx=row_idx: self.update_dtype(idx, text))
            dtype_combo.setStyleSheet("background-color: #333333; color: white;")
            self.columns_table.setCellWidget(row_idx, 1, dtype_combo)
            
            # Способ генерации
            mode_combo = QComboBox()
            mode_combo.currentTextChanged.connect(lambda text, idx=row_idx: self.update_mode(idx, text))
            mode_combo.setStyleSheet("background-color: #333333; color: white;")
            self.columns_table.setCellWidget(row_idx, 2, mode_combo)
            
            # Параметры
            params_btn = QPushButton("Настроить...")
            params_btn.clicked.connect(lambda _, idx=row_idx: self.configure_params(idx))
            params_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2f2f2f;
                    color: white;
                    border-radius: 10px;
                    padding: 5px 10px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #444444;
                }
            """)
            self.columns_table.setCellWidget(row_idx, 3, params_btn)
            
            # Пример значений
            example_item = QTableWidgetItem(str(config['example']))
            example_item.setBackground(Qt.GlobalColor.transparent)
            example_item.setForeground(Qt.GlobalColor.white)
            self.columns_table.setItem(row_idx, 4, example_item)
            
            # Действия
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(5)
            
            reset_btn = QPushButton("Сбросить")
            reset_btn.clicked.connect(lambda _, idx=row_idx: self.reset_column(idx))
            reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2f2f2f;
                    color: white;
                    border-radius: 10px;
                    padding: 5px 10px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #444444;
                }
            """)
            actions_layout.addWidget(reset_btn)
            
            
            self.columns_table.setCellWidget(row_idx, 5, actions_widget)
            
            # Обновляем доступные режимы в зависимости от типа данных
            self.update_modes_for_column(row_idx)
    
    def update_dtype(self, column_idx, dtype):
        """Обновление типа данных для столбца"""
        self.column_configs[column_idx]['dtype'] = dtype
        self.update_modes_for_column(column_idx)
        
        # Также нужно обновить пример
        self.update_example(column_idx)
    
    def update_modes_for_column(self, column_idx):
        """Обновление доступных режимов для столбца в зависимости от типа данных"""
        dtype = self.column_configs[column_idx]['dtype']
        cell_widget = self.columns_table.cellWidget(column_idx, 2)
        
        if not isinstance(cell_widget, QComboBox):
            return
        
        # Очищаем текущие значения
        cell_widget.clear()
        
        # Добавляем доступные режимы
        available_modes = ['constant']
        
        if dtype in ['int', 'float', 'datetime', 'string']:
            available_modes.append('range')
        
        if dtype in ['int', 'float', 'datetime', 'boolean', 'string']:
            available_modes.append('random')
        
        if SynteticDataGeneration.OPENROUTER_API_KEY:
            available_modes.append('ai')
        
        cell_widget.addItems(available_modes)
        
        # Устанавливаем значение по умолчанию или первое доступное
        current_mode = self.column_configs[column_idx]['mode']
        if current_mode in available_modes:
            cell_widget.setCurrentText(current_mode)
        else:
            cell_widget.setCurrentIndex(0)
            self.column_configs[column_idx]['mode'] = available_modes[0]
    
    def update_mode(self, column_idx, mode):
        """Обновление способа генерации для столбца"""
        self.column_configs[column_idx]['mode'] = mode
    
    def configure_params(self, column_idx):
        """Открытие диалога настройки параметров"""
        config = self.column_configs[column_idx]
        dialog = ColumnParamsDialog(config, self)
        
        if dialog.exec() == QDialog.Accepted:
            self.column_configs[column_idx] = dialog.get_config()
            self.update_example(column_idx)
    
    def update_example(self, column_idx):
        """Обновление примера данных для столбца"""
        config = self.column_configs[column_idx]
        n_rows = self.rows_input.value()
        
        try:
            ai_api_key = SynteticDataGeneration.OPENROUTER_API_KEY if SynteticDataGeneration.OPENROUTER_API_KEY else None
            example_data = SynteticDataGeneration.generate_ersatz_column(
                n_rows=min(n_rows, 5),
                name=config['name'],
                dtype=config['dtype'],
                mode=config['mode'],
                params=config['params'],
                ai_api_key=ai_api_key
            )
            self.column_configs[column_idx]['example'] = example_data
            
            # Обновляем ячейку примера в таблице
            example_item = QTableWidgetItem(str(example_data))
            example_item.setBackground(Qt.GlobalColor.transparent)
            example_item.setForeground(Qt.GlobalColor.white)
            self.columns_table.setItem(column_idx, 4, example_item)
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось обновить пример: {str(e)}")
    
    def reset_column(self, column_idx):
        """Сброс настроек столбца к значениям по умолчанию"""
        default_config = {
            'name': self.column_configs[column_idx]['name'],
            'dtype': 'int',
            'mode': 'constant',
            'params': {'value': 0},
            'example': [0] * min(5, self.rows_input.value())
        }
        
        self.column_configs[column_idx] = default_config
        
        # Обновляем соответствующую строку в таблице
        self.update_row_in_table(column_idx)
    
    def update_row_in_table(self, row_idx):
        """Обновление конкретной строки в таблице"""
        config = self.column_configs[row_idx]
        
        # Обновляем тип данных
        dtype_combo = self.columns_table.cellWidget(row_idx, 1)
        if isinstance(dtype_combo, QComboBox):
            dtype_combo.setCurrentText(config['dtype'])
        
        # Обновляем режим генерации
        mode_combo = self.columns_table.cellWidget(row_idx, 2)
        if isinstance(mode_combo, QComboBox):
            mode_combo.setCurrentText(config['mode'])
            self.update_modes_for_column(row_idx)
        
        # Обновляем пример
        example_item = QTableWidgetItem(str(config['example']))
        example_item.setBackground(Qt.GlobalColor.transparent)
        example_item.setForeground(Qt.GlobalColor.white)
        self.columns_table.setItem(row_idx, 4, example_item)
    
    def delete_column(self, column_idx):
        """Удаление столбца"""
        if len(self.column_configs) <= 1:
            QMessageBox.information(self, "Внимание", "Нельзя удалить последний столбец.")
            return
        
        del self.column_configs[column_idx]
        self.update_columns_table()
    
    def generate_data(self):
        # Обновляем имена столбцов из таблицы
        for col_idx in range(len(self.column_configs)):
            name_widget = self.columns_table.cellWidget(col_idx, 0)
            if isinstance(name_widget, QLineEdit):
                new_name = name_widget.text().strip()
                if new_name:
                    self.column_configs[col_idx]['name'] = new_name
            
        """Генерация данных"""
        n_rows = self.rows_input.value()
        n_cols = self.cols_input.value()
        
        if len(self.column_configs) != n_cols:
            QMessageBox.critical(self, "Ошибка", "Конфигурация столбцов не соответствует количеству столбцов.")
            return
        
        try:
            # Создаем пустой DataFrame
            self.generated_data = pd.DataFrame(np.nan, index=range(n_rows), columns=[config['name'] for config in self.column_configs])
            
            # Заполняем данными
            for col_idx, config in enumerate(self.column_configs):
                ai_api_key = SynteticDataGeneration.OPENROUTER_API_KEY if SynteticDataGeneration.OPENROUTER_API_KEY else None
                ersatz_col = SynteticDataGeneration.generate_ersatz_column(
                    n_rows=n_rows,
                    name=config['name'],
                    dtype=config['dtype'],
                    mode=config['mode'],
                    params=config['params'],
                    ai_api_key=ai_api_key
                )
                
                self.generated_data.loc[:, config['name']] = ersatz_col
            
            QMessageBox.information(self, "Успех", "Данные успешно сгенерированы!")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сгенерировать данные: {str(e)}")
    
    def save_to_csv(self):
        """Сохранение данных в CSV-файл"""
        if self.generated_data is None:
            QMessageBox.information(self, "Внимание", "Сначала необходимо сгенерировать данные.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV-файл", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.generated_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Успех", f"Файл успешно сохранен: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")


class ColumnParamsDialog(QDialog):
    """Диалог настройки параметров столбца"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Настройка параметров для {config['name']}")
        self.setWindowIcon(QIcon("Icon.png"))
        self.setFixedSize(600, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: #212121;
            }
            QLabel {
                color: white;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #333333;
                color: white;
            }
            QGroupBox {
                border: 1px solid #444444;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subline-control: false;
            }
        """)
        
        self.config = config.copy()
        self.init_ui()
    
    def init_ui(self):
        """Инициализация интерфейса диалога"""
        main_layout = QVBoxLayout(self)
        
        # Информационная панель
        info_layout = QHBoxLayout()
        
        name_label = QLabel(f"<b>Название:</b> {self.config['name']}")
        dtype_label = QLabel(f"<b>Тип:</b> {self.config['dtype']}")
        mode_label = QLabel(f"<b>Способ:</b> {self.config['mode']}")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(dtype_label)
        info_layout.addWidget(mode_label)
        
        # Форма параметров
        self.params_form = QFormLayout()
        
        # Блок для динамического добавления полей
        self.params_container = QWidget()
        self.params_container.setLayout(self.params_form)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.params_container)
        
        # Кнопки
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("Ок")
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #2f2f2f;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        # Добавляем все элементы
        main_layout.addLayout(info_layout)
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(button_layout)
        
        # Заполняем параметры
        self.populate_params()
    
    def populate_params(self):
        """Заполнение формы параметрами в зависимости от типа и режима"""
        # Очищаем предыдущие поля
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
        
        dtype = self.config['dtype']
        mode = self.config['mode']
        params = self.config['params']
        
        if mode == 'constant':
            # Поле для константного значения
            self.constant_input = self.create_input_for_dtype(dtype, params.get('value'))
            self.params_form.addRow("Значение:", self.constant_input)
        
        elif mode == 'range':
            # Поля для диапазона
            self.start_input = self.create_input_for_dtype(dtype, params.get('start'))
            self.stop_input = self.create_input_for_dtype(dtype, params.get('stop'))
            
            self.step_input = QSpinBox()
            self.step_input.setMinimum(1)
            self.step_input.setMaximum(1000000)
            self.step_input.setValue(params.get('step', 1))
            
            self.cycle_checkbox = QCheckBox("Циклически повторять значения")
            self.cycle_checkbox.setChecked(params.get('cycle', True))
            
            self.params_form.addRow("Начало:", self.start_input)
            self.params_form.addRow("Конец:", self.stop_input)
            self.params_form.addRow("Шаг:", self.step_input)
            self.params_form.addRow("", self.cycle_checkbox)
        
        elif mode == 'random':
            if dtype in ['int', 'float', 'datetime']:
                # Выбор распределения
                self.dist_combo = QComboBox()
                self.dist_combo.addItems(['normal', 'uniform', 'exponential'])
                self.dist_combo.setCurrentText(params.get('dist', 'normal'))
                self.dist_combo.currentTextChanged.connect(self.populate_distribution_params)
                
                self.params_form.addRow("Распределение:", self.dist_combo)
                
                # Контейнер для параметров распределения
                self.dist_params_container = QWidget()
                self.dist_params_layout = QFormLayout()
                self.dist_params_container.setLayout(self.dist_params_layout)
                self.params_form.addRow(self.dist_params_container)
                
                # Инициализируем параметры для текущего распределения
                self.populate_distribution_params(params.get('dist', 'normal'))
            
            elif dtype == 'boolean':
                # Вероятность True
                self.p_true_input = QDoubleSpinBox()
                self.p_true_input.setMinimum(0.0)
                self.p_true_input.setMaximum(1.0)
                self.p_true_input.setSingleStep(0.05)
                self.p_true_input.setValue(params.get('dist_params', {}).get('p_true', 0.5))
                
                self.params_form.addRow("Вероятность True:", self.p_true_input)
            
            elif dtype == 'string':
                # Категории и пропорции
                self.categories_list = QListWidget()
                categories = params.get('categories', [])
                proportions = params.get('proportions', [])
                
                for category, proportion in zip(categories, proportions):
                    item = QListWidgetItem(f"{category} ({proportion:.2%})")
                    item.setData(Qt.ItemDataRole.UserRole, (category, proportion))
                    self.categories_list.addItem(item)
                
                add_category_btn = QPushButton("Добавить категорию")
                add_category_btn.clicked.connect(self.add_category)
                
                remove_category_btn = QPushButton("Удалить выбранную")
                remove_category_btn.clicked.connect(self.remove_category)
                
                buttons_layout = QHBoxLayout()
                buttons_layout.addWidget(add_category_btn)
                buttons_layout.addWidget(remove_category_btn)
                
                self.params_form.addRow("Категории:", self.categories_list)
                self.params_form.addRow(buttons_layout)
        
        elif mode == 'ai':
            # Промпт для ИИ
            self.prompt_input = QLineEdit()
            self.prompt_input.setText(params.get('prompt', ""))
            
            self.params_form.addRow("Промпт:", self.prompt_input)
    
    def create_input_for_dtype(self, dtype, value=None):
        """Создает подходящее поле ввода в зависимости от типа данных"""
        if dtype == 'int':
            input_widget = QSpinBox()
            input_widget.setMinimum(-1000000)
            input_widget.setMaximum(1000000)
            input_widget.setValue(int(value) if value is not None else 0)
        
        elif dtype == 'float':
            input_widget = QDoubleSpinBox()
            input_widget.setMinimum(-1000000.0)
            input_widget.setMaximum(1000000.0)
            input_widget.setSingleStep(0.1)
            input_widget.setValue(float(value) if value is not None else 0.0)
        
        elif dtype == 'string':
            input_widget = QLineEdit()
            input_widget.setText(str(value) if value is not None else "")
        
        elif dtype == 'boolean':
            input_widget = QComboBox()
            input_widget.addItems(['True', 'False'])
            input_widget.setCurrentText(str(value) if value is not None else 'True')
        
        elif dtype == 'datetime':
            input_widget = QLineEdit()
            input_widget.setText(str(value) if value is not None else "1970-01-01")
        
        return input_widget
    
    def populate_distribution_params(self, dist_name):
        """Заполняет параметры распределения"""
        # Очищаем контейнер
        while self.dist_params_layout.rowCount() > 0:
            self.dist_params_layout.removeRow(0)
        
        params = self.config['params'].get('dist_params', {})
        
        if dist_name == 'normal':
            mean_input = QDoubleSpinBox()
            mean_input.setMinimum(-1000000.0)
            mean_input.setMaximum(1000000.0)
            mean_input.setSingleStep(0.1)
            mean_input.setValue(params.get('mean', 0.0))
            
            std_input = QDoubleSpinBox()
            std_input.setMinimum(0.0)
            std_input.setMaximum(1000000.0)
            std_input.setSingleStep(0.1)
            std_input.setValue(params.get('std', 1.0))
            
            self.dist_params_layout.addRow("Матожидание:", mean_input)
            self.dist_params_layout.addRow("Стандартное отклонение:", std_input)
            
            self.normal_params = {
                'mean': mean_input,
                'std': std_input
            }
        
        elif dist_name == 'uniform':
            low_input = QDoubleSpinBox()
            low_input.setMinimum(-1000000.0)
            low_input.setMaximum(1000000.0)
            low_input.setSingleStep(0.1)
            low_input.setValue(params.get('low', 0.0))
            
            high_input = QDoubleSpinBox()
            high_input.setMinimum(-1000000.0)
            high_input.setMaximum(1000000.0)
            high_input.setSingleStep(0.1)
            high_input.setValue(params.get('high', 1.0))
            
            self.dist_params_layout.addRow("Минимум:", low_input)
            self.dist_params_layout.addRow("Максимум:", high_input)
            
            self.uniform_params = {
                'low': low_input,
                'high': high_input
            }
        
        elif dist_name == 'exponential':
            scale_input = QDoubleSpinBox()
            scale_input.setMinimum(0.0)
            scale_input.setMaximum(1000000.0)
            scale_input.setSingleStep(0.1)
            scale_input.setValue(params.get('scale', 1.0))
            
            self.dist_params_layout.addRow("Параметр масштабирования:", scale_input)
            
            self.exponential_params = {
                'scale': scale_input
            }
        
        # Добавляем поле для базовой даты, если это datetime
        if self.config['dtype'] == 'datetime':
            base_date_input = QLineEdit()
            base_date_input.setText(params.get('base_date', '1970-01-01'))
            
            self.dist_params_layout.addRow("Базовая дата:", base_date_input)
            
            if not hasattr(self, 'dist_params'):
                self.dist_params = {}
            
            self.dist_params['base_date_input'] = base_date_input
    
    def add_category(self):
        """Добавление новой категории"""
        dialog = InputDialog(title="Добавить категорию", label="Введите название категории:")
        if dialog.exec() == QDialog.Accepted:
            category = dialog.get_input()
            
            if category.strip() == "":
                return
            
            # Рассчитываем пропорции
            current_count = self.categories_list.count()
            total = 1.0
            remaining = total - sum([item.data(Qt.ItemDataRole.UserRole)[1] for item in [
                self.categories_list.item(i) for i in range(current_count)
            ]])
            
            prop_dialog = InputDialog(
                title="Пропорция категории",
                label=f"Введите долю для категории '{category}' (оставшееся: {remaining:.2f}):",
                default=str(remaining)
            )
            
            if prop_dialog.exec() == QDialog.Accepted:
                try:
                    proportion = float(prop_dialog.get_input())
                    if proportion < 0 or proportion > remaining:
                        raise ValueError("Неверная пропорция")
                    
                    item = QListWidgetItem(f"{category} ({proportion:.2%})")
                    item.setData(Qt.ItemDataRole.UserRole, (category, proportion))
                    self.categories_list.addItem(item)
                
                except ValueError as e:
                    QMessageBox.critical(self, "Ошибка", f"Неверное значение пропорции: {str(e)}")
    
    def remove_category(self):
        """Удаление выбранной категории"""
        selected_items = self.categories_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            self.categories_list.takeItem(self.categories_list.row(item))
    
    def get_config(self):
        """Получение обновленной конфигурации"""
        dtype = self.config['dtype']
        mode = self.config['mode']
        self.config['params'] = {}
        
        if mode == 'constant':
            value = self.get_value_from_input(self.constant_input, dtype)
            self.config['params']['value'] = value
        
        elif mode == 'range':
            start = self.get_value_from_input(self.start_input, dtype)
            stop = self.get_value_from_input(self.stop_input, dtype)
            step = self.step_input.value()
            cycle = self.cycle_checkbox.isChecked()
            
            self.config['params'].update({
                'start': start,
                'stop': stop,
                'step': step,
                'cycle': cycle
            })
        
        elif mode == 'random':
            if dtype in ['int', 'float', 'datetime']:
                dist = self.dist_combo.currentText()
                self.config['params']['dist'] = dist
                
                dist_params = {}
                
                if dist == 'normal':
                    dist_params['mean'] = self.normal_params['mean'].value()
                    dist_params['std'] = self.normal_params['std'].value()
                
                elif dist == 'uniform':
                    dist_params['low'] = self.uniform_params['low'].value()
                    dist_params['high'] = self.uniform_params['high'].value()
                
                elif dist == 'exponential':
                    dist_params['scale'] = self.exponential_params['scale'].value()
                
                if dtype == 'datetime':
                    dist_params['base_date'] = self.dist_params['base_date_input'].text()
                
                self.config['params']['dist_params'] = dist_params
            
            elif dtype == 'boolean':
                self.config['params']['dist_params'] = {
                    'p_true': self.p_true_input.value()
                }
            
            elif dtype == 'string':
                categories = []
                proportions = []
                
                for i in range(self.categories_list.count()):
                    item = self.categories_list.item(i)
                    category, proportion = item.data(Qt.ItemDataRole.UserRole)
                    categories.append(category)
                    proportions.append(proportion)
                
                self.config['params']['categories'] = categories
                self.config['params']['proportions'] = proportions
        
        elif mode == 'ai':
            self.config['params']['prompt'] = "Данные должны быть сгенерированы в виде {n} чисел через запятую. Не выдавай ничего, кроме этих данных. Тип данных должен быть {column_type}." + self.prompt_input.text()
        
        return self.config
    
    def get_value_from_input(self, input_widget, dtype):
        """Получение значения из виджета ввода в соответствии с типом данных"""
        if dtype == 'int':
            return int(input_widget.value())
        
        elif dtype == 'float':
            return float(input_widget.value())
        
        elif dtype == 'string':
            return str(input_widget.text())
        
        elif dtype == 'boolean':
            return input_widget.currentText().lower() == 'true'
        
        elif dtype == 'datetime':
            return str(input_widget.text())