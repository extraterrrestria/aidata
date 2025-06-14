from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QFileDialog, QTableWidget, QTableWidgetItem, QCheckBox, QDialog, QListWidget, QListWidgetItem, QComboBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Импорты бэкенд-функций
import csvloader
import correlation_functions
import grouping_functions
import registry_functions
import unique_values_functions
import logging

import SynteticDataGeneration
import SyntheticDataGenerator_main

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Настройка основного окна
        self.setWindowTitle("Анализ данных")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #212121;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2f2f2f;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QLineEdit {
                background-color: #333333;
                color: white;
                border-radius: 10px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: none;
                background-color: #212121;
            }
            QTabBar::tab {
                background-color: #2f2f2f;
                color: white;
                min-width: 8ex;
                padding: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #444444;
            }
        """)
        
        # Хедер
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setPixmap(QIcon("logotype.png").pixmap(200, 50))
        header_layout.addWidget(logo_label)
        
        tab_widget = QTabWidget()
        self.analysis_tab = AnalysisTab(main_window=self)
        self.generation_tab = SyntheticDataGenerator_main.SyntheticDataGenerator()
        tab_widget.addTab(self.analysis_tab, "Анализ")
        tab_widget.addTab(self.generation_tab, "Генерация")
        header_layout.addWidget(tab_widget)
        
        # Основной контейнер
        main_container = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addWidget(tab_widget)
        main_container.setLayout(main_layout)
        self.setCentralWidget(main_container)


class AnalysisTab(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        
        self.main_window = main_window
        
        # Основной лэйаут
        main_layout = QVBoxLayout()
        
        # Заголовок и описание
        title_label = QLabel("Анализ")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        description_label = QLabel(
            "Раздел «Анализ» предоставляет широкие возможности для исследования и обработки данных в удобной визуальной форме. Вы можете работать как с реестром записей, так и с дашбордом, получая глубокое понимание структуры и характеристик вашего датасета."
        )
        description_label.setWordWrap(True)
        
        # Форма загрузки файла
        file_form_layout = QVBoxLayout()
        
        # Разделитель
        separator_label = QLabel("Укажите путь к CSV-файлу базы данных")
        file_path_layout = QHBoxLayout()
        self.file_path_input = QLineEdit(placeholderText="Путь к БД")
        self.file_path_input.setStyleSheet("color: grey;")
        browse_button = QPushButton("Обзор")
        browse_button.clicked.connect(self.browse_file)
        file_path_layout.addWidget(self.file_path_input)
        file_path_layout.addWidget(browse_button)
        
        # Выбор разделителя
        delimiter_layout = QHBoxLayout()
        delimiter_label = QLabel("Разделитель:")
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([",", ";", "\t", "|"])  # Допустимые разделители
        delimiter_layout.addWidget(delimiter_label)
        delimiter_layout.addWidget(self.delimiter_combo)
        
        # Кнопки проверки и импорта
        button_layout = QHBoxLayout()
        self.validate_button = QPushButton("Проверить файл")
        self.validate_button.clicked.connect(self.validate_file)
        self.import_button = QPushButton("Импортировать")
        self.import_button.clicked.connect(self.import_file)
        self.import_button.setEnabled(False)  # Изначально отключена
        button_layout.addWidget(self.validate_button)
        button_layout.addWidget(self.import_button)
        
        # Добавление элементов в форму
        file_form_layout.addWidget(separator_label)
        file_form_layout.addLayout(file_path_layout)
        file_form_layout.addLayout(delimiter_layout)
        file_form_layout.addLayout(button_layout)
        
        # Добавление всего в основной лэйаут
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addLayout(file_form_layout)
        self.setLayout(main_layout)
    
    def browse_file(self):
        """Открыть диалог выбора файла"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV-файл", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path_input.setText(file_path)
    
    def validate_file(self):
        """Проверить файл на корректность"""
        file_path = self.file_path_input.text()
        if not file_path:
            return
        
        # Получаем выбранный разделитель
        delimiter = self.delimiter_combo.currentText()
        
        # Использование бэкенд-функции для проверки файла
        loader = csvloader.CSVLoader()
        loader.file_path = file_path
        loader.set_reading_parameters(delimiter=delimiter)
        if loader.validate_structure():
            self.import_button.setEnabled(True)
            print("Файл прошел валидацию.")
        else:
            self.import_button.setEnabled(False)
            print("Ошибка при валидации файла.")
    
    def import_file(self):
        file_path = self.file_path_input.text()
        if not file_path:
            return

        delimiter = self.delimiter_combo.currentText()
        loader = csvloader.CSVLoader()
        loader.file_path = file_path
        loader.set_reading_parameters(delimiter=delimiter)

        if loader.load_data():
            print("Файл успешно загружен.")

            # Открыть модальное окно выбора столбцов
            column_dialog = ColumnSelectionDialog(loader.get_column_names())
            if column_dialog.exec():
                selected_columns = column_dialog.selected_columns()
                print(f"Выбранные столбцы: {selected_columns}")
                # Перейти на страницу анализа
                self.main_window.setCentralWidget(AnalysisDataPage(selected_columns, loader))


from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QListWidgetItem, QHBoxLayout


class ColumnSelectionDialog(QDialog):
    def __init__(self, columns):
        super().__init__()
        self.setWindowTitle("Выберите столбцы")
        self.setFixedSize(400, 500)

        layout = QVBoxLayout(self)

        # Список столбцов
        self.column_list = QListWidget()
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # Все по умолчанию выбраны
            self.column_list.addItem(item)

        layout.addWidget(self.column_list)

        # Кнопки
        button_layout = QHBoxLayout()
        confirm_button = QPushButton("Применить")
        confirm_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def selected_columns(self):
        """Возвращает список выбранных столбцов"""
        selected = []
        for index in range(self.column_list.count()):
            item = self.column_list.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected



class AnalysisDataPage(QWidget):
    def __init__(self, selected_columns, loader):
        super().__init__()

        # Основной лэйаут
        main_layout = QVBoxLayout()

        # Навигационные табы (Реестр записей и Дашборд)
        tab_widget = QTabWidget()
        tab_widget.addTab(DataRegistry(selected_columns, loader), "Реестр записей")
        tab_widget.addTab(Dashboard(selected_columns, loader), "Дашборд")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
from PySide6.QtCore import Qt


class DataRegistry(QWidget):
    def __init__(self, selected_columns, loader):
        super().__init__()
        self.loader = loader
        self.selected_columns = selected_columns
        self.init_ui()

    def init_ui(self):
        
        # Основной лэйаут
        main_layout = QHBoxLayout()

        # Сайдбар с функциями
        sidebar_layout = QVBoxLayout()
        sidebar_widget = QWidget()
        sidebar_widget.setStyleSheet("background-color: #171717;")
        sidebar_layout.addSpacing(10)

        functions = [
            "Удаление строк с пропусками",
            "Заполнение пропусков: средним/медианой/модой",
            "Нормализация данных",
            "Стандартизация данных",
            "Удаление выбросов через IQR-метод",
            "Фильтрация строк по условию",
            "One-Hot Encoding для категориальных переменных",
            "Сортировка по заранее заданному массиву-порядку",
            "Фильтрация по заранее заданному массиву-фильтру",
            "Корреляция Пирсона",
            "Асимметрия распределения",
            "Эксцесс распределения",
            "Процентное соотношение уникальных значений",
            "Подсчет пропущенных значений"
        ]

        for func in functions:
            button = QPushButton(func)
            button.setStyleSheet("QPushButton { background-color: #2f2f2f; } QPushButton:checked { background-color: #444444; }")
            button.setCheckable(True)
            button.clicked.connect(self.handle_function_click)
            sidebar_layout.addWidget(button)

        save_button = QPushButton("Сохранить изменения")
        save_button.setStyleSheet("QPushButton { background-color: white; color: black; border-radius: 10px; padding: 10px 20px; }")
        save_button.clicked.connect(self.save_data)
        sidebar_layout.addWidget(save_button)

        sidebar_widget.setLayout(sidebar_layout)

        # Кнопка "Выбор столбцов"
        select_columns_button = QPushButton("Выбор столбцов")
        select_columns_button.setStyleSheet("QPushButton { background-color: #2f2f2f; color: white; border-radius: 10px; padding: 10px 20px; }")
        select_columns_button.clicked.connect(self.select_columns)
        sidebar_layout.addWidget(select_columns_button)

        sidebar_widget.setLayout(sidebar_layout)

        # Таблица данных
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self.selected_columns))
        self.table_widget.setHorizontalHeaderLabels(self.selected_columns)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.load_data_to_table()

        # Добавление элементов
        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.table_widget)
        self.setLayout(main_layout)

    def select_columns(self):
        """Открывает диалог выбора столбцов"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            selected_columns = dialog.selected_columns()
            self.loader.select_columns(selected_columns)  # Обновляем активные столбцы в CSVLoader
            self.selected_columns = selected_columns  # Обновляем список выбранных столбцов
            self.table_widget.setColumnCount(len(selected_columns))  # Обновляем количество колонок в таблице
            self.table_widget.setHorizontalHeaderLabels(selected_columns)  # Обновляем заголовки колонок
            self.load_data_to_table()  # Обновляем данные в таблице
        
        

    def load_data_to_table(self):
        data = self.loader.get_data_for_display()
        if not data:
            return

        self.table_widget.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, column in enumerate(self.selected_columns):
                item = QTableWidgetItem(str(row_data.get(column, "")))
                self.table_widget.setItem(row_idx, col_idx, item)
    def save_data(self):
        """Вызывает функцию сохранения файла"""
        from csv_loader import save_to_csv
        save_to_csv(self.loader)
        
    def load_data_to_table(self):
        """Обновляет данные в таблице"""
        data = self.loader.get_active_data()
        if not data:
            return

        self.table_widget.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, column in enumerate(self.selected_columns):
                item = QTableWidgetItem(str(row_data.get(column, "")))
                self.table_widget.setItem(row_idx, col_idx, item)

    @Slot()
    def handle_function_click(self):
        sender = self.sender()
        function_name = sender.text()

        if function_name == "Удаление строк с пропусками":
            self.remove_missing_rows()
        elif function_name == "Заполнение пропусков: средним/медианой/модой":
            self.fill_missing_values()
        elif function_name == "Нормализация данных":
            self.normalize_data()
        elif function_name == "Стандартизация данных":
            self.standardize_data()
        elif function_name == "Удаление выбросов через IQR-метод":
            self.remove_outliers_iqr()
        elif function_name == "Фильтрация строк по условию":
            self.filter_rows_by_condition()
        elif function_name == "One-Hot Encoding для категориальных переменных":
            self.one_hot_encode()
        elif function_name == "Сортировка по заранее заданному массиву-порядку":
            self.sort_data_by_order()
        elif function_name == "Фильтрация по заранее заданному массиву-фильтру":
            self.filter_data_by_filter()
        elif function_name == "Корреляция Пирсона":
            self.calculate_pearson_correlation()
        elif function_name == "Асимметрия распределения":
            self.calculate_skewness()
        elif function_name == "Эксцесс распределения":
            self.calculate_kurtosis()
        elif function_name == "Процентное соотношение уникальных значений":
            self.calculate_category_percentages()
        elif function_name == "Подсчет пропущенных значений":
            self.count_missing_values()

        self.load_data_to_table()

    def calculate_pearson_correlation(self):
        """Вычисляет корреляцию Пирсона"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            columns = dialog.selected_columns()
            correlation_functions.calculate_pearson_correlation(self.loader.active_data, columns)

    def calculate_skewness(self):
        """Вычисляет асимметрию распределения"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            columns = dialog.selected_columns()
            skew_dict = correlation_functions.calculate_skewness(self.loader.active_data, columns)
            print(f"Асимметрия: {skew_dict}")

    def calculate_kurtosis(self):
        """Вычисляет эксцесс распределения"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            columns = dialog.selected_columns()
            kurt_dict = correlation_functions.calculate_kurtosis(self.loader.active_data, columns)
            print(f"Эксцесс: {kurt_dict}")

    def calculate_category_percentages(self):
        """Процентное соотношение уникальных значений"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            columns = dialog.selected_columns()
            result = unique_values_functions.calculate_category_percentages(self.loader.active_data, columns)
            print(f"Процентное соотношение: {result}")

    def count_missing_values(self):
        """Подсчет пропущенных значений"""
        dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if dialog.exec():
            columns = dialog.selected_columns()
            missing = unique_values_functions.count_missing_values(self.loader.active_data, columns)
            print(f"Пропущенные значения: {missing}")
    def remove_missing_rows(self):
        """Удаление строк с пропусками"""
        try:
            data = registry_functions.remove_rows_with_missing_values(self.loader.active_data)
            self.loader.active_data = data
            logging.info("Строки с пропусками успешно удалены.")
        except Exception as e:
            logging.error(f"Ошибка при удалении строк: {e}")

    def fill_missing_values(self):
        dialog = InputDialog(
            title="Метод заполнения",
            label="Выберите метод:",
            options=["mean", "median", "mode"]
        )
        if dialog.exec():
            method = dialog.get_input()
            column_dialog = ColumnSelectionDialog(self.loader.get_column_names())
            if column_dialog.exec():
                columns = column_dialog.selected_columns()
                try:
                    data = registry_functions.fill_missing_values(
                        self.loader.active_data,
                        columns=columns,
                        method=method
                    )
                    self.loader.active_data = data
                    logging.info(f"Пропуски заполнены методом '{method}' в столбцах: {columns}")
                except Exception as e:
                    logging.error(f"Ошибка при заполнении пропусков: {e}")

    def normalize_data(self):
        min_dialog = InputDialog(title="Минимальное значение", label="Введите минимальное значение:")
        if min_dialog.exec():
            max_dialog = InputDialog(title="Максимальное значение", label="Введите максимальное значение:")
            if max_dialog.exec():
                min_val = float(min_dialog.get_input())
                max_val = float(max_dialog.get_input())
                column_dialog = ColumnSelectionDialog(self.loader.get_column_names())
                if column_dialog.exec():
                    columns = column_dialog.selected_columns()
                    try:
                        data = registry_functions.normalize_data(
                            self.loader.active_data,
                            columns=columns,
                            min_val=min_val,
                            max_val=max_val
                        )
                        self.loader.active_data = data
                        logging.info(f"Данные нормализованы в диапазоне [{min_val}, {max_val}] для столбцов: {columns}")
                    except Exception as e:
                        logging.error(f"Ошибка при нормализации: {e}")

    def standardize_data(self):
        column_dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if column_dialog.exec():
            columns = column_dialog.selected_columns()
            try:
                data = registry_functions.standardize_data(self.loader.active_data, columns=columns)
                self.loader.active_data = data
                logging.info(f"Данные стандартизированы для столбцов: {columns}")
            except Exception as e:
                logging.error(f"Ошибка при стандартизации: {e}")

    def remove_outliers_iqr(self):
        column_dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if column_dialog.exec():
            columns = column_dialog.selected_columns()
            try:
                data = registry_functions.remove_outliers_iqr(self.loader.active_data, columns=columns)
                self.loader.active_data = data
                logging.info(f"Выбросы удалены методом IQR для столбцов: {columns}")
            except Exception as e:
                logging.error(f"Ошибка при удалении выбросов: {e}")

    def filter_rows_by_condition(self):
        condition_dialog = InputDialog(title="Условие фильтрации", label="Введите условие (например, 'Age > 30'):")
        if condition_dialog.exec():
            condition = condition_dialog.get_input()
            try:
                data = registry_functions.modify_rows_by_condition(
                    self.loader.active_data,
                    condition=condition,
                    column="",
                    value=None
                )
                self.loader.active_data = data
                logging.info(f"Фильтрация выполнена по условию: {condition}")
            except Exception as e:
                logging.error(f"Ошибка при фильтрации: {e}")

    def one_hot_encode(self):
        column_dialog = ColumnSelectionDialog(self.loader.get_column_names())
        if column_dialog.exec():
            columns = column_dialog.selected_columns()
            try:
                data = registry_functions.one_hot_encode(self.loader.active_data, columns=columns)
                self.loader.active_data = data
                logging.info(f"One-Hot Encoding выполнен для столбцов: {columns}")
            except Exception as e:
                logging.error(f"Ошибка при One-Hot кодировании: {e}")

    def sort_data_by_order(self):
        column_dialog = InputDialog(title="Столбец для сортировки", label="Введите столбец:")
        if column_dialog.exec():
            ascending_dialog = InputDialog(
                title="Порядок сортировки",
                label="Сортировать по возрастанию?",
                options=["True", "False"]
            )
            if ascending_dialog.exec():
                column = column_dialog.get_input()
                ascending = ascending_dialog.get_input().lower() == "true"
                try:
                    data = registry_functions.sort_data(
                        self.loader.active_data,
                        columns=[column],
                        ascending=ascending
                    )
                    self.loader.active_data = data
                    logging.info(f"Сортировка выполнена по столбцу '{column}', порядок: {'возрастание' if ascending else 'убывание'}")
                except Exception as e:
                    logging.error(f"Ошибка при сортировке: {e}")

    def filter_data_by_filter(self):
        array_dialog = InputDialog(title="Массив фильтра", label="Введите значения через запятую:")
        if array_dialog.exec():
            filter_array = array_dialog.get_input().split(",")
            try:
                data = registry_functions.filter_data_by_array(self.loader.active_data, filter_array=filter_array)
                self.loader.active_data = data
                logging.info(f"Фильтрация выполнена по массиву: {filter_array}")
            except Exception as e:
                logging.error(f"Ошибка при фильтрации: {e}")

    def load_data_to_table(self):
        """Загружает данные в таблицу"""
        data = self.loader.get_data_for_display()
        if not data:
            return

        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(self.selected_columns))

        for row_idx, row_data in enumerate(data):
            for col_idx, column in enumerate(self.selected_columns):
                item = QTableWidgetItem(str(row_data.get(column, "")))
                self.table_widget.setItem(row_idx, col_idx, item)
                
    def load_data_to_table(self):
        """Загружает данные в таблицу"""
        data = self.loader.get_data_for_display()
        if not data:
            return
        
        # Очищаем текущую таблицу
        self.table_widget.clearContents()
        self.table_widget.setRowCount(len(data))
        
        # Заполняем таблицу новыми данными
        for row_idx, row_data in enumerate(data):
            for col_idx, column in enumerate(self.selected_columns):
                item = QTableWidgetItem(str(row_data.get(column, "")))
                self.table_widget.setItem(row_idx, col_idx, item)
        

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_histogram(self, data, column):
        """Построение гистограммы"""
        self.axes.clear()
        self.axes.hist(data[column], bins=20, color='skyblue', edgecolor='black')
        self.axes.set_title(f'Гистограмма: {column}')
        self.draw()

    def plot_line(self, data, column):
        """Построение линейного графика"""
        self.axes.clear()
        self.axes.plot(data[column])
        self.axes.set_title(f'Линейный график: {column}')
        self.draw()

    def plot_pie(self, data, column):
        """Построение круговой диаграммы"""
        counts = data[column].value_counts()
        self.axes.clear()
        self.axes.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        self.axes.set_title(f'Круговая диаграмма: {column}')
        self.draw()

    def plot_bar(self, data, column):
        """Построение столбчатой диаграммы"""
        counts = data[column].value_counts()
        self.axes.clear()
        counts.plot.bar(ax=self.axes)
        self.axes.set_title(f'Столбчатый график: {column}')
        self.draw()

    def plot_correlation(self, data, columns):
        """Построение тепловой карты корреляции"""
        import seaborn as sns
        numeric_data = data[columns].select_dtypes(include=['number'])
        if len(numeric_data.columns) < 2:
            return
        corr = numeric_data.corr()
        self.axes.clear()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=self.axes)
        self.axes.set_title("Тепловая карта корреляции")
        self.draw()

    def clear(self):
        """Очистка графика"""
        self.axes.clear()
        self.draw()


class Dashboard(QWidget):
    def __init__(self, selected_columns, loader):
        super().__init__()
        self.loader = loader
        self.selected_columns = selected_columns

        layout = QVBoxLayout()

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems([
            "Гистограмма",
            "Линейный график",
            "Круговая диаграмма",
            "Столбчатый график",
            "Тепловая карта корреляции",
            "Группированная сумма",
            "Группированное среднее",
            "Процент категории"
        ])
        self.graph_type_combo.currentIndexChanged.connect(self.plot_selected_graph)

        self.canvas = PlotCanvas(self)

        layout.addWidget(self.graph_type_combo)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_selected_graph(self):
        graph_type = self.graph_type_combo.currentText()
        active_data = self.loader.get_active_data()
        if active_data is None:
            logging.error("Нет данных для построения графика.")
            return

        if graph_type in ["Гистограмма", "Линейный график", "Круговая диаграмма", "Столбчатый график"]:
            dialog = ColumnSelectionDialog(self.loader.get_column_names())
            dialog.setWindowTitle("Выберите один столбец")
            if dialog.exec():
                selected = dialog.selected_columns()
                if len(selected) != 1:
                    logging.warning("Выберите ровно один столбец.")
                    return
                if graph_type == "Гистограмма":
                    self.canvas.plot_histogram(active_data, selected[0])
                elif graph_type == "Линейный график":
                    self.canvas.plot_line(active_data, selected[0])
                elif graph_type == "Круговая диаграмма":
                    self.canvas.plot_pie(active_data, selected[0])
                elif graph_type == "Столбчатый график":
                    self.canvas.plot_bar(active_data, selected[0])

        elif graph_type == "Тепловая карта корреляции":
            dialog = ColumnSelectionDialog(self.loader.get_column_names())
            if dialog.exec():
                columns = dialog.selected_columns()
                numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(active_data[col])]
                if len(numeric_cols) < 2:
                    logging.warning("Выберите минимум 2 числовых столбца.")
                    return
                self.canvas.plot_correlation(active_data, numeric_cols)

        elif graph_type == "Группированная сумма":
            group_col = InputDialog(title="Группирующий столбец", label="Введите группирующий столбец:")
            if group_col.exec():
                group_col = group_col.get_input()
                cols_dialog = ColumnSelectionDialog(self.loader.get_column_names())
                if cols_dialog.exec():
                    cols = cols_dialog.selected_columns()
                    result = grouping_functions.group_sum(active_data, group_col, cols)
                    print("Группированная сумма:", result)

        elif graph_type == "Группированное среднее":
            group_col = InputDialog(title="Группирующий столбец", label="Введите группирующий столбец:")
            if group_col.exec():
                group_col = group_col.get_input()
                cols_dialog = ColumnSelectionDialog(self.loader.get_column_names())
                if cols_dialog.exec():
                    cols = cols_dialog.selected_columns()
                    result = grouping_functions.group_mean(active_data, group_col, cols)
                    print("Группированное среднее:", result)

        elif graph_type == "Процент категории":
            group_col = InputDialog(title="Группирующий столбец", label="Введите группирующий столбец:")
            if group_col.exec():
                group_col = group_col.get_input()
                cat_col = InputDialog(title="Категориальный столбец", label="Введите категориальный столбец:")
                if cat_col.exec():
                    category_col = cat_col.get_input()
                    result = grouping_functions.group_category_percentage(active_data, group_col, category_col)
                    print("Процент категории:", result)


from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox


class InputDialog(QDialog):
    def __init__(self, title="Введите данные", label="Значение:", default="", options=None):
        super().__init__()
        self.setWindowTitle(title)
        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)

        self.label = QLabel(label)
        layout.addWidget(self.label)

        if options:
            self.input_field = QComboBox()
            self.input_field.addItems(options)
        else:
            self.input_field = QLineEdit(default)

        layout.addWidget(self.input_field)

        button_layout = QHBoxLayout()
        confirm_button = QPushButton("OK")
        confirm_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def get_input(self):
        """Возвращает введённое значение"""
        if isinstance(self.input_field, QComboBox):
            return self.input_field.currentText()
        else:
            return self.input_field.text()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()