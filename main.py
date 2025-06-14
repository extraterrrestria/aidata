import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from csvloader import CSVLoader

# Создаем глобальный объект для хранения данных
class DataStorage:
    def __init__(self):
        self.data = None  # Здесь будут храниться данные

data_storage = DataStorage()

class FileAnalysisWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ файла")
        self.root.geometry("800x600")
        self.csv_loader = CSVLoader()  # Создаем экземпляр CSVLoader

        # Панель управления
        self.create_top_panel()

        # Область для пути к файлу
        self.create_file_path_section()

        # Настройки файла
        self.create_file_settings()

        # Кнопки действия
        self.create_action_buttons()

        # Модальное окно для предпросмотра
        self.preview_window = None

    def create_top_panel(self):
        """Создает панель с логотипом и дополнительными кнопками."""
        top_panel = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=5)
        top_panel.pack(fill="x")

        # Логотип (заменяется словом "ЛОГО")
        logo_label = tk.Label(top_panel, text="ЛОГО", font=("Arial", 14, "bold"), bg="#f0f0f0")
        logo_label.pack(side="left")

        # Дополнительные кнопки
        analysis_button = tk.Button(top_panel, text="Анализ", state=tk.DISABLED)
        generation_button = tk.Button(top_panel, text="Генерация", state=tk.DISABLED)
        analysis_button.pack(side="right", padx=10)
        generation_button.pack(side="right", padx=10)

    def create_file_path_section(self):
        """Создает поле для пути к файлу."""
        file_path_frame = tk.Frame(self.root, padx=10, pady=10)
        file_path_frame.pack(fill="x")

        # Поле ввода пути к файлу
        self.file_path_entry = tk.Entry(file_path_frame, width=70)
        self.file_path_entry.insert(0, "Путь к файлу")
        self.file_path_entry.pack(side="left", padx=10)

        # Кнопка выбора файла
        browse_button = tk.Button(file_path_frame, text="Обзор", command=self.browse_file)
        browse_button.pack(side="left", padx=10)

    def browse_file(self):
        """Открывает диалоговое окно для выбора файла."""
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def create_file_settings(self):
        """Создает настройки файла (кодировка и разделитель)."""
        settings_frame = tk.Frame(self.root, padx=10, pady=10)
        settings_frame.pack(fill="x")

        # Выбор кодировки
        encoding_label = tk.Label(settings_frame, text="Кодировка:")
        encoding_label.pack(side="left", padx=10)
        self.encoding_combobox = ttk.Combobox(
            settings_frame,
            values=["utf-8", "cp1251", "ISO-8859-1"],
            width=15
        )
        self.encoding_combobox.set("utf-8")  # Значение по умолчанию
        self.encoding_combobox.pack(side="left", padx=10)

        # Выбор разделителя
        delimiter_label = tk.Label(settings_frame, text="Разделитель:")
        delimiter_label.pack(side="left", padx=10)
        self.delimiter_combobox = ttk.Combobox(
            settings_frame,
            values=[",", ";", "\t", "|"],
            width=15
        )
        self.delimiter_combobox.set(",")  # Значение по умолчанию
        self.delimiter_combobox.pack(side="left", padx=10)

    def validate_file(self):
        """Проверяет файл и показывает предпросмотр."""
        file_path = self.file_path_entry.get().strip()
        encoding = self.encoding_combobox.get()
        delimiter = self.delimiter_combobox.get()
        if not file_path:
            messagebox.showerror("Ошибка", "Укажите путь к файлу.")
            return

        # Устанавливаем параметры чтения в CSVLoader
        self.csv_loader.file_path = file_path
        self.csv_loader.set_reading_parameters(delimiter=delimiter, encoding=encoding)

        # Проверяем структуру файла
        if not self.csv_loader.validate_structure():
            messagebox.showerror("Ошибка", "Структура файла невалидна.")
            return

        # Загружаем данные
        if not self.csv_loader.load_data():
            messagebox.showerror("Ошибка", "Не удалось загрузить данные.")
            return

        # Получаем первые несколько строк для предпросмотра
        df = self.csv_loader.get_active_data()
        if df is not None:
            self.show_preview(df)

        # Активируем кнопку "Импортировать"
        self.import_button.config(state=tk.NORMAL)

    def show_preview(self, df):
        """Отображает модальное окно с предпросмотром данных."""
        if self.preview_window:
            self.preview_window.destroy()

        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Предпросмотр данных")
        self.preview_window.geometry("600x400")

        # Таблица для отображения данных
        tree = ttk.Treeview(self.preview_window)
        tree["columns"] = list(df.columns)
        tree.heading("#0", text="Index")
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Добавляем данные в таблицу
        for index, row in df.iterrows():
            tree.insert("", "end", text=index, values=list(row))

        tree.pack(fill="both", expand=True)

    def import_file(self):
        """Импортирует файл."""
        file_path = self.file_path_entry.get().strip()
        encoding = self.encoding_combobox.get()
        delimiter = self.delimiter_combobox.get()
        try:
            # Загружаем полные данные из файла
            df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
            messagebox.showinfo("Импорт", "Файл успешно импортирован.")
            # Сохраняем данные в глобальное хранилище
            data_storage.data = df
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось импортировать файл: {str(e)}")

    def create_action_buttons(self):
        """Создает кнопки действия."""
        buttons_frame = tk.Frame(self.root, padx=10, pady=10)
        buttons_frame.pack()

        # Кнопка "Проверить файл"
        validate_button = tk.Button(
            buttons_frame,
            text="Проверить файл",
            command=self.validate_file
        )
        validate_button.pack(side="left", padx=10)

        # Кнопка "Импортировать"
        self.import_button = tk.Button(
            buttons_frame,
            text="Импортировать",
            state=tk.DISABLED,
            command=self.import_file
        )
        self.import_button.pack(side="left", padx=10)

        # Кнопка "Перейти к анализу"
        analysis_button = tk.Button(
            buttons_frame,
            text="Перейти к анализу",
            command=self.open_data_analysis_window
        )
        analysis_button.pack(side="left", padx=10)

    def open_data_analysis_window(self):
        """Открывает окно анализа данных."""
        self.root.withdraw()  # Скрываем текущее окно
        analysis_root = tk.Tk()
        app = DataAnalysisWindow(analysis_root)
        app.update_data()  # Обновляем данные в новом окне
        analysis_root.mainloop()


class DataAnalysisWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Дашборд анализа данных")
        self.root.geometry("800x600")
        # Проверяем, есть ли данные в глобальном хранилище
        if data_storage.data is not None:
            self.data = data_storage.data
        else:
            self.data = pd.DataFrame()  # Пустой DataFrame, если данных нет
        # Изначально отображаемые столбцы (например, все столбцы)
        self.displayed_columns = list(self.data.columns)
        # Панель управления
        self.create_top_panel()
        # Табы для переключения между разделами
        self.create_tabs()
        # Раздел "Реестр записей"
        self.create_registry_tab()
        # Раздел "Главный дашборд" (заглушка)
        self.create_dashboard_tab()
        # Загрузка данных в таблицу
        self.load_data_to_table(self.data[self.displayed_columns])

    def create_top_panel(self):
        """Создает панель с логотипом и дополнительными кнопками."""
        top_panel = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=5)
        top_panel.pack(fill="x")

        # Логотип (заменяется словом "ЛОГО")
        logo_label = tk.Label(top_panel, text="ЛОГО", font=("Arial", 14, "bold"), bg="#f0f0f0")
        logo_label.pack(side="left")

        # Дополнительные кнопки
        analysis_button = tk.Button(top_panel, text="Анализ", state=tk.DISABLED)
        generation_button = tk.Button(top_panel, text="Генерация", state=tk.DISABLED)
        analysis_button.pack(side="right", padx=10)
        generation_button.pack(side="right", padx=10)

    def create_tabs(self):
        """Создает табы для переключения между разделами."""
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.pack(fill="both", expand=True)

        # Таб "Реестр записей"
        self.registry_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.registry_tab, text="Реестр записей")

        # Таб "Главный дашборд"
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.dashboard_tab, text="Главный дашборд")

    def create_registry_tab(self):
        """Создает содержимое таба "Реестр записей"."""
        # Рамка для кнопки выбора столбцов
        button_frame = tk.Frame(self.registry_tab)
        button_frame.pack(pady=10)

        # Кнопка "Выбор столбцов"
        select_columns_button = tk.Button(
            button_frame,
            text="Выбор столбцов",
            command=self.open_column_selection_dialog
        )
        select_columns_button.pack(side="left", padx=10)

        # Таблица для отображения данных
        self.table_frame = tk.Frame(self.registry_tab)
        self.table_frame.pack(fill="both", expand=True)

        # Создаем Treeview для таблицы
        self.tree = ttk.Treeview(self.table_frame)
        self.tree.pack(side="left", fill="both", expand=True)

        # Добавляем полосу прокрутки
        yscrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        yscrollbar.pack(side="right", fill="y")
        xscrollbar = ttk.Scrollbar(self.registry_tab, orient="horizontal", command=self.tree.xview)
        xscrollbar.pack(fill="x")

        # Настройка прокрутки
        self.tree.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)

        # Подсказка о количестве строк
        self.row_count_label = tk.Label(self.registry_tab, text="")
        self.row_count_label.pack()

    def create_dashboard_tab(self):
        """Создает содержимое таба "Главный дашборд" (заглушка)."""
        label = tk.Label(self.dashboard_tab, text="В разработке")
        label.pack(pady=50)

    def load_data_to_table(self, data):
        """Загружает данные в таблицу."""
        # Очищаем текущую таблицу
        self.tree.delete(*self.tree.get_children())
        # Устанавливаем колонки
        self.tree["columns"] = list(data.columns)
        self.tree.heading("#0", text="Index")
        for col in data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        # Добавляем данные в таблицу
        for index, row in data.iterrows():
            self.tree.insert("", "end", text=index, values=list(row))
        # Обновляем информацию о количестве строк
        self.update_row_count(len(data))

    def apply_column_selection(self, selected_columns):
        """Применяет выбранные столбцы к таблице."""
        if not selected_columns:
            messagebox.showwarning("Предупреждение", "Не выбраны столбцы.")
            return

        # Фильтруем данные по выбранным столбцам через CSVLoader
        if self.csv_loader.select_columns(selected_columns):
            filtered_data = self.csv_loader.get_active_data()
            self.load_data_to_table(filtered_data)
            self.displayed_columns = selected_columns

    def update_row_count(self, count):
        """Обновляет подсказку о количестве строк."""
        self.row_count_label.config(text=f"Найдено {count} строк")

    def open_column_selection_dialog(self):
        """Открывает модальное окно для выбора столбцов."""
        dialog = ColumnSelectionDialog(
            self.root,
            self.data.columns,
            self.displayed_columns,
            self.apply_column_selection
        )
        self.root.wait_window(dialog.top)

    def update_data(self):
        """Обновляет данные в DataAnalysisWindow из глобального хранилища."""
        if data_storage.data is not None:
            self.data = data_storage.data
            self.displayed_columns = list(self.data.columns)
            self.load_data_to_table(self.data[self.displayed_columns])
        else:
            messagebox.showwarning("Предупреждение", "Нет данных для отображения.")


# Модальное окно для выбора столбцов
class ColumnSelectionDialog:
    def __init__(self, parent, all_columns, initial_selected_columns, on_apply):
        self.top = tk.Toplevel(parent)
        self.top.title("Выбор столбцов")
        self.top.geometry("300x400")

        self.selected_columns = []
        self.on_apply = on_apply

        # Создаем список чекбоксов для столбцов
        self.checkbuttons = []
        for column in all_columns:
            var = tk.IntVar()
            cb = tk.Checkbutton(
                self.top,
                text=column,
                variable=var,
                command=lambda c=column, v=var: self.toggle_column(c, v)
            )
            cb.pack(anchor="w")
            self.checkbuttons.append((cb, var))

            # Если столбец уже выбран, устанавливаем галочку
            if column in initial_selected_columns:
                var.set(1)
                self.selected_columns.append(column)

        # Кнопка "Выбрать"
        apply_button = tk.Button(
            self.top,
            text="Выбрать",
            command=self.apply_selection
        )
        apply_button.pack(pady=10)

    def toggle_column(self, column, var):
        """Добавляет/удаляет столбец при изменении состояния чекбокса."""
        if var.get():
            self.selected_columns.append(column)
        else:
            self.selected_columns.remove(column)

    def apply_selection(self):
        """Применяет выбранные столбцы и закрывает диалог."""
        self.on_apply(self.selected_columns)
        self.top.destroy()

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = FileAnalysisWindow(root)
    root.mainloop()


