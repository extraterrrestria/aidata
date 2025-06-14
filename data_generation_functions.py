import pandas as pd
from logging_logic import logging
import numpy as np
import random
from datetime import datetime, timedelta
import requests
# Ключ API OpenRouter
OPENROUTER_API_KEY = "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"

def create_empty_dataframe():
    """Создает пустой датафрейм размером n x m."""
    while True:
        try:
            n = input("Введите количество строк (n): ")
            if not n.isdigit() or int(n) < 1:
                print("Количество строк должно быть целым числом больше 0. Попробуйте снова.")
                continue
            n = int(n)
            
            m = input("Введите количество столбцов (m): ")
            if not m.isdigit() or int(m) < 1:
                print("Количество столбцов должно быть целым числом больше 0. Попробуйте снова.")
                continue
            m = int(m)
            break
        except ValueError:
            print("Введите корректное целое число.")
    
    # Создаем пустой датафрейм с именами столбцов col_0, col_1, ..., col_m-1
    columns = [f"col_{i}" for i in range(m)]
    df = pd.DataFrame(index=range(n), columns=columns)
    logging.info("Создан пустой датафрейм.")
    print("\nСгенерированный пустой датафрейм:")
    print(df)
    return df

def generate_column(df):
    """Генерация данных для столбца."""
    while True:
        try:
            column_index = input(f"Введите номер столбца для генерации (0-{len(df.columns) - 1}): ")
            if not column_index.isdigit() or int(column_index) < 0 or int(column_index) >= len(df.columns):
                logging.error(f"Некорректный номер столбца: {column_index}")
                print("Некорректный номер столбца. Попробуйте снова.")
                continue
            column_index = int(column_index)
            break
        except ValueError:
            logging.error("Ошибка: Введено некорректное значение для номера столбца.")
            print("Введите корректное целое число.")
    
    column_name = df.columns[column_index]
    while True:
        column_type = input(f"Введите тип данных для столбца '{column_name}' (int, float, string, data, bool): ").strip().lower()
        if column_type not in ['int', 'float', 'string', 'data', 'bool']:
            logging.error(f"Некорректный тип данных: {column_type}")
            print("Некорректный тип данных. Попробуйте снова.")
            continue
        break
    
    print(f"Выберите способ генерации данных для столбца '{column_name}':")
    print("1. Конкретные значения")
    print("2. Циклическая генерация (только для int, float, data)")
    print("3. Случайная генерация (только для int, float, data)")
    print("4. Генерация через ИИ (Deepseek)")
    
    while True:
        choice = input("Введите номер выбранного способа: ").strip()
        if choice not in ['1', '2', '3', '4']:
            logging.error(f"Некорректный выбор способа генерации: {choice}")
            print("Некорректный выбор. Попробуйте снова.")
            continue
        break
    
    generated_values = None
    while True:
        if choice == '1':
            values = input(f"Введите значения для столбца '{column_name}' через запятую: ").split(',')
            generated_values = [convert_value(v.strip(), column_type) for v in values * (len(df) // len(values) + 1)][:len(df)]
        elif choice == '2' and column_type in ['int', 'float', 'data']:
            if column_type == 'data':
                while True:
                    start_date = input("Введите начальную дату (YYYY-MM-DD): ")
                    end_date = input("Введите конечную дату (YYYY-MM-DD): ")
                    try:
                        pd.to_datetime(start_date)
                        pd.to_datetime(end_date)
                        break
                    except ValueError:
                        logging.error(f"Некорректный формат даты: start_date={start_date}, end_date={end_date}")
                        print("Некорректный формат даты. Попробуйте снова.")
                step_days = input("Введите шаг в днях: ")
                if not step_days.isdigit() or int(step_days) <= 0:
                    logging.error(f"Некорректный шаг в днях: {step_days}")
                    print("Шаг должен быть целым числом больше 0. Попробуйте снова.")
                    return df
                step_days = int(step_days)
                dates = pd.date_range(start=start_date, end=end_date, freq=f'{step_days}D')
                generated_values = [dates[i % len(dates)] for i in range(len(df))]
            else:
                while True:
                    start = input("Введите начало диапазона: ")
                    end = input("Введите конец диапазона: ")
                    step = input("Введите шаг: ")
                    try:
                        start = float(start)
                        end = float(end)
                        step = float(step)
                        if step <= 0:
                            logging.error(f"Некорректный шаг: {step}")
                            print("Шаг должен быть положительным числом. Попробуйте снова.")
                            continue
                        break
                    except ValueError:
                        logging.error(f"Некорректный формат числа: start={start}, end={end}, step={step}")
                        print("Некорректный формат числа. Попробуйте снова.")
                values = np.arange(start, end + step, step)
                generated_values = [values[i % len(values)] for i in range(len(df))]
        elif choice == '3' and column_type in ['int', 'float', 'data']:
            distribution = input("Выберите распределение (uniform, normal, exponential): ").strip().lower()
            if distribution not in ['uniform', 'normal', 'exponential']:
                logging.error(f"Некорректный выбор распределения: {distribution}")
                print("Некорректный выбор распределения. Попробуйте снова.")
                return df
            
            if distribution == 'uniform':
                while True:
                    low = input("Введите нижнюю границу: ")
                    high = input("Введите верхнюю границу: ")
                    try:
                        low = float(low)
                        high = float(high)
                        if low >= high:
                            logging.error(f"Нижняя граница должна быть меньше верхней: low={low}, high={high}")
                            print("Нижняя граница должна быть меньше верхней. Попробуйте снова.")
                            continue
                        break
                    except ValueError:
                        logging.error(f"Некорректный формат числа: low={low}, high={high}")
                        print("Некорректный формат числа. Попробуйте снова.")
                if column_type == 'data':
                    start_date = datetime.fromtimestamp(low).date()
                    end_date = datetime.fromtimestamp(high).date()
                    generated_values = [start_date + timedelta(days=random.uniform(0, (end_date - start_date).days)) for _ in range(len(df))]
                else:
                    generated_values = np.random.uniform(low, high, len(df))
            elif distribution == 'normal':
                while True:
                    mean = input("Введите математическое ожидание: ")
                    std_dev = input("Введите стандартное отклонение: ")
                    try:
                        mean = float(mean)
                        std_dev = float(std_dev)
                        if std_dev <= 0:
                            logging.error(f"Стандартное отклонение должно быть положительным числом: std_dev={std_dev}")
                            print("Стандартное отклонение должно быть положительным числом. Попробуйте снова.")
                            continue
                        break
                    except ValueError:
                        logging.error(f"Некорректный формат числа: mean={mean}, std_dev={std_dev}")
                        print("Некорректный формат числа. Попробуйте снова.")
                if column_type == 'data':
                    base_date = datetime.fromtimestamp(mean).date()
                    generated_values = [base_date + timedelta(days=int(random.gauss(0, std_dev))) for _ in range(len(df))]
                else:
                    generated_values = np.random.normal(mean, std_dev, len(df))
            elif distribution == 'exponential':
                while True:
                    scale = input("Введите параметр лямбда (в днях для дат): ")
                    try:
                        scale = float(scale)
                        if scale <= 0:
                            logging.error(f"Лямбда должна быть положительным числом: scale={scale}")
                            print("Лямбда должна быть положительным числом. Попробуйте снова.")
                            continue
                        break
                    except ValueError:
                        logging.error(f"Некорректный формат числа: scale={scale}")
                        print("Некорректный формат числа. Попробуйте снова.")
                if column_type == 'data':
                    zero_date = datetime.now().date()
                    generated_values = [zero_date + timedelta(days=int(random.expovariate(1 / scale))) for _ in range(len(df))]
                else:
                    generated_values = np.random.exponential(scale, len(df))
        elif choice == '4':
            prompt = input(f"Введите промт для генерации данных для столбца '{column_name}': ")
            generated_values = generate_with_ai(prompt, len(df), column_type)
        
        # Приведение данных к указанному типу
        generated_values = [convert_value(v, column_type) for v in generated_values]
        
        # Вывод сгенерированных данных
        print("\nСгенерированные данные:")
        print(generated_values)
        
        # Запрос на подтверждение
        confirm = input("Оставляем данные или генерируем заново? (да/нет): ").strip().lower()
        if confirm == 'да':
            df[column_name] = generated_values
            logging.info(f"Данные добавлены в столбец '{column_name}'.")
            print(f"Данные добавлены в столбец '{column_name}'.")
            break
        elif confirm == 'нет':
            logging.info("Пользователь запросил повторную генерацию данных.")
            print("Генерация данных будет повторена.")
        else:
            logging.warning("Некорректный ответ на подтверждение данных.")
            print("Некорректный ответ. Генерация данных будет повторена.")
    
    return df

def convert_value(value, column_type):
    """Преобразует значение в указанный тип."""
    try:
        if column_type == 'int':
            return int(float(value)) if isinstance(value, (str, float)) else int(value)
        elif column_type == 'float':
            return float(value)
        elif column_type == 'string':
            return str(value)
        elif column_type == 'data':
            return pd.to_datetime(value)
        elif column_type == 'bool':
            return value.lower() in ['true', '1', 'yes'] if isinstance(value, str) else bool(value)
        else:
            return value
    except ValueError as e:
        logging.error(f"Ошибка при преобразовании значения '{value}' в тип {column_type}: {e}")
        print(f"Ошибка при преобразовании значения '{value}' в тип {column_type}: {e}")
        return None

def generate_with_ai(prompt, n, column_type):
    """Генерирует данные через ИИ."""
    url = "https://openrouter.ai/api/v1/chat/completions" 
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [{
        "role": "user",
        "content": (
            f"{prompt}. "
            f"Данные должны быть сгенерированы в виде {n} чисел через запятую."
            f"Не выдавай ничего, кроме этих данных. Тип данных должен быть {column_type}."
        )
    }]
    response = requests.post(url, headers=headers, json={"model": "deepseek/deepseek-chat", "messages": messages})
    
    if response.status_code == 200:
        result = response.json()
        generated_text = result['choices'][0]['message']['content']
        try:
            raw_values = [v.strip() for v in generated_text.split(',') if v.strip()]
            if len(raw_values) < n:
                logging.warning(f"ИИ вернул только {len(raw_values)} значений вместо {n}. Дополнительные значения будут заполнены None.")
                print(f"Внимание: ИИ вернул только {len(raw_values)} значений вместо {n}. Дополнительные значения будут заполнены None.")
                raw_values.extend([None] * (n - len(raw_values)))
            values = []
            for v in raw_values[:n]:
                try:
                    converted_value = convert_value(v, column_type)
                    values.append(converted_value)
                except Exception as e:
                    logging.error(f"Ошибка при преобразовании значения '{v}' в тип {column_type}: {e}")
                    print(f"Ошибка при преобразовании значения '{v}' в тип {column_type}: {e}")
                    values.append(None)
            return values
        except Exception as e:
            logging.error(f"Ошибка при обработке данных от ИИ: {e}")
            print(f"Ошибка при обработке данных от ИИ: {e}")
            return [None] * n
    else:
        logging.error(f"Ошибка при генерации данных через ИИ: status_code={response.status_code}")
        print("Ошибка при генерации данных через ИИ.")
        return [None] * n

def rename_column(df):
    """Переименовывает столбец."""
    while True:
        column_index = input(f"Введите номер столбца для переименования (0-{len(df.columns) - 1}): ")
        if not column_index.isdigit() or int(column_index) < 0 or int(column_index) >= len(df.columns):
            logging.error(f"Некорректный номер столбца: {column_index}")
            print("Некорректный номер столбца. Попробуйте снова.")
            continue
        column_index = int(column_index)
        break
    old_name = df.columns[column_index]
    new_name = input(f"Введите новое имя для столбца '{old_name}': ").strip()
    if not new_name:
        logging.error("Имя столбца не может быть пустым.")
        print("Имя столбца не может быть пустым. Попробуйте снова.")
        return df
    df.rename(columns={old_name: new_name}, inplace=True)
    logging.info(f"Столбец '{old_name}' переименован в '{new_name}'.")
    print(f"Столбец '{old_name}' переименован в '{new_name}'.")
    return df

def add_column(df):
    """Добавляет новый столбец."""
    while True:
        column_name = input("Введите имя нового столбца: ").strip()
        if not column_name:
            logging.error("Имя столбца не может быть пустым.")
            print("Имя столбца не может быть пустым. Попробуйте снова.")
            continue
        break
    while True:
        column_type = input(f"Введите тип данных для столбца '{column_name}' (int, float, string, data, bool): ").strip().lower()
        if column_type not in ['int', 'float', 'string', 'data', 'bool']:
            logging.error(f"Некорректный тип данных: {column_type}")
            print("Некорректный тип данных. Попробуйте снова.")
            continue
        break
    default_value = input(f"Введите значение по умолчанию для столбца '{column_name}' (или оставьте пустым для NaN): ").strip()
    if not default_value:
        default_value = np.nan
    else:
        default_value = convert_value(default_value, column_type)
    df[column_name] = [default_value] * len(df)
    logging.info(f"Добавлен новый столбец '{column_name}' с типом данных {column_type}.")
    print(f"Столбец '{column_name}' добавлен.")
    return df

def delete_column(df):
    """Удаляет столбец."""
    while True:
        column_index = input(f"Введите номер столбца для удаления (0-{len(df.columns) - 1}): ")
        if not column_index.isdigit() or int(column_index) < 0 or int(column_index) >= len(df.columns):
            logging.error(f"Некорректный номер столбца: {column_index}")
            print("Некорректный номер столбца. Попробуйте снова.")
            continue
        column_index = int(column_index)
        break
    column_name = df.columns[column_index]
    df = df.drop(columns=[column_name])
    logging.info(f"Столбец '{column_name}' удален.")
    print(f"Столбец '{column_name}' удален.")
    return df