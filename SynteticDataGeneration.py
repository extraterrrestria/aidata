import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import re
import sys
import time

# =============== Настройка: ключ OpenRouter API ===============
OPENROUTER_API_KEY = "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"  # Вставьте свой ключ здесь или оставьте пустым для теста без ИИ

# =============== Создание пустого DataFrame с NaN ===============

def create_empty_dataframe(n_rows, n_cols):
    df = pd.DataFrame(np.nan, index=range(n_rows), columns=[f"col_{i}" for i in range(n_cols)])
    return df

# =============== Генерация эрзац-столбца ===============

def generate_ersatz_column(n_rows, name, dtype, mode, params, ai_api_key=None):
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ['true', '1', 'yes']
        return bool(val)

    if mode == 'constant':
        value = params.get('value')
        if dtype == 'int':
            return [int(float(value))] * n_rows
        elif dtype == 'float':
            return [float(value)] * n_rows
        elif dtype == 'boolean':
            bval = to_bool(value)
            return [bval] * n_rows
        elif dtype == 'string':
            return [str(value)] * n_rows
        elif dtype == 'datetime':
            dt = pd.to_datetime(value)
            return [dt] * n_rows
        else:
            raise ValueError(f"Неизвестный тип данных: {dtype}")

    elif mode == 'range':
        start = params.get('start')
        stop = params.get('stop')
        step = params.get('step', 1)
        cycle = params.get('cycle', True)

        if dtype == 'int':
            arr = list(range(int(start), int(stop), int(step)))
        elif dtype == 'float':
            arr = list(np.arange(float(start), float(stop), float(step)))
        elif dtype == 'datetime':
            start_dt = pd.to_datetime(start)
            stop_dt = pd.to_datetime(stop)
            step_days = int(step)
            arr = []
            current = start_dt
            while current < stop_dt:
                arr.append(current)
                current += timedelta(days=step_days)
        elif dtype == 'string':
            arr = [str(x) for x in range(int(start), int(stop), int(step))]
        elif dtype == 'boolean':
            arr = [True, False]
        else:
            raise ValueError(f"Range не поддерживается для типа {dtype}")

        if len(arr) == 0:
            raise ValueError("Диапазон пустой.")

        if len(arr) >= n_rows:
            arr = arr[:n_rows]
        else:
            if cycle:
                repeats = (n_rows // len(arr)) + 1
                arr = (arr * repeats)[:n_rows]
            else:
                arr = arr + [arr[-1]] * (n_rows - len(arr))

        # Строгое приведение типов
        if dtype == 'int':
            arr = [int(float(x)) for x in arr]
        elif dtype == 'float':
            arr = [float(x) for x in arr]
        elif dtype == 'boolean':
            arr = [to_bool(x) for x in arr]
        elif dtype == 'datetime':
            arr = [pd.to_datetime(x) for x in arr]
        elif dtype == 'string':
            arr = [str(x) for x in arr]
        return arr

    elif mode == 'random':
        dist = params.get('dist')
        dist_params = params.get('dist_params', {})

        if dtype == 'boolean':
            p_true = float(dist_params.get('p_true', 0.5))
            return list(np.random.choice([True, False], size=n_rows, p=[p_true, 1-p_true]))
        elif dtype == 'string':
            # Категориальная генерация
            categories = params.get('categories', [])
            proportions = params.get('proportions', [])
            n_per_cat = [int(round(p * n_rows)) for p in proportions]
            # Корректируем, чтобы сумма была ровно n_rows
            diff = n_rows - sum(n_per_cat)
            for i in range(abs(diff)):
                n_per_cat[i % len(n_per_cat)] += 1 if diff > 0 else -1
            result = []
            for cat, count in zip(categories, n_per_cat):
                result.extend([cat] * count)
            np.random.shuffle(result)
            return result
        elif dtype == 'datetime':
            base_date = pd.to_datetime(dist_params.get('base_date', '1970-01-01'))
            if dist == 'normal':
                mean = float(dist_params.get('mean', 0))
                std = float(dist_params.get('std', 1))
                days = np.random.normal(mean, std, n_rows).astype(int)
            elif dist == 'uniform':
                low = float(dist_params.get('low', 0))
                high = float(dist_params.get('high', 1))
                days = np.random.uniform(low, high, n_rows).astype(int)
            elif dist == 'exponential':
                scale = float(dist_params.get('scale', 1))
                days = np.random.exponential(scale, n_rows).astype(int)
            else:
                raise ValueError(f"Неизвестное распределение: {dist}")
            return [base_date + timedelta(days=int(d)) for d in days]
        else:
            if dist == 'normal':
                mean = float(dist_params.get('mean', 0))
                std = float(dist_params.get('std', 1))
                data = np.random.normal(mean, std, n_rows)
            elif dist == 'uniform':
                low = float(dist_params.get('low', 0))
                high = float(dist_params.get('high', 1))
                data = np.random.uniform(low, high, n_rows)
            elif dist == 'exponential':
                scale = float(dist_params.get('scale', 1))
                data = np.random.exponential(scale, n_rows)
            else:
                raise ValueError(f"Неизвестное распределение: {dist}")

            if dtype == 'int':
                return data.round().astype(int).tolist()
            elif dtype == 'float':
                return data.tolist()
            else:
                raise ValueError(f"Неизвестный тип данных: {dtype}")

    elif mode == 'ai':
        if ai_api_key is None:
            raise ValueError("Для генерации через ИИ требуется ai_api_key")

        prompt = params.get('prompt', '')
        if not prompt:
            raise ValueError("Для AI генерации требуется параметр 'prompt'")

        full_prompt = (
            f"Сгенерируй {n_rows} значений столбца '{name}' типа {dtype}. "
            f"Значения должны соответствовать типу и быть в формате JSON-массива. "
            f"Пользовательский промпт: {prompt}"
        )

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {ai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "Ты — генератор синтетических данных."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }

        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                resp_json = response.json()
                content = resp_json["choices"][0]["message"]["content"]

                values = try_parse_json_array(content)
                if not isinstance(values, list):
                    raise ValueError("Ответ ИИ не является списком.")
                if len(values) < n_rows:
                    raise ValueError(f"ИИ вернул {len(values)} значений, ожидалось {n_rows}.")

                # Приведение типов с обработкой ошибок
                casted = []
                errors = 0
                for v in values[:n_rows]:
                    try:
                        if dtype == 'int':
                            casted.append(int(float(v)))
                        elif dtype == 'float':
                            casted.append(float(v))
                        elif dtype == 'boolean':
                            casted.append(to_bool(v))
                        elif dtype == 'datetime':
                            casted.append(pd.to_datetime(v))
                        elif dtype == 'string':
                            casted.append(str(v))
                        else:
                            casted.append(v)
                    except Exception:
                        # Дублируем последнее корректное значение или NaN
                        errors += 1
                        casted.append(casted[-1] if casted else np.nan)
                if errors > n_rows // 5:
                    print(f"Внимание: {errors} значений не удалось привести к типу {dtype}.")
                    fix = input_bool("Переделать генерацию столбца через ИИ? (иначе ошибки будут заменены на дубликаты)", default=True)
                    if fix:
                        raise ValueError("Пользователь выбрал повторить генерацию.")
                return casted

            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"Ошибка при генерации через ИИ: {e}")

    else:
        raise ValueError(f"Неизвестный режим генерации: {mode}")

def try_parse_json_array(text):
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            lines = text.strip().split('\n')
            cleaned = []
            for line in lines:
                line = line.strip().strip('",')
                if line:
                    cleaned.append(line)
            return cleaned
        json_str = match.group(0)
        json_str = json_str.replace("'", '"')
        return json.loads(json_str)
    except Exception:
        return [text.strip()]

# =============== Замена столбца в DataFrame ===============

def replace_column_in_dataframe(df, col_name, ersatz_column):
    if col_name not in df.columns:
        raise KeyError(f"Столбец '{col_name}' отсутствует в датафрейме.")
    if len(ersatz_column) != len(df):
        raise ValueError(f"Длина ersatz_column ({len(ersatz_column)}) не совпадает с числом строк датафрейма ({len(df)}).")
    df.loc[:, col_name] = ersatz_column
    return df

# =============== Вспомогательные функции для интерактивного ввода ===============

def input_int(prompt, default=None):
    while True:
        val = input(prompt).strip()
        if val == '' and default is not None:
            return default
        try:
            return int(val)
        except ValueError:
            print("Ошибка: введите целое число.")

def input_float(prompt, default=None):
    while True:
        val = input(prompt).strip()
        if val == '' and default is not None:
            return default
        try:
            return float(val)
        except ValueError:
            print("Ошибка: введите число.")

def input_str(prompt, default=None):
    val = input(prompt).strip()
    if val == '' and default is not None:
        return default
    return val

def input_bool(prompt, default=None):
    while True:
        val = input(prompt + " (y/n): ").strip().lower()
        if val == '' and default is not None:
            return default
        if val in ['y', 'yes']:
            return True
        elif val in ['n', 'no']:
            return False
        else:
            print("Ошибка: введите 'y' или 'n'.")

# =============== Главная функция взаимодействия с пользователем ===============

def interactive_synthetic_data_generator():
    print("=== Генератор синтетических данных ===")

    n_rows = input_int("Введите количество строк (n_rows): ")
    n_cols = input_int("Введите количество столбцов (n_cols): ")

    df = create_empty_dataframe(n_rows, n_cols)
    print(f"Создан пустой DataFrame размером {n_rows}x{n_cols} с NaN.")

    ai_api_key = OPENROUTER_API_KEY if OPENROUTER_API_KEY else None

    dtype_options = ['int', 'float', 'string', 'boolean', 'datetime']

    for col_idx in range(n_cols):
        col_name = input_str(f"\nВведите название столбца #{col_idx+1} (текущий: {df.columns[col_idx]}): ", default=df.columns[col_idx])
        # Выбор типа данных
        print("Выберите тип данных:")
        for i, dt in enumerate(dtype_options, 1):
            print(f"{i} - {dt}")
        dtype_num = 0
        while dtype_num not in range(1, len(dtype_options)+1):
            dtype_num = input_int("Введите номер типа данных: ")
        dtype = dtype_options[dtype_num-1]

        # Доступные режимы генерации для типа
        mode_options = []
        mode_map = {}
        print("Выберите способ генерации:")
        mode_idx = 1

        # constant всегда доступен
        print(f"{mode_idx} - Константное значение")
        mode_map[mode_idx] = 'constant'
        mode_idx += 1

        # range для всех кроме boolean
        if dtype in ['int', 'float', 'datetime', 'string']:
            print(f"{mode_idx} - Диапазон (range)")
            mode_map[mode_idx] = 'range'
            mode_idx += 1

        # random для int, float, datetime, boolean, string (категории)
        if dtype in ['int', 'float', 'datetime', 'boolean', 'string']:
            print(f"{mode_idx} - Случайные значения (random)")
            mode_map[mode_idx] = 'random'
            mode_idx += 1

        # ai всегда доступен если есть ключ
        if ai_api_key:
            print(f"{mode_idx} - Генерация через ИИ (ai)")
            mode_map[mode_idx] = 'ai'
            mode_idx += 1

        mode_num = 0
        while mode_num not in mode_map:
            mode_num = input_int("Введите номер способа генерации: ")
        mode = mode_map[mode_num]

        params = {}

        if mode == 'constant':
            val = input_str("Введите фиксированное значение для столбца: ")
            params['value'] = val

        elif mode == 'range':
            start = input_str("Введите начало диапазона (start): ")
            stop = input_str("Введите конец диапазона (stop): ")
            step = input_int("Введите шаг (step, целое число, по умолчанию 1): ", default=1)
            cycle = input_bool("Циклически повторять значения, если не хватает?", default=True)
            params.update({'start': start, 'stop': stop, 'step': step, 'cycle': cycle})

        elif mode == 'random':
            if dtype in ['int', 'float', 'datetime']:
                print("Выберите распределение:")
                dist_options = ['normal', 'uniform', 'exponential']
                for i, d in enumerate(dist_options, 1):
                    print(f"{i} - {d}")
                dist_num = 0
                while dist_num not in range(1, len(dist_options)+1):
                    dist_num = input_int("Введите номер распределения: ")
                dist = dist_options[dist_num-1]
                dist_params = {}
                if dist == 'normal':
                    dist_params['mean'] = input_float("Введите математическое ожидание (mean): ")
                    dist_params['std'] = input_float("Введите стандартное отклонение (std): ")
                elif dist == 'uniform':
                    dist_params['low'] = input_float("Введите минимум (low): ")
                    dist_params['high'] = input_float("Введите максимум (high): ")
                elif dist == 'exponential':
                    dist_params['scale'] = input_float("Введите параметр scale: ")
                if dtype == 'datetime':
                    base_date = input_str("Введите базовую дату (base_date, например 1970-01-01): ", default='1970-01-01')
                    dist_params['base_date'] = base_date
                params['dist'] = dist
                params['dist_params'] = dist_params
            elif dtype == 'boolean':
                p_true = input_float("Введите вероятность True (от 0 до 1, по умолчанию 0.5): ", default=0.5)
                params['dist_params'] = {'p_true': p_true}
            elif dtype == 'string':
                n_cat = input_int("Введите количество категорий (от 2 до 5): ", default=2)
                n_cat = max(2, min(5, n_cat))
                categories = []
                proportions = []
                total = 0.0
                for i in range(n_cat):
                    cat = input_str(f"Введите название категории #{i+1}: ")
                    if i == n_cat-1:
                        prop = 1.0 - sum(proportions)
                        print(f"Доля категории #{i+1} автоматически установлена как {prop:.2f}")
                    else:
                        remaining = 1.0 - sum(proportions)
                        prop = input_float(f"Введите долю для категории #{i+1} (оставшееся {remaining:.2f}): ")
                        prop = max(0.0, min(remaining, prop))
                    categories.append(cat)
                    proportions.append(prop)
                params['categories'] = categories
                params['proportions'] = proportions

        elif mode == 'ai':
            prompt = input_str("Введите промпт для генерации через ИИ: ")
            params['prompt'] = prompt

        while True:
            try:
                ersatz_col = generate_ersatz_column(n_rows, col_name, dtype, mode, params, ai_api_key)
                break
            except Exception as e:
                print(f"Ошибка генерации столбца: {e}")
                retry = input_bool("Повторить ввод параметров для этого столбца?", default=True)
                if retry:
                    continue
                else:
                    print(f"Пропускаем столбец '{col_name}'.")
                    ersatz_col = [np.nan] * n_rows
                    break

        print(f"\nПредпросмотр первых 5 значений столбца '{col_name}':")
        for i, val in enumerate(ersatz_col[:5]):
            print(f"  [{i}]: {val}")

        accept = input_bool("Принять этот столбец?", default=True)
        if accept:
            df = replace_column_in_dataframe(df, df.columns[col_idx], ersatz_col)
            if df.columns[col_idx] != col_name:
                df.rename(columns={df.columns[col_idx]: col_name}, inplace=True)
        else:
            print("Повторите ввод параметров для этого столбца.")
            col_idx -= 1

    print("\nГенерация завершена. Итоговый DataFrame:")
    print(df.head())
    return df

# =============== Запуск ===============

if __name__ == "__main__":
    try:
        df_result = interactive_synthetic_data_generator()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(0)
