import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_generation.log"),  # Логи сохраняются в файл
        logging.StreamHandler()  # Логи выводятся в консоль
    ]
)