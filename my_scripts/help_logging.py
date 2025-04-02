import os
import logging
import torch

def setup_logging():
    """Настройка раздельного логирования"""
    # Очищаем логи перед началом
    clear_log_files()
    
    # Логгер для loss (просто числа, одно на строку)
    loss_logger = logging.getLogger('loss_logger')
    loss_logger.setLevel(logging.INFO)
    loss_handler = logging.FileHandler('loss_values.log')
    loss_handler.setFormatter(logging.Formatter('%(message)s'))
    loss_logger.addHandler(loss_handler)
    loss_logger.propagate = False
    
    # Логгер для памяти (пиковое использование на каждом шаге)
    memory_logger = logging.getLogger('memory_logger')
    memory_logger.setLevel(logging.INFO)
    memory_handler = logging.FileHandler('memory_usage.log')
    memory_handler.setFormatter(logging.Formatter('%(message)s'))
    memory_logger.addHandler(memory_handler)
    memory_logger.propagate = False
    
    # Основной логгер для остальной информации
    main_logger = logging.getLogger('main_logger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.FileHandler('training.log')
    main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)
    
    # Консольный логгер
    console_logger = logging.getLogger('console_logger')
    console_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    console_logger.addHandler(console_handler)
    console_logger.propagate = False
    
    return loss_logger, memory_logger, main_logger, console_logger

def clear_log_files():
    """Очистка файлов логов перед началом работы"""
    log_files = ['loss_values.log', 'memory_usage.log', 'training.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')

def format_time(seconds):
    """Форматирование времени в читаемый вид"""
    return str(timedelta(seconds=int(seconds)))

def get_memory_stats():
    """Получение статистики по использованию памяти"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return allocated, reserved, peak

