#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы с vLLM.
Запустите этот скрипт для проверки совместимости.
"""

import os
import sys
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь к API
sys.path.append('./api')

def test_vllm_connection():
    """Тестирует соединение с vLLM сервером."""
    try:
        from api.openai_client import OpenAIClient
        from adalflow.core.types import ModelType
        
        # Создаем клиент с vLLM совместимостью
        client = OpenAIClient(
            vllm_compatible=True,
            force_streaming=False
        )
        
        # Тестовые параметры
        test_kwargs = {
            "model": "mistral-24B",  # Замените на имя вашей модели
            "messages": [
                {"role": "user", "content": "Привет! Как дела?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        # Преобразуем параметры через клиент
        api_kwargs = client.convert_inputs_to_api_kwargs(
            input="Привет! Как дела?",
            model_kwargs=test_kwargs,
            model_type=ModelType.LLM
        )
        
        logger.info("Параметры запроса после обработки:")
        for key, value in api_kwargs.items():
            if key == "messages":
                logger.info(f"  {key}: {len(value)} сообщений")
            else:
                logger.info(f"  {key}: {value}")
        
        # Попытка вызова
        logger.info("Отправляем запрос к vLLM...")
        response = client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        logger.info("Успешно получен ответ от vLLM!")
        logger.info(f"Тип ответа: {type(response)}")
        
        # Парсим ответ
        parsed_response = client.parse_chat_completion(response)
        logger.info(f"Обработанный ответ: {parsed_response.raw_response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании vLLM: {str(e)}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        return False

def test_simple_request():
    """Тестирует простой HTTP запрос к vLLM."""
    try:
        import requests
        
        url = "http://10.138.16.219:8666/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-key"  # vLLM может требовать фиктивный ключ
        }
        
        data = {
            "model": "mistral-24B",  # Используем правильное имя модели
            "messages": [
                {"role": "user", "content": "Привет!"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        logger.info("Отправляем прямой HTTP запрос к vLLM...")
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        logger.info(f"Статус ответа: {response.status_code}")
        logger.info(f"Заголовки ответа: {dict(response.headers)}")
        
        if response.status_code == 200:
            logger.info("Успешный ответ от vLLM!")
            result = response.json()
            logger.info(f"Содержимое ответа: {result}")
            return True
        else:
            logger.error(f"Ошибка {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка при прямом запросе: {str(e)}")
        return False

def main():
    """Основная функция тестирования."""
    logger.info("Запуск тестов для vLLM...")
    
    # Проверяем переменные окружения
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY не установлен. Используем фиктивный ключ.")
        os.environ["OPENAI_API_KEY"] = "dummy-key"
    
    logger.info("=== Тест 1: Прямой HTTP запрос ===")
    test1_result = test_simple_request()
    
    logger.info("=== Тест 2: Через OpenAIClient ===")
    test2_result = test_vllm_connection()
    
    logger.info("=== Результаты тестирования ===")
    logger.info(f"Прямой HTTP запрос: {'✓ ПРОШЕЛ' if test1_result else '✗ НЕ ПРОШЕЛ'}")
    logger.info(f"Через OpenAIClient: {'✓ ПРОШЕЛ' if test2_result else '✗ НЕ ПРОШЕЛ'}")
    
    if test1_result and test2_result:
        logger.info("🎉 Все тесты прошли успешно!")
        return 0
    else:
        logger.error("❌ Некоторые тесты не прошли. Проверьте конфигурацию vLLM.")
        return 1

if __name__ == "__main__":
    exit(main()) 