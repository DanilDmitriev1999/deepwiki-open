"""
Конфигурация для оптимальной работы с vLLM.
"""

import os
from typing import Dict, Any

# Настройки по умолчанию для vLLM
VLLM_DEFAULT_CONFIG = {
    "vllm_compatible": True,
    "force_streaming": False,
    "default_model_kwargs": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        # Убираем проблемные параметры для vLLM
        "stream": False,  # По умолчанию не используем streaming
    }
}

# Параметры, несовместимые с vLLM
VLLM_INCOMPATIBLE_PARAMS = [
    'logprobs',
    'top_logprobs', 
    'response_format',
    'tools',
    'tool_choice',
    'function_call',
    'functions',
    'seed',
    'logit_bias',
    'user',
    'presence_penalty',
    'frequency_penalty',
    'n',  # vLLM может не поддерживать множественные выборы
]

def get_vllm_optimized_kwargs(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Оптимизирует параметры модели для работы с vLLM.
    
    Args:
        model_kwargs: Исходные параметры модели
        
    Returns:
        Оптимизированные параметры для vLLM
    """
    optimized_kwargs = model_kwargs.copy()
    
    # Удаляем несовместимые параметры
    for param in VLLM_INCOMPATIBLE_PARAMS:
        optimized_kwargs.pop(param, None)
    
    # Добавляем дефолтные значения если не указаны
    for key, value in VLLM_DEFAULT_CONFIG["default_model_kwargs"].items():
        if key not in optimized_kwargs:
            optimized_kwargs[key] = value
    
    return optimized_kwargs

def create_vllm_client():
    """
    Создает оптимизированный для vLLM клиент OpenAI.
    
    Returns:
        Настроенный OpenAIClient
    """
    from .openai_client import OpenAIClient
    
    return OpenAIClient(
        vllm_compatible=VLLM_DEFAULT_CONFIG["vllm_compatible"],
        force_streaming=VLLM_DEFAULT_CONFIG["force_streaming"]
    )

def log_vllm_request(api_kwargs: Dict[str, Any]):
    """
    Логирует запрос к vLLM для отладки.
    
    Args:
        api_kwargs: Параметры API запроса
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("vLLM request parameters:")
    for key, value in api_kwargs.items():
        if key == "messages":
            logger.info(f"  {key}: {len(value)} messages")
            # Логируем общую длину сообщений для отладки max_tokens
            total_length = sum(len(str(msg.get('content', ''))) for msg in value)
            logger.info(f"  total_message_content_length: {total_length} characters")
        elif key == "max_tokens":
            logger.info(f"  {key}: {value} (type: {type(value).__name__})")
            # Предупреждение если max_tokens отрицательный
            if isinstance(value, (int, float)) and value < 1:
                logger.error(f"CRITICAL: max_tokens is invalid: {value}")
        else:
            logger.info(f"  {key}: {value}") 