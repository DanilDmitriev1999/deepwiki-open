# DeepWiki-Open

[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

export OLLAMA_HOST=http://localhost:11434

**DeepWiki-Open** — это генератор документации с открытым исходным кодом на основе ИИ, который создает полные интерактивные wiki для любого репозитория GitHub, GitLab, или Bitbucket. Превратите сложные кодовые базы в красивую, навигационную документацию всего за несколько кликов!

## ✨ Основные функции

- **🤖 ИИ-анализ**: Автоматически анализирует и понимает структуру вашего кода
- **📖 Интерактивная Wiki**: Создает полноценную wiki с навигацией и связанными страницами
- **🔍 Умный поиск**: Находит и связывает связанные файлы и концепции
- **💬 Чат с репозиторием**: Задавайте вопросы о коде и получайте ответы на основе контекста
- **📊 Визуальные диаграммы**: Автоматически генерирует диаграммы Mermaid для архитектуры
- **🌐 Поддержка нескольких платформ**: Работает с GitHub, GitLab, Bitbucket и локальными репозиториями
- **🔒 Приватные репозитории**: Поддерживает приватные репозитории с токенами доступа
- **🌍 Многоязычность**: Создает документацию на разных языках
- **🔄 DeepResearch**: Многоступенчатый процесс глубокого исследования для комплексного анализа тем
- **🎯 Поддержка множества моделей**: Поддерживает Google Gemini, OpenAI, OpenRouter и локальные модели Ollama
- **🔗 Локальные ссылки**: Поддержка file:// URL и символических ссылок для локальных репозиториев

## 🚀 Быстрый старт (Супер просто :))

### Вариант 1: Использование Docker

```bash
# Клонировать репозиторий
git clone https://github.com/AsyncFuncAI/deepwiki-open.git
cd deepwiki-open

# Создать файл .env с API ключами
echo "GOOGLE_API_KEY=your_google_api_key" > .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env
# Опционально: Добавить OpenRouter API ключ для моделей OpenRouter
echo "OPENROUTER_API_KEY=your_openrouter_api_key" >> .env

# Запустить с Docker Compose
docker-compose up
```

> 💡 **Как получить ключи:**
> - Получить Google API ключ из [Google AI Studio](https://makersuite.google.com/app/apikey)
> - Получить OpenAI API ключ из [OpenAI Platform](https://platform.openai.com/api-keys)

### Вариант 2: Ручная настройка (Рекомендуется)

#### Шаг 1: Настройка API ключей

Создайте файл `.env` в корневой папке проекта с вашими API ключами:

```
GOOGLE_API_KEY=ваш_google_api_ключ
OPENAI_API_KEY=ваш_openai_api_ключ
# Опционально: Добавить OpenRouter API ключ для моделей OpenRouter
OPENROUTER_API_KEY=ваш_openrouter_api_ключ
```

#### Шаг 2: Запуск Backend

```bash
# Установить Python зависимости
pip install -r api/requirements.txt

# Запустить API сервер
python -m api.main
```

#### Шаг 3: Запуск Frontend

```bash
# Установить JavaScript зависимости
npm install
# или
yarn install

# Запустить веб-приложение
npm run dev
# или  
yarn dev
```

#### Шаг 4: Использовать DeepWiki!

1. Откройте [http://localhost:3000](http://localhost:3000) в браузере
2. Введите репозиторий в одном из форматов:
   - **GitHub/GitLab/Bitbucket URL**: `https://github.com/openai/codex`, `https://gitlab.com/gitlab-org/gitlab`
   - **Локальный путь**: `/путь/к/локальному/репозиторию` или `C:\путь\к\репозиторию`
   - **File URL**: `file:///абсолютный/путь/к/репозиторию`
3. Для приватных репозиториев нажмите "+ Добавить токены доступа" и введите ваш GitHub или GitLab персональный токен доступа
4. Нажмите "Создать Wiki" и наслаждайтесь магией!

## 🔗 Поддержка локальных репозиториев

DeepWiki-Open теперь поддерживает несколько способов работы с локальными репозиториями:

### 1. Прямые локальные пути
```
/home/user/projects/my-repo
C:\Projects\my-repo
```

### 2. File:// URL (Новая функция!)
```
file:///home/user/projects/my-repo
file:///C:/Projects/my-repo
```

### 3. Символические ссылки
DeepWiki автоматически создает символические ссылки на ваши локальные репозитории в `~/.adalflow/repos/`, что позволяет:
- Избежать дублирования данных
- Автоматически отслеживать изменения в исходном репозитории
- Оптимизировать использование дискового пространства

### Как это работает:
1. При использовании file:// URL или абсолютного пути, DeepWiki проверяет существование директории
2. Создает символическую ссылку в `~/.adalflow/repos/` (или копирует, если символические ссылки не поддерживаются)
3. Индексирует содержимое для создания документации

## 🔍 Как работает DeepWiki

DeepWiki использует ИИ для:

1. Клонирования и анализа репозиториев GitHub, GitLab, Bitbucket или локальных папок
2. Создания эмбеддингов для кода (поддержка RAG)
3. Генерации документации с помощью контекстно-зависимого ИИ (используя Google Gemini, OpenAI, OpenRouter или локальные модели Ollama)
4. Создания диаграмм для объяснения взаимосвязей в коде
5. Организации информации в wiki
6. Обеспечения Q&A с репозиторием
7. Предоставления возможностей DeepResearch

## 🛠️ Структура проекта

```
deepwiki/
├── api/                  # Backend API сервер
│   ├── config/          # Конфигурационные файлы
│   ├── tools/           # Утилиты и инструменты
│   └── main.py          # Точка входа API
├── src/                 # Frontend Next.js приложение
│   ├── app/            # App Router страницы
│   ├── components/     # React компоненты
│   ├── messages/       # Файлы локализации (включая ru.json)
│   └── utils/          # Утилиты
└── docs/               # Документация
```

## 🌍 Поддерживаемые языки

DeepWiki поддерживает создание документации на следующих языках:
- 🇺🇸 English
- 🇷🇺 Русский (Russian)
- 🇯🇵 日本語 (Japanese)
- 🇨🇳 中文 (Chinese)
- 🇹🇼 繁體中文 (Traditional Chinese)
- 🇪🇸 Español (Spanish)
- 🇰🇷 한국어 (Korean)
- 🇻🇳 Tiếng Việt (Vietnamese)
- 🇧🇷 Português Brasileiro (Brazilian Portuguese)

## 🤖 Система выбора модели на основе провайдера

DeepWiki реализует гибкую систему выбора модели на основе нескольких провайдеров LLM:

### Поддерживаемые провайдеры и модели

- **Google**: По умолчанию `gemini-2.0-flash`, также поддерживает `gemini-1.5-flash`, `gemini-1.0-pro` и др.
- **OpenAI**: По умолчанию `gpt-4o`, также поддерживает `o4-mini` и др.
- **OpenRouter**: Доступ к множеству моделей через единый API, включая Claude, Llama, Mistral и др.
- **Ollama**: Поддержка локальных открытых моделей типа `llama3`

## 📚 Примеры использования

### Создание документации для GitHub проекта
```bash
# В веб-интерфейсе введите:
https://github.com/microsoft/typescript
```

### Создание документации для локального проекта
```bash
# Вариант 1: Прямой путь
/home/user/projects/my-awesome-project

# Вариант 2: File URL  
file:///home/user/projects/my-awesome-project

# Вариант 3: Windows путь
C:\Projects\my-awesome-project
```

### Создание документации для приватного репозитория
1. Получите персональный токен доступа от GitHub/GitLab
2. Нажмите "+ Добавить токены доступа"
3. Выберите платформу и введите токен
4. Введите URL приватного репозитория

## 🔧 API

API сервер предоставляет:
- Клонирование и индексирование репозиториев
- RAG (Retrieval Augmented Generation)
- Потоковый чат

Подробности смотрите в [API README](./api/README.md).

## 🤝 Вклад в развитие

Мы приветствуем вклад в развитие проекта! Пожалуйста:

1. Сделайте форк репозитория
2. Создайте ветку для функции (`git checkout -b feature/AmazingFeature`)
3. Закоммитьте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект лицензирован под лицензией MIT - смотрите файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- Базируется на фреймворке [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)
- UI компоненты от [shadcn/ui](https://ui.shadcn.com/)
- Иконки от [React Icons](https://react-icons.github.io/react-icons/)
- Диаграммы генерируются с помощью [Mermaid](https://mermaid.js.org/)

---

**Превратите свой код в красивую документацию с DeepWiki-Open! 🚀** 