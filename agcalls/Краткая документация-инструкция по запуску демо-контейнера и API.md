### Краткая документация-инструкция по запуску демо-контейнера и API

Эта инструкция обеспечивает быстрый запуск и тестирование решения.

#### Общие сведения
- Репозиторий: `gravekgit/small_projects`
- Имя образа: `agcalls-app`
- Порты: 8000 (API), 7860 (Gradio демо)
- Требования: Docker установлен и запущен, доступ к интернету для первоначальной загрузки образа.

#### Инструкция по запуску

1. **Получение образа**
   Выполните команду для загрузки образа из Docker Hub:
   ```bash
   docker pull gravekgit/small_projects:agcalls-app
   ```

2. **Запуск контейнера**
   - **Linux/macOS:**
     ```bash
     docker run -d -p 8000:8000 -p 7860:7860 gravekgit/small_projects:agcalls-app
     ```
   - **Windows (PowerShell):**
     ```powershell
     docker run -d -p 8000:8000 -p 7860:7860 gravekgit/small_projects:agcalls-app
     ```
   - Описание: Флаг `-d` запускает контейнер в фоновом режиме, `-p` мапит порты хоста на порты контейнера.

3. **Проверка работы**
   - **Демо-интерфейс (Gradio):** Откройте браузер и перейдите по адресу `http://localhost:7860`. Вы увидите веб-интерфейс для классификации текста и аудио.
   - **API:** Убедитесь, что сервер доступен, отправив тестовый запрос (см. ниже).

4. **Тестовый запрос к API**
   - Подготовьте WAV-файл (до 30 секунд) и используйте следующую команду:
     - **Linux/macOS:**
       ```bash
       curl -X POST "http://localhost:8000/classify-call/?call_id=test123" \
            -H "X-API-Key: demo_key_123" \
            -F "audio_file=@test.wav"
       ```
     - **Windows (PowerShell):**
       ```powershell
       Invoke-RestMethod -Uri "http://localhost:8000/classify-call/?call_id=test123" `
                         -Method Post `
                         -Headers @{"X-API-Key"="demo_key_123"} `
                         -InFile "test.wav" `
                         -ContentType "multipart/form-data"
       ```
   - Ожидаемый результат: JSON-ответ с полями `category`, `confidence` и `probabilities`.

5. **Остановка контейнера**
   - Найдите ID контейнера:
     ```bash
     docker ps
     ```
   - Остановите контейнер:
     ```bash
     docker stop <container_id>
     ```
   - (Команда одинакова для Linux и Windows).

#### Примечания
- API-ключ: Используйте `demo_key_123` (лимит 100 запросов).
- Логи: Доступны в контейнере в файле `/app/api_server.log`.
- Для локальной разработки: Убедитесь, что порты 8000 и 7860 свободны.

#### Устранение неполадок
- Если образ не загружается, проверьте подключение к интернету и доступ к Docker Hub.
- При ошибке портов измените маппинг (например, `-p 8080:8000`).

