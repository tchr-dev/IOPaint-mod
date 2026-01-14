Ниже — единый TODO/roadmap для расширения функционала IOPaint, собранный и
нормализованный на основе загруженных файлов: текущего roadmap, списка
фикс-файдингов и плана OpenAI-compatible MVP генерации/редактирования с
бюджет-контролем. ￼ ￼ ￼ ￼

⸻

Техническое задание: IOPaint — расширение функционала (Roadmap)

Epic 1 — OpenAI-compatible слой (генерация + редактирование) для IOPaint

Цель: добавить унифицированный клиент для OpenAI-совместимых провайдеров
(OpenAI/ProxyAPI/OpenRouter/…). •	E1.1 Внедрить/адаптировать OpenAI-compatible
client в контекст IOPaint •	list_models() → для UI выбора модели
•	refine_prompt() → “дёшево уточнить” перед дорогой операцией •	generate_image()
→ text-to-image •	edit_image() → image+mask edit (inpaint/outpaint вариации)
•	Единая структура ошибок (status, retryable, detail) •	Конфиг через env:
•	AIE_BACKEND=openai •	AIE_OPENAI_API_KEY, AIE_OPENAI_BASE_URL,
AIE_OPENAI_MODEL, AIE_OPENAI_TIMEOUT_S ￼ ￼ •	E1.2 Привязать к инструментам
IOPaint •	Inpaint (이미 есть) → перевести на общий клиент •	Добавить инструменты
по мере готовности: •	Outpaint •	Variations •	Upscale •	Background removal •	Для
каждого инструмента: единый ToolRequest/ToolResult контракт и тест-валидатор ￼

⸻

Epic 2 — Budget Safety: защита бюджета, дедупликация, rate limit

Цель: чтобы пользователь не сжёг бюджет случайно (двойной клик, повтор запроса,
“Final” режим без предупреждения). •	E2.1 BudgetGuard (hard caps) •	Daily cap /
monthly cap / optional session cap •	Любой paid-вызов проходит через BudgetGuard
•	Статус blocked_budget + понятное сообщение в UI ￼ •	E2.2 Dedupe по fingerprint
•	fingerprint =
sha256(normalized(model+action+prompt+negative+params+input_hashes)) •	Если
fingerprint уже успешен недавно → показать существующий результат, без нового
вызова (по умолчанию) ￼ •	E2.3 Rate limit + защита от double-click •	1 “дорогая”
операция / 10с / session (пример) •	UI: дизейбл кнопок на время выполнения
•	Блокировка на уровне БД/runner по fingerprint ￼ •	E2.4 Cost awareness UI
•	Показ “low/med/high” по preset (Draft/Final), size/quality •	Опционально:
estimated_cost_usd (правило-базированно) ￼

⸻

Epic 3 — Хранилище: SQLite + файлы + история

Цель: аудит, повторы, галерея, восстановление истории и бюджет-ledger. •	E3.1
SQLite схема и миграции •	jobs/images/history_snapshots как минимум
•	budget_ledger (estimated/actual) •	models_cache (кэш списка моделей) ￼ •	E3.2
Layout файлов •	data/images/{job_id}/{image_id}.{ext}
•	data/input/{sha256}.{ext} для исходников •	data/thumbs/{image_id}.jpg
(опционально) ￼ •	E3.3 История и проекты •	История сохраняется/восстанавливается
без “дыр” •	Переносимость проектов (связано с E0.5) ￼

⸻

Epic 4 — UI/UX: “Refine → Generate/Edit”, пресеты, галерея, rerun

Цель: сделать новый функционал понятным и быстрым. •	E4.1 Новый поток генерации
•	Поле “Intent” (сырой запрос) •	Кнопка “Refine prompt” (дешёвый текстовый
вызов) •	Поле “Final prompt” (редактируемое) + Negative prompt •	Presets:
•	Draft: 512, low/standard, n=1 •	Final: 1024, medium/high, n=1 •	Confirm modal
для “дорогих” параметров ￼ •	E4.2 Edit flow •	Выбор изображения из
галереи/открытого проекта •	Маска из текущего холста •	Refine → Edit
(/v1/images/edits) ￼ •	E4.3 История/галерея •	Список последних jobs: prompt
snippet, params, статус, thumbnail, время •	Actions: Open, Copy prompt, Re-run
(с уважением к dedupe), Delete local ￼

⸻

Epic 5 — Runner/Jobs: очередь, статусы, прогресс, отмена

Цель: не блокировать UI и иметь управляемый pipeline. •	E5.1 Job runner
(локальный MVP) •	statuses: queued/running/succeeded/failed/blocked_budget
•	progress updates в UI •	audit: сохранять job input/params до вызова API ￼
•	E5.2 Cancel •	Cancellation token на уровне клиента/runner (если
поддерживается) •	UI cancel button во время генерации/редакта ￼ ￼

⸻

Epic 6 — Тесты: unit + integration + (опционально) E2E

Цель: зафиксировать контракт и не сломать историю/бюджет. •	E6.1 Unit
•	fingerprint детерминизм •	BudgetGuard: блок/разрешение •	Retry logic:
429/timeout retry, 400 no retry •	Undo/redo для кисти и AI-правок ￼ ￼ •	E6.2
Integration (mock OpenAI-compatible API) •	/v1/images/generations: файл
сохранён, DB записи созданы, dedupe работает •	/v1/images/edits: корректная
маска/параметры ￼ •	E6.3 Playwright E2E (MVP+) •	load → refine → draft generate
→ final generate → edit ￼
