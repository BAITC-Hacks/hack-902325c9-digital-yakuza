# Миграции

Все команды выполнять из backend при запущенных контейнерах.

1. Применить готовую миграцию 0001 с примером таблицы example_items:

```sh
docker compose exec backend alembic upgrade head
```

2. Добавить модель, наследующую Base, в app/models и импортировать её в app/models/__init__.py.
3. Создать миграцию, проверить сгенерированные upgrade/downgrade и применить:

```sh
docker compose exec backend alembic revision --autogenerate -m "message"
docker compose exec backend alembic upgrade head
```

4. Проверить текущую версию, историю и соответствие моделей БД:

```sh
docker compose exec backend alembic current
docker compose exec backend alembic history
docker compose exec backend alembic check
```

5. Откатить последнюю миграцию:

```sh
docker compose exec backend alembic downgrade -1
```

Откат 0001 удаляет example_items вместе с её данными. Миграции не запускаются автоматически при старте API. Каталог alembic подключён с хоста: созданные файлы остаются в проекте.

Для нового проекта заменить ExampleItem своими моделями и создать следующую миграцию. Уже применённые миграции не переписывать. Таблицы создаются только Alembic, без create_all. get_db закрывает сессию; сохранение изменений выполнять явно через await session.commit() либо session.begin().
