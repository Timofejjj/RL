#!/usr/bin/env bash
set -euo pipefail

# Переходим в директорию, где лежит этот скрипт (корень проекта).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

# Проверяем наличие python3.
if ! command -v python3 >/dev/null 2>&1; then
  echo "Ошибка: python3 не найден в PATH."
  exit 1
fi

# Проверяем наличие requirements.txt.
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Ошибка: файл $REQ_FILE не найден в $PROJECT_DIR."
  exit 1
fi

# Создаем виртуальное окружение, если его еще нет.
# Ключевой флаг --system-site-packages позволяет видеть системные ROS-пакеты (например, rclpy).
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Создаю виртуальное окружение $VENV_DIR с доступом к системным пакетам..."
  python3 -m venv --system-site-packages "$VENV_DIR"
else
  echo "Виртуальное окружение $VENV_DIR уже существует."
fi

# Активируем окружение.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Обновляем инструменты установки пакетов.
echo "Обновляю pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# Проверяем/устанавливаем зависимости из requirements.txt.
# pip сам пропускает уже установленные совместимые версии.
echo "Проверяю зависимости из $REQ_FILE и доустанавливаю недостающие..."
python -m pip install -r "$REQ_FILE"

# Неблокирующая проверка видимости ROS Python-пакетов.
if ! python -c "import rclpy" >/dev/null 2>&1; then
  echo "Предупреждение: модуль rclpy не найден."
  echo "Перед запуском скрипта выполните source /opt/ros/<ваш_дистрибутив>/setup.bash, затем снова запустите setup_local_env.bash."
fi

echo "Окружение готово. Перед каждым запуском скрипта обучения не забывайте писать: source .venv/bin/activate"
