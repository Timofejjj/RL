#!/usr/bin/env bash
# РћРґРёРЅ Р·Р°РїСѓСЃРє: Gazebo diff_drive + РјРѕСЃС‚ + СЃРїР°РІРЅ course_robot + TF + URDF РІ С‚РѕРїРёРє + RViz СЃ РІР°С€РµР№ РјРѕРґРµР»СЊСЋ.
# Р—Р°РїСѓСЃРєР°С‚СЊ С‚РѕР»СЊРєРѕ С‚Р°Рє:  bash start_everything.bash   (РєРѕРјР°РЅРґР° В«sh вЂ¦В» РґР°СЃС‚ РѕС€РёР±РєРё вЂ” РЅСѓР¶РµРЅ bash).
# WSL/Linux, СѓСЃС‚Р°РЅРѕРІР»РµРЅС‹ ROS 2 Jazzy Рё Gazebo (gz). Р—Р°РїСѓСЃРє: bash start_everything.bash
#
# Р’Р°Р¶РЅРѕ: РЅРµ РІСЃС‚Р°РІР»СЏР№С‚Рµ СЌС‚Сѓ РєРѕРјР°РЅРґСѓ РІ РѕРєРЅРѕ, РіРґРµ СѓР¶Рµ РёРґС‘С‚ В«ros2 launchВ» вЂ” С‚Р°Рј РЅРµС‚ РїСЂРёРіР»Р°С€РµРЅРёСЏ shell,
# РІРІРѕРґ СѓС…РѕРґРёС‚ РІ Р·Р°РїСѓС‰РµРЅРЅС‹Р№ РїСЂРѕС†РµСЃСЃ. РЎРЅР°С‡Р°Р»Р° Ctrl+C РІ С‚РѕРј РѕРєРЅРµ, Р»РёР±Рѕ РѕС‚РєСЂРѕР№С‚Рµ РќРћР’РЈР® РІРєР»Р°РґРєСѓ/РѕРєРЅРѕ WSL.
# Р‘РµР· В«set -uВ»: РёРЅР°С‡Рµ source /opt/ros/.../setup.bash РїР°РґР°РµС‚ (AMENT_TRACE_SETUP_FILES Рё РґСЂ.).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CR_DIR="$ROOT/course_robot"
URDF="$CR_DIR/course_robot.urdf"
RVIZ_CFG="$CR_DIR/diff_drive_plus_course_robot.rviz"
WORLD_FILE="$ROOT/course_robot_world.sdf"
WORLD_NAME="course_world"

# WSL/Linux graphics can be flaky. For RL training you usually don't need Gazebo GUI.
#
# Default is stable split-mode:
# - gz sim server: headless rendering (sensors + physics)
# - optional gz sim GUI client (separate process)
#
# Env vars:
# - COURSE_WITH_GUI=1            start separate `gz sim -g <тот же .sdf что у -s>` (без SDF у клиента дерево сущностей часто пустое)
# - COURSE_LEGACY_GUI=1         old single-process `gz sim -r` (known to crash with sensors+GUI on some setups)
# - COURSE_SOFTWARE_GL=1        force software OpenGL (GUI client + RViz; при отключённом headless — и для сервера)
# - COURSE_SERVER_HEADLESS=0|1   принудительно: сервер с/без --headless-rendering; не задано = авто
#   (авто: на WSL всегда headless=1 — стабильнее с gpu_lidar + отдельным gz -g; на обычном Linux при GUI+DISPLAY — headless=0)
# - COURSE_RENDER_ENGINE_SERVER / COURSE_RENDER_ENGINE_GUI (по умолчанию ogre2; на WSL для -g часто ogre — см. ниже)
# - WSLg + окно gz -g без llvmpipe: по умолчанию GALLIUM_DRIVER=d3d12 для клиента (Mesa 24.3+ иначе часто не берёт D3D12 — чёрный экран).
#   Переопределение: COURSE_GUI_GALLIUM_DRIVER=d3d12|llvmpipe|…; гибрид GPU: COURSE_MESA_D3D12_ADAPTER=NVIDIA|Intel|…
# - COURSE_SERVER_SOFTWARE_GL=0|1 — применять llvmpipe к gz sim -s (не к -g). Имеет смысл при COURSE_SOFTWARE_GL=1.
#   По умолчанию: на WSL при headless-сервере =0 (окно -g на системном GL/WSLg — меньше чёрного экрана; RViz всё равно на llvmpipe).
#   На WSL при сервере без headless: 0; принудительно весь стек в софте: =1
COURSE_WITH_GUI="${COURSE_WITH_GUI:-1}"
COURSE_LEGACY_GUI="${COURSE_LEGACY_GUI:-0}"
# Если не задано: на WSL/WSLg RViz и gz GUI часто падают через секунды на «железном» GL — по умолчанию llvmpipe.
COURSE_SOFTWARE_GL="${COURSE_SOFTWARE_GL:-}"
if [[ -z "$COURSE_SOFTWARE_GL" ]]; then
  if [[ -f /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version; then
    COURSE_SOFTWARE_GL=1
  else
    COURSE_SOFTWARE_GL=0
  fi
fi
is_wsl=0
if [[ -f /proc/version ]] && grep -qiE 'microsoft|wsl' /proc/version; then
  is_wsl=1
  export QT_X11_NO_MITSHM=1
  # WSLg + X11: стабильнее, чем умолчание; при чёрном окне Qt попробуйте явно: export QT_QPA_PLATFORM=xcb
  if [[ -n "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  fi
fi
COURSE_RENDER_ENGINE_SERVER="${COURSE_RENDER_ENGINE_SERVER:-ogre2}"
# COURSE_RENDER_ENGINE_GUI задаётся ниже, после COURSE_SERVER_SOFTWARE_GL (см. комментарий там).

course_apply_gui_gl_env() {
  if [[ "$COURSE_SOFTWARE_GL" -eq 1 ]]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-4.2}"
    export MESA_GLSL_VERSION_OVERRIDE="${MESA_GLSL_VERSION_OVERRIDE:-420}"
    export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
  fi
}

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/jazzy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
else
  echo "РќРµ РЅР°Р№РґРµРЅ /opt/ros/jazzy/setup.bash (РёР»Рё humble). РЈРєР°Р¶РёС‚Рµ ROS РІСЂСѓС‡РЅСѓСЋ: source /opt/ros/<РґРёСЃС‚СЂРёР±СѓС‚РёРІ>/setup.bash" >&2
  exit 1
fi

for need in ros2 gz python3; do
  if ! command -v "$need" &>/dev/null; then
    echo "Р’ PATH РЅРµС‚ РєРѕРјР°РЅРґС‹: $need" >&2
    exit 1
  fi
done

if [[ -z "${COURSE_PYTHON:-}" ]]; then
  if [[ -x "$CR_DIR/.venv_wsl/bin/python3" ]]; then
    COURSE_PYTHON="$CR_DIR/.venv_wsl/bin/python3"
  else
    COURSE_PYTHON="$(command -v python3)"
  fi
fi
if [[ ! -x "$COURSE_PYTHON" ]]; then
  echo "Не найден исполняемый Python: COURSE_PYTHON=$COURSE_PYTHON" >&2
  exit 1
fi
echo "python runtime: $COURSE_PYTHON"
# Явный раздел: иначе gz sim -s и gz sim -g иногда оказываются в разных «кластерах» — в GUI пустое дерево сущностей, а мосты/RViz по-прежнему видят сервер.
export GZ_PARTITION="${GZ_PARTITION:-course_stack_${WORLD_NAME}_$$}"
# Важно: при включённом pipefail команда `ros2 pkg list | grep -q ...` часто даёт ложный "не найден"
# из-за SIGPIPE (grep завершился после совпадения). Поэтому проверяем через `ros2 pkg prefix`.
if ! ros2 pkg prefix ros_gz_bridge &>/dev/null; then
  echo "РќСѓР¶РµРЅ РїР°РєРµС‚ ros_gz_bridge (РЅР°РїСЂРёРјРµСЂ: sudo apt install ros-${ROS_DISTRO:-jazzy}-ros-gz-bridge)" >&2
  exit 1
fi

if [[ ! -f "$URDF" || ! -f "$RVIZ_CFG" ]]; then
  echo "РќРµС‚ С„Р°Р№Р»РѕРІ РІ $CR_DIR" >&2
  exit 1
fi
if [[ ! -f "$WORLD_FILE" ]]; then
  echo "РќРµС‚ С„Р°Р№Р»Р° РјРёСЂР°: $WORLD_FILE" >&2
  exit 1
fi

cleanup() {
  [[ -n "${GUI_PID:-}" ]] && kill "$GUI_PID" 2>/dev/null || true
  [[ -n "${SIM_PID:-}" ]] && kill "$SIM_PID" 2>/dev/null || true
  if [[ -f "${CR_DIR:-}/ros_tf_stack.bash" ]]; then
    # shellcheck source=course_robot/ros_tf_stack.bash
    source "$CR_DIR/ros_tf_stack.bash"
    course_robot_stop_ros_tf_stack
  fi
  [[ -n "${LIDAR_DRIVE_PID:-}" ]] && kill "$LIDAR_DRIVE_PID" 2>/dev/null || true
  [[ -n "${RL_PID:-}" ]] && kill "$RL_PID" 2>/dev/null || true
  [[ -n "${RVIZ_PID:-}" ]] && kill "$RVIZ_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap cleanup INT
trap cleanup TERM

echo "== Р—Р°РїСѓСЃРє course_world (Р±РµР· vehicle_blue/vehicle_green) =="
export GZ_SIM_RESOURCE_PATH="${ROOT}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
# Лог пишем в /tmp, т.к. /mnt/<disk> иногда смонтирован без прав на создание скрытых файлов.
GZ_LOG="/tmp/gz_sim_course_world_${USER:-user}_$$.log"
GZ_GUI_LOG="/tmp/gz_gui_course_world_${USER:-user}_$$.log"
RVIZ_LOG="/tmp/rviz_course_world_${USER:-user}_$$.log"
echo "gz server log: $GZ_LOG"
echo "gz gui log:    $GZ_GUI_LOG"
echo "rviz log:      $RVIZ_LOG"

# Сервер с --headless-rendering + отдельный gz sim -g на Jazzy/ogre2 часто даёт падение сенсоров:
# Ogre::ItemIdentityException ... material datablock ... already exists.
SERVER_USE_HEADLESS=1
if [[ "${COURSE_SERVER_HEADLESS:-}" == "0" || "${COURSE_SERVER_HEADLESS:-}" == "1" ]]; then
  SERVER_USE_HEADLESS="$COURSE_SERVER_HEADLESS"
elif [[ "$COURSE_WITH_GUI" -eq 1 ]] && { [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; }; then
  # На WSL не отключаем headless у сервера: иначе часто падает/не поднимается мир с сенсорами + отдельный GUI.
  if [[ "$is_wsl" -ne 1 ]]; then
    SERVER_USE_HEADLESS=0
  fi
fi

COURSE_SERVER_SOFTWARE_GL="${COURSE_SERVER_SOFTWARE_GL:-}"
if [[ -z "$COURSE_SERVER_SOFTWARE_GL" ]]; then
  if [[ "$is_wsl" -eq 1 && "$SERVER_USE_HEADLESS" -eq 0 && "$COURSE_SOFTWARE_GL" -eq 1 ]]; then
    COURSE_SERVER_SOFTWARE_GL=0
  elif [[ "$is_wsl" -eq 1 && "$SERVER_USE_HEADLESS" -eq 1 && "$COURSE_SOFTWARE_GL" -eq 1 ]]; then
    # Headless-сервер не рисует окно; клиент -g с llvmpipe+Ogre часто даёт чёрный экран — оставляем GUI без LIBGL_ALWAYS_SOFTWARE.
    COURSE_SERVER_SOFTWARE_GL=0
  else
    COURSE_SERVER_SOFTWARE_GL="$COURSE_SOFTWARE_GL"
  fi
fi

# Движок GUI: при WSL + llvmpipe на весь стек — ogre (1) стабильнее для окна -g; при сервере на
# «системном» GL (COURSE_SERVER_SOFTWARE_GL=0) клиент без llvmpipe: ogre2+D3D12 на части Lenovo/Intel даёт
# чёрный экран (см. gazebosim/gz-sim#2670) — по умолчанию ogre (1) для -g; переопределите COURSE_RENDER_ENGINE_GUI=ogre2 при необходимости.
if [[ -z "${COURSE_RENDER_ENGINE_GUI:-}" ]]; then
  if [[ "$is_wsl" -eq 1 && "$COURSE_SOFTWARE_GL" -eq 1 && "$COURSE_SERVER_SOFTWARE_GL" -eq 0 ]]; then
    COURSE_RENDER_ENGINE_GUI="ogre"
  elif [[ "$is_wsl" -eq 1 && "$COURSE_SOFTWARE_GL" -eq 1 ]]; then
    COURSE_RENDER_ENGINE_GUI="ogre"
  else
    COURSE_RENDER_ENGINE_GUI="ogre2"
  fi
fi

course_gui_uses_software_gl() {
  [[ "$COURSE_SOFTWARE_GL" -ne 1 ]] && return 1
  # На WSLg "железный" рендер для gz -g часто даёт чёрный экран (особенно если D3D12 не подхватился).
  # Делаем более надёжный дефолт: если COURSE_SOFTWARE_GL=1, то и GUI тоже в llvmpipe.
  # При желании вернуть D3D12-путь: COURSE_SOFTWARE_GL=0 (или переопределить ниже веткой env -u ...).
  return 0
}

# true (exit 0): WSL и gz -g на WSLg GL (не полный llvmpipe для GUI) — задаём GALLIUM_DRIVER=d3d12 для клиента.
course_wsl_gz_gui_wants_d3d12_env() {
  [[ "$is_wsl" -eq 1 ]] || return 1
  if course_gui_uses_software_gl; then
    return 1
  fi
  return 0
}

if [[ "$COURSE_SOFTWARE_GL" -eq 1 ]]; then
  echo "graphics: COURSE_SOFTWARE_GL=1 (llvmpipe для RViz; для gz -g — см. строку ниже)"
  if [[ "$is_wsl" -eq 1 ]]; then
    if course_gui_uses_software_gl; then
      echo "WSL: gz -g = llvmpipe + engine ${COURSE_RENDER_ENGINE_GUI} (server = ${COURSE_RENDER_ENGINE_SERVER})"
    else
      echo "WSL: gz -g = D3D12 (GALLIUM_DRIVER=${COURSE_GUI_GALLIUM_DRIVER:-d3d12}) + engine ${COURSE_RENDER_ENGINE_GUI}; llvmpipe только для RViz"
    fi
    echo "WSL: при чёрном окне: COURSE_MESA_D3D12_ADAPTER=NVIDIA (дискретка); COURSE_GUI_GALLIUM_DRIVER=llvmpipe; COURSE_WITH_GUI=0; см. start_everything.bash (шапка)"
  fi
fi

if [[ "$COURSE_LEGACY_GUI" -eq 1 ]]; then
  echo "gz sim mode: LEGACY single-process GUI (may crash with sensors rendering on some setups)"
  export GZ_SIM_RENDER_ENGINE="${COURSE_RENDER_ENGINE_SERVER}"
  course_apply_gui_gl_env
  gz sim -r "$WORLD_FILE" >"$GZ_LOG" 2>&1 &
  SIM_PID=$!
else
  if [[ "$SERVER_USE_HEADLESS" -eq 1 ]]; then
    echo "gz sim mode: SERVER + --headless-rendering (SERVER_USE_HEADLESS=1); GUI client (if any) after world is ready"
  else
    echo "gz sim mode: SERVER без --headless-rendering (SERVER_USE_HEADLESS=0; на WSL см. COURSE_SERVER_HEADLESS=0 вручную); нужен DISPLAY"
    if [[ "$COURSE_SOFTWARE_GL" -eq 1 && "$COURSE_SERVER_SOFTWARE_GL" -eq 1 ]]; then
      course_apply_gui_gl_env
      echo "server GL: llvmpipe (COURSE_SERVER_SOFTWARE_GL=1)"
    elif [[ "$COURSE_SOFTWARE_GL" -eq 1 && "$COURSE_SERVER_SOFTWARE_GL" -eq 0 ]]; then
      echo "server GL: системный по умолчанию (COURSE_SERVER_SOFTWARE_GL=0); llvmpipe только для RViz, gz -g на системном GL"
    fi
  fi
  gz_cmd=(gz sim -r -s)
  if [[ "$SERVER_USE_HEADLESS" -eq 1 ]]; then
    # без "=1": в gz-sim из ROS Jazzy иначе OptionParser::NeedlessArgument
    gz_cmd+=(--headless-rendering)
  fi
  gz_cmd+=(--render-engine-server "$COURSE_RENDER_ENGINE_SERVER" "$WORLD_FILE")
  "${gz_cmd[@]}" >"$GZ_LOG" 2>&1 &
  SIM_PID=$!
fi

echo "== РћР¶РёРґР°РЅРёРµ РјРёСЂР° ${WORLD_NAME} (РґРѕ ~45 СЃ) =="
ready=0
for i in {1..45}; do
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "gz server process died while waiting for world services (pid=$SIM_PID)." >&2
    echo "== tail gz server log ==" >&2
    tail -n 80 "$GZ_LOG" 2>/dev/null || true
    exit 1
  fi

  svc_out=$(gz service --list 2>/dev/null || gz service -ls 2>/dev/null || true)
  if echo "$svc_out" | grep -qF "/world/${WORLD_NAME}/create"; then
    ready=1
    break
  fi

  # lightweight progress (so it doesn't look "stuck")
  if (( i % 5 == 0 )); then
    echo "... still waiting (${i}/45) for /world/${WORLD_NAME}/create"
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "РЎРµСЂРІРёСЃ /world/${WORLD_NAME}/create РЅРµ РїРѕСЏРІРёР»СЃСЏ Р·Р° 45СЃ." >&2
  echo "Подсказка: проверьте, что world name в SDF совпадает с WORLD_NAME=${WORLD_NAME}" >&2
  echo "Доступные /world/*/create (первые совпадения):" >&2
  (gz service --list 2>/dev/null || gz service -ls 2>/dev/null || true) | grep -F '/world/' | grep -F '/create' | head -n 20 >&2 || true
  echo "== tail gz server log ==" >&2
  tail -n 120 "$GZ_LOG" 2>/dev/null || true
  exit 1
fi

# Ранний gz sim -g (до готовности мира) на части сборок WSL даёт окно на пару секунд и вылет.
if [[ "$COURSE_LEGACY_GUI" -ne 1 && "$COURSE_WITH_GUI" -eq 1 ]] && { [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; }; then
  echo "== gz GUI client (gz sim -g), тот же SDF, что и у сервера: $WORLD_FILE =="
  if course_gui_uses_software_gl; then
    course_apply_gui_gl_env
    gz sim -g --render-engine-gui "$COURSE_RENDER_ENGINE_GUI" "$WORLD_FILE" >"$GZ_GUI_LOG" 2>&1 &
  elif course_wsl_gz_gui_wants_d3d12_env; then
    # Сброс принудительного llvmpipe из окружения + явный d3d12 для WSLg (Mesa 24.3+; см. gz-sim#2670, WSL#12584).
    COURSE_GUI_GALLIUM_DRIVER="${COURSE_GUI_GALLIUM_DRIVER:-d3d12}"
    mesa_d3d12=()
    [[ -n "${COURSE_MESA_D3D12_ADAPTER:-}" ]] && mesa_d3d12=(MESA_D3D12_DEFAULT_ADAPTER_NAME="${COURSE_MESA_D3D12_ADAPTER}")
    env -u LIBGL_ALWAYS_SOFTWARE -u MESA_GL_VERSION_OVERRIDE -u MESA_GLSL_VERSION_OVERRIDE \
      -u MESA_LOADER_DRIVER_OVERRIDE \
      GALLIUM_DRIVER="${COURSE_GUI_GALLIUM_DRIVER}" \
      "${mesa_d3d12[@]}" \
      gz sim -g --render-engine-gui "$COURSE_RENDER_ENGINE_GUI" "$WORLD_FILE" >"$GZ_GUI_LOG" 2>&1 &
  else
    # Обычный Linux (на WSL сюда не попадаем — см. course_wsl_gz_gui_wants_d3d12_env).
    gz sim -g --render-engine-gui "$COURSE_RENDER_ENGINE_GUI" "$WORLD_FILE" >"$GZ_GUI_LOG" 2>&1 &
  fi
  GUI_PID=$!
  sleep 2
  if ! kill -0 "$GUI_PID" 2>/dev/null; then
    echo "Gazebo GUI (gz sim -g) завершился сразу после старта — см. $GZ_GUI_LOG" >&2
    tail -n 50 "$GZ_GUI_LOG" 2>/dev/null || true
  elif [[ "$is_wsl" -eq 1 ]]; then
    echo "Подсказка: чёрный экран — COURSE_MESA_D3D12_ADAPTER=NVIDIA|Intel; chmod 0700 \"\$XDG_RUNTIME_DIR\"; группа render для /dev/dri; COURSE_RENDER_ENGINE_GUI=ogre2; COURSE_WITH_GUI=0; лог: $GZ_GUI_LOG"
  fi
elif [[ "$COURSE_LEGACY_GUI" -ne 1 ]]; then
  echo "gz gui: skipped (COURSE_WITH_GUI=0 or no DISPLAY/WAYLAND_DISPLAY)"
fi

sleep 2
# В world-файле robot уже включён (include) с позой 0 0 0.36.
export COURSE_ROBOT_SPAWN_X="${COURSE_ROBOT_SPAWN_X:-0.0}"
export COURSE_ROBOT_SPAWN_Y="${COURSE_ROBOT_SPAWN_Y:-0.0}"
export COURSE_ROBOT_SPAWN_Z="${COURSE_ROBOT_SPAWN_Z:-0.36}"
echo "== course_robot: используется из world-файла (позиция COURSE_ROBOT_SPAWN_X/Y/Z; сейчас ${COURSE_ROBOT_SPAWN_X}, ${COURSE_ROBOT_SPAWN_Y}, ${COURSE_ROBOT_SPAWN_Z}) =="

echo "== ros_gz_bridge + TF (см. course_robot/ros_gz_course_robot.yaml, ros_tf_stack.bash) =="
# Все TF-узлы и RViz — с use_sim_time; мост /clock — в yaml (иначе RViz/tf2 не склеивают дерево).
# shellcheck source=course_robot/ros_tf_stack.bash
source "$CR_DIR/ros_tf_stack.bash"
SKIP_BRIDGE=0
course_robot_launch_ros_tf_stack

COURSE_RUN_RL="${COURSE_RUN_RL:-0}"
if [[ "$COURSE_RUN_RL" -eq 1 ]]; then
  RL_BACKEND="${COURSE_RL_BACKEND:-ros}"
  RL_TIMESTEPS="${COURSE_RL_TIMESTEPS:-100000}"
  RL_MODEL_NAME="${COURSE_RL_MODEL_NAME:-ppo_course_robot_100k}"
  RL_MAX_LINEAR_SPEED_MPS="${COURSE_RL_MAX_LINEAR_SPEED_MPS:-0.35}"
  RL_MAX_ANGULAR_SPEED_RADPS="${COURSE_RL_MAX_ANGULAR_SPEED_RADPS:-0.8}"
  RL_CONTROL_DT_SEC="${COURSE_RL_CONTROL_DT_SEC:-0.12}"
  RL_MAX_LINEAR_ACCEL_MPS2="${COURSE_RL_MAX_LINEAR_ACCEL_MPS2:-0.8}"
  RL_MAX_ANGULAR_ACCEL_RADPS2="${COURSE_RL_MAX_ANGULAR_ACCEL_RADPS2:-2.5}"
  RL_ANGULAR_DEADBAND="${COURSE_RL_ANGULAR_DEADBAND:-0.08}"
  RL_RESET_WORLD_ON_EPISODE="${COURSE_RL_RESET_WORLD_ON_EPISODE:-0}"
  RL_OBSTACLE_RANDOMIZE_EVERY="${COURSE_RL_OBSTACLE_RANDOMIZE_EVERY:-20}"
  RL_SAVE_ROOT="${COURSE_RL_SAVE_DIR:-$CR_DIR/models}"
  RL_LOG_ROOT="${COURSE_RL_LOG_ROOT:-$CR_DIR/training_logs}"
  RL_RUN_ID="${COURSE_RL_RUN_ID:-}"

  if ! mkdir -p "$RL_SAVE_ROOT" 2>/dev/null; then
    RL_SAVE_ROOT="/tmp/course_robot_models"
    mkdir -p "$RL_SAVE_ROOT"
    echo "WARN: save root on /mnt/d is not writable, fallback to $RL_SAVE_ROOT" >&2
  fi

  if ! mkdir -p "$RL_LOG_ROOT" 2>/dev/null; then
    RL_LOG_ROOT="/tmp/course_robot_training_logs"
    mkdir -p "$RL_LOG_ROOT"
    echo "WARN: log root on /mnt/d is not writable, fallback to $RL_LOG_ROOT" >&2
  fi

  if ! touch "$RL_LOG_ROOT/.write_test" 2>/dev/null; then
    RL_LOG_ROOT="/tmp/course_robot_training_logs"
    mkdir -p "$RL_LOG_ROOT"
    echo "WARN: cannot write to selected log root, fallback to $RL_LOG_ROOT" >&2
  else
    rm -f "$RL_LOG_ROOT/.write_test" 2>/dev/null || true
  fi

  echo "== RL training: course_robot/rl_train.py (COURSE_RUN_RL=1) =="
  echo "backend=$RL_BACKEND timesteps=$RL_TIMESTEPS model=$RL_MODEL_NAME"
  echo "save_root=$RL_SAVE_ROOT"
  echo "log_root=$RL_LOG_ROOT"
  if [[ -n "$RL_RUN_ID" ]]; then
    echo "run_id=$RL_RUN_ID"
  fi
  rl_cmd=(
    "$COURSE_PYTHON" "$CR_DIR/rl_train.py"
    --backend "$RL_BACKEND"
    --total-timesteps "$RL_TIMESTEPS"
    --model-name "$RL_MODEL_NAME"
    --save-dir "$RL_SAVE_ROOT"
    --log-dir "$RL_LOG_ROOT"
    --max-linear-speed-mps "$RL_MAX_LINEAR_SPEED_MPS"
    --max-angular-speed-radps "$RL_MAX_ANGULAR_SPEED_RADPS"
    --control-dt-sec "$RL_CONTROL_DT_SEC"
    --max-linear-accel-mps2 "$RL_MAX_LINEAR_ACCEL_MPS2"
    --max-angular-accel-radps2 "$RL_MAX_ANGULAR_ACCEL_RADPS2"
    --angular-deadband "$RL_ANGULAR_DEADBAND"
    --obstacle-randomize-every "$RL_OBSTACLE_RANDOMIZE_EVERY"
  )
  if [[ "$RL_RESET_WORLD_ON_EPISODE" -eq 1 ]]; then
    rl_cmd+=(--reset-world-on-episode)
  else
    rl_cmd+=(--no-reset-world-on-episode)
  fi
  if [[ -n "$RL_RUN_ID" ]]; then
    rl_cmd+=(--run-id "$RL_RUN_ID")
  fi
  "${rl_cmd[@]}" &
  RL_PID=$!
else
  echo "== Obstacle avoidance: course_robot/lidar_drive.py (drive forward + обход) =="
  "$COURSE_PYTHON" "$CR_DIR/lidar_drive.py" --ros-args -p use_sim_time:=true &
  LIDAR_DRIVE_PID=$!
fi

sleep 1
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "== RViz: пропуск (нет DISPLAY/WAYLAND_DISPLAY) =="
else
  echo "== RViz (мир Gazebo будет жить, даже если RViz закрыть) =="
  echo "Подсказка: если робот не едет / нет одометрии — проверьте мост (топик /model/course_robot/odometry)."
  echo "Диагностика TF лидара: bash $CR_DIR/verify_course_robot_stack.bash (после source /opt/ros/.../setup.bash)."
  if [[ "$is_wsl" -eq 1 ]]; then
    # На WSLg лучше, чтобы RViz рисовал на GPU через Mesa/D3D12, иначе llvmpipe грузит CPU и всё "тормозит".
    # Пробуем GPU-вариант, и если RViz сразу падает — откатываемся на llvmpipe.
    COURSE_RVIZ_GALLIUM_DRIVER="${COURSE_RVIZ_GALLIUM_DRIVER:-d3d12}"
    mesa_d3d12=()
    [[ -n "${COURSE_MESA_D3D12_ADAPTER:-}" ]] && mesa_d3d12=(MESA_D3D12_DEFAULT_ADAPTER_NAME="${COURSE_MESA_D3D12_ADAPTER}")
    env -u LIBGL_ALWAYS_SOFTWARE -u MESA_GL_VERSION_OVERRIDE -u MESA_GLSL_VERSION_OVERRIDE \
      -u MESA_LOADER_DRIVER_OVERRIDE \
      GALLIUM_DRIVER="${COURSE_RVIZ_GALLIUM_DRIVER}" \
      "${mesa_d3d12[@]}" \
      rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true >"$RVIZ_LOG" 2>&1 &
    RVIZ_PID=$!
    sleep 2
    if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
      echo "RViz (GPU via ${COURSE_RVIZ_GALLIUM_DRIVER}) завершился сразу — пробуем llvmpipe. Лог: $RVIZ_LOG" >&2
      tail -n 80 "$RVIZ_LOG" 2>/dev/null || true
      course_apply_gui_gl_env
      rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true >"$RVIZ_LOG" 2>&1 &
      RVIZ_PID=$!
      sleep 2
      if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
        echo "RViz завершился сразу после старта — см. $RVIZ_LOG" >&2
        tail -n 80 "$RVIZ_LOG" 2>/dev/null || true
      fi
    fi
  else
    course_apply_gui_gl_env
    rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true >"$RVIZ_LOG" 2>&1 &
    RVIZ_PID=$!
    sleep 2
    if ! kill -0 "$RVIZ_PID" 2>/dev/null; then
      echo "RViz завершился сразу после старта — см. $RVIZ_LOG" >&2
      tail -n 80 "$RVIZ_LOG" 2>/dev/null || true
    fi
  fi
fi

echo "== Ожидание закрытия Gazebo server (gz sim -s) =="
set +e
wait "$SIM_PID"
rc=$?
set -e
echo "gz server exited with code: $rc"
echo "== Последние строки из gz server лога =="
tail -n 60 "$GZ_LOG" 2>/dev/null || true

# If Gazebo died, stop RViz to avoid ROS-context errors.
if [[ "$rc" -ne 0 ]]; then
  [[ -n "${RVIZ_PID:-}" ]] && kill "$RVIZ_PID" 2>/dev/null || true
  # shellcheck source=course_robot/ros_tf_stack.bash
  source "$CR_DIR/ros_tf_stack.bash" 2>/dev/null && course_robot_stop_ros_tf_stack || true
  [[ -n "${LIDAR_DRIVE_PID:-}" ]] && kill "$LIDAR_DRIVE_PID" 2>/dev/null || true
  [[ -n "${RL_PID:-}" ]] && kill "$RL_PID" 2>/dev/null || true
  [[ -n "${GUI_PID:-}" ]] && kill "$GUI_PID" 2>/dev/null || true
  exit "$rc"
fi

# If server exited "cleanly" but log shows a fatal render abort, treat as failure.
if grep -qE 'Ogre::ItemIdentityException|Another item already exists with name: scene|Aborted' "$GZ_LOG" 2>/dev/null; then
  echo "gz server log indicates a fatal crash even though wait returned $rc" >&2
  [[ -n "${RVIZ_PID:-}" ]] && kill "$RVIZ_PID" 2>/dev/null || true
  # shellcheck source=course_robot/ros_tf_stack.bash
  source "$CR_DIR/ros_tf_stack.bash" 2>/dev/null && course_robot_stop_ros_tf_stack || true
  [[ -n "${LIDAR_DRIVE_PID:-}" ]] && kill "$LIDAR_DRIVE_PID" 2>/dev/null || true
  [[ -n "${RL_PID:-}" ]] && kill "$RL_PID" 2>/dev/null || true
  [[ -n "${GUI_PID:-}" ]] && kill "$GUI_PID" 2>/dev/null || true
  exit 1
fi

if [[ -n "${GUI_PID:-}" ]]; then
  echo "== Ожидание закрытия Gazebo GUI (gz sim -g) =="
  set +e
  wait "$GUI_PID"
  grc=$?
  set -e
  echo "gz gui exited with code: $grc"
  echo "== Последние строки из gz gui лога =="
  tail -n 60 "$GZ_GUI_LOG" 2>/dev/null || true
fi
