# course_robot — навигация PPO на реальном роботе

Добавлен скрипт `course_robot/ppo_real_nav.py`: он повторяет логику наблюдения и сглаживания команд из `RobotEnv` в `rl_train.py` и подхватывает гиперпараметры из `run_metadata.json` вашего прогона.

## Что важно понимать

1. **Сеть видела ту же размерность, что и при обучении:** 72 луча + 4 числа (дистанция до цели / 13 м, угол / π, нормированные текущие `v` и `ω`). Параметр `max_goal_distance_norm_m` по умолчанию **13** — как в `RobotEnv`; для цели **3 м** это даёт нормализованную дистанцию \(3/13 \approx 0.23\), что попадает в диапазон обучения.

2. **Цель «в 3 метрах»** задаётся так: при первом успешном цикле (есть лидара и одометрия) в **плоскости выбранного кадра одометрии** запоминается точка  
   `(x + 3·cos(yaw), y + 3·sin(yaw))` — то есть **3 м вперёд по начальному yaw**, а не «евклидово 3 м от текущей позиции позже». Если нужна фиксированная точка в odom, используйте `--goal-x` и `--goal-y`.

3. **`cmd_vel`:** в `run_metadata.json` указано `/model/course_robot/cmd_vel` (симулятор). На Pi задайте кадры и топик, например:  
   `-p cmd_vel_topic:=/cmd_vel -p odom_frame:=course_robot_odom -p base_frame:=course_robot_base_link`  
   (подставьте свои имена кадров, если другие.)

4. **ESP32:** узел только публикует `geometry_msgs/Twist`. Между ROS и ESP32 нужен ваш мост (например `serial`/`micro-ROS`), который подписан на тот же топик и переводит в протокол драйвера.

5. **Лидар:** ожидается `sensor_msgs/PointCloud2` в **системе базы** (как в симуляторе: x вперёд, y влево). Если облако в другом кадре — добавьте `tf2_ros` `MessageFilter` или публикуйте облако уже в `base_link`.

## Запуск (на Raspberry Pi, после `source install/setup.bash`)

```bash
python3 course_robot/ppo_real_nav.py \
  --model "/path/to/course_robot/models/run_20260512_011044_ppo_course_robot_100k/final_model/ppo_course_robot_100k.zip" \
  --metadata "/path/to/course_robot/models/run_20260512_011044_ppo_course_robot_100k/run_metadata.json" \
  --goal-distance 3.0 \
  --ros-args \
  -p cmd_vel_topic:=/cmd_vel \
  -p odom_frame:=course_robot_odom \
  -p base_frame:=course_robot_base_link
```

Флаг `--metadata` можно опустить: если рядом лежит `run_metadata.json` в каталоге запуска (`…/run_…/run_metadata.json`), он подхватится автоматически.

В коде есть **аварийный стоп**, если минимальный луч лидара ≤ `emergency_stop_lidar_m` (по умолчанию 0.22 м), по аналогии с виртуальным бампером в обучении.

Если пришлёте протокол обмена с ESP32 (строки/поля), можно точечно добавить маленький узел-публикатор под него без лишней абстракции.
