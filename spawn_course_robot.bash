#!/usr/bin/env bash
# РЎРїР°РІРЅ course_robot РІ СѓР¶Рµ Р·Р°РїСѓС‰РµРЅРЅС‹Р№ gz sim.
#   bash spawn_course_robot.bash [РёРјСЏ_РјРёСЂР°] [РІС‹СЃРѕС‚Р°_z_РІ_РјРёСЂРµ]
# РџРµСЂРµРјРµРЅРЅР°СЏ РѕРєСЂСѓР¶РµРЅРёСЏ: COURSE_ROBOT_SPAWN_Z (РµСЃР»Рё РІС‚РѕСЂРѕР№ Р°СЂРіСѓРјРµРЅС‚ РЅРµ Р·Р°РґР°РЅ)
# РќРµ РёСЃРїРѕР»СЊР·СѓР№С‚Рµ В«shВ» вЂ” РЅСѓР¶РµРЅ bash.
set -eo pipefail

WORLD="${1:-diff_drive}"
if [[ -n "${2:-}" ]]; then
  SPAWN_Z="$2"
elif [[ -n "${COURSE_ROBOT_SPAWN_Z:-}" ]]; then
  SPAWN_Z="$COURSE_ROBOT_SPAWN_Z"
else
  # Р’С‹СЃРѕС‚Р° С†РµРЅС‚СЂР° РјРѕРґРµР»Рё РІ РјРёСЂРµ (Рј). Р СЏРґРѕРј СЃ vehicle_blue (~0.325). РўРѕРЅРµС‚ в†’ +0.05; РІ РЅРµР±Рµ в†’ в€’0.05
  SPAWN_Z="0.36"
fi
SPAWN_X="${COURSE_ROBOT_SPAWN_X:-1.5}"
SPAWN_Y="${COURSE_ROBOT_SPAWN_Y:-2.0}"
MODEL_NAME="${3:-course_robot}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ":${GZ_SIM_RESOURCE_PATH:-}:" != *":${ROOT}:"* ]]; then
  export GZ_SIM_RESOURCE_PATH="${ROOT}${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
fi

SDF_SRC="${ROOT}/course_robot/model.sdf"
if [[ ! -f "$SDF_SRC" ]]; then
  echo "РќРµС‚ С„Р°Р№Р»Р°: $SDF_SRC" >&2
  exit 1
fi

TMP_SDF="/tmp/course_robot_spawn_$$.sdf"
cp -f "$SDF_SRC" "$TMP_SDF"
MNAME="$MODEL_NAME"
sed -i "s/<model name=\"course_robot\">/<model name=\"${MNAME}\">/" "$TMP_SDF"

echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo "SDF РґР»СЏ gz: $TMP_SDF"
echo "РРјСЏ СЌРєР·РµРјРїР»СЏСЂР°: $MNAME | РІС‹СЃРѕС‚Р° С†РµРЅС‚СЂР° РјРѕРґРµР»Рё РІ РјРёСЂРµ Z=${SPAWN_Z} Рј"

echo "=== РњРѕРґРµР»Рё (РґРѕ) ==="
gz model --list 2>/dev/null || true

REQ="sdf_filename: \"${TMP_SDF}\", name: \"${MNAME}\", allow_renaming: true, pose: {position: {x: ${SPAWN_X}, y: ${SPAWN_Y}, z: ${SPAWN_Z}}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"
echo "EntityFactory pose: x=${SPAWN_X} y=${SPAWN_Y} z=${SPAWN_Z}"

if ! gz service -s "/world/${WORLD}/create" \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 10000 \
  --req "$REQ"; then
  echo "Р’С‹Р·РѕРІ gz service РЅРµ СѓРґР°Р»СЃСЏ." >&2
  gz service --list 2>/dev/null | grep -F '/create' || true
  exit 1
fi

sleep 0.5
echo "=== РњРѕРґРµР»Рё (РїРѕСЃР»Рµ) ==="
gz model --list 2>/dev/null || true

echo "РўРѕРЅРµС‚ РІ РїРѕР»: вЂ¦ ${WORLD} 0.42  |  Р’ РЅРµР±Рµ: вЂ¦ ${WORLD} 0.30"
echo "РљРѕСЂРїСѓСЃ РѕС‚РЅРѕСЃРёС‚РµР»СЊРЅРѕ В«РЅРѕР¶РµРєВ»: РїСЂР°РІСЊС‚Рµ model.sdf в†’ base_link <pose> z (С€Р°Рі 0.02)"
