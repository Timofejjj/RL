# Spawns course_robot into a running gz sim world (e.g. diff_drive from ros_gz_sim_demos).
# Requires: gz in PATH, same machine as the running simulator, UserCommands / create service enabled.
# Usage: .\spawn_course_robot.ps1
#        .\spawn_course_robot.ps1 -WorldName my_world

param(
    [string]$WorldName = "diff_drive"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Join-Path $root "course_robot"
$sdf = Join-Path $modelDir "model.sdf"
$sdfUri = ($sdf -replace '\\', '/')

if (-not (Test-Path $sdf)) {
    Write-Error "Missing model: $sdf"
    exit 1
}

$env:GZ_SIM_RESOURCE_PATH = $root

$gz = Get-Command gz -ErrorAction SilentlyContinue
if (-not $gz) {
    Write-Host "gz not found in PATH. If you use WSL/Linux, run spawn_course_robot.bash there instead."
    Write-Host "Or in Gazebo Sim GUI: insert/spawn model from file:"
    Write-Host "  $sdf"
    Write-Host "Set resource path before starting sim (same terminal or system env):"
    Write-Host "  GZ_SIM_RESOURCE_PATH=$root"
    exit 1
}

$service = "/world/$WorldName/create"
$req = "sdf_filename: `"$sdfUri`", name: `"course_robot_1`""

Write-Host "GZ_SIM_RESOURCE_PATH=$root"
Write-Host "gz service -s $service ... $sdfUri"

& gz service -s $service --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 --req $req
