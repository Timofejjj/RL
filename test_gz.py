import subprocess
cmd = [
    'gz', 'service', '-s', '/world/course_world/set_pose',
    '--reqtype', 'gz.msgs.Pose',
    '--reptype', 'gz.msgs.Boolean',
    '--timeout', '2000',
    '--req', 'name: "course_robot" position { x: 0 y: -4.6 z: 0.36 } orientation { x: 0 y: 0 z: 0 w: 1 }'
]
proc = subprocess.run(cmd, capture_output=True, text=True)
print('STDOUT:', repr(proc.stdout))
print('STDERR:', repr(proc.stderr))
print('EXIT CODE:', proc.returncode)
