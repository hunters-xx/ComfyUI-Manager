module.exports = {
  apps: [{
    name: 'comfyui-installer',
    script: 'custom_nodes/comfyui-manager/direct_installer.py',
    args: '--resource-url https://pub-79bc862635254c60af6fca612486fdb9.r2.dev/install.json --interval 120',
    interpreter: '.venv/bin/python',
    cwd: '.',
    env: {
      COMFYUI_PATH: '.',
      PYTHON_EXECUTABLE: '.venv/bin/python',
      NODE_ENV: 'production'
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    time: true,
    min_uptime: '10s',
    max_restarts: 10,
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    kill_timeout: 5000,
    wait_ready: true,
    listen_timeout: 10000
  }]
};
