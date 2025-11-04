#!/usr/bin/env python3
"""
File Unified Resource Installation Service - Direct Call Version
Directly calls manager_server functions, avoiding HTTP API calls
"""

import asyncio
import json
import logging
import argparse
import time
import sys
import os
import subprocess
import zipfile
import tempfile
import concurrent.futures
import aiohttp
import platform
import shutil
from typing import Dict, Set, List, Optional, Tuple
from datetime import datetime

# Constants
CACHE_TIMEOUT = 30  # Cache timeout in seconds
DEFAULT_INTERVAL = 60  # Default check interval in seconds
DEFAULT_CHECK_INTERVAL_ON_ERROR = 5  # Wait time on error
MAX_WORKERS = 8  # Max workers for thread pool
NODE_PREFIXES = ['comfyui-', 'ComfyUI_']  # Node ID prefixes to normalize
HUGGINGFACE_PLACEHOLDER = '<huggingface>'
DEFAULT_OS_TYPE = "ubuntu"
HTTP_TIMEOUT = 30  # HTTP request timeout in seconds
HTTP_MAX_RETRIES = 3  # Maximum retry attempts for HTTP requests
SUBPROCESS_TIMEOUT_SHORT = 5  # Short timeout for subprocess commands
SUBPROCESS_TIMEOUT_MEDIUM = 60  # Medium timeout for subprocess commands
SUBPROCESS_TIMEOUT_LONG = 300  # Long timeout for subprocess commands

# Set paths first before importing modules from glob directory
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager_path = os.path.join(comfy_path, "custom_nodes", "comfyui-manager")

for path in [comfy_path, manager_path, os.path.join(manager_path, "glob")]:
    sys.path.insert(0, path)

# Import modules after paths are set
import manager_core as core
import manager_util
import cm_global
import folder_paths

# Ensure using virtual environment Python path
manager_util.add_python_path_to_env()

# Set environment variables to ensure subprocess uses correct Python
os.environ['PYTHON_EXECUTABLE'] = sys.executable

# Ensure sys.executable points to virtual environment Python
virtual_env_python = os.path.join(comfy_path, '.venv', 'bin', 'python')
if os.path.exists(virtual_env_python):
    sys.executable = virtual_env_python
    print(f"Set sys.executable to: {sys.executable}")

# Initialize ComfyUI core settings
cm_global.pip_overrides = {}
cm_global.pip_blacklist = {'torch', 'torchaudio', 'torchsde', 'torchvision'}
cm_global.pip_downgrade_blacklist = ['torch', 'torchaudio', 'torchsde', 'torchvision', 'transformers', 'safetensors', 'kornia']
core.comfy_ui_revision = "Unknown"
core.comfy_ui_commit_datetime = datetime(1900, 1, 1, 0, 0, 0)

# Configure logging
for logger_name in ["ComfyUI-Manager", "manager_util"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("directInstaller")

# Load configuration files
def _load_config_files():
    """Load configuration files and override defaults"""
    def load_config_file(filename, default_value, loader_func):
        file_path = os.path.join(manager_util.comfyui_manager_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                    return loader_func(f)
            except (IOError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load config file {filename}: {e}")
        return default_value

    cm_global.pip_overrides.update(load_config_file("pip_overrides.json", {}, json.load))
    cm_global.pip_blacklist.update(load_config_file("pip_blacklist.list", [], lambda f: [line.strip() for line in f if line.strip()]))

_load_config_files()


class FileDirectInstaller:
    """File Unified Resource Installation Service"""
    # Model directory names used in folder_paths (valid keys only, no 'checkpoint')
    MODEL_DIR_NAMES = [
        'checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models',
        'clip_vision', 'embeddings', 'diffusers', 'vae_approx', 'controlnet',
        'gligen', 'upscale_models', 'hypernetworks', 'photomaker', 'classifiers'
    ]
    
    # Model directory name mapping (maps model type to folder_paths key)
    # This matches manager_server.py model_dir_name_map
    MODEL_DIR_NAME_MAP = {
        "checkpoints": "checkpoints",
        "checkpoint": "checkpoints",  # Map 'checkpoint' to 'checkpoints'
        "unclip": "checkpoints",
        "text_encoders": "text_encoders",
        "clip": "text_encoders",
        "vae": "vae",
        "lora": "loras",
        "t2i-adapter": "controlnet",
        "t2i-style": "controlnet",
        "controlnet": "controlnet",
        "clip_vision": "clip_vision",
        "gligen": "gligen",
        "upscale": "upscale_models",
        "embedding": "embeddings",
        "embeddings": "embeddings",
        "unet": "diffusion_models",
        "diffusion_model": "diffusion_models",
    }
    def __init__(self, file_url: str, interval: int = DEFAULT_INTERVAL, restart_command: Optional[str] = None):
        self.file_url = file_url
        self.interval = interval
        self.restart_command = restart_command or os.environ.get('COMFYUI_RESTART_COMMAND')
        self.installed_nodes: Set[str] = set()
        self.installed_models: Set[str] = set()
        self.running = False
        self._cached_node_packs: Optional[Dict] = None
        self._cache_timestamp = 0
        self._last_version: Optional[str] = None  # Record last processed version number
        self._custom_nodes_dir: Optional[str] = None  # Cache custom nodes directory
        self._system_deps_checked: bool = False  # Cache system dependency check result
        self._system_deps_ok: bool = True  # Default to True to allow installation
        self._pending_restart: bool = False  # Flag to indicate if restart is needed

    def _get_custom_nodes_dir(self) -> str:
        """Get custom nodes directory (with cache)"""
        if self._custom_nodes_dir is None:
            self._custom_nodes_dir = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
        return self._custom_nodes_dir

    def _get_model_dir(self, save_path: str, model_type: str) -> str:
        """Get model directory path based on save_path and model_type - matches manager_server.py get_model_dir"""
        # Get models base directory
        if 'download_model_base' in folder_paths.folder_names_and_paths:
            models_base = folder_paths.folder_names_and_paths['download_model_base'][0][0]
        else:
            models_base = folder_paths.models_dir
        
        if save_path == "default":
            # Use model_dir_name_map to get the correct folder_paths key
            model_dir_name = self.MODEL_DIR_NAME_MAP.get(model_type.lower())
            if model_dir_name is not None:
                # Use folder_paths to get the actual directory path
                return folder_paths.folder_names_and_paths[model_dir_name][0][0]
            else:
                # Unknown type, save to etc
                return os.path.join(models_base, "etc")
        else:
            # Use custom path
            # Validate to prevent path traversal
            if '..' in save_path or save_path.startswith('/'):
                logger.warning(f"Invalid save_path '{save_path}', saving to models/etc")
                return os.path.join(models_base, "etc")
            return os.path.join(models_base, save_path)

    @staticmethod
    def check_model_installed(json_obj):
        """Check if model is already installed - matches manager_server.py logic"""
        def is_exists(model_dir_name, filename, url):
            if filename == HUGGINGFACE_PLACEHOLDER:
                filename = os.path.basename(url)
            
            try:
                dirs = folder_paths.get_folder_paths(model_dir_name)
                for x in dirs:
                    if os.path.exists(os.path.join(x, filename)):
                        return True
            except (KeyError, AttributeError):
                pass
            return False

        # Get all installed model files from valid folder_paths directories
        total_models_files = set()
        for dir_name in FileDirectInstaller.MODEL_DIR_NAMES:
            try:
                for filename in folder_paths.get_filename_list(dir_name):
                    total_models_files.add(filename)
            except (AttributeError, KeyError, OSError):
                pass

        def process_model_phase(item):
            """Process a single model - matches manager_server.py logic"""
            try:
                # Check common filename (non-general name case)
                if 'diffusion' not in item['filename'] and 'pytorch' not in item['filename'] and 'model' not in item['filename']:
                    if item['filename'] in total_models_files:
                        item['installed'] = 'True'
                        return

                # Check default path
                if item.get('save_path') == 'default':
                    model_dir_name = FileDirectInstaller.MODEL_DIR_NAME_MAP.get(item.get('type', '').lower())
                    if model_dir_name is not None:
                        item['installed'] = str(is_exists(model_dir_name, item['filename'], item.get('url', '')))
                    else:
                        item['installed'] = 'False'
                else:
                    # Check custom path
                    model_dir_name = item.get('save_path', '').split('/')[0]
                    if model_dir_name in folder_paths.folder_names_and_paths:
                        if is_exists(model_dir_name, item['filename'], item.get('url', '')):
                            item['installed'] = 'True'

                    if 'installed' not in item:
                        if item.get('filename') == HUGGINGFACE_PLACEHOLDER:
                            filename = os.path.basename(item.get('url', ''))
                        else:
                            filename = item['filename']
                        fullpath = os.path.join(folder_paths.models_dir, item.get('save_path', ''), filename)
                        item['installed'] = 'True' if os.path.exists(fullpath) else 'False'
            except (KeyError, AttributeError, TypeError) as e:
                # If any error occurs, mark as not installed
                item['installed'] = 'False'

        with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as executor:
            for item in json_obj['models']:
                executor.submit(process_model_phase, item)

    def _log_install_result(self, item_id: str, success: bool, action: str = None, error_msg: str = None):
        """Unified installation result logging"""
        if action == 'skip':
            logger.info(f"Already exists, skipping installation: {item_id}")
        elif success:
            logger.info(f"✅ Installation successful: {item_id}")
        else:
            logger.error(f"❌ Installation failed: {item_id}" + (f", error: {error_msg}" if error_msg else ""))

    async def _download_file(self, url: str, max_retries: int = HTTP_MAX_RETRIES) -> bytes:
        """Download file from URL with retry mechanism"""
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Download attempt {attempt + 1} failed for {url}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
        
        raise last_error

    async def _install_copy_node(self, node_id: str, file_url: str) -> bool:
        """Install copy type node"""
        try:
            custom_nodes_dir = self._get_custom_nodes_dir()
            node_file_path = os.path.join(custom_nodes_dir, f"{node_id}.py")
            
            if os.path.exists(node_file_path):
                self._log_install_result(node_id, False, 'skip')
                return False
            
            content = await self._download_file(file_url)
            
            os.makedirs(custom_nodes_dir, exist_ok=True)
            with open(node_file_path, 'wb') as f:
                f.write(content)
            
            self._log_install_result(node_id, True)
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, IOError, OSError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_unzip_node(self, node_id: str, zip_url: str) -> bool:
        """Install unzip type node"""
        temp_zip_path = None
        try:
            custom_nodes_dir = self._get_custom_nodes_dir()
            node_dir_path = os.path.join(custom_nodes_dir, node_id)
            
            if os.path.exists(node_dir_path):
                self._log_install_result(node_id, False, 'skip')
                return False
            
            content = await self._download_file(zip_url)
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_file.write(content)
                temp_zip_path = temp_file.name
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(custom_nodes_dir)
            
            self._log_install_result(node_id, True)
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError, zipfile.BadZipFile, IOError, OSError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False
        finally:
            if temp_zip_path and os.path.exists(temp_zip_path):
                try:
                    os.unlink(temp_zip_path)
                except OSError:
                    pass

    async def _install_pip_node(self, node_id: str, pip_packages: List[str]) -> bool:
        """Install pip type node"""
        try:
            if not pip_packages:
                self._log_install_result(node_id, False, error_msg="No pip packages specified")
                return False
            
            core.pip_install(pip_packages)
            self._log_install_result(node_id, True)
            return True
        except (subprocess.CalledProcessError, RuntimeError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_cnr_node(self, node_id: str, version_spec: Optional[str] = None) -> bool:
        """Install cnr type node with optional version specification"""
        try:
            result = await core.unified_manager.install_by_id(
                node_id, version_spec=version_spec, channel='default', mode='cache'
            )
            self._log_install_result(node_id, result.result, result.action, result.msg)
            # If skipped, return False indicating no new installation
            return result.result and result.action != 'skip'
        except (AttributeError, RuntimeError, ValueError) as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    def _normalize_node_id(self, node_id: str) -> Set[str]:
        """Normalize node ID to various formats for matching"""
        normalized = set()
        normalized.update([node_id, node_id.lower()])
        
        for prefix in NODE_PREFIXES:
            if node_id.startswith(prefix):
                clean_id = node_id.replace(prefix, '').lower()
                normalized.add(clean_id)
                normalized.add(clean_id.replace('_', '-'))
                if '_' in clean_id:
                    parts = clean_id.split('_')
                    for i in range(1, len(parts) + 1):
                        normalized.add('-'.join(parts[:i]))
        
        return normalized

    async def get_installed_nodes(self) -> Set[str]:
        """Get list of installed custom nodes (with cache)"""
        try:
            # Use cache to avoid repeated calls
            current_time = time.time()
            if self._cached_node_packs is None or (current_time - self._cache_timestamp) > CACHE_TIMEOUT:
                self._cached_node_packs = core.get_installed_node_packs()
                self._cache_timestamp = current_time
            
            nodes = set()
            for node_id, node_info in self._cached_node_packs.items():
                nodes.update(self._normalize_node_id(node_id))
                
                cnr_id = node_info.get('cnr_id', '')
                if cnr_id:
                    nodes.update(self._normalize_node_id(cnr_id))
                    
            return nodes
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to get installed nodes: {e}")
            return set()

    async def get_installed_models(self) -> Set[str]:
        """Get list of installed models"""
        try:
            total_models_files = set()
            # Filter out 'checkpoint' as it's not a valid folder_paths key (use 'checkpoints' instead)
            valid_dir_names = [d for d in FileDirectInstaller.MODEL_DIR_NAMES if d != 'checkpoint']
            for dir_name in valid_dir_names:
                try:
                    total_models_files.update(folder_paths.get_filename_list(dir_name))
                except (AttributeError, KeyError, OSError) as e:
                    logger.debug(f"Failed to get files from {dir_name}: {e}")
                    pass
                    
            return total_models_files
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to get installed models: {e}")
            return set()

    def detect_os_type(self) -> str:
        """Detect operating system type"""
        system = platform.system().lower()
        if system == "linux":
            try:
                with open("/etc/os-release", "r", encoding='utf-8') as f:
                    content = f.read().lower()
                    if "ubuntu" in content or "debian" in content:
                        return "ubuntu"
                    elif "centos" in content or "rhel" in content or "fedora" in content:
                        return "centos"
            except (IOError, OSError):
                pass
            return DEFAULT_OS_TYPE  # Default to ubuntu for Linux
        elif system == "darwin":
            return "macos"
        else:
            return DEFAULT_OS_TYPE  # Default fallback

    def check_package_manager(self) -> str:
        """Check which package manager is available"""
        if shutil.which("apt"):
            return "apt"
        elif shutil.which("yum"):
            return "yum"
        elif shutil.which("dnf"):
            return "dnf"
        elif shutil.which("brew"):
            return "brew"
        else:
            return None

    def check_single_package(self, package: str, os_type: str) -> bool:
        """Check if a single package is installed"""
        try:
            if os_type == "ubuntu":
                result = subprocess.run(['dpkg', '-l', package], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SHORT)
                return 'ii' in result.stdout
            elif os_type == "centos":
                result = subprocess.run(['rpm', '-q', package], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SHORT)
                return result.returncode == 0
            elif os_type == "macos":
                result = subprocess.run(['brew', 'list', package], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SHORT)
                return result.returncode == 0
            return False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return False

    def _needs_sudo(self) -> bool:
        """Check if sudo is needed (not root user)"""
        try:
            return os.geteuid() != 0
        except AttributeError:
            # On Windows, geteuid doesn't exist, assume no sudo needed
            return False

    async def install_system_dependencies(self, dependencies: List[str], os_type: str) -> bool:
        """Install system dependencies automatically"""
        try:
            package_manager = self.check_package_manager()
            if not package_manager:
                logger.error("No suitable package manager found")
                return False

            # Check if sudo is needed and available
            use_sudo = self._needs_sudo()
            if use_sudo and not shutil.which("sudo"):
                logger.warning("sudo is required but not found, skipping automatic installation")
                return False

            logger.info(f"Installing system dependencies using {package_manager}...")
            
            def run_command(cmd, timeout):
                """Helper to run command with or without sudo"""
                if use_sudo:
                    cmd = ['sudo'] + cmd
                return subprocess.run(cmd, check=True, timeout=timeout)
            
            if os_type == "ubuntu" and package_manager == "apt":
                run_command(['apt', 'update'], SUBPROCESS_TIMEOUT_MEDIUM)
                result = run_command(['apt', 'install', '-y'] + dependencies, SUBPROCESS_TIMEOUT_LONG)
                return result.returncode == 0
            elif os_type == "centos" and package_manager in ["yum", "dnf"]:
                result = run_command([package_manager, 'install', '-y'] + dependencies, SUBPROCESS_TIMEOUT_LONG)
                return result.returncode == 0
            elif os_type == "macos" and package_manager == "brew":
                result = subprocess.run(['brew', 'install'] + dependencies, check=True, timeout=SUBPROCESS_TIMEOUT_LONG)
                return result.returncode == 0
            
            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout while installing system dependencies")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install system dependencies: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"Command not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing system dependencies: {e}")
            return False

    async def check_system_dependencies(self, data: Dict = None, force_recheck: bool = False) -> bool:
        """Check if system dependencies meet requirements"""
        # Use cached result if already checked
        if self._system_deps_checked and not force_recheck:
            return self._system_deps_ok
            
        try:
            # Get system dependencies from data if available
            if data and 'system_dependencies' in data:
                os_type = self.detect_os_type()
                required_deps = data['system_dependencies'].get(os_type, [])
            else:
                # Fallback to default dependencies
                required_deps = ['libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libfreetype-dev']
                os_type = DEFAULT_OS_TYPE
            
            if not required_deps:
                logger.info("No system dependencies required")
                self._system_deps_checked = True
                self._system_deps_ok = True
                return True
            
            logger.info(f"Checking system dependencies for {os_type}...")
            
            missing_deps = []
            for dep in required_deps:
                if not self.check_single_package(dep, os_type):
                    missing_deps.append(dep)
            
            if missing_deps:
                logger.warning(f"Missing system dependencies: {', '.join(missing_deps)}")
                
                # Try to install automatically
                logger.info("Attempting to install missing dependencies automatically...")
                if await self.install_system_dependencies(missing_deps, os_type):
                    logger.info("✅ System dependencies installed successfully")
                    self._system_deps_checked = True
                    self._system_deps_ok = True
                    return True
                else:
                    logger.error("❌ Failed to install system dependencies automatically")
                    logger.info(f"Please install manually: {', '.join(missing_deps)}")
                    # Don't block installation if dependencies are missing, just warn
                    logger.warning("Continuing with installation despite missing dependencies...")
                    self._system_deps_checked = True
                    self._system_deps_ok = True  # Allow installation to continue
                    return True
            
            logger.info("✅ All system dependencies are satisfied")
            self._system_deps_checked = True
            self._system_deps_ok = True
            return True
            
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to check system dependencies: {e}")
            self._system_deps_checked = True
            self._system_deps_ok = True  # Continue execution even if dependency check fails
            return True

    async def install_custom_node(self, node_data: Dict, system_data: Dict = None) -> bool:
        """Install custom node"""
        try:
            node_id = node_data.get("id", "") or node_data.get("file_name", "").lower()
            install_type = node_data.get("install_type", "")
            files = node_data.get("files", [])

            if not node_id:
                return False
            
            # System dependencies are checked once in process_unified_data, skip here to avoid duplicate checks
            # Only check if not already checked (shouldn't happen, but just in case)
            if not self._system_deps_checked:
                await self.check_system_dependencies(system_data)

            # Check installation type and select appropriate installation method
            if install_type == "git-clone" and files:
                # Support version selection via version field
                # Format: url@branch, url@tag, or url@commit
                git_url = files[0]
                version = node_data.get("version", "")
                if version:
                    # If URL already contains @, don't add another one
                    if "@" not in git_url:
                        git_url = f"{git_url}@{version}"
                        logger.info(f"Installing {node_id} with version: {version}")
                
                result = await core.gitclone_install(git_url, instant_execution=True, no_deps=False)
                self._log_install_result(node_id, result.result, result.action, result.msg)
                # If skipped, return False indicating no new installation
                return result.result and result.action != 'skip'
            elif install_type == "copy" and files:
                return await self._install_copy_node(node_id, files[0])
            elif install_type == "unzip" and files:
                return await self._install_unzip_node(node_id, files[0])
            elif install_type == "pip":
                return await self._install_pip_node(node_id, node_data.get("pip", []))
            elif install_type == "cnr":
                # Support version selection for CNR nodes
                version = node_data.get("version", None)
                version_spec = None if version in ["latest", ""] else version
                return await self._install_cnr_node(node_id, version_spec)
            else:
                # For other installation types, skip for now
                logger.warning(f"Unsupported installation type: {install_type} for {node_id}")
                return False

        except (KeyError, AttributeError, ValueError) as e:
            logger.error(f"Failed to install custom node: {e}, node: {node_data}")
            return False

    async def install_model(self, model_data: Dict) -> bool:
        """Install model"""
        try:
            model_name = model_data.get("filename", "")
            if not model_name or model_name in self.installed_models:
                return False

            logger.info(f"Starting model download: {model_name}")
            
            # Get model save path and type
            save_path = model_data.get("save_path", "default")
            model_type = model_data.get("type", "checkpoints")  # Use 'checkpoints' instead of 'checkpoint'
            
            # Get model directory using helper method
            model_dir = self._get_model_dir(save_path, model_type)
            
            # Ensure directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            # Download file
            try:
                import manager_downloader
                manager_downloader.download_url(
                    model_data['url'], 
                    model_dir, 
                    model_name
                )
                
                # Check if file downloaded successfully
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    self.installed_models.add(model_name)
                    logger.info(f"✅ Model download successful: {model_name}")
                    return True
                else:
                    logger.error(f"❌ Model file does not exist or is empty: {model_name}")
                    return False
                    
            except (IOError, OSError, RuntimeError) as e:
                logger.error(f"❌ Model download exception: {model_name}, error: {e}")
                return False

        except (KeyError, AttributeError, ValueError) as e:
            logger.error(f"Failed to install model: {e}, model: {model_data}")
            return False

    async def fetch_file_data(self) -> Dict:
        """Fetch unified resource data from url with timeout"""
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.file_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get('version', 'unknown')
                        node_count = len(data.get('custom_nodes', []))
                        model_count = len(data.get('models', []))
                        logger.info(f"Successfully fetched unified resource list, version: {version}, nodes: {node_count}, models: {model_count}")
                        return data
                    else:
                        logger.error(f"Failed to fetch file data: HTTP {response.status}")
                        return {}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching file data from {self.file_url}")
            return {}
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.error(f"File data fetch exception: {e}")
            return {}

    async def process_unified_data(self, data: Dict):
        """Process unified resource data"""
        try:
            logger.info("Starting to process unified resource data...")
            
            # Check system dependencies first
            logger.info("Checking system dependencies...")
            if not await self.check_system_dependencies(data):
                logger.warning("System dependency check failed, but continuing with installation...")
            
            # Update installed list
            self.installed_nodes = await self.get_installed_nodes()
            self.installed_models = await self.get_installed_models()
            
            logger.info(f"Installed: {len(self.installed_nodes)} nodes, {len(self.installed_models)} models")

            # Process custom nodes
            custom_nodes = data.get("custom_nodes", [])
            new_nodes = 0
            skipped_nodes = 0
            
            for node_data in custom_nodes:
                # Use id if available, otherwise use file_name in lowercase
                node_id = node_data.get("id", "") or node_data.get("file_name", "").lower()
                # Check if already installed
                if node_id in self.installed_nodes:
                    skipped_nodes += 1
                    logger.debug(f"Custom node already exists, skipping installation: {node_id}")
                    continue
                    
                if await self.install_custom_node(node_data, data):
                    new_nodes += 1

            logger.info(f"Custom node processing completed, new: {new_nodes}, skipped: {skipped_nodes}")

            # Process models - using manager_server check logic
            new_models = 0
            skipped_models = 0
            for model_data in data.get("models", []):
                try:
                    model_name = model_data.get("filename", "")
                    if not model_name:
                        logger.warning(f"Skipping model with empty filename: {model_data}")
                        continue
                    
                    # Use manager_server check_model_installed logic to check if model is installed
                    temp_json_obj = {"models": [model_data.copy()]}
                    try:
                        self.check_model_installed(temp_json_obj)
                    except Exception as e:
                        logger.warning(f"Failed to check if model is installed: {model_name}, error: {e}, will try to install")
                        # If check fails, assume not installed and try to install
                        temp_json_obj["models"][0]['installed'] = 'False'
                    
                    # Check if model is already installed
                    if temp_json_obj["models"][0].get('installed') == 'True':
                        skipped_models += 1
                        logger.debug(f"Model already exists, skipping installation: {model_name}")
                        continue
                    
                    logger.info(f"Model {model_name} not found, starting download...")
                    if await self.install_model(model_data):
                        new_models += 1
                except Exception as e:
                    logger.error(f"Failed to process model {model_data.get('filename', 'unknown')}: {e}")
                    continue

            logger.info(f"Model processing completed: new {new_models}, skipped {skipped_models}")
            
            # Mark restart needed if any new installations
            if new_nodes > 0 or new_models > 0:
                self._pending_restart = True
                logger.info(f"New installations detected: {new_nodes} nodes, {new_models} models. Restart may be needed.")

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to process unified resource data: {e}")
    
    async def execute_restart_command(self):
        """Execute restart command if configured"""
        if not self.restart_command:
            return
        
        if not self._pending_restart:
            logger.debug("No pending restart needed")
            return
        
        try:
            logger.info(f"Executing restart command: {self.restart_command}")
            # Use shell=True to support complex commands
            result = subprocess.run(
                self.restart_command,
                shell=True,
                timeout=30,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✅ Restart command executed successfully")
                self._pending_restart = False
            else:
                logger.warning(f"Restart command returned non-zero exit code: {result.returncode}")
                if result.stderr:
                    logger.warning(f"Restart command stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("Restart command execution timeout")
        except Exception as e:
            logger.error(f"Failed to execute restart command: {e}")

    async def run(self):
        """Run main loop"""
        logger.info(f"file direct installation service started, file URL: {self.file_url}, check interval: {self.interval} seconds")
        self.running = True

        while self.running:
            try:
                # Fetch file data
                data = await self.fetch_file_data()
                if data:
                    current_version = data.get('version', 'unknown')
                    
                    # Check if version has changed
                    if self._last_version is None:
                        logger.info(f"First run, processing version: {current_version}")
                        await self.process_unified_data(data)
                        self._last_version = current_version
                    elif self._last_version != current_version:
                        logger.info(f"Version change detected: {self._last_version} -> {current_version}, starting to process update")
                        await self.process_unified_data(data)
                        self._last_version = current_version
                        
                        # Execute restart command if needed and configured
                        if self._pending_restart:
                            await self.execute_restart_command()
                    else:
                        logger.info(f"No version change ({current_version}), skipping processing")
                else:
                    logger.warning("Failed to fetch file data, skipping this check")

                # Wait for next check
                await asyncio.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("Received stop signal, shutting down service...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop exception: {e}")
                await asyncio.sleep(DEFAULT_CHECK_INTERVAL_ON_ERROR)

        logger.info("File direct installation service stopped")

def main():
    parser = argparse.ArgumentParser(description="File Unified Resource Installation Service - Direct Call Version")
    parser.add_argument("--resource-url", required=True, help="File JSON file URL")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Check interval (seconds)")
    parser.add_argument("--comfyui-path", help="ComfyUI path (deprecated, kept for compatibility)")
    parser.add_argument("--restart-command", help="Restart command to execute after new installations (e.g., 'pm2 restart comfyui' or 'systemctl restart comfyui')")

    args = parser.parse_args()

    installer = FileDirectInstaller(
        file_url=args.resource_url,
        interval=args.interval,
        restart_command=args.restart_command
    )

    try:
        asyncio.run(installer.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service runtime exception: {e}")

if __name__ == "__main__":
    main()
