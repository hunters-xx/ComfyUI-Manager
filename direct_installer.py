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
import manager_core as core
import manager_util
import cm_global
import folder_paths

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

# Set paths
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager_path = os.path.join(comfy_path, "custom_nodes", "comfyui-manager")

for path in [comfy_path, manager_path, os.path.join(manager_path, "glob")]:
    sys.path.insert(0, path)

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
    # Model directory names used across the codebase
    MODEL_DIR_NAMES = [
        'checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models',
        'clip_vision', 'embeddings', 'diffusers', 'vae_approx', 'controlnet',
        'gligen', 'upscale_models', 'hypernetworks', 'photomaker', 'classifiers', 'checkpoint'
    ]
    
    # Model directory name mapping
    MODEL_DIR_NAME_MAP = {
        "checkpoints": "checkpoints", "checkpoint": "checkpoints", "unclip": "checkpoints",
        "text_encoders": "text_encoders", "clip": "text_encoders",
        "vae": "vae", "lora": "loras",
        "t2i-adapter": "controlnet", "t2i-style": "controlnet", "controlnet": "controlnet",
        "clip_vision": "clip_vision", "gligen": "gligen", "upscale": "upscale_models",
        "embedding": "embeddings", "embeddings": "embeddings",
        "unet": "diffusion_models", "diffusion_model": "diffusion_models",
    }
    def __init__(self, file_url: str, interval: int = DEFAULT_INTERVAL):
        self.file_url = file_url
        self.interval = interval
        self.installed_nodes: Set[str] = set()
        self.installed_models: Set[str] = set()
        self.running = False
        self._cached_node_packs: Optional[Dict] = None
        self._cache_timestamp = 0
        self._last_version: Optional[str] = None  # Record last processed version number
        self._custom_nodes_dir: Optional[str] = None  # Cache custom nodes directory

    def _get_custom_nodes_dir(self) -> str:
        """Get custom nodes directory (with cache)"""
        if self._custom_nodes_dir is None:
            self._custom_nodes_dir = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
        return self._custom_nodes_dir

    def _get_model_dir(self, save_path: str, model_type: str) -> str:
        """Get model directory path based on save_path and model_type"""
        if save_path == "default":
            # Use default path mapping
            model_dir_name = self.MODEL_DIR_NAME_MAP.get(model_type.lower(), model_type)
            if model_dir_name == "upscale":
                model_dir_name = "upscale_models"
            elif model_dir_name == "checkpoint":
                model_dir_name = "checkpoints"
            return os.path.join(folder_paths.models_dir, model_dir_name)
        else:
            # Use custom path, correct path mapping
            if save_path.startswith('checkpoints/'):
                corrected_path = save_path.replace('checkpoints/', 'checkpoint/')
                return os.path.join(folder_paths.models_dir, corrected_path)
            return os.path.join(folder_paths.models_dir, save_path)

    @staticmethod
    def check_model_installed(json_obj):
        """Check if model is already installed"""
        def is_exists(model_dir_name, filename, url):
            if filename == HUGGINGFACE_PLACEHOLDER:
                filename = os.path.basename(url)
            return any(os.path.exists(os.path.join(d, filename)) for d in folder_paths.get_folder_paths(model_dir_name))

        # Get all installed model files
        total_models_files = {f for dir_name in FileDirectInstaller.MODEL_DIR_NAMES for f in folder_paths.get_filename_list(dir_name)}

        def process_model(item):
            # Check common filename
            if not any(x in item['filename'] for x in ['diffusion', 'pytorch', 'model']):
                if item['filename'] in total_models_files:
                    item['installed'] = 'True'
                    return

            # Check default path
            if item['save_path'] == 'default':
                model_dir = FileDirectInstaller.MODEL_DIR_NAME_MAP.get(item['type'].lower())
                item['installed'] = str(is_exists(model_dir, item['filename'], item['url'])) if model_dir else 'False'
            else:
                # Check custom path - directly check full path
                filename = os.path.basename(item['url']) if item['filename'] == HUGGINGFACE_PLACEHOLDER else item['filename']
                save_path = item['save_path'].replace('checkpoints/', 'checkpoint/')
                fullpath = os.path.join(folder_paths.models_dir, save_path, filename)
                item['installed'] = 'True' if os.path.exists(fullpath) else 'False'

        with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as executor:
            futures = [executor.submit(process_model, item) for item in json_obj['models']]
            concurrent.futures.wait(futures)  # Wait for all tasks to complete

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
            for dir_name in FileDirectInstaller.MODEL_DIR_NAMES:
                try:
                    total_models_files.update(folder_paths.get_filename_list(dir_name))
                except (AttributeError, OSError) as e:
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

    async def install_system_dependencies(self, dependencies: List[str], os_type: str) -> bool:
        """Install system dependencies automatically"""
        try:
            package_manager = self.check_package_manager()
            if not package_manager:
                logger.error("No suitable package manager found")
                return False

            logger.info(f"Installing system dependencies using {package_manager}...")
            
            if os_type == "ubuntu" and package_manager == "apt":
                subprocess.run(['sudo', 'apt', 'update'], check=True, timeout=SUBPROCESS_TIMEOUT_MEDIUM)
                result = subprocess.run(['sudo', 'apt', 'install', '-y'] + dependencies, check=True, timeout=SUBPROCESS_TIMEOUT_LONG)
                return result.returncode == 0
            elif os_type == "centos" and package_manager in ["yum", "dnf"]:
                result = subprocess.run(['sudo', package_manager, 'install', '-y'] + dependencies, check=True, timeout=SUBPROCESS_TIMEOUT_LONG)
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
        except Exception as e:
            logger.error(f"Unexpected error installing system dependencies: {e}")
            return False

    async def check_system_dependencies(self, data: Dict = None) -> bool:
        """Check if system dependencies meet requirements"""
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
                    return True
                else:
                    logger.error("❌ Failed to install system dependencies automatically")
                    logger.info(f"Please install manually: {', '.join(missing_deps)}")
                    return False
            
            logger.info("✅ All system dependencies are satisfied")
            return True
            
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to check system dependencies: {e}")
            return True  # Continue execution even if dependency check fails

    async def install_custom_node(self, node_data: Dict, system_data: Dict = None) -> bool:
        """Install custom node"""
        try:
            node_id = node_data.get("id", "") or node_data.get("file_name", "").lower()
            install_type = node_data.get("install_type", "")
            files = node_data.get("files", [])

            if not node_id:
                return False
            
            # Check system dependencies with data from install.json
            if not await self.check_system_dependencies(system_data):
                logger.warning(f"System dependency check failed, skipping node installation: {node_id}")
                return False

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
            model_type = model_data.get("type", "checkpoint")
            
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
                model_name = model_data.get("filename", "")
                
                # Use manager_server check_model_installed logic to check if model is installed
                temp_json_obj = {"models": [model_data.copy()]}
                self.check_model_installed(temp_json_obj)
                
                # Check if model is already installed
                if temp_json_obj["models"][0].get('installed') == 'True':
                    skipped_models += 1
                    logger.debug(f"Model already exists, skipping installation: {model_name}")
                    continue
                
                if await self.install_model(model_data):
                    new_models += 1

            logger.info(f"Model processing completed: new {new_models}, skipped {skipped_models}")

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to process unified resource data: {e}")

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
    parser.add_argument("--restart-command", help="Restart command (deprecated, kept for compatibility)")

    args = parser.parse_args()

    installer = FileDirectInstaller(
        file_url=args.resource_url,
        interval=args.interval
    )

    try:
        asyncio.run(installer.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service runtime exception: {e}")

if __name__ == "__main__":
    main()
