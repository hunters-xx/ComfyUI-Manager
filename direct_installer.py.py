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
import requests
import zipfile
import tempfile
import concurrent.futures
import aiohttp
from typing import Dict, Set
from datetime import datetime

# Set paths
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager_path = os.path.join(comfy_path, "custom_nodes", "comfyui-manager")

for path in [comfy_path, manager_path, os.path.join(manager_path, "glob")]:
    sys.path.insert(0, path)

# Import ComfyUI-Manager core modules
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

# Initialize configuration
cm_global.pip_overrides = {}
cm_global.pip_blacklist = {'torch', 'torchaudio', 'torchsde', 'torchvision'}
cm_global.pip_downgrade_blacklist = ['torch', 'torchaudio', 'torchsde', 'torchvision', 'transformers', 'safetensors', 'kornia']

core.comfy_ui_revision = "Unknown"
core.comfy_ui_commit_datetime = datetime(1900, 1, 1, 0, 0, 0)

# Configure logging level
for logger_name in ["ComfyUI-Manager", "manager_util"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Load configuration files
def load_config_file(filename, default_value, loader_func):
    file_path = os.path.join(manager_util.comfyui_manager_path, filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                return loader_func(f)
        except:
            pass
    return default_value

cm_global.pip_overrides = load_config_file("pip_overrides.json", {}, json.load)
cm_global.pip_blacklist.update(load_config_file("pip_blacklist.list", [], lambda f: [line.strip() for line in f if line.strip()]))

# Model directory mapping
model_dir_name_map = {
    "checkpoints": "checkpoints", "checkpoint": "checkpoints", "unclip": "checkpoints",
    "text_encoders": "text_encoders", "clip": "text_encoders",
    "vae": "vae", "lora": "loras",
    "t2i-adapter": "controlnet", "t2i-style": "controlnet", "controlnet": "controlnet",
    "clip_vision": "clip_vision", "gligen": "gligen", "upscale": "upscale_models",
    "embedding": "embeddings", "embeddings": "embeddings",
    "unet": "diffusion_models", "diffusion_model": "diffusion_models",
}

# Model check function
def check_model_installed(json_obj):
    """Check if model is already installed"""
    def is_exists(model_dir_name, filename, url):
        if filename == '<huggingface>':
            filename = os.path.basename(url)
        return any(os.path.exists(os.path.join(d, filename)) for d in folder_paths.get_folder_paths(model_dir_name))

    # Get all installed model files
    model_dir_names = ['checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models', 'clip_vision', 'embeddings',
                       'diffusers', 'vae_approx', 'controlnet', 'gligen', 'upscale_models', 'hypernetworks',
                       'photomaker', 'classifiers']
    total_models_files = {f for dir_name in model_dir_names for f in folder_paths.get_filename_list(dir_name)}

    def process_model(item):
        # Check common filename
        if not any(x in item['filename'] for x in ['diffusion', 'pytorch', 'model']):
            if item['filename'] in total_models_files:
                item['installed'] = 'True'
                return

        # Check default path
        if item['save_path'] == 'default':
            model_dir = model_dir_name_map.get(item['type'].lower())
            item['installed'] = str(is_exists(model_dir, item['filename'], item['url'])) if model_dir else 'False'
        else:
            # Check custom path - directly check full path
            filename = os.path.basename(item['url']) if item['filename'] == '<huggingface>' else item['filename']
            save_path = item['save_path'].replace('checkpoints/', 'checkpoint/')
            fullpath = os.path.join(folder_paths.models_dir, save_path, filename)
            item['installed'] = 'True' if os.path.exists(fullpath) else 'False'

    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for item in json_obj['models']:
            executor.submit(process_model, item)

# Configure logging - output to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("directInstaller")

class DirectInstaller:
    def __init__(self, file_url: str, interval: int = 60):
        self.file_url = file_url
        self.interval = interval
        self.installed_nodes: Set[str] = set()
        self.installed_models: Set[str] = set()
        self.running = False
        self._cached_node_packs = None
        self._cache_timestamp = 0
        self._last_version = None  # Record last processed version number

    def _log_install_result(self, item_id: str, success: bool, action: str = None, error_msg: str = None):
        """Unified installation result logging"""
        if action == 'skip':
            logger.info(f"Already exists, skipping installation: {item_id}")
        elif success:
            logger.info(f"✅ Installation successful: {item_id}")
        else:
            logger.error(f"❌ Installation failed: {item_id}" + (f", error: {error_msg}" if error_msg else ""))

    async def _install_copy_node(self, node_id: str, file_url: str) -> bool:
        """Install copy type node"""
        try:
            custom_nodes_dir = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
            node_file_path = os.path.join(custom_nodes_dir, f"{node_id}.py")
            
            if os.path.exists(node_file_path):
                self._log_install_result(node_id, False, 'skip')
                return False
            
            response = requests.get(file_url)
            response.raise_for_status()
            
            os.makedirs(custom_nodes_dir, exist_ok=True)
            with open(node_file_path, 'wb') as f:
                f.write(response.content)
            
            self._log_install_result(node_id, True)
            return True
        except Exception as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_unzip_node(self, node_id: str, zip_url: str) -> bool:
        """Install unzip type node"""
        try:
            custom_nodes_dir = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
            node_dir_path = os.path.join(custom_nodes_dir, node_id)
            
            if os.path.exists(node_dir_path):
                self._log_install_result(node_id, False, 'skip')
                return False
            
            response = requests.get(zip_url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_zip_path = temp_file.name
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(custom_nodes_dir)
            
            os.unlink(temp_zip_path)
            self._log_install_result(node_id, True)
            return True
        except Exception as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_pip_node(self, node_id: str, pip_packages: list) -> bool:
        """Install pip type node"""
        try:
            if not pip_packages:
                self._log_install_result(node_id, False, error_msg="No pip packages specified")
                return False
            
            core.pip_install(pip_packages)
            self._log_install_result(node_id, True)
            return True
        except Exception as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def _install_cnr_node(self, node_id: str) -> bool:
        """Install cnr type node"""
        try:
            result = await core.unified_manager.install_by_id(
                node_id, version_spec=None, channel='default', mode='cache'
            )
            self._log_install_result(node_id, result.result, result.action, result.msg)
            # If skipped, return False indicating no new installation
            return result.result and result.action != 'skip'
        except Exception as e:
            self._log_install_result(node_id, False, error_msg=str(e))
            return False

    async def get_installed_nodes(self) -> Set[str]:
        """Get list of installed custom nodes (with cache)"""
        try:
            # Use cache to avoid repeated calls
            current_time = time.time()
            if self._cached_node_packs is None or (current_time - self._cache_timestamp) > 30:
                self._cached_node_packs = core.get_installed_node_packs()
                self._cache_timestamp = current_time
            
            nodes = set()
            for node_id, node_info in self._cached_node_packs.items():
                cnr_id = node_info.get('cnr_id', '')
                
                # Add basic ID
                nodes.update([node_id, node_id.lower()])
                if cnr_id:
                    nodes.update([cnr_id, cnr_id.lower()])
                
                # Add version without prefix
                for prefix in ['comfyui-', 'ComfyUI_']:
                    if node_id.startswith(prefix):
                        clean_id = node_id.replace(prefix, '').lower()
                        nodes.add(clean_id)
                        nodes.add(clean_id.replace('_', '-'))
                        if '_' in clean_id:
                            parts = clean_id.split('_')
                            for i in range(1, len(parts) + 1):
                                nodes.add('-'.join(parts[:i]))
                    
                    if cnr_id and cnr_id.startswith(prefix):
                        clean_cnr = cnr_id.replace(prefix, '').lower()
                        nodes.add(clean_cnr)
                        nodes.add(clean_cnr.replace('_', '-'))
                        if '_' in clean_cnr:
                            parts = clean_cnr.split('_')
                            for i in range(1, len(parts) + 1):
                                nodes.add('-'.join(parts[:i]))
                    
            return nodes
        except Exception as e:
            logger.error(f"Failed to get installed nodes: {e}")
            return set()

    async def get_installed_models(self) -> Set[str]:
        """Get list of installed models"""
        try:
            model_dir_names = ['checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models', 'clip_vision', 'embeddings',
                              'diffusers', 'vae_approx', 'controlnet', 'gligen', 'upscale_models', 'hypernetworks',
                              'photomaker', 'classifiers', 'checkpoint']
            
            total_models_files = set()
            for dir_name in model_dir_names:
                try:
                    total_models_files.update(folder_paths.get_filename_list(dir_name))
                except:
                    pass
                    
            return total_models_files
        except Exception as e:
            logger.error(f"Failed to get installed models: {e}")
            return set()

    async def check_system_dependencies(self) -> bool:
        """Check if system dependencies meet requirements"""
        try:
            required_libs = ['libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libfreetype-dev']
            
            missing_libs = []
            for lib in required_libs:
                try:
                    result = subprocess.run(['dpkg', '-l', lib], capture_output=True, text=True, check=True)
                    if 'ii' not in result.stdout:
                        missing_libs.append(lib)
                except subprocess.CalledProcessError:
                    missing_libs.append(lib)
            
            if missing_libs:
                logger.warning(f"Missing system dependency libraries: {', '.join(missing_libs)}")
                logger.info(f"Please run: sudo apt update && sudo apt install -y {' '.join(missing_libs)}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check system dependencies: {e}")
            return True

    async def install_custom_node(self, node_data: Dict) -> bool:
        """Install custom node"""
        try:
            node_id = node_data.get("id", "")
            install_type = node_data.get("install_type", "")
            files = node_data.get("files", [])

            if not node_id:
                return False
            
            # Check system dependencies
            if not await self.check_system_dependencies():
                logger.warning(f"System dependency check failed, skipping node installation: {node_id}")
                return False

            # Check installation type and select appropriate installation method
            if install_type == "git-clone" and files:
                result = await core.gitclone_install(files[0], instant_execution=True, no_deps=False)
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
                return await self._install_cnr_node(node_id)
            else:
                # For other installation types, skip for now
                logger.warning(f"Unsupported installation type: {install_type} for {node_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to install custom node: {e}, node: {node_data}")
            return False

    async def install_model(self, model_data: Dict) -> bool:
        """Install model"""
        try:
            model_name = model_data.get("filename", "")
            if not model_name or model_name in self.installed_models:
                return False

            logger.info(f"Starting model download: {model_name}")
            
            # Get model save path
            save_path = model_data.get("save_path", "default")
            model_type = model_data.get("type", "checkpoint")
            
            if save_path == "default":
                # Use default path
                model_dir_name = model_type
                if model_dir_name == "upscale":
                    model_dir_name = "upscale_models"
                elif model_dir_name == "checkpoint":
                    model_dir_name = "checkpoints"
                
                model_dir = os.path.join(folder_paths.models_dir, model_dir_name)
            else:
                # Use custom path, correct path mapping
                if save_path.startswith('checkpoints/'):
                    # checkpoints/upscale -> checkpoint/upscale
                    corrected_path = save_path.replace('checkpoints/', 'checkpoint/')
                    model_dir = os.path.join(folder_paths.models_dir, corrected_path)
                else:
                    model_dir = os.path.join(folder_paths.models_dir, save_path)
            
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
                    
            except Exception as e:
                logger.error(f"❌ Model download exception: {model_name}, error: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to install model: {e}, model: {model_data}")
            return False

    async def fetch_file_data(self) -> Dict:
        """Fetch unified resource data from url"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.file_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get('version', 'unknown')
                        node_count = len(data.get('custom_nodes', []))
                        model_count = len(data.get('models', []))
                        logger.info(f"Successfully fetched unified resource list, version: {version}, nodes: {node_count}, models: {model_count}")
                        return data
                    else:
                        logger.error(f"Failed to fetch file data: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"file data fetch exception: {e}")
            return {}

    async def process_unified_data(self, data: Dict):
        """Process unified resource data"""
        try:
            logger.info("Starting to process unified resource data...")
            
            # Update installed list
            self.installed_nodes = await self.get_installed_nodes()
            self.installed_models = await self.get_installed_models()
            
            logger.info(f"Installed: {len(self.installed_nodes)} nodes, {len(self.installed_models)} models")

            # Process custom nodes
            new_nodes = 0
            for node_data in data.get("custom_nodes", []):
                node_id = node_data.get("id", "")
                # Check if already installed
                if node_id in self.installed_nodes:
                    logger.info(f"Custom node already exists, skipping installation: {node_id}")
                    continue
                    
                if await self.install_custom_node(node_data):
                    new_nodes += 1

            logger.info(f"Custom node processing completed, new: {new_nodes}, skipped: {len(data.get('custom_nodes', [])) - new_nodes}")

            # Process models - using manager_server check logic
            new_models = 0
            skipped_models = 0
            for model_data in data.get("models", []):
                model_name = model_data.get("filename", "")
                
                # Use manager_server check_model_installed logic to check if model is installed
                temp_json_obj = {"models": [model_data.copy()]}
                check_model_installed(temp_json_obj)
                
                # Check if model is already installed
                if temp_json_obj["models"][0].get('installed') == 'True':
                    skipped_models += 1
                    continue
                
                if await self.install_model(model_data):
                    new_models += 1

            logger.info(f"Model processing completed: new {new_models}, skipped {skipped_models}")

        except Exception as e:
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
                await asyncio.sleep(5)

        logger.info("File direct installation service stopped")

def main():
    parser = argparse.ArgumentParser(description="File Unified Resource Installation Service - Direct Call Version")
    parser.add_argument("--resource-url", required=True, help="File JSON file URL")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
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
