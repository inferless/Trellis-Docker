import os
import sys
import io
import time
import json
import requests
import uuid
import shutil
import warnings
import tempfile
import numpy as np
from PIL import Image
from urllib.parse import urlparse
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
from botocore.exceptions import ClientError

# ---- Custom imports ----
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import postprocessing_utils
from scripts.storage import get_storage_provider
from scripts.config import load_config
from scripts.logger_setup import setup_logging

# --------------------------------------------------------
# 1) Setup logger 
# --------------------------------------------------------
logger = setup_logging()

# --------------------------------------------------------
# 2) Ignore xFormers warnings 
# --------------------------------------------------------
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2.layers")

# --------------------------------------------------------
# 3) Configuration Management Class
# --------------------------------------------------------
class TrellisConfig:
    def __init__(self):
        # Load config file
        config = load_config("./config.yaml")
        
        # Task processing settings
        self.max_concurrent_tasks = config["task"]["max_concurrent_tasks"]
        self.queue_poll_interval = config["task"]["queue_poll_interval"]
        self.queue_timeout = config["task"]["queue_timeout"]
        
        # Model defaults
        self.model_defaults = config["model"]
        
        # System paths
        self.TEMP_DIR = config["paths"].get("temp_dir", tempfile.gettempdir())
        self.IMAGES_DIR = os.path.join(self.TEMP_DIR, config["paths"]["images_dir"])
        self.TASKS_DIR = os.path.join(self.TEMP_DIR, config["paths"]["tasks_dir"])
        self.MAX_SEED = 2**32 - 1
        
        # Storage settings
        self.storage_input_dir = config["storage"]['prefix'] + config["paths"]["images_dir"]
        self.storage_output_dir = config["storage"]['prefix'] + config["paths"]["tasks_dir"]
        
        # Create necessary directories
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.TASKS_DIR, exist_ok=True)
        
        # Initialize storage
        self.storage = get_storage_provider(config["storage"])
        self.storage_type = config["storage"]['provider']
        
        # Log configuration
        logger.info(f"TEMP_DIR: {self.TEMP_DIR}")
        logger.info(f"IMAGES_DIR: {self.IMAGES_DIR}")
        logger.info(f"TASKS_DIR: {self.TASKS_DIR}")
        logger.info(f"AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not Set'}")
        logger.info(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not Set'}")
        logger.info(f"AWS_REGION: {os.getenv('AWS_REGION') or 'Not Set'}")


# --------------------------------------------------------
# 4) Model Management Class
# --------------------------------------------------------
class TrellisModel:
    """Model management class"""
    def __init__(self):
        # Load your pipeline
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()
        
    def process_image(self, image: Image.Image, params: dict, progress_callback=None) -> dict:
        """
        Process an image using Trellis pipeline with dynamic progress tracking.
        """
        outputs = self.pipeline.run_with_progress(
            image=image,
            seed=params.get('geometry_seed', 42),
            formats=["gaussian", "mesh"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": params.get('sparse_structure_steps', 20),
                "cfg_strength": params.get('sparse_structure_strength', 7.5),
            },
            slat_sampler_params={
                "steps": params.get('slat_steps', 20),
                "cfg_strength": params.get('slat_strength', 3.0),
            },
            progress_callback=progress_callback
        )
        return outputs


# --------------------------------------------------------
# 5) Task Management Class
# --------------------------------------------------------
class TaskManager:
    """Task management class"""
    def __init__(self, config: TrellisConfig, model: TrellisModel):
        self.config = config
        self.model = model
        self.tasks = {}
        self.task_queue = Queue()
        self.max_concurrent_tasks = config.max_concurrent_tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self.active_tasks = 0
        self.tasks_lock = threading.Lock()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        logger.info(f"TaskManager initialized with max_concurrent_tasks: {config.max_concurrent_tasks}")
        self.worker_thread.start()
        logger.info("Task queue worker thread started")

    def _update_queue_positions(self):
        """Update queue positions for all queued tasks."""
        queued_tasks = []
        for task_id, task in self.tasks.items():
            if task.get('status') == 'queued' and 'created_at' in task:
                queued_tasks.append((task_id, task['created_at']))
        
        # Sort by creation time
        queued_tasks.sort(key=lambda x: x[1])
        
        # Update queue positions
        for position, (task_id, _) in enumerate(queued_tasks):
            self.tasks[task_id]['queue_position'] = position

    def _process_queue(self):
        """Background worker to process tasks from the queue."""
        while True:
            try:
                # Check if we can process more tasks
                can_process = False
                with self.tasks_lock:
                    if self.active_tasks < self.max_concurrent_tasks:
                        can_process = True
                
                if not can_process:
                    time.sleep(self.config.queue_poll_interval)  # Wait before checking again
                    continue

                try:
                    # Non-blocking get with configured timeout
                    task = self.task_queue.get(timeout=self.config.queue_timeout)
                except Empty:
                    continue

                # Process the task
                with self.tasks_lock:
                    self.active_tasks += 1
                    if task['task_id'] in self.tasks:
                        self.tasks[task['task_id']]['status'] = 'processing'
                        self._update_queue_positions()

                # Submit task to thread pool
                future = self.thread_pool.submit(
                    self.process_task,
                    task['task_id'],
                    task['image_path'],
                    task['params']
                )

                def task_done_callback(future):
                    with self.tasks_lock:
                        self.active_tasks = max(0, self.active_tasks - 1)
                        self._update_queue_positions()
                    self.task_queue.task_done()

                future.add_done_callback(task_done_callback)

            except Exception as e:
                logger.error(f"Error in queue processing: {str(e)}", exc_info=True)
                time.sleep(0.1)

    def get_task_dir(self, task_id: str) -> str:
        """Get or create the local directory for a task."""
        task_dir = os.path.join(self.config.TASKS_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
        
    def save_task_metadata(self, task_id: str, metadata: dict):
        """Save task metadata locally and to the storage provider."""
        task_dir = self.get_task_dir(task_id)
        metadata_path = os.path.join(task_dir, 'metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        storage_path = f"{self.config.storage_output_dir}/{task_id}/metadata.json"
        return self.config.storage.upload_file(metadata_path, storage_path)
        
    def process_task(self, task_id: str, image_path: str, params: dict):
        """
        Process a task in the background. 
        Replicates the logic from your original code.
        """
        try:
            start_time = time.time()
            logger.info(f"Starting task processing for task_id: {task_id}")
            self.tasks[task_id]['status'] = 'processing'
            self.tasks[task_id]['progress'] = 0
            
            # Phase 1: Setup
            task_dir = self.get_task_dir(task_id)
            task_image_path = os.path.join(task_dir, 'input.png')
            shutil.copy2(image_path, task_image_path)
            self.tasks[task_id]['progress'] = 1
            
            storage_input_path = f"{self.config.storage_output_dir}/{task_id}/input.png"
            self.config.storage.upload_file(task_image_path, storage_input_path)
            self.tasks[task_id]['progress'] = 2
            
            params_path = os.path.join(task_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)
            storage_params_path = f"{self.config.storage_output_dir}/{task_id}/params.json"
            self.config.storage.upload_file(params_path, storage_params_path)
            self.tasks[task_id]['progress'] = 3
            
            setup_time = time.time() - start_time
            
            # Phase 2: Model Processing
            def model_progress_callback(progress):
                """Maps model progress (0-100) to Phase 2 range (2-35)."""
                overall_progress = 2 + (progress * 0.33)
                self.tasks[task_id]['progress'] = overall_progress
            
            image = Image.open(task_image_path)
            model_start = time.time()
            outputs = self.model.process_image(image, params, progress_callback=model_progress_callback)
            model_time = time.time() - model_start
            self.tasks[task_id]['progress'] = 35
            
            # Phase 3: Post-processing
            def postprocessing_progress_callback(progress):
                """Maps postprocessing progress (0-100) to Phase 3 range (35-90)."""
                overall_progress = 35 + (progress * 0.55)
                self.tasks[task_id]['progress'] = overall_progress
            
            glb_start = time.time()
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=params.get('simplify', 0.95),
                texture_size=params.get('texture_size', 1024),
                verbose=False,
                use_vertex_colors=False,
                progress_callback=postprocessing_progress_callback
            )
            glb_time = time.time() - glb_start
            self.tasks[task_id]['progress'] = 90
            
            glb_path = os.path.join(task_dir, 'model.glb')
            save_start = time.time()
            glb.export(glb_path)
            save_time = time.time() - save_start
            self.tasks[task_id]['progress'] = 95
            
            upload_start = time.time()
            storage_model_path = f"{self.config.storage_output_dir}/{task_id}/model.glb"
            download_temp_url = self.config.storage.upload_file(glb_path, storage_model_path)
            upload_time = time.time() - upload_start
            
            post_time = glb_time + save_time + upload_time
            total_time = time.time() - start_time
            
            # Finalize
            if self.config.storage_type == "s3":
                s3_url = f"s3://{self.config.storage.bucket_name}/{storage_model_path}"
            else:
                s3_url = ""
            if self.config.storage_type == "gcs":
                gcs_url = f"https://storage.googleapis.com/{self.config.storage.bucket_name}/{storage_model_path}"  
            else:
                gcs_url = ""
            
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['progress'] = 100
            self.tasks[task_id]['output'] = {'model': download_temp_url, "s3_url": s3_url, "gcs_url": gcs_url}
            
            self.save_task_metadata(task_id, self.tasks[task_id])
            shutil.rmtree(task_dir)
            
            logger.info(
                f"Task {task_id} completed in {total_time:.2f}s "
                f"(setup: {setup_time:.2f}s, model: {model_time:.2f}s, post-processing: {post_time:.2f}s)"
            )
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['error'] = str(e)
            self.save_task_metadata(task_id, self.tasks[task_id])
            task_dir = self.get_task_dir(task_id)
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
                
    def create_task(self, file_token: str, download_path: str, params: dict) -> str:
        """Create a new task and add it to the queue."""
        task_id = str(uuid.uuid4())
        logger.info(f"Creating new task {task_id}")
        
        with self.tasks_lock:
            queue_size = self.task_queue.qsize()
            self.tasks[task_id] = {
                'status': 'queued',
                'progress': 0,
                'created_at': time.time(),
                'params': params,
                'queue_position': queue_size,
            }
            
        task_data = {
            'task_id': task_id,
            'image_path': download_path,
            'params': params
        }
        self.task_queue.put(task_data)
        
        logger.info(
            f"Task {task_id} added to queue. "
            f"Active tasks: {self.active_tasks}/{self.max_concurrent_tasks}, "
            f"Queued: {queue_size}"
        )
        return task_id
        
    def is_url_expired(self, url: str) -> bool:
        """Check if a URL is expired by making a HEAD request."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code != 200
        except requests.RequestException:
            return True

    def get_task_status(self, task_id: str) -> dict:
        """Get task status (including updated presigned URLs if needed)."""
        if task_id not in self.tasks:
            return None
            
        with self.tasks_lock:
            status_data = self.tasks[task_id].copy()
            
            queue_info = {
                'active_tasks': self.active_tasks,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'tasks_in_queue': self.task_queue.qsize(),
                'queue_position': status_data.get('queue_position', 0) if status_data['status'] == 'queued' else 0
            }
            
            status_mapping = {
                'queued': 'queued',
                'processing': 'running',
                'completed': 'success',
                'failed': 'failed'
            }
            
            # If completed, check if model URL is still valid
            if status_data['status'] == 'completed':
                try:
                    model_url = status_data["output"].get("model")
                    if not model_url or self.is_url_expired(model_url):
                        storage_url = f"{self.config.storage_output_dir}/{task_id}/model.glb"
                        status_data['output']['model'] = self.config.storage.get_url(storage_url)
                except Exception as e:
                    logger.error(f"Failed to generate presigned URL for task {task_id}: {str(e)}")
            
            return {
                'message': f"Task is {status_mapping.get(status_data['status'], 'unknown')}",
                'task_id': task_id,
                'type': 'image_to_model',
                'status': status_mapping.get(status_data['status'], 'unknown'),
                'input': status_data['params'],
                'output': status_data.get('output', {}),
                'progress': status_data.get('progress', 0),
                'create_time': int(status_data['created_at']),
                'error': status_data.get('error') if status_data['status'] == 'failed' else None,
                'queue_info': queue_info
            }


# --------------------------------------------------------
# 6) Inferless Model Wrapper
# --------------------------------------------------------
class InferlessPythonModel:
    def initialize(self):
        """
        Runs once when your inference container is started.
        """
        logger.info("Initializing InferlessPythonModel...")
        
        # Equivalent to TrellisAPI.__init__()
        self.config = TrellisConfig()         # Load config
        self.model = TrellisModel()           # Initialize model pipeline
        self.task_manager = TaskManager(      # Create background task manager
            self.config, 
            self.model
        )
        
        logger.info("InferlessPythonModel initialized successfully.")

    def infer(self, inputs: dict) -> dict:
        """
        This method replaces the Flask-based /trellis/inference endpoint.
        Expects a dictionary similar to:
        
            {
                "image_url": "http://...",
                "geometry_seed": 42,
                "sparse_structure_steps": 20,
                ...
                "timeout": 300
            }
        
        Returns a dictionary that mimics your JSON response.
        """
        try:
            logger.info("Starting inference request via InferlessPythonModel.infer")

            # -----------------------------------------------------
            # 1) Validate inputs
            # -----------------------------------------------------
            if "image_url" not in inputs:
                logger.warning("Inference attempt with no image_url provided.")
                return {
                    'code': 2003,
                    'data': {'message': 'No image_url provided'}
                }

            image_url = inputs["image_url"]
            timeout = float(inputs.get('timeout', 300))  # Default 5 minutes
            poll_interval = 0.5  # Poll every 0.5 seconds

            # -----------------------------------------------------
            # 2) Download image
            # -----------------------------------------------------
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                image_format = image.format.lower()
                
                if image_format not in ['jpeg', 'jpg', 'png']:
                    raise ValueError(f"Unsupported file type: {image_format}")
                
                width, height = image.size
                file_size = len(response.content)
                logger.info(f"Image downloaded & validated: {width}x{height}, {file_size} bytes")

            except Exception as e:
                logger.error(f"Failed to download or validate image: {str(e)}")
                return {
                    'code': 2004,
                    'data': {'message': f'Failed to download or validate image: {str(e)}'}
                }

            # -----------------------------------------------------
            # 3) Save and upload file
            # -----------------------------------------------------
            image_token = str(uuid.uuid4())
            temp_path = os.path.join(self.config.IMAGES_DIR, f"{image_token}.png")
            image.save(temp_path)
            
            storage_path = f"{self.config.storage_input_dir}/{image_token}.png"
            storage_url = self.config.storage.upload_file(temp_path, storage_path)
            logger.info(f"Image uploaded to storage: {storage_path}")

            # -----------------------------------------------------
            # 4) Create & validate task parameters
            # -----------------------------------------------------
            params = {
                'type': 'image_to_model',
                'file': {
                    'file_token': image_token,
                    'type': "image"
                },
                # Model parameters
                'geometry_seed': inputs.get('geometry_seed', 0),
                'sparse_structure_steps': inputs.get('sparse_structure_steps', 20),
                'sparse_structure_strength': inputs.get('sparse_structure_strength', 7.5),
                'slat_steps': inputs.get('slat_steps', 20),
                'slat_strength': inputs.get('slat_strength', 3.0),
                'simplify': inputs.get('simplify', 0.95),
                'texture_size': inputs.get('texture_size', 1024),
            }
            
            # Minimal validation logic from your existing validate_task_params:
            # (You can copy the entire function if you want the same checks, 
            #  or keep it simpler.)
            
            # Example: geometry_seed must be between 0 and MAX_SEED
            geometry_seed = int(params['geometry_seed'])
            if geometry_seed < 0 or geometry_seed > self.config.MAX_SEED:
                os.remove(temp_path)
                return {
                    'code': 2002,
                    'data': {'message': f"geometry_seed must be between 0 and {self.config.MAX_SEED}"}
                }

            # -----------------------------------------------------
            # 5) Create task
            # -----------------------------------------------------
            start_time = time.time()
            task_id = self.task_manager.create_task(image_token, temp_path, params)
            logger.info(f"Task created: {task_id}")

            # -----------------------------------------------------
            # 6) Poll for completion or timeout
            # -----------------------------------------------------
            while True:
                status_data = self.task_manager.get_task_status(task_id)
                
                if not status_data:
                    # If the task somehow doesn't exist
                    os.remove(temp_path)
                    return {
                        'code': 2001,
                        'data': {'message': 'Task not found'}
                    }
                
                if status_data['status'] == 'failed':
                    os.remove(temp_path)
                    return {
                        'code': 2005,
                        'data': {
                            'message': 'Processing failed',
                            'error': status_data.get('error', 'Unknown error'),
                            'task_id': task_id
                        }
                    }

                if status_data['status'] == 'success':
                    total_time = time.time() - start_time
                    os.remove(temp_path)
                    return {
                        'code': 0,
                        'data': {
                            'message': 'Processing completed successfully',
                            'task_id': task_id,
                            'processing_time': total_time,
                            'input': {
                                'image_token': image_token,
                                'width': width,
                                'height': height,
                                'size': file_size,
                                'parameters': params
                            },
                            'output': status_data['output'],
                            'metrics': {
                                'total_time': total_time,
                                'progress': 100
                            }
                        }
                    }

                # Check timeout
                if time.time() - start_time > timeout:
                    os.remove(temp_path)
                    return {
                        'code': 2006,
                        'data': {
                            'message': 'Processing timeout',
                            'task_id': task_id,
                            'status': status_data['status'],
                            'progress': status_data['progress']
                        }
                    }

                time.sleep(poll_interval)

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            # Cleanup if file still exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            return {
                'code': 500,
                'data': {'message': f'Internal server error: {str(e)}'}
            }

    def finalize(self):
        """
        Called just before the container is shut down.
        Use this to clean up or close resources if needed.
        """
        logger.info("Finalizing InferlessPythonModel...done.")
        pass
