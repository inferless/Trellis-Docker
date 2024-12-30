import os
import sys
from collections import OrderedDict

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.test_local import *
except ImportError:
    from test_local import *

import threading
import concurrent.futures
from datetime import datetime

def get_task_status_safe(task_id, task_number):
    """Safely get task status without raising exceptions"""
    try:
        data = get_task_status(task_id)
        if data is None:
            return {
                'status': 'unknown',
                'progress': 0,
                'queue_info': {},
                'task_number': task_number,
                'task_id': task_id
            }
        return {
            'status': data.get('status', 'unknown'),
            'progress': data.get('progress', 0),
            'queue_info': data.get('queue_info', {}),
            'task_number': task_number,
            'task_id': task_id
        }
    except Exception as e:
        return {
            'status': 'error',
            'progress': 0,
            'queue_info': {},
            'task_number': task_number,
            'task_id': task_id,
            'error': str(e)
        }

def print_task_group_status(task_info, completed_tasks=None):
    """Print status for all tasks in a formatted way"""
    completed_tasks = completed_tasks or []
    print("\n=== Current Tasks Status ===")
    print("Task# | Status     | Progress  | Active/Max | Queue Pos | ID       | Time")
    print("-" * 80)  # Increased width for the new column
    
    all_statuses = []
    for task_id, task_number in task_info.items():
        status = get_task_status_safe(task_id, task_number)
        all_statuses.append(status)
    
    # Sort by task number
    all_statuses.sort(key=lambda x: x['task_number'])
    
    for status in all_statuses:
        queue_info = status['queue_info']
        active = queue_info.get('active_tasks', '?')
        max_tasks = queue_info.get('max_concurrent_tasks', '?')
        queue_pos = queue_info.get('queue_position', '?')
        
        progress_str = f"{status['progress']:>7.1f}%"
        
        # Add completion time if task is finished
        time_str = ""
        if status['status'] == 'success':
            completion_time = next((t['completion_time'] for t in completed_tasks 
                                 if t['task_id'] == status['task_id']), None)
            if completion_time:
                time_str = f"| {completion_time:>6.1f}s"
        
        print(f"#{status['task_number']:<4} | {status['status']:<10} | {progress_str} | "
              f"{active}/{max_tasks:<8} | {queue_pos:<9} | {status['task_id'][:8]} {time_str}")
    
    print("-" * 80)
    
    # Check if all tasks are completed or failed
    completed = all(s['status'] in ['success', 'failed'] for s in all_statuses)
    return completed, all_statuses

def test_concurrent_tasks(num_tasks=5):
    """Test submitting multiple tasks concurrently"""
    print(f"\n=== Testing {num_tasks} Concurrent Tasks (max_concurrent_tasks=2) ===")
    
    # 1. Upload image once and use it for all tasks
    image_path = "./TRELLIS/assets/example_image/T.png"
    print(f"\n1. Uploading image: {image_path}")
    image_token = upload_image(image_path)
    print(f"✓ Image uploaded successfully. Token: {image_token}")
    
    # 2. Submit multiple tasks
    print(f"\n2. Submitting {num_tasks} tasks...")
    task_info = OrderedDict()  # Keep track of task_id to task_number mapping
    for i in range(num_tasks):
        task_number = i + 1
        try:
            task_id = submit_image_to_model(
                image_token=image_token,
                model_version="Trellis-image-large",
                texture_seed=i,
                geometry_seed=i,
                face_limit=10000,
                sparse_structure_steps=12,
                sparse_structure_strength=7.5,
                slat_steps=12,
                slat_strength=3.0,
                simplify=0.7,
                texture_size=2048,
            )
            task_info[task_id] = task_number
            print(f"✓ Task {task_number} submitted. ID: {task_id[:8]}")
        except Exception as e:
            print(f"✗ Failed to submit task {task_number}: {str(e)}")
            raise
    
    # 3. Monitor all tasks together
    print("\n3. Monitoring tasks execution...")
    start_time = time.time()
    completed_tasks = []
    
    while True:
        all_completed, statuses = print_task_group_status(task_info, completed_tasks)
        
        # Collect completed tasks
        for status in statuses:
            if status['status'] == 'success' and status['task_id'] not in [t['task_id'] for t in completed_tasks]:
                completed_tasks.append({
                    'task_id': status['task_id'],
                    'task_number': status['task_number'],
                    'completion_time': time.time() - start_time
                })
        
        if all_completed:
            break
            
        time.sleep(5)
    
    # 4. Download completed models
    print("\n4. Downloading completed models...")
    for task in sorted(completed_tasks, key=lambda x: x['task_number']):
        task_id = task['task_id']
        task_number = task['task_number']
        try:
            data = get_task_status(task_id)
            if data and 'output' in data and 'model' in data['output']:
                url = data['output']['model']
                output_path = Path(f"./result_{task_number}.glb")
                download_model(url, task_id, output_path)
                print(f"✓ Model {task_number} downloaded successfully to: {output_path.absolute()}")
            else:
                print(f"✗ No model URL found for task {task_number}")
        except Exception as e:
            print(f"✗ Failed to download model {task_number}: {str(e)}")
    
    # 5. Print summary
    print("\n=== Test Summary ===")
    print(f"Total tasks submitted: {num_tasks}")
    print(f"Tasks completed: {len(completed_tasks)}")
    if completed_tasks:
        completion_times = [r['completion_time'] for r in completed_tasks]
        print(f"Average completion time: {sum(completion_times)/len(completion_times):.2f} seconds")
        print(f"Min completion time: {min(completion_times):.2f} seconds")
        print(f"Max completion time: {max(completion_times):.2f} seconds")
    
    print("\n✓ Concurrent test completed!")

if __name__ == "__main__":
    try:
        test_concurrent_tasks(5)  # Test with 5 concurrent tasks
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        raise

