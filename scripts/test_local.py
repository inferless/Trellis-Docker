import requests
from pathlib import Path
import os
import time
# import boto3
from urllib.parse import urlparse


API_HOST = os.getenv('API_HOST', 'localhost')
url_upload = f"http://{API_HOST}:5000/trellis/upload"
url_task = f"http://{API_HOST}:5000/trellis/task"
url_status = f"http://{API_HOST}:5000/trellis/task/{{}}"

# Add environment variable to control storage mode
os.environ['STORAGE_MODE'] = 'local'  # Options: 'local' or 's3'

def upload_image(file_path):
    """Upload image to API and return image token"""
    files = {'file': open(file_path, 'rb')}
    headers = {
        'X-Storage-Mode': os.getenv('STORAGE_MODE', 'local')  # Tell the API to use local storage
    }
    response = requests.post(url_upload, files=files, headers=headers)
    response_data = response.json()
    if response.status_code != 200 or response_data.get("code") != 0:
        raise Exception(f"Image upload failed: {response_data}")

    image_token = response_data["data"]["image_token"]
    return image_token

def submit_image_to_model(
          image_token, model_version, texture_seed, geometry_seed, face_limit=10000,
          sparse_structure_steps=20, sparse_structure_strength=7.5, slat_steps=20, slat_strength=3.0,
          simplify=0.7, texture_size=2048,
    ):
        payload = {
            "type": "image_to_model",
            "model_version": model_version,
            "file": {
                "type": "image",
                "file_token": image_token
            },
            "face_limit": face_limit,
            "texture": True,
            "pbr": True,
            "texture_seed": texture_seed,
            "geometry_seed": geometry_seed,
            "sparse_structure_steps": sparse_structure_steps,
            "sparse_structure_strength": sparse_structure_strength,
            "slat_steps": slat_steps,
            "slat_strength": slat_strength,
            "simplify": simplify,
            "texture_size": texture_size,
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url_task, headers=headers, json=payload)
        print(response)
        response_data = response.json()
        if response.status_code != 200 or response_data.get("code") != 0:
            raise Exception(f"Task submission failed: {response_data}")
        task_id = response_data["data"]["task_id"]
        return task_id 


def get_task_status(task_id):
    url = url_status.format(task_id)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('data', {})
        return data
    else:
        raise Exception(f"Failed to get task status: {response.text}")

def download_model(url, task_id, out_file_path):
    """Download the model file from either presigned URL or S3 URL and save it to the specified path."""
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if it's a presigned URL or S3 URL
    if url.startswith('https://'):
        try:
            # Direct download using presigned URL
            print(f"Downloading from presigned URL")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                # fetch a new url using task_id with get_task_status
                data = get_task_status(task_id)
                url = data['output']['model']
                response = requests.get(url, stream=True)

            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Write the file
            with open(out_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Successfully downloaded to: {out_file_path}")
            return out_file_path
            
        except Exception as e:
            raise Exception(f"Failed to download model from presigned URL: {str(e)}")
            
    elif url.startswith('s3://'):
        try:
            import boto3
            # Parse S3 URL
            parsed_url = urlparse(url)
            bucket_name = parsed_url.netloc
            s3_key = parsed_url.path.lstrip('/')
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            print(f"Downloading from S3: bucket={bucket_name}, key={s3_key}")
            
            # Download the file
            s3_client.download_file(
                Bucket=bucket_name,
                Key=s3_key,
                Filename=str(out_file_path)
            )
            
            print(f"Successfully downloaded to: {out_file_path}")
            return out_file_path
            
        except Exception as e:
            raise Exception(f"Failed to download model from S3: {str(e)}")
    else:
        raise ValueError(f"Invalid URL format. URL must start with 'https://' or 's3://'. Got: {url}")

if __name__ == "__main__":
    # Check AWS environment variables
    import os
    print("\n=== Starting Full Test Flow ===")
    try:
        # 1. Upload image
        image_path = "./TRELLIS/assets/example_image/T.png"  # Adjust path as needed
        print(f"\n1. Uploading image: {image_path}")
        image_token = upload_image(image_path)
        print(f"✓ Image uploaded successfully. Token: {image_token}")

        # image_token = '381c81fd-c766-480b-83ee-70c0ecfe2788'

        # 2. Create task
        print("\n2. Creating task for 3D conversion")
        task_id = submit_image_to_model(
            image_token=image_token,
            model_version="Trellis-image-large",
            texture_seed=1,
            geometry_seed=1,
            face_limit=10000,
            sparse_structure_steps=12,
            sparse_structure_strength=7.5,
            slat_steps=12,
            slat_strength=3.0
        )
        print(f"✓ Task created successfully. ID: {task_id}")

        # 3. Poll for completion
        print("\n3. Waiting for task completion...")
        start_time = time.time()
        while True:
            data = get_task_status(task_id)
            status = data.get('status')
            progress = data.get('progress', 0)
            
            print(f"   Status: {status}, Progress: {progress}%")
            
            if status == 'success':
                print("✓ Task completed successfully!")
                print(data)
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
                break
            elif status == 'failed':
                error = data.get('error', 'Unknown error')
                print(f"✗ Task failed: {error}")
                raise Exception(f"Task processing failed: {error}")
            
            time.sleep(2)  # Wait 0.5 seconds before next check

        time.sleep(65)
        # 4. Download result
        print("\n4. Downloading result")
        if 'model' in data.get('output', {}):
            url = data['output']['model']
            output_path = Path("./result.glb")
            download_model(url, task_id, output_path)
            print(f"✓ Model downloaded successfully to: {output_path.absolute()}")
        else:
            print("✗ No model URL in response")
            raise Exception("No model URL in response")

        print("\n✓ Test completed successfully!")
        print("=== Test Flow Completed ===\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        print("=== Test Flow Failed ===\n")
        raise




