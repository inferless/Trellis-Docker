from abc import ABC, abstractmethod
import os
from typing import Optional
import logging

# Create logger for storage module
logger = logging.getLogger('trellis-api.storage')

class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload a file and return its public URL"""
        pass
    
    @abstractmethod
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        """Get a signed/public URL for a file"""
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Delete a file"""
        pass

class GCSProvider(StorageProvider):
    """Google Cloud Storage implementation"""
    
    def __init__(self, bucket_name: str):
        from google.cloud import storage
        from google.cloud.exceptions import GoogleCloudError
        
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        return self.get_url(remote_path)
        
    def get_url(self, remote_path: str, expires_in: int = 60) -> str:
        blob = self.bucket.blob(remote_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=expires_in,
            method="GET"
        )
        return url
        
    def delete_file(self, remote_path: str) -> bool:
        blob = self.bucket.blob(remote_path)
        blob.delete()
        return True

class S3Provider(StorageProvider):
    """AWS S3 implementation"""
    
    def __init__(self, bucket_name: str, region_name: Optional[str] = None):
        import boto3
        from botocore.exceptions import ClientError
        
        self.bucket_name = bucket_name
        self.client = boto3.client('s3', region_name=region_name)
        logger.info(f"Initialized S3Provider with bucket: {bucket_name}, region: {region_name}")
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        try:
            logger.debug(f"Uploading file {local_path} to S3 path {remote_path}")
            self.client.upload_file(local_path, self.bucket_name, remote_path)
            url = self.get_url(remote_path)
            logger.info(f"Successfully uploaded file to S3: {remote_path}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {str(e)}", exc_info=True)
            raise
        
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        try:
            logger.debug(f"Generating presigned URL for {remote_path}")
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': remote_path
                },
                ExpiresIn=expires_in
            )
            logger.debug(f"Generated presigned URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}", exc_info=True)
            raise
        
    def delete_file(self, remote_path: str) -> bool:
        try:
            logger.debug(f"Deleting file from S3: {remote_path}")
            self.client.delete_object(Bucket=self.bucket_name, Key=remote_path)
            logger.info(f"Successfully deleted file from S3: {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {str(e)}", exc_info=True)
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            # First check if the file exists
            logger.debug(f"Checking if file exists in S3: {remote_path}")
            try:
                self.client.head_object(Bucket=self.bucket_name, Key=remote_path)
            except Exception as e:
                logger.error(f"File not found in S3: {remote_path}")
                raise FileNotFoundError(f"File not found in S3: {remote_path}")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            logger.debug(f"Created local directory: {os.path.dirname(local_path)}")
            
            # Download the file
            logger.debug(f"Downloading file from S3: {remote_path} to {local_path}")
            self.client.download_file(self.bucket_name, remote_path, local_path)
            logger.info(f"Successfully downloaded file from S3: {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file from S3: {str(e)}", exc_info=True)
            raise

class LocalProvider(StorageProvider):
    """Local filesystem implementation for development/testing"""
    
    def __init__(self, base_path: str, base_url: str):
        self.base_path = base_path
        self.base_url = base_url
        os.makedirs(base_path, exist_ok=True)
        
    def upload_file(self, local_path: str, remote_path: str) -> str:
        target_path = os.path.join(self.base_path, remote_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(local_path, 'rb') as src, open(target_path, 'wb') as dst:
            dst.write(src.read())
        return self.get_url(remote_path)
        
    def get_url(self, remote_path: str, expires_in: int = 3600) -> str:
        return f"{self.base_url}/{remote_path}"
        
    def delete_file(self, remote_path: str) -> bool:
        target_path = os.path.join(self.base_path, remote_path)
        if os.path.exists(target_path):
            os.remove(target_path)
            return True
        return False

def get_storage_provider(provider_config: dict = None) -> StorageProvider:
    """Factory function to get the configured storage provider"""
    provider_type = provider_config.get('provider', 'local') if provider_config else 'local'
    logger.info(f"Initializing storage provider: {provider_type}")
    
    try:
        if provider_type == 's3':
            # Get credentials from environment with detailed error checking
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            # Try to get region from multiple sources
            region = (
                os.getenv('AWS_REGION') or 
                os.getenv('AWS_DEFAULT_REGION') or 
                provider_config.get('region')
            )
            
            missing_creds = []
            if not aws_access_key:
                missing_creds.append('AWS_ACCESS_KEY_ID')
            if not aws_secret_key:
                missing_creds.append('AWS_SECRET_ACCESS_KEY')
                
            if missing_creds:
                error_msg = f"Missing AWS credentials: {', '.join(missing_creds)}"
                logger.error(error_msg)
                raise ValueError(f"{error_msg}. Please set these environment variables.")
                
            if not region:
                error_msg = "AWS region must be set in environment (AWS_REGION or AWS_DEFAULT_REGION) or in config.yaml"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Get bucket from config
            bucket_name = provider_config.get('bucket')
            if not bucket_name:
                error_msg = "bucket name is required in config for S3 storage"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return S3Provider(bucket_name, region)
            
        else:
            error_msg = f"Unsupported storage provider: {provider_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Failed to initialize storage provider: {str(e)}", exc_info=True)
        raise 
