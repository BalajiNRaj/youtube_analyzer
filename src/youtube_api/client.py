"""
YouTube API Client for handling authentication and basic API operations.
"""

import os
import json
import yaml
from typing import Optional, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class YouTubeClient:
    """
    YouTube API client for handling authentication and basic API operations.
    """
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize YouTube client.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.service = None
        self.credentials = None
        os.environ['YOUTUBE_API_KEY'] = os.getenv('YOUTUBE_API_KEY', '')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
    
    def authenticate(self, credentials_path: str = 'config/credentials.json') -> bool:
        """
        Authenticate with YouTube API using OAuth2 or API key.
        
        Args:
            credentials_path: Path to credentials JSON file
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Try OAuth2 flow first
            if os.path.exists(credentials_path):
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, self.SCOPES
                )
                self.credentials = flow.run_local_server(port=0)
                
                self.service = build(
                    self.API_SERVICE_NAME, 
                    self.API_VERSION, 
                    credentials=self.credentials
                )
                return True
                
        except Exception as e:
            print(f"OAuth2 authentication failed: {e}")
            
        # Fallback to API key
        api_key = os.environ.get('YOUTUBE_API_KEY')
        if api_key:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print(f"Using API key from environment: {masked_key}")
        else:
            print("API key not found in environment")
        if api_key:
            try:
                self.service = build(
                    self.API_SERVICE_NAME,
                    self.API_VERSION,
                    developerKey=api_key
                )
                return True
            except Exception as e:
                print(f"API key authentication failed: {e}")
                
        print("Authentication failed. Please provide valid credentials or API key.")
        return False
    
    def get_service(self):
        """
        Get authenticated YouTube service object.
        
        Returns:
            YouTube service object or None
        """
        if not self.service and not self.authenticate():
            raise Exception("Failed to authenticate with YouTube API")
        return self.service
    
    def test_connection(self) -> bool:
        """
        Test API connection by making a simple request.
        
        Returns:
            bool: True if connection successful
        """
        try:
            service = self.get_service()
            request = service.videos().list(
                part='snippet',
                id='dQw4w9WgXcQ',  # Rick Roll video ID for testing
                maxResults=1
            )
            response = request.execute()
            return len(response.get('items', [])) > 0
        except HttpError as e:
            print(f"API connection test failed: {e}")
            return False
        except Exception as e:
            print(f"Connection test error: {e}")
            return False
