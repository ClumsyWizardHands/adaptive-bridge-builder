"""
API Key Generator for Alex Familiar Agent

This script generates secure API keys that can be shared with other agents
for authentication purposes.
"""

import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import base64


class APIKeyManager:
    """Manages API keys for agent authentication."""
    
    def __init__(self, storage_file: str = "data/api_keys.json"):
        self.storage_file = storage_file
        self.keys_data = self._load_keys()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
    
    def generate_api_key(
        self, 
        name: str, 
        description: str = "",
        permissions: List[str] = None,
        expires_in_days: int = 365
    ) -> Dict[str, str]:
        """
        Generate a new API key.
        
        Args:
            name: Name/identifier for this key
            description: Description of what this key is for
            permissions: List of permissions (default: all)
            expires_in_days: Days until expiration (default: 365)
            
        Returns:
            Dictionary with key information
        """
        # Generate secure random key
        key_bytes = secrets.token_bytes(32)
        api_key = f"afx-{base64.urlsafe_b64encode(key_bytes).decode('utf-8').rstrip('=')}"
        
        # Generate key ID
        key_id = f"key-{secrets.token_hex(8)}"
        
        # Calculate expiration
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(days=expires_in_days)
        
        # Default permissions
        if permissions is None:
            permissions = ["read", "write", "connect"]
        
        # Create key metadata
        key_data = {
            "id": key_id,
            "name": name,
            "description": description,
            "api_key": api_key,
            "permissions": permissions,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_used": None,
            "active": True
        }
        
        # Store the key
        self.keys_data[key_id] = key_data
        self._save_keys()
        
        return {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "expires_at": expires_at.isoformat()
        }
    
    def validate_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Key data if valid, None if invalid
        """
        # Find the key
        for key_id, key_data in self.keys_data.items():
            if key_data["api_key"] == api_key:
                # Check if active
                if not key_data["active"]:
                    return None
                
                # Check expiration
                expires_at = datetime.fromisoformat(key_data["expires_at"])
                if datetime.utcnow() > expires_at:
                    return None
                
                # Update last used
                key_data["last_used"] = datetime.utcnow().isoformat()
                self._save_keys()
                
                return key_data
        
        return None
    
    def list_keys(self) -> List[Dict]:
        """List all API keys."""
        keys = []
        for key_id, key_data in self.keys_data.items():
            # Don't include the actual key in listings
            safe_data = {
                "id": key_id,
                "name": key_data["name"],
                "description": key_data["description"],
                "permissions": key_data["permissions"],
                "created_at": key_data["created_at"],
                "expires_at": key_data["expires_at"],
                "last_used": key_data["last_used"],
                "active": key_data["active"]
            }
            keys.append(safe_data)
        return keys
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: ID of the key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if key_id in self.keys_data:
            self.keys_data[key_id]["active"] = False
            self._save_keys()
            return True
        return False
    
    def _load_keys(self) -> Dict:
        """Load keys from storage."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_keys(self) -> None:
        """Save keys to storage."""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        with open(self.storage_file, 'w') as f:
            json.dump(self.keys_data, f, indent=2)


def main():
    """Main function to generate and manage API keys."""
    manager = APIKeyManager()
    
    print("Alex Familiar Agent - API Key Manager")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Generate new API key")
        print("2. List all keys")
        print("3. Validate a key")
        print("4. Revoke a key")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            name = input("Enter key name/identifier: ").strip()
            description = input("Enter description (optional): ").strip()
            
            result = manager.generate_api_key(name, description)
            
            print("\n" + "=" * 70)
            print("NEW API KEY GENERATED")
            print("=" * 70)
            print(f"Key ID: {result['key_id']}")
            print(f"Name: {result['name']}")
            print(f"Expires: {result['expires_at']}")
            print("\nAPI Key (save this securely, it won't be shown again):")
            print(f"\n{result['api_key']}\n")
            print("=" * 70)
            
        elif choice == "2":
            keys = manager.list_keys()
            if keys:
                print("\nActive API Keys:")
                print("-" * 50)
                for key in keys:
                    status = "Active" if key["active"] else "Revoked"
                    print(f"ID: {key['id']}")
                    print(f"Name: {key['name']}")
                    print(f"Status: {status}")
                    print(f"Created: {key['created_at']}")
                    print(f"Expires: {key['expires_at']}")
                    print(f"Last Used: {key['last_used'] or 'Never'}")
                    print("-" * 50)
            else:
                print("\nNo API keys found.")
        
        elif choice == "3":
            api_key = input("Enter API key to validate: ").strip()
            result = manager.validate_key(api_key)
            if result:
                print(f"\n✓ Valid key: {result['name']}")
                print(f"Permissions: {', '.join(result['permissions'])}")
            else:
                print("\n✗ Invalid or expired key")
        
        elif choice == "4":
            key_id = input("Enter key ID to revoke: ").strip()
            if manager.revoke_key(key_id):
                print(f"\n✓ Key {key_id} has been revoked")
            else:
                print(f"\n✗ Key {key_id} not found")
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    main()
