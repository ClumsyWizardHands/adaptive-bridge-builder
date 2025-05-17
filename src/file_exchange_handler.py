"""
File Exchange Handler

This module provides capabilities for exchanging files and artifacts between agents.
It manages file transfers with integrity verification, chunking for large files,
tracking transfer status, and implementing security features.
"""

import os
import hashlib
import json
import base64
import uuid
import logging
import shutil
import time
from typing import Dict, List, Any, Optional, Union, BinaryIO, Tuple
from enum import Enum
from datetime import datetime, timedelta
import threading
import mimetypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FileExchangeHandler")


class TransferStatus(Enum):
    """Possible statuses for file transfers."""
    PENDING = "pending"          # Transfer waiting to begin
    IN_PROGRESS = "in_progress"  # Transfer in progress
    COMPLETED = "completed"      # Transfer completed successfully
    FAILED = "failed"            # Transfer failed
    PAUSED = "paused"            # Transfer paused
    CANCELLED = "cancelled"      # Transfer cancelled
    VERIFYING = "verifying"      # Transfer completed, verifying integrity


class TransferType(Enum):
    """Types of file transfers."""
    UPLOAD = "upload"            # Agent is uploading a file
    DOWNLOAD = "download"        # Agent is downloading a file
    EXCHANGE = "exchange"        # Bidirectional file exchange


class FileExchangeHandler:
    """
    Handles file exchanges between agents.
    
    This class provides methods for:
    1. Uploading and downloading files
    2. Chunking large files for efficient transfer
    3. Tracking transfer status and progress
    4. Verifying file integrity
    5. Implementing security measures for safe file exchange
    """
    
    def __init__(
        self,
        agent_id: str,
        storage_dir: str = "file_exchange",
        max_chunk_size: int = 1024 * 1024,  # 1MB default chunk size
        verify_integrity: bool = True,
        max_concurrent_transfers: int = 5
    ):
        """
        Initialize the file exchange handler.
        
        Args:
            agent_id: ID of the agent using this handler
            storage_dir: Directory for storing files
            max_chunk_size: Maximum size of file chunks in bytes
            verify_integrity: Whether to verify file integrity
            max_concurrent_transfers: Maximum number of concurrent transfers
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.max_chunk_size = max_chunk_size
        self.verify_integrity = verify_integrity
        self.max_concurrent_transfers = max_concurrent_transfers
        
        # Create storage directory structure
        self.incoming_dir = os.path.join(storage_dir, "incoming")
        self.outgoing_dir = os.path.join(storage_dir, "outgoing")
        self.temp_dir = os.path.join(storage_dir, "temp")
        
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.outgoing_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Track transfers
        self.transfers: Dict[str, Dict[str, Any]] = {}
        self.transfer_lock = threading.Lock()
        
        # Transfer queue and worker threads
        self.transfer_queue: List[str] = []
        self.active_transfers = 0
        
        logger.info(f"FileExchangeHandler initialized for agent {agent_id}")
    
    def upload_file(
        self,
        file_path: str,
        recipient_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        transfer_id: Optional[str] = None,
        chunk: bool = True
    ) -> str:
        """
        Start uploading a file to another agent.
        
        Args:
            file_path: Path to the file to upload
            recipient_id: ID of the recipient agent
            metadata: Additional metadata about the file
            transfer_id: Optional custom transfer ID
            chunk: Whether to chunk the file if large
            
        Returns:
            Transfer ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Generate transfer ID if not provided
        if not transfer_id:
            transfer_id = f"transfer-{uuid.uuid4().hex}"
            
        # Get file information
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            content_type = "application/octet-stream"
            
        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(file_path)
        
        # Determine if file should be chunked
        should_chunk = chunk and file_size > self.max_chunk_size
        
        # Store transfer information
        transfer_info = {
            "transfer_id": transfer_id,
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "content_type": content_type,
            "file_hash": file_hash,
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "metadata": metadata or {},
            "status": TransferStatus.PENDING.value,
            "transfer_type": TransferType.UPLOAD.value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "progress": 0,
            "chunks": {
                "total": 0,
                "sent": 0,
                "size": self.max_chunk_size
            },
            "should_chunk": should_chunk,
            "error": None
        }
        
        # Calculate chunk information if needed
        if should_chunk:
            total_chunks = (file_size + self.max_chunk_size - 1) // self.max_chunk_size
            transfer_info["chunks"]["total"] = total_chunks
            
        # Store transfer info
        with self.transfer_lock:
            self.transfers[transfer_id] = transfer_info
            self.transfer_queue.append(transfer_id)
            
        # Process transfer queue if capacity available
        self._process_transfer_queue()
        
        return transfer_id
    
    def download_file(
        self,
        transfer_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Download a file that another agent has sent.
        
        Args:
            transfer_id: ID of the transfer
            output_path: Optional custom output path
            
        Returns:
            Path to the downloaded file
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            raise ValueError(f"Transfer not found: {transfer_id}")
            
        transfer_info = self.transfers[transfer_id]
        
        # Check if this is a download/receive
        if transfer_info["transfer_type"] != TransferType.DOWNLOAD.value:
            raise ValueError(f"Transfer {transfer_id} is not a download")
            
        # Check if transfer is complete
        if transfer_info["status"] != TransferStatus.COMPLETED.value:
            raise ValueError(f"Transfer {transfer_id} is not complete")
            
        # Get file path
        temp_file_path = os.path.join(self.temp_dir, transfer_id)
        file_name = transfer_info["file_name"]
        
        # Determine output path
        if not output_path:
            output_path = os.path.join(self.incoming_dir, file_name)
        
        # Copy file to final location
        shutil.copy2(temp_file_path, output_path)
        
        # Update transfer info
        transfer_info["output_path"] = output_path
        transfer_info["updated_at"] = datetime.utcnow().isoformat()
        
        return output_path
    
    def receive_file_chunk(
        self,
        transfer_id: str,
        chunk_index: int,
        chunk_data: str,
        sender_id: str,
        total_chunks: int,
        file_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Receive a chunk of a file from another agent.
        
        Args:
            transfer_id: ID of the transfer
            chunk_index: Index of the chunk (0-based)
            chunk_data: Base64-encoded chunk data
            sender_id: ID of the sending agent
            total_chunks: Total number of chunks
            file_info: File metadata (required for first chunk)
            
        Returns:
            Transfer status information
        """
        # Check if transfer exists, create if it's the first chunk
        is_new_transfer = transfer_id not in self.transfers
        
        if is_new_transfer:
            if chunk_index != 0:
                raise ValueError("First chunk must have index 0")
            if not file_info:
                raise ValueError("File info must be provided for the first chunk")
                
            # Create new transfer record
            transfer_info = {
                "transfer_id": transfer_id,
                "file_name": file_info.get("file_name", f"received_file_{transfer_id}"),
                "file_size": file_info.get("file_size", 0),
                "content_type": file_info.get("content_type", "application/octet-stream"),
                "file_hash": file_info.get("file_hash"),
                "sender_id": sender_id,
                "recipient_id": self.agent_id,
                "metadata": file_info.get("metadata", {}),
                "status": TransferStatus.IN_PROGRESS.value,
                "transfer_type": TransferType.DOWNLOAD.value,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "progress": 0,
                "chunks": {
                    "total": total_chunks,
                    "received": 0,
                    "size": file_info.get("chunk_size", self.max_chunk_size)
                },
                "should_chunk": total_chunks > 1,
                "error": None
            }
            
            # Create temporary file
            temp_file_path = os.path.join(self.temp_dir, transfer_id)
            
            with open(temp_file_path, 'wb') as f:
                pass  # Just create an empty file
                
            # Store transfer info
            with self.transfer_lock:
                self.transfers[transfer_id] = transfer_info
        else:
            # Update existing transfer
            transfer_info = self.transfers[transfer_id]
            
            # Check if transfer is in a valid state
            if transfer_info["status"] != TransferStatus.IN_PROGRESS.value:
                raise ValueError(f"Transfer {transfer_id} is not in progress")
                
            # Check if sender matches
            if transfer_info["sender_id"] != sender_id:
                raise ValueError(f"Sender mismatch for transfer {transfer_id}")
                
            # Check chunk index
            received_chunks = transfer_info["chunks"]["received"]
            if chunk_index != received_chunks:
                raise ValueError(f"Expected chunk {received_chunks}, got {chunk_index}")
        
        # Decode chunk data
        try:
            binary_data = base64.b64decode(chunk_data)
        except Exception as e:
            error_msg = f"Failed to decode chunk data: {str(e)}"
            transfer_info["status"] = TransferStatus.FAILED.value
            transfer_info["error"] = error_msg
            transfer_info["updated_at"] = datetime.utcnow().isoformat()
            return self._get_transfer_status(transfer_id)
        
        # Write chunk to file
        temp_file_path = os.path.join(self.temp_dir, transfer_id)
        try:
            with open(temp_file_path, 'ab') as f:
                f.write(binary_data)
        except Exception as e:
            error_msg = f"Failed to write chunk data: {str(e)}"
            transfer_info["status"] = TransferStatus.FAILED.value
            transfer_info["error"] = error_msg
            transfer_info["updated_at"] = datetime.utcnow().isoformat()
            return self._get_transfer_status(transfer_id)
        
        # Update chunks received and progress
        transfer_info["chunks"]["received"] += 1
        received_chunks = transfer_info["chunks"]["received"]
        total_chunks = transfer_info["chunks"]["total"]
        transfer_info["progress"] = int(100 * received_chunks / total_chunks)
        transfer_info["updated_at"] = datetime.utcnow().isoformat()
        
        # Check if all chunks received
        if received_chunks == total_chunks:
            # Set status to verifying
            transfer_info["status"] = TransferStatus.VERIFYING.value
            
            # Verify file integrity if hash is provided
            if self.verify_integrity and transfer_info["file_hash"]:
                calculated_hash = self._calculate_file_hash(temp_file_path)
                if calculated_hash != transfer_info["file_hash"]:
                    transfer_info["status"] = TransferStatus.FAILED.value
                    transfer_info["error"] = "File integrity check failed"
                    return self._get_transfer_status(transfer_id)
            
            # Set status to completed
            transfer_info["status"] = TransferStatus.COMPLETED.value
        
        return self._get_transfer_status(transfer_id)
    
    def send_file_chunk(
        self,
        transfer_id: str,
        chunk_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prepare and send a chunk of a file.
        
        Args:
            transfer_id: ID of the transfer
            chunk_index: Optional specific chunk index to send
            
        Returns:
            Chunk data and metadata
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            raise ValueError(f"Transfer not found: {transfer_id}")
            
        transfer_info = self.transfers[transfer_id]
        
        # Check if this is an upload
        if transfer_info["transfer_type"] != TransferType.UPLOAD.value:
            raise ValueError(f"Transfer {transfer_id} is not an upload")
            
        # Determine chunk index to send
        if chunk_index is None:
            chunk_index = transfer_info["chunks"]["sent"]
            
        # Check if chunk index is valid
        total_chunks = max(1, transfer_info["chunks"]["total"])
        if chunk_index >= total_chunks:
            raise ValueError(f"Chunk index {chunk_index} out of range (0-{total_chunks-1})")
            
        # Get file information
        file_path = transfer_info["file_path"]
        file_size = transfer_info["file_size"]
        chunk_size = transfer_info["chunks"]["size"]
        
        # Calculate chunk boundaries
        start_pos = chunk_index * chunk_size
        end_pos = min(start_pos + chunk_size, file_size)
        
        # Read chunk data
        try:
            with open(file_path, 'rb') as f:
                f.seek(start_pos)
                chunk_data = f.read(end_pos - start_pos)
        except Exception as e:
            error_msg = f"Failed to read chunk data: {str(e)}"
            transfer_info["status"] = TransferStatus.FAILED.value
            transfer_info["error"] = error_msg
            transfer_info["updated_at"] = datetime.utcnow().isoformat()
            raise ValueError(error_msg)
        
        # Encode chunk data
        encoded_data = base64.b64encode(chunk_data).decode('utf-8')
        
        # Update transfer info
        transfer_info["chunks"]["sent"] += 1
        transfer_info["progress"] = int(100 * transfer_info["chunks"]["sent"] / total_chunks)
        
        if transfer_info["status"] == TransferStatus.PENDING.value:
            transfer_info["status"] = TransferStatus.IN_PROGRESS.value
            
        # Check if all chunks sent
        if transfer_info["chunks"]["sent"] == total_chunks:
            transfer_info["status"] = TransferStatus.COMPLETED.value
            
        transfer_info["updated_at"] = datetime.utcnow().isoformat()
        
        # Prepare response
        chunk_info = {
            "transfer_id": transfer_id,
            "chunk_index": chunk_index,
            "chunk_data": encoded_data,
            "total_chunks": total_chunks,
            "is_last_chunk": chunk_index == total_chunks - 1,
            "status": transfer_info["status"]
        }
        
        # Add file info for first chunk
        if chunk_index == 0:
            chunk_info["file_info"] = {
                "file_name": transfer_info["file_name"],
                "file_size": transfer_info["file_size"],
                "content_type": transfer_info["content_type"],
                "file_hash": transfer_info["file_hash"],
                "chunk_size": chunk_size,
                "metadata": transfer_info["metadata"]
            }
        
        return chunk_info
    
    def get_transfer_status(self, transfer_id: str) -> Dict[str, Any]:
        """
        Get the current status of a transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            Transfer status information
        """
        return self._get_transfer_status(transfer_id)
    
    def cancel_transfer(self, transfer_id: str) -> bool:
        """
        Cancel an in-progress transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            True if successful
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            return False
            
        transfer_info = self.transfers[transfer_id]
        
        # Check if transfer can be cancelled
        if transfer_info["status"] in [
            TransferStatus.COMPLETED.value,
            TransferStatus.FAILED.value,
            TransferStatus.CANCELLED.value
        ]:
            return False
            
        # Update transfer info
        transfer_info["status"] = TransferStatus.CANCELLED.value
        transfer_info["updated_at"] = datetime.utcnow().isoformat()
        
        # Remove from queue if present
        with self.transfer_lock:
            if transfer_id in self.transfer_queue:
                self.transfer_queue.remove(transfer_id)
                
        # Clean up temporary files
        self._cleanup_transfer(transfer_id)
        
        return True
    
    def list_transfers(
        self,
        status: Optional[TransferStatus] = None,
        transfer_type: Optional[TransferType] = None,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List transfers with optional filtering.
        
        Args:
            status: Filter by transfer status
            transfer_type: Filter by transfer type
            agent_id: Filter by agent ID (sender or recipient)
            limit: Maximum number of transfers to return
            
        Returns:
            List of transfer info dictionaries
        """
        transfers = []
        
        for transfer_id, transfer_info in self.transfers.items():
            # Apply filters
            if status and transfer_info["status"] != status.value:
                continue
                
            if transfer_type and transfer_info["transfer_type"] != transfer_type.value:
                continue
                
            if agent_id and (
                transfer_info["sender_id"] != agent_id and
                transfer_info["recipient_id"] != agent_id
            ):
                continue
                
            # Add to results
            transfers.append(self._get_transfer_status(transfer_id))
            
            # Check limit
            if len(transfers) >= limit:
                break
                
        return transfers
    
    def resume_transfer(self, transfer_id: str) -> bool:
        """
        Resume a paused transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            True if successful
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            return False
            
        transfer_info = self.transfers[transfer_id]
        
        # Check if transfer is paused
        if transfer_info["status"] != TransferStatus.PAUSED.value:
            return False
            
        # Update transfer info
        transfer_info["status"] = TransferStatus.PENDING.value
        transfer_info["updated_at"] = datetime.utcnow().isoformat()
        
        # Add to queue
        with self.transfer_lock:
            if transfer_id not in self.transfer_queue:
                self.transfer_queue.append(transfer_id)
                
        # Process queue
        self._process_transfer_queue()
        
        return True
    
    def cleanup_completed_transfers(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed transfers.
        
        Args:
            max_age_hours: Maximum age in hours for completed transfers
            
        Returns:
            Number of transfers cleaned up
        """
        cleanup_count = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        transfers_to_remove = []
        
        # Find transfers to clean up
        for transfer_id, transfer_info in self.transfers.items():
            if transfer_info["status"] not in [
                TransferStatus.COMPLETED.value,
                TransferStatus.FAILED.value,
                TransferStatus.CANCELLED.value
            ]:
                continue
                
            # Check age
            updated_at = datetime.fromisoformat(transfer_info["updated_at"])
            if updated_at < cutoff_time:
                transfers_to_remove.append(transfer_id)
                
        # Clean up transfers
        for transfer_id in transfers_to_remove:
            self._cleanup_transfer(transfer_id)
            with self.transfer_lock:
                if transfer_id in self.transfers:
                    del self.transfers[transfer_id]
            cleanup_count += 1
            
        return cleanup_count
    
    def create_file_manifest(
        self,
        files: List[str],
        recipient_id: str,
        manifest_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a manifest for a group of files to be exchanged.
        
        Args:
            files: List of file paths to include
            recipient_id: ID of the recipient agent
            manifest_id: Optional custom manifest ID
            
        Returns:
            Manifest information
        """
        # Generate manifest ID if not provided
        if not manifest_id:
            manifest_id = f"manifest-{uuid.uuid4().hex}"
            
        # Collect file information
        file_infos = []
        for file_path in files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            file_info = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "content_type": mimetypes.guess_type(file_path)[0] or "application/octet-stream",
                "file_hash": self._calculate_file_hash(file_path)
            }
            
            file_infos.append(file_info)
            
        # Create manifest
        manifest = {
            "manifest_id": manifest_id,
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "created_at": datetime.utcnow().isoformat(),
            "files": file_infos,
            "total_size": sum(info["file_size"] for info in file_infos),
            "file_count": len(file_infos),
            "status": "created",
            "transfers": []
        }
        
        # Save manifest
        manifest_path = os.path.join(self.outgoing_dir, f"{manifest_id}.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest
    
    def process_file_manifest(
        self,
        manifest: Dict[str, Any],
        accept: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a file manifest received from another agent.
        
        Args:
            manifest: The manifest to process
            accept: Whether to accept the files
            output_dir: Optional custom output directory
            
        Returns:
            Updated manifest information
        """
        manifest_id = manifest["manifest_id"]
        sender_id = manifest["sender_id"]
        
        # Verify recipient
        if manifest["recipient_id"] != self.agent_id:
            raise ValueError(f"Manifest not addressed to this agent")
            
        # Update manifest status
        manifest["status"] = "accepted" if accept else "rejected"
        manifest["processed_at"] = datetime.utcnow().isoformat()
        
        if not accept:
            # Save manifest
            manifest_path = os.path.join(self.incoming_dir, f"{manifest_id}.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            return manifest
            
        # Determine output directory
        if not output_dir:
            output_dir = os.path.join(self.incoming_dir, manifest_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Update manifest with output directory
        manifest["output_dir"] = output_dir
        manifest["transfers"] = []
        
        # Save manifest
        manifest_path = os.path.join(self.incoming_dir, f"{manifest_id}.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest
    
    def get_transfer_progress(self, transfer_id: str) -> Dict[str, Any]:
        """
        Get detailed progress information for a transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            Progress information
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            raise ValueError(f"Transfer not found: {transfer_id}")
            
        transfer_info = self.transfers[transfer_id]
        
        # Get basic progress info
        progress_info = {
            "transfer_id": transfer_id,
            "status": transfer_info["status"],
            "progress": transfer_info["progress"],
            "file_name": transfer_info["file_name"],
            "file_size": transfer_info["file_size"],
            "transfer_type": transfer_info["transfer_type"]
        }
        
        # Add chunk information for chunked transfers
        if transfer_info["should_chunk"]:
            chunks_info = transfer_info["chunks"]
            progress_info["chunks"] = {
                "total": chunks_info["total"],
                "processed": chunks_info["sent"] if transfer_info["transfer_type"] == TransferType.UPLOAD.value else chunks_info["received"],
                "remaining": chunks_info["total"] - (chunks_info["sent"] if transfer_info["transfer_type"] == TransferType.UPLOAD.value else chunks_info["received"])
            }
            
        # Add timing information
        created_at = datetime.fromisoformat(transfer_info["created_at"])
        updated_at = datetime.fromisoformat(transfer_info["updated_at"])
        elapsed = (updated_at - created_at).total_seconds()
        
        progress_info["timing"] = {
            "elapsed_seconds": elapsed,
            "created_at": transfer_info["created_at"],
            "updated_at": transfer_info["updated_at"]
        }
        
        # Estimate remaining time if in progress
        if transfer_info["status"] == TransferStatus.IN_PROGRESS.value and transfer_info["progress"] > 0:
            progress_percent = transfer_info["progress"] / 100.0
            if progress_percent > 0:
                total_estimated_time = elapsed / progress_percent
                remaining_time = total_estimated_time - elapsed
                progress_info["timing"]["estimated_remaining_seconds"] = remaining_time
                
        return progress_info
    
    # --- Helper Methods ---
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hex digest of the file hash
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read and update hash in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
                
        return sha256.hexdigest()
    
    def _get_transfer_status(self, transfer_id: str) -> Dict[str, Any]:
        """
        Get status information for a transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            Transfer status information
        """
        if transfer_id not in self.transfers:
            raise ValueError(f"Transfer not found: {transfer_id}")
            
        transfer_info = self.transfers[transfer_id]
        
        # Create a simplified status object
        status_info = {
            "transfer_id": transfer_id,
            "status": transfer_info["status"],
            "progress": transfer_info["progress"],
            "file_name": transfer_info["file_name"],
            "file_size": transfer_info["file_size"],
            "content_type": transfer_info["content_type"],
            "sender_id": transfer_info["sender_id"],
            "recipient_id": transfer_info["recipient_id"],
            "transfer_type": transfer_info["transfer_type"],
            "created_at": transfer_info["created_at"],
            "updated_at": transfer_info["updated_at"],
            "error": transfer_info["error"]
        }
        
        return status_info
    
    def _process_transfer_queue(self) -> None:
        """Process the transfer queue."""
        with self.transfer_lock:
            # Check if we have capacity for more transfers
            while (
                self.transfer_queue and 
                self.active_transfers < self.max_concurrent_transfers
            ):
                # Get next transfer
                transfer_id = self.transfer_queue.pop(0)
                
                # Skip if transfer doesn't exist anymore
                if transfer_id not in self.transfers:
                    continue
                
                # Skip if not in pending state
                transfer_info = self.transfers[transfer_id]
                if transfer_info["status"] != TransferStatus.PENDING.value:
                    continue
                
                # Update status and increment active count
                transfer_info["status"] = TransferStatus.IN_PROGRESS.value
                transfer_info["updated_at"] = datetime.utcnow().isoformat()
                self.active_transfers += 1
                
                # No actual threading needed for this implementation
                # since we're relying on the agent to call methods
    
    def _cleanup_transfer(self, transfer_id: str) -> None:
        """
        Clean up temporary files for a transfer.
        
        Args:
            transfer_id: ID of the transfer to clean up
        """
        # Check if transfer exists
        if transfer_id not in self.transfers:
            return
            
        # Clean up temporary file
        temp_file_path = os.path.join(self.temp_dir, transfer_id)
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file for transfer {transfer_id}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file for transfer {transfer_id}: {str(e)}")
                
        # Decrement active transfers count if this was an active transfer
        transfer_info = self.transfers[transfer_id]
        if transfer_info["status"] == TransferStatus.IN_PROGRESS.value:
            with self.transfer_lock:
                self.active_transfers = max(0, self.active_transfers - 1)
