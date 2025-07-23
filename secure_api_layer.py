# Secure API Layer for SIPD AI Chatbot
# Implementasi lapisan keamanan dengan data masking dan fitur GRC

import os
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from datetime import datetime
import uuid

class SecureAPILayer:
    """Secure API Layer with data masking and GRC features"""
    
    def __init__(self, audit_log_path: str = "logs/audit_trail.jsonl"):
        self.audit_log_path = audit_log_path
        self.audit_trail = []
        self.max_memory_logs = 1000  # Maximum number of logs to keep in memory
        self.sensitive_patterns = {
            "password": r'\b(?:password|kata\s*sandi|katasandi|pwd|pass)\s*[:=]\s*([^\s,;]{3,})',
            "nik": r'\b\d{16}\b',  # 16 digit NIK
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+62|62|0)\d{9,12}\b',  # Indonesian phone numbers
            "credit_card": r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            "api_key": r'\b[A-Za-z0-9_\-]{20,}\b'
        }
        
        # Setup logging
        logger.add("logs/secure_api.log", rotation="10 MB", level="INFO")
        
        # Ensure audit log directory exists
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data like passwords, NIK, etc."""
        if not text:
            return text
            
        masked_text = text
        
        # Apply each pattern
        for data_type, pattern in self.sensitive_patterns.items():
            if data_type == "password":
                # For passwords, keep first character and mask the rest
                masked_text = re.sub(
                    pattern, 
                    lambda m: m.group(0).replace(m.group(1), m.group(1)[0] + "*" * (len(m.group(1))-1)),
                    masked_text
                )
            else:
                # For other sensitive data, replace with [MASKED]
                replacement = f"[{data_type.upper()} MASKED]"
                masked_text = re.sub(pattern, replacement, masked_text)
        
        return masked_text
    
    async def log_audit_trail(self, user_id: str, action: str, details: Dict[str, Any], persist: bool = True):
        """Log audit trail for compliance"""
        try:
            # Generate log entry
            timestamp = datetime.now().isoformat()
            log_entry = {
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "user_id": user_id,
                "action": action,
                "details": details
            }
            
            # Add to memory
            self.audit_trail.append(log_entry)
            
            # Limit memory logs
            if len(self.audit_trail) > self.max_memory_logs:
                self.audit_trail = self.audit_trail[-self.max_memory_logs:]
            
            # Persist to file if requested
            if persist:
                await self._persist_log(log_entry)
            
            logger.info(f"Audit trail: {action} by {user_id} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging audit trail: {e}")
            return False
    
    async def _persist_log(self, log_entry: Dict[str, Any]):
        """Persist log entry to file"""
        try:
            # Write to file asynchronously
            async with asyncio.Lock():
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception as e:
            logger.error(f"Error persisting log: {e}")
            return False
    
    async def get_audit_logs(self, user_id: Optional[str] = None, action: Optional[str] = None, 
                           start_time: Optional[str] = None, end_time: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get filtered audit logs"""
        try:
            # Filter logs
            filtered_logs = self.audit_trail
            
            if user_id:
                filtered_logs = [log for log in filtered_logs if log["user_id"] == user_id]
            
            if action:
                filtered_logs = [log for log in filtered_logs if log["action"] == action]
            
            if start_time:
                filtered_logs = [log for log in filtered_logs if log["timestamp"] >= start_time]
            
            if end_time:
                filtered_logs = [log for log in filtered_logs if log["timestamp"] <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_logs = sorted(filtered_logs, key=lambda x: x["timestamp"], reverse=True)
            
            # Apply limit
            return filtered_logs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []
    
    def check_access_permission(self, user_id: str, resource: str, action: str = "read") -> bool:
        """Check if user has permission to access resource"""
        try:
            # In a real implementation, this would check against a permission database
            # For now, we'll use a simple rule-based approach
            
            # Admin users have access to everything
            if user_id.startswith("admin"):
                return True
            
            # Guest users have limited access
            if user_id.startswith("guest"):
                if action == "read":
                    return True
                else:
                    return False
            
            # Default to allow access
            return True
            
        except Exception as e:
            logger.error(f"Error checking access permission: {e}")
            # Default to deny access on error
            return False
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information (PII) in text"""
        try:
            pii_found = {}
            
            # Check for each pattern
            for data_type, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    pii_found[data_type] = matches
            
            return pii_found
            
        except Exception as e:
            logger.error(f"Error detecting PII: {e}")
            return {}
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status"""
        try:
            # In a real implementation, this would check various compliance metrics
            return {
                "audit_trail_enabled": True,
                "data_masking_enabled": True,
                "pii_detection_enabled": True,
                "access_control_enabled": True,
                "audit_log_count": len(self.audit_trail),
                "audit_log_path": self.audit_log_path,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    async def test_secure_api():
        # Initialize secure API
        secure_api = SecureAPILayer()
        
        # Test data masking
        test_text = "My password: secret123 and my NIK is 1234567890123456 and my email is user@example.com"
        masked_text = secure_api.mask_sensitive_data(test_text)
        
        print("\nOriginal text:")
        print(test_text)
        print("\nMasked text:")
        print(masked_text)
        
        # Test audit trail
        await secure_api.log_audit_trail(
            user_id="test_user",
            action="login",
            details={
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0",
                "success": True
            }
        )
        
        await secure_api.log_audit_trail(
            user_id="test_user",
            action="query",
            details={
                "query": "How to fix login issues",
                "response_time": 0.5
            }
        )
        
        # Get audit logs
        logs = await secure_api.get_audit_logs(user_id="test_user")
        
        print("\nAudit logs:")
        for log in logs:
            print(f"ID: {log['id']}")
            print(f"Timestamp: {log['timestamp']}")
            print(f"User: {log['user_id']}")
            print(f"Action: {log['action']}")
            print(f"Details: {log['details']}")
            print()
        
        # Test PII detection
        pii = secure_api.detect_pii(test_text)
        
        print("\nPII detected:")
        for data_type, matches in pii.items():
            print(f"{data_type}: {matches}")
        
        # Test compliance status
        status = secure_api.get_compliance_status()
        
        print("\nCompliance status:")
        for key, value in status.items():
            print(f"{key}: {value}")
    
    # Run the test
    asyncio.run(test_secure_api())