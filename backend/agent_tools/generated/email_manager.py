"""
Email manager tool for Gmail using IMAP and local LLM classification.
Handles safe deletion of marketing/notification emails while preserving personal correspondence.
"""

import imaplib
import email
from email.header import decode_header
from typing import Optional, Any
import json
from pathlib import Path
from datetime import datetime, timedelta

from agent_tools import register_tool

# Global state for connection persistence across function calls
_gmail_connection = None


async def gmail_connect(username: str, app_password: str) -> dict[str, Any]:
    """
    Connect to Gmail via IMAP using an app password.
    Stores connection globally for use by other functions.
    """
    global _gmail_connection
    
    try:
        imap = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        imap.login(username, app_password)
        _gmail_connection = imap
        return {
            'success': True,
            'username': username,
            'message': f'Connected to Gmail as {username}'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to connect: {str(e)}'
        }


async def gmail_scan_inbox(max_emails: int = 100, days_old: Optional[int] = None) -> dict[str, Any]:
    """
    Scan Gmail inbox and extract email metadata.
    Returns: list of emails with subject, sender, date, and uid.
    """
    global _gmail_connection
    
    try:
        if not _gmail_connection:
            return {'success': False, 'message': 'Not connected to Gmail. Call gmail_connect first.'}
        
        _gmail_connection.select('INBOX')
        
        # Build search query
        search_query = 'ALL'
        if days_old:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%d-%b-%Y')
            search_query = f'BEFORE {cutoff_date}'
        
        status, messages = _gmail_connection.search(None, search_query)
        email_ids = messages[0].split()[-max_emails:]  # Get last N emails
        
        emails = []
        for email_id in email_ids:
            status, msg_data = _gmail_connection.fetch(email_id, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])
            
            # Decode subject
            subject = msg.get('Subject', '(no subject)')
            if isinstance(subject, str):
                try:
                    decoded_parts = decode_header(subject)
                    subject = ''.join(
                        part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part
                        for part, encoding in decoded_parts
                    )
                except:
                    pass
            
            sender = msg.get('From', '(unknown sender)')
            date = msg.get('Date', '(unknown date)')
            
            emails.append({
                'uid': email_id.decode() if isinstance(email_id, bytes) else email_id,
                'subject': str(subject)[:100],  # Limit length
                'sender': str(sender)[:100],
                'date': str(date)[:30],
                'has_body': bool(msg.get_payload())
            })
        
        return {
            'success': True,
            'count': len(emails),
            'emails': emails,
            'message': f'Scanned {len(emails)} emails from inbox'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to scan inbox: {str(e)}'
        }


async def gmail_classify_with_llm(emails: list) -> dict[str, Any]:
    """
    Use local LLM to classify emails as DELETE or KEEP.
    Returns categorization plan with reasoning.
    """
    try:
        from ollama import Client
        client = Client(host='http://localhost:11434')
        
        # Build classification prompt
        prompt = """You are an email classification expert. Analyze the following emails and decide which should be DELETED.

RULES:
- DELETE: Marketing, promotional, newsletters, automated notifications, spam, notifications from services/apps, social media emails
- KEEP: Personal emails from individuals, important account information, receipts you might need, work emails
- NEVER DELETE: Emails from people writing directly to you (not automated)

For each email, respond with ONLY: EMAIL_[uid]|DELETE or EMAIL_[uid]|KEEP with a brief reason (max 10 words).

EMAILS TO CLASSIFY:
"""
        for mail in emails[:50]:  # Limit to 50 for performance
            prompt += f"\nEMAIL_{mail['uid']}|From: {mail['sender']}|Subject: {mail['subject']}"
        
        prompt += "\n\nRespond with one classification per line, nothing else:"
        
        response = client.generate(
            model='mistral',
            prompt=prompt,
            stream=False,
        )
        
        # Parse response
        classifications = {}
        for line in response['response'].split('\n'):
            line = line.strip()
            if '|' in line and 'EMAIL_' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    uid = parts[0].replace('EMAIL_', '').strip()
                    action = 'DELETE' if 'DELETE' in parts[1].upper() else 'KEEP'
                    reason = parts[1] if len(parts) > 2 else parts[1]
                    classifications[uid] = {'action': action, 'reason': reason}
        
        delete_list = [uid for uid, info in classifications.items() if info['action'] == 'DELETE']
        keep_list = [uid for uid, info in classifications.items() if info['action'] == 'KEEP']
        
        return {
            'success': True,
            'total_classified': len(classifications),
            'to_delete': delete_list,
            'to_keep': keep_list,
            'classifications': classifications,
            'message': f'Classified {len(classifications)} emails: {len(delete_list)} to delete, {len(keep_list)} to keep'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to classify emails: {str(e)}'
        }


async def gmail_preview_deletion(emails: list, classifications: dict) -> dict[str, Any]:
    """
    Create a human-readable preview of what will be deleted.
    """
    deletion_preview = []
    for mail in emails:
        uid = mail['uid']
        if uid in classifications and classifications[uid]['action'] == 'DELETE':
            deletion_preview.append({
                'uid': uid,
                'subject': mail['subject'],
                'sender': mail['sender'],
                'date': mail['date'],
                'reason': classifications[uid].get('reason', 'Marketing/Notification')
            })
    
    return {
        'success': True,
        'total_to_delete': len(deletion_preview),
        'preview': deletion_preview,
        'message': f'Preview: {len(deletion_preview)} emails ready for deletion'
    }


async def gmail_delete_batch(email_ids: list, dry_run: bool = True) -> dict[str, Any]:
    """
    Delete a batch of emails by UID. Set dry_run=False to actually delete.
    """
    global _gmail_connection
    
    try:
        if not _gmail_connection:
            return {'success': False, 'message': 'Not connected to Gmail. Call gmail_connect first.'}
        
        _gmail_connection.select('INBOX')
        deleted_count = 0
        
        for email_id in email_ids:
            try:
                if not dry_run:
                    _gmail_connection.store(email_id, '+FLAGS', '\\Deleted')
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting email {email_id}: {str(e)}")
        
        if not dry_run:
            _gmail_connection.expunge()
        
        action = 'Would delete' if dry_run else 'Deleted'
        return {
            'success': True,
            'dry_run': dry_run,
            'count': len(email_ids),
            'message': f'{action} {len(email_ids)} emails'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to delete emails: {str(e)}'
        }


async def gmail_disconnect() -> dict[str, Any]:
    """
    Close Gmail IMAP connection.
    """
    global _gmail_connection
    
    try:
        if _gmail_connection:
            _gmail_connection.close()
            _gmail_connection.logout()
            _gmail_connection = None
        return {
            'success': True,
            'message': 'Disconnected from Gmail'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error closing connection: {str(e)}'
        }


# Tool registration
def register_email_manager_tools() -> None:
    """Register all email manager tools with the agent."""
    
    register_tool(
        name='gmail_connect',
        description='Connect to Gmail via IMAP using app password. Username should be your Gmail address.',
        input_schema={
            'type': 'object',
            'properties': {
                'username': {'type': 'string', 'description': 'Gmail address'},
                'app_password': {'type': 'string', 'description': 'App-specific password from Google Account'}
            },
            'required': ['username', 'app_password']
        },
        handler=gmail_connect,
        is_destructive=False
    )
    
    register_tool(
        name='gmail_scan_inbox',
        description='Scan Gmail inbox and extract email metadata (subject, sender, date). Returns list of recent emails.',
        input_schema={
            'type': 'object',
            'properties': {
                'max_emails': {'type': 'integer', 'description': 'Max emails to scan', 'default': 100},
                'days_old': {'type': 'integer', 'description': 'Only get emails older than X days (optional)'}
            },
            'required': []
        },
        handler=gmail_scan_inbox,
        is_destructive=False
    )
    
    register_tool(
        name='gmail_classify_with_llm',
        description='Use local Mistral model to classify emails as DELETE or KEEP based on content',
        input_schema={
            'type': 'object',
            'properties': {
                'emails': {'type': 'array', 'description': 'List of email dicts with uid, subject, sender fields'}
            },
            'required': ['emails']
        },
        handler=gmail_classify_with_llm,
        is_destructive=False
    )
    
    register_tool(
        name='gmail_preview_deletion',
        description='Create human-readable preview of which emails will be deleted',
        input_schema={
            'type': 'object',
            'properties': {
                'emails': {'type': 'array', 'description': 'Original email list'},
                'classifications': {'type': 'object', 'description': 'Classifications dict from LLM'}
            },
            'required': ['emails', 'classifications']
        },
        handler=gmail_preview_deletion,
        is_destructive=False
    )
    
    register_tool(
        name='gmail_delete_batch',
        description='Delete a batch of emails by UID. Use dry_run=True first to preview.',
        input_schema={
            'type': 'object',
            'properties': {
                'email_ids': {'type': 'array', 'description': 'List of email UIDs to delete'},
                'dry_run': {'type': 'boolean', 'description': 'If True, preview only. Default: True', 'default': True}
            },
            'required': ['email_ids']
        },
        handler=gmail_delete_batch,
        is_destructive=True
    )
    
    register_tool(
        name='gmail_disconnect',
        description='Close Gmail IMAP connection',
        input_schema={
            'type': 'object',
            'properties': {},
            'required': []
        },
        handler=gmail_disconnect,
        is_destructive=False
    )
