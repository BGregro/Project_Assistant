import re
from typing import Dict, List

async def classify_emails_for_deletion(emails: List[Dict]) -> Dict:
    """
    Classify emails as DELETE or KEEP based on content patterns.
    
    Rules:
    - DELETE: Marketing, newsletters, notifications, promotions
    - KEEP: Personal emails from individuals, transactional receipts you need, 
            university emails, system notifications for YOUR services
    """
    
    # Patterns that indicate DELETE
    delete_patterns = {
        'marketing': [
            r'marketing|promo|promotional|discount|sale|offer|special deal|limited time',
            r'100% off|50% off|coupon|voucher|redeem',
        ],
        'newsletter': [
            r'newsletter|news\s+update|weekly|monthly|digest|unsubscribe',
            r'update|new release|what\'s new|feature announcement',
        ],
        'notification_spam': [
            r'you have.*new messages?|notification',
            r'trending|recommended for you|check out|suggested',
            r'social media|follow us|like us|share',
        ],
        'unsolicited': [
            r'make money|earn|crypto|mining|investment|get rich',
            r'click here|act now|don\'t miss|hurry|limited spots',
        ],
        'automated_generic': [
            r'ryanair|kajabi|shopify|store\+|decathlon|headout|ajouter',
            r'balatonman|ötproba|greengo|analog devices marketing',
        ]
    }
    
    # Patterns that indicate KEEP (take priority)
    keep_patterns = {
        'personal': [
            r'chris williamson',  # Individual content creator
        ],
        'important_notification': [
            r'receipt|invoice|payment|billing',
            r'security alert|account|access|verify|password|login',
            r'github.*token|github.*application|docker.*linked',
        ],
        'educational': [
            r'neptun|university|university|course|exam|grade|scholarship',
            r'ppke|itk-hallgatok',  # University mailing list
        ],
        'service_important': [
            r'ollama|anthropic.*receipt|anthropic.*payment|supabase.*update',
            r'onedrive|discord|duolingo|notion',
            r'patreon.*terms|paypal.*terms',
        ]
    }
    
    classifications = {}
    
    for email in emails:
        uid = email['uid']
        subject = email.get('subject', '').lower()
        sender = email.get('sender', '').lower()
        
        # Combine subject and sender for analysis
        full_text = f"{subject} {sender}".lower()
        
        # Check KEEP patterns first (higher priority)
        should_keep = False
        for category, patterns in keep_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text):
                    should_keep = True
                    break
            if should_keep:
                break
        
        # If not explicitly kept, check DELETE patterns
        if should_keep:
            classifications[uid] = 'KEEP'
        else:
            should_delete = False
            for category, patterns in delete_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, full_text):
                        should_delete = True
                        break
                if should_delete:
                    break
            
            classifications[uid] = 'DELETE' if should_delete else 'KEEP'
    
    return {
        'success': True,
        'classifications': classifications,
        'total': len(classifications),
        'delete_count': sum(1 for v in classifications.values() if v == 'DELETE'),
        'keep_count': sum(1 for v in classifications.values() if v == 'KEEP'),
    }


def register_email_classifier_tools():
    """Register email classification tool"""
    from agent_tools import register_tool
    
    register_tool(
        name='classify_emails_for_deletion',
        description='Classify emails as DELETE or KEEP using intelligent pattern matching. '
                    'Takes a list of email dicts with uid, subject, sender fields.',
        input_schema={
            'type': 'object',
            'properties': {
                'emails': {
                    'type': 'array',
                    'description': 'List of email dicts with uid, subject, sender fields',
                    'items': {'type': 'object'}
                }
            },
            'required': ['emails']
        },
        handler=classify_emails_for_deletion,
        is_destructive=False,
    )
