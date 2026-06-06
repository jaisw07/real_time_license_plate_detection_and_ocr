import re

def clean_ocr_text(raw_text: str) -> str:
    """
    Generic alphanumeric cleanup.
    Strips symbols. Fixes common O/0, I/1, S/5, B/8 confusion based on neighbors.
    """
    # 1. Strip all non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper().strip())
    
    if not cleaned:
        return ""
        
    # 2. Fix isolated characters based on surrounding context
    # Convert string to list for mutability
    chars = list(cleaned)
    
    for i in range(len(chars)):
        c = chars[i]
        
        # Check neighbors to determine if we are in a 'letter' block or 'number' block
        prev_is_digit = chars[i-1].isdigit() if i > 0 else False
        next_is_digit = chars[i+1].isdigit() if i < len(chars) - 1 else False
        
        prev_is_alpha = chars[i-1].isalpha() if i > 0 else False
        next_is_alpha = chars[i+1].isalpha() if i < len(chars) - 1 else False
        
        looks_like_number_block = prev_is_digit or next_is_digit
        looks_like_letter_block = prev_is_alpha or next_is_alpha
        
        # If surrounded by numbers, it's likely a number
        if looks_like_number_block and not looks_like_letter_block:
            if c == 'O': chars[i] = '0'
            elif c == 'I': chars[i] = '1'
            elif c == 'S': chars[i] = '5'
            elif c == 'B': chars[i] = '8'
            elif c == 'G': chars[i] = '6'
            elif c == 'Z': chars[i] = '2'
            
        # If surrounded by letters, it's likely a letter
        elif looks_like_letter_block and not looks_like_number_block:
            if c == '0': chars[i] = 'O'
            elif c == '1': chars[i] = 'I'
            elif c == '5': chars[i] = 'S'
            elif c == '8': chars[i] = 'B'
            elif c == '6': chars[i] = 'G'
            elif c == '2': chars[i] = 'Z'
            
    return "".join(chars)
