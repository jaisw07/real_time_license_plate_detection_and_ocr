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

def validate_plate(plate_text: str) -> bool:
    """Validate plate text against Indian formats."""
    # BH Series: e.g. 21BH2345AA
    bh_pattern = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")
    if bh_pattern.match(plate_text):
        return True
        
    # Standard RTO: e.g. MH12AB1234
    std_pattern = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")
    if std_pattern.match(plate_text) and 8 <= len(plate_text) <= 10:
        return True
        
    return False

def calculate_iou(box1: dict, box2: dict) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
    x1_inter = max(box1['x1'], box2['x1'])
    y1_inter = max(box1['y1'], box2['y1'])
    x2_inter = min(box1['x2'], box2['x2'])
    y2_inter = min(box1['y2'], box2['y2'])
    
    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h
    
    area1 = max(0, box1['x2'] - box1['x1']) * max(0, box1['y2'] - box1['y1'])
    area2 = max(0, box2['x2'] - box2['x1']) * max(0, box2['y2'] - box2['y1'])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def try_merge(a: str, b: str) -> list[str]:
    """Try raw concatenation and overlap-removed concatenation."""
    merges = [a + b, b + a]
    for k in range(1, min(len(a), len(b)) + 1):
        if a[-k:] == b[:k]:
            merges.append(a + b[k:])
        if b[-k:] == a[:k]:
            merges.append(b + a[k:])
    return list(set(merges))

def deduplicate_plates(candidates: list[dict]) -> list[dict]:
    """Group by IoU, deduplicate via substrings/merges, validate, and return best per group."""
    if not candidates:
        return []
        
    # Step 2a: Group by bounding box IoU
    groups = []
    visited = set()
    for i in range(len(candidates)):
        if i in visited:
            continue
        group = [candidates[i]]
        visited.add(i)
        for j in range(i + 1, len(candidates)):
            if j in visited:
                continue
            if calculate_iou(candidates[i]['bounding_box'], candidates[j]['bounding_box']) > 0.3:
                group.append(candidates[j])
                visited.add(j)
        groups.append(group)
        
    final_results = []
    
    # Step 2b: Within group, attempt merge/dedup
    for group in groups:
        valid_cands = []
        
        # 1. Substring dedup + initial validation
        for i, c in enumerate(group):
            is_substring = False
            for j, oc in enumerate(group):
                if i != j and c['plate_number'] in oc['plate_number']:
                    is_substring = True
                    break
            
            if not is_substring:
                if validate_plate(c['plate_number']):
                    valid_cands.append(c)
                    
        # 2. Fragment merge (if multiple fragments exist)
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a_txt = group[i]['plate_number']
                    b_txt = group[j]['plate_number']
                    if not a_txt or not b_txt: continue
                    
                    merged_opts = try_merge(a_txt, b_txt)
                    for m in merged_opts:
                        if validate_plate(m):
                            m_cand = group[i].copy()
                            m_cand['plate_number'] = m
                            m_cand['raw_ocr'] = m
                            m_cand['confidence'] = {
                                "detection": max(group[i]['confidence']['detection'], group[j]['confidence']['detection']),
                                "ocr": max(group[i]['confidence']['ocr'], group[j]['confidence']['ocr'])
                            }
                            m_cand['bounding_box'] = {
                                "x1": min(group[i]['bounding_box']['x1'], group[j]['bounding_box']['x1']),
                                "y1": min(group[i]['bounding_box']['y1'], group[j]['bounding_box']['y1']),
                                "x2": max(group[i]['bounding_box']['x2'], group[j]['bounding_box']['x2']),
                                "y2": max(group[i]['bounding_box']['y2'], group[j]['bounding_box']['y2']),
                            }
                            valid_cands.append(m_cand)
                            
        # 3. Pick best valid by OCR confidence
        if valid_cands:
            best = max(valid_cands, key=lambda x: x['confidence']['ocr'])
            final_results.append(best)
            
    return final_results
