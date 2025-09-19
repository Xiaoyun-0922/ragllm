import re

def extract_key_info(texts):
    seq, mic, struct = None, None, None
    for t in texts:
        if not seq:
            m = re.search(r'(sequence|序列)[:：]?\s*([A-Za-z\-]+)', t, re.I)
            if m:
                seq = m.group(2)
        if not mic:
            m = re.search(r'(MIC)[:：]?\s*([\d\.]+ ?(μ?g/?mL|mg/?L|ug/ml))', t, re.I)
            if m:
                mic = m.group(2)
        if not struct:
            m = re.search(r'(structure|结构)[:：]?\s*([A-Za-z0-9\- ]+)', t, re.I)
            if m:
                struct = m.group(2)
    return seq, mic, struct
