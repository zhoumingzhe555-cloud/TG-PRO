# 使用 core/customer.py 里更完整的解析器（支持繁简混合、更多字段别名）
from core.customer import parse_customer_info, is_customer_record


def scan_text(text):
    data = parse_customer_info(text)
    if is_customer_record(data):
        return data
    return None
