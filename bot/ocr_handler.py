
from ocr.extractor import extract_text
from core.customer import parse_customer_info


def process_customer_image(path):

    text = extract_text(path)

    info = parse_customer_info(text)

    return {
        "text": text,
        "info": info
    }
