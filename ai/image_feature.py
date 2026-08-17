# 向后兼容旧模块名。
from core.image_match import check_image

def check_ai_similarity(image):
    r=check_image(image)
    return r if r.get("type") in {"same","similar"} else None
