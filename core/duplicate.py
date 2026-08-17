# V1.4 兼容层：旧代码如果仍调用 duplicate.check_image/save_image，不会失效。
from core.image_match import check_image as _check_image, save_customer_and_image


def check_image(path):
    result=_check_image(path)
    if result["type"]=="new": return None
    if result["type"]=="same":
        return {"type":"same","image_id":result["matched_image_id"],"score":100.0}
    return {"type":"similar","image_id":result["matched_image_id"],"score":result["score"]}


def save_image(path,file_id,submitter,chat_id):
    # 旧接口没有客户资料；按新版规则“单图片只检测不入库”，因此不再把裸图片正式入客户库。
    return None
