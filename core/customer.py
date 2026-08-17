import re

FIELD_RULES = {
    "name": ["姓名", "名字"],
    "age": ["年龄", "年齡"],
    "job": ["职业", "職業"],
    "income": ["收入"],
    "work_year": ["工作年限", "工作年數", "工作年数"],
    "software": ["引流软件", "引流軟件"],
    "receiver": ["接粉人员", "接粉人員", "接粉人"],
}


def _clean_value(v):
    return re.sub(r"^[\s:：=\-]+", "", v or "").strip()


def parse_customer_info(text):
    data = {k: "" for k in FIELD_RULES}
    labels_found = set()
    if not text:
        return data

    for raw_line in str(text).replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for key, aliases in FIELD_RULES.items():
            for label in aliases:
                # 只把明确出现字段标签的行作为正式资料，避免普通聊天误入库。
                m = re.search(rf"(?:^|\s){re.escape(label)}\s*[:：]?\s*(.*)$", line, flags=re.I)
                if m:
                    data[key] = _clean_value(m.group(1))
                    labels_found.add(key)
                    break

    data["_labels_found"] = labels_found
    return data


def is_customer_record(data):
    labels = set(data.get("_labels_found") or [])
    # 用户的正式资料模板以“姓名”为核心；至少再出现两个字段标签才入正式客户库。
    return "name" in labels and len(labels) >= 3


def public_customer_data(data):
    return {k: data.get(k, "") for k in FIELD_RULES}
