from core.image_features import extract_features

def extract_feature(path):
    return extract_features(path).get("ai_feature")
