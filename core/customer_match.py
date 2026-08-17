def match_customer(old, new):
    score=0

    fields=['name','age','job','software','receiver']

    for f in fields:
        if old.get(f) and old.get(f)==new.get(f):
            score+=20

    return score
