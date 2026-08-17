from core.image_match import confirm_collision

def save_confirm(collision_id,result,user,user_id=""):
    status="confirmed" if result in {"confirm","confirmed","撞客"} else "false_positive"
    return confirm_collision(collision_id,status,user,user_id)
