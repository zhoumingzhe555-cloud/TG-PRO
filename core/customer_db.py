from datetime import datetime
from core.database import get_conn


def save_customer(data, submitter="", chat_id=""):
    conn = get_conn()
    conn.execute("""
    INSERT INTO customers
    (name, age, job, income, work_year, software, receiver, submitter, chat_id, created_time)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("name"),
        data.get("age"),
        data.get("job"),
        data.get("income"),
        data.get("work_year"),
        data.get("software"),
        data.get("receiver"),
        submitter,
        chat_id,
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()
