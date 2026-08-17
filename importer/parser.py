def parse_customer(text):
    data = {
        'name':'',
        'age':'',
        'job':'',
        'income':'',
        'work_year':'',
        'software':'',
        'receiver':''
    }

    rules = {
        'name':['姓名'],
        'age':['年龄','年齡'],
        'job':['职业','職業'],
        'income':['收入'],
        'work_year':['工作年限'],
        'software':['引流软件','引流軟件'],
        'receiver':['接粉人员','接粉人員']
    }

    for line in text.splitlines():
        for key, words in rules.items():
            for w in words:
                if w in line:
                    value=line.replace(w,'').replace(':','').replace('：','').strip()
                    data[key]=value

    return data
