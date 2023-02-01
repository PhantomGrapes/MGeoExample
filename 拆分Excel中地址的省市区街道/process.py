from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pandas as pd


def get_pcdt(inputs):
    task = Tasks.token_classification
    model = 'damo/mgeo_geographic_elements_tagging_chinese_base'
    pipeline_ins = pipeline(task=task, model=model)
    res = pipeline_ins(input=inputs)
    pcdt = {'prov': '', 'city': '', 'district': '', 'town': ''}
    for r in res['output']:
        if r['type'] in pcdt:
            pcdt[r['type']] = r['span']
    return pcdt

df = pd.read_excel('test.xlsx')
total_pcdt = {'prov': [], 'city': [], 'district': [], 'town': []}
for line in df['address']:
    res = get_pcdt(line)
    for k in res:
        total_pcdt[k].append(res[k])
for k in total_pcdt:
    df[k] = total_pcdt[k] 
df.to_excel('test_out.xlsx', index=False, header=True)


