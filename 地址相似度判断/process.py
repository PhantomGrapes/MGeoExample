import numpy as np
import pandas as pd
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def get_sim(inputs):
    task = Tasks.sentence_similarity
    model = "damo/mgeo_geographic_entity_alignment_chinese_base"
    pipeline_ins = pipeline(task=task, model=model, model_revision='v1.1.2')

    res = pipeline_ins(input=inputs)
    sim_label = res["labels"][np.argmax(res["scores"])]
    sim_score = res["scores"][np.argmax(res["scores"])]
    return sim_label, sim_score


df = pd.read_excel("test.xlsx")
total_res = {"label": [], "score": []}
for address_1, address_2 in zip(df["address_1"], df["address_2"]):
    inputs = (address_1, address_2)
    sim_label, sim_score = get_sim(inputs)
    total_res["label"].append(sim_label)
    total_res["score"].append(sim_score)

for k in total_res:
    df[k] = total_res[k]
df.to_excel("test_out.xlsx", index=False, header=True)
