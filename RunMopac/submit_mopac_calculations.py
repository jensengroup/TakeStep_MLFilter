import os
import copy
import pandas as pd
import numpy as np
import sys
from jinja2 import Template


nbatches = 30


submit_properties = {
    "mem": "2gb",
    "ncpus": "1",
    "partition": "chem",
    "pwd": os.getcwd(),
}

template = Template(open(sys.argv[1]).read())

def write_submit_script(subdf, batchname):
    subdf.to_csv(batchname + ".csv", index=False)

    sub_props = copy.deepcopy(submit_properties)
    sub_props.update({"jobname": batchname, "input_file": batchname + ".csv"})
    sbatch = template.render(**sub_props)
    with open(batchname + ".submit.sh", 'w') as s:
        s.write(sbatch)


fragement_batches = pd.read_csv(sys.argv[2])
for i, batch in enumerate(np.array_split(fragement_batches, nbatches)):
    name = f"fragment_batch_{i}"
    write_submit_script(batch, name)




