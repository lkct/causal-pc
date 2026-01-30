import bnlearn
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference

from repro import set_determinism, seed
set_determinism(check_env=True)
seed(3407)


model = bnlearn.import_DAG('src/Experiments/bayesian_network/sachs.bif')
infer = CausalInference(model["model"])

print("Ground truth causal effect: ", infer.query(variables=["Raf"], do={
    "PKA": 0
}))

vars = bnlearn.topological_sort(model)

print("Topological order of variables: ", vars)


df = bnlearn.sampling(model, n=1000, methodtype='bayes')

# make variables binary
df = df.replace(2, 1)
df = df.replace(3, 1)
df = df.replace(4, 1)
df = df.replace(5, 1)
df = df.replace(6, 1)

print(df)




df.to_pickle("src/Experiments/data/sachs.pkl")

