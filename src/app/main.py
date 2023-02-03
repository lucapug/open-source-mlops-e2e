import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import datetime
import json
import os
import warnings

import pandas as pd
from alibi_detect.saving import load_detector
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from jsonschema import validate
from sqlalchemy import create_engine
from utils.load_params import load_params

app = FastAPI()
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATABASE_URL = os.environ['DATABASE_URL'].replace('postgres://', 'postgresql://')

params = load_params(params_path='params.yaml')
model_path = params.train.model_path
feat_cols = params.base.feat_cols
min_batch_size = params.drift_detect.min_batch_size

cd = load_detector(Path('models')/'drift_detector')
model = load(filename=model_path)

schema = {
    "type" : "array",
    "items": 
        {
    "type": "object",
    "properties" : 
                {
        "CreditScore" : {"type" : "number"},
        "Age" : {"type" : "number"},
        "Tenure" : {"type" : "number"},
        "Balance" : {"type" : "number"},
        "NumOfProducts" : {"type" : "number"},
        "HasCrCard" : {"type" : "number"},
        "IsActiveMember" : {"type" : "number"},
        "EstimatedSalary" : {"type" : "number"}
                },
        }
}


@app.post("/predict")
async def predict(info : Request, background_tasks: BackgroundTasks):
    json_list = await info.json()
    try:
        background_tasks.add_task(collect_batch, json_list)
    except:
        warnings.warn("Unable to process batch data for drift detection")
    validate(instance=json_list, schema=schema)
    input_data = pd.DataFrame(json_list)
    probs = model.predict_proba(input_data)[:,0]
    probs = probs.tolist()
    return probs


@app.get("/drift_data")
async def get_drift_data(info : Request):
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        sql_query = "SELECT * FROM p_val_table"
        df_p_val = pd.read_sql(sql_query, con=conn)
    engine.dispose()
    parsed = json.loads(df_p_val.to_json())
    return json.dumps(parsed) 


def collect_batch(json_list, batch_size_thres = min_batch_size, batch = []):
    for req_json in json_list:
        batch.append(req_json)
    L = len(batch)
    if L >= batch_size_thres:
        X = pd.DataFrame.from_records(batch)
        preds = cd.predict(X)
        p_val = preds['data']['p_val']
        now = datetime.datetime.now()
        data = [[now] + p_val.tolist()]
        columns = ['time'] + feat_cols
        df_p_val = pd.DataFrame(data=data, columns=columns)
        print('Writing to database')
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            df_p_val.to_sql('p_val_table', con=conn, if_exists='append', index=False)
        engine.dispose()
        batch.clear()