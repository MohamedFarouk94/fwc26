import pandas as pd
from xgboost import XGBRegressor
import joblib
from tqdm import tqdm

entries70 = pd.read_csv('data/entries70.csv')
to_keep = ['team_rate', 'opponent_rate', 'team_asi', 'opponent_dwi', 'importance', 'venue']

pxgbr = XGBRegressor(objective='count:poisson', eval_metric='poisson-nloglik',
                                   n_estimators=100, learning_rate=0.1)
pxgbr.fit(entries70[to_keep], entries70.goals)

pxgbr.save_model("models/model.json")
joblib.dump(pxgbr.get_params(), "models/params.joblib")
