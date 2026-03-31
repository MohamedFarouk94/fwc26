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

teams_data = {'team': [], 'rate': [], 'asi': [], 'dwi': []}
teams = entries70.team.unique()
for team in tqdm(teams, desc=f'FIFA World Cup {2026}'):
    team_rate = entries70[entries70.team == team].team_rate.to_list()[-1]
    team_asi = entries70[entries70.team == team].team_asi.to_list()[-1]
    team_dwi = entries70[entries70.opponent == team].opponent_dwi.to_list()[-1]
    teams_data['team'].append(team)
    teams_data['rate'].append(team_rate)
    teams_data['asi'].append(team_asi)
    teams_data['dwi'].append(team_dwi)

teams_data = pd.DataFrame(teams_data)
teams_data.to_csv('data/teams_data.csv', index=False)
