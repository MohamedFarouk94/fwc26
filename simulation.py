from time import time

import joblib
import numpy as np
import pandas as pd
import random
from scipy.stats import skellam
from collections import defaultdict
from numbers import Number
from plottable import Table, ColumnDefinition
from plottable.plots import image
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBRegressor
from pandas.api.types import is_float_dtype
import io
import base64


VENUE_ = None
RATINGS_ = None
ASI_ = None
DWI_ = None
GG_ = {}
PXGBR_ = None
RANDOM_ = None
FLAGS = {}

def venue_weight_26(team1, team2):
    if team1 == 'USA':
        return 1, 0
    if team2 == 'USA':
        return 0, 1
    if team1 == 'Mexico':
        return 1, 0
    if team2 == 'Mexico':
        return 0, 1
    if team1 == 'Canada':
        return 1, 0
    if team2 == 'Canada':
        return 0, 1
    return 0.5, 0.5

def update_rating(r, r_opp, i, g1, g2, c=0.1, rho=1):
    d = r - r_opp
    w = 0.5 if g1 == g2 else int(g1 > g2)
    denom = 10 ** (-d) + 1
    we = 1 / denom
    return r + rho * c * i * (w - we)

def update_asi(asi, dwi_opp, x, i, alpha=0.25, k=2, rho=1):
    sig = 1 / (k ** (-dwi_opp) + 1)
    mu = i * alpha * (1.5 - sig)
    return rho * mu * x + (1 - rho * mu) * asi

def update_dwi(dwi, asi_opp, y, i, alpha=0.25, k=2, rho=1):
    sig = 1 / (k ** (-asi_opp) + 1)
    mu = i * alpha * (1.5 - sig)
    return rho * mu * y + (1 - rho * mu) * dwi


class GoalGenerator:
    def __init__(self):
        self.rng = random.Random(int(time()))

    def fit(self, lambdas_in_pairs, scores):
        pass

    def predict(self, lambdas_in_pairs):
        raise NotImplementedError


class Poisson1X2Generator(GoalGenerator):
    def __init__(self, mu=0.5, n=10):
        self.original_rng = random.Random(int(time()))
        self.rng = np.random.default_rng(int(time()))
        self.shuffle_rng = random.Random(int(time()))
        self.mu = mu
        self.n = n

    def _predict_poisson(self, lambdas_in_pairs, oneXtwos):
        preds = []
        for (lam1, lam2), oneXtwo in zip(lambdas_in_pairs, oneXtwos):
            for _ in range(self.n):
                g1, g2 = int(self.rng.poisson(lam1)), int(self.rng.poisson(lam2))
                if self._get_1X2(g1, g2) == oneXtwo:
                    break
            preds.append((g1, g2))

        return preds 

    def _get_skellam(self, lambdas_in_pairs, eps=1e-15):
        skellam_probs = []
        for lam1, lam2 in lambdas_in_pairs:
            p_draw = skellam.pmf(0, lam1, lam2)
            p_home = 1 - skellam.cdf(0, lam1, lam2)
            p_away = skellam.cdf(-1, lam1, lam2)

            if p_home > p_away:
                p_away *= self.mu
                p_home = 1 - p_draw - p_away
            else:
                p_home *= self.mu
                p_away = 1 - p_draw - p_home
    
            p_home = np.clip(p_home, eps, 1 - eps)
            p_draw = np.clip(p_draw, eps, 1 - eps)
            p_away = np.clip(p_away, eps, 1 - eps)
            skellam_probs.append((p_home, p_draw, p_away))

        return skellam_probs

    def _get_1X2(self, g1, g2):
        return '1' if g1 > g2 else '2' if g1 < g2 else 'X'

    def _predict_1X2(self, skellam_probs):
        oneXtwos = []
        for p_home, p_draw, p_away in skellam_probs:
            outcome = self.original_rng.choices(['1', 'X', '2'], [p_home, p_draw, p_away])[0]
            oneXtwos.append(outcome)
        return oneXtwos

    def _fix_score(self, g1, g2, oneXtwo):
        queue = [(g1, g2)]
        visited = {(g1, g2)}
        while True:
            gi, gj = queue.pop(0)
            if self._get_1X2(gi, gj) == oneXtwo:
                return gi, gj

            children = [(gi + 1, gj), 
                        (gi, gj + 1),
                        (max(gi - 1, 0), gj),
                        (gi, max(gj - 1, 0))]

            children = [child for child in children if child not in visited]
            visited.update(children)
            self.shuffle_rng.shuffle(children)
            queue.extend(children)

    def predict(self, lambdas_in_pairs):
        skellam_probs = self._get_skellam(lambdas_in_pairs)
        oneXtwos = self._predict_1X2(skellam_probs)
        initial_scores = self._predict_poisson(lambdas_in_pairs, oneXtwos)
        return [self._fix_score(g1, g2, oneXtwo) for (g1, g2), oneXtwo in zip(initial_scores, oneXtwos)]


class ArithmeticDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.default_factory:
            self.default_factory = int

    def __add__(self, other):
        if not isinstance(other, dict):
            return NotImplemented

        result = ArithmeticDict(self.default_factory)
        result.update(self)
        for k, v in other.items():
            result[k] += v
        return result

    def __sub__(self, other):
        if not isinstance(other, dict):
            return NotImplemented

        result = ArithmeticDict(self.default_factory)
        result.update(self)
        for k, v in other.items():
            result[k] -= v
        return result

    def __iadd__(self, other):
        if not isinstance(other, dict):
            return NotImplemented

        for k, v in other.items():
            self[k] += v
                
        return self

    def __isub__(self, other):
        if not isinstance(other, dict):
            return NotImplemented

        for k, v in other.items():
            self[k] -= v
        return self

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented

        result = ArithmeticDict(self.default_factory)
        for k, v in self.items():
            result[k] = v * scalar
        return result

    def __truediv__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented

        result = ArithmeticDict(self.default_factory)
        for k, v in self.items():
            result[k] = v / scalar
        return result

    def __pow__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented

        result = ArithmeticDict(self.default_factory)
        for k, v in self.items():
            result[k] = v ** scalar
        return result

    def __imul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented

        for k in self:
            self[k] *= scalar
        return self

    def __itruediv__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented

        for k in self:
            self[k] /= scalar
        return self

    def __rmul__(self, scalar):
        return self * scalar
# alias
adict = ArithmeticDict


class Team:
    def __init__(self, name: str, flag: str='', continent: str='', nwcs: int=0):
        self.name_ = name
        self.flag_ = flag
        self.continent_ = continent
        self.nwcs_ = nwcs

    def __str__(self):
        return self.name_

    def __repr__(self):
        return self.name_

    def __invert__(self):
        side = Side()
        side.set_team(self)
        return side

    def __hash__(self):
        return hash(self.name_)

    def __eq__(self, other):
        if type(other) == str:
            return self.name_ == other
        return self.name_ == other.name_


class Side:
    def __init__(self):
        self._team = None
        self.coming_from_ = None

    def get_team(self):
        if self._team is None:
            raise ValueError('Cannot get team of an undecided side')

        return self._team

    def set_team(self, team):
        if self._team is not None:
            raise ValueError('Cannot set team to an already decided side')

        self._team = team

    def __mul__(self, other):
        return Match(side1=self, side2=other)


class Match:
    def __init__(self, side1: Side, side2: Side):
        self.side1_ = side1
        self.side2_ = side2
        self.must_win_ = False
        self.i_ = 1
        self.rho_ = 0
        self.gg_ = 'poisson'

        self.Winner = Side()
        self.Winner.coming_from_ = self

        self._ft_lambda_1 = -1
        self._ft_lambda_2 = -1

        self._et_lambda_1 = -1
        self._et_lambda_2 = -1

        self._ft_goals_1 = -1
        self._ft_goals_2 = -1

        self._et_goals_1 = -1
        self._et_goals_2 = -1

        self.ft_ = False
        self.et_ = False
        self.pk_ = -1

    def get_teams(self):
        return {self.side1_.get_team(), self.side2_.get_team()}

    def _create_entries(self):
        team1 = self.side1_.get_team() 
        team2 = self.side2_.get_team()
        venue1, venue2 = VENUE_(team1, team2)
        entry_1 = {'team_rate': RATINGS_[team1],
                   'opponent_rate': RATINGS_[team2],
                   'team_asi': ASI_[team1],
                   'opponent_dwi': DWI_[team2],
                   'importance': self.i_, 'venue': venue1}
        entry_2 = {'team_rate': RATINGS_[team2],
                   'opponent_rate': RATINGS_[team1],
                   'team_asi': ASI_[team2],
                   'opponent_dwi': DWI_[team1],
                   'importance': self.i_, 'venue': venue2}
        return pd.DataFrame([entry_1, entry_2])
    
    def _calc_ft_lambdas(self):
        self._ft_lambda_1, self._ft_lambda_2 = list(PXGBR_.predict(self._create_entries()))

    def _gen_ft_goals(self):
        self._ft_goals_1, self._ft_goals_2 = list(GG_[self.gg_].predict([(self._ft_lambda_1, self._ft_lambda_2)]))[0]

    def _calc_et_lambdas(self):
        self._et_lambda_1 = 0.25 * self._ft_lambda_1
        self._et_lambda_2 = 0.25 * self._ft_lambda_2

    def _gen_et_goals(self):
        self._et_goals_1, self._et_goals_2 = list(GG_[self.gg_].predict([(self._et_lambda_1, self._et_lambda_2)]))[0]

    def _get_1X2_probs(self, eps=1e-15):
        lam1 = self._ft_lambda_1
        lam2 = self._ft_lambda_2

        p_draw = skellam.pmf(0, lam1, lam2)
        p_home = 1 - skellam.cdf(0, lam1, lam2)
        p_away = skellam.cdf(-1, lam1, lam2)

        p_home = np.clip(p_home, eps, 1 - eps)
        p_draw = np.clip(p_draw, eps, 1 - eps)
        p_away = np.clip(p_away, eps, 1 - eps)

        return p_home, p_draw, p_away
        

    def _play_pk(self):
        self.pk_ = 1 if RANDOM_.random() > RANDOM_.random() else 2

    def _get_winner_side(self):
        if self.pk_ == 1:
            return self.side1_
        if self.pk_ == 2:
            return self.side2_
        a, b = self.get_score()
        if a == b:
            raise ValueError('Cannot have a winner when the match ends with draw and no penalty shootouts are played')
        return self.side1_ if a > b else self.side2_

    def _set_winner(self):
        self.Winner.set_team(self._get_winner_side().get_team())

    def get_score(self):
        if self.et_:
            return (self._ft_goals_1 + self._et_goals_1, self._ft_goals_2 + self._et_goals_2)

        if self.ft_:
            return (self._ft_goals_1, self._ft_goals_2)

        raise ValueError('Cannot get score of an non-played match')

    def is_draw(self):
        a, b = self.get_score()
        return a == b

    def _play_ft(self):
        self._calc_ft_lambdas()
        self._gen_ft_goals()
        self.ft_ = True

    def _play_et(self):
        self._calc_et_lambdas()
        self._gen_et_goals()
        self.et_ = True

    def _update_dynamic(self):
        a, b = self.get_score()
        team1 = self.side1_.get_team() 
        team2 = self.side2_.get_team()
        r1 = RATINGS_[team1]
        r2 = RATINGS_[team2]
        asi1 = ASI_[team1]
        asi2 = ASI_[team2]
        dwi1 = DWI_[team1]
        dwi2 = DWI_[team2]
        r1_new = update_rating(r1, r2, 1, a, b, rho=self.rho_)
        r2_new = update_rating(r2, r1, 1, b, a, rho=self.rho_)
        asi1_new = update_asi(asi1, dwi2, a, 1, rho=self.rho_)
        asi2_new = update_asi(asi2, dwi1, b, 1, rho=self.rho_)
        dwi1_new = update_dwi(dwi1, asi2, b, 1, rho=self.rho_)
        dwi2_new = update_dwi(dwi2, asi1, a, 1, rho=self.rho_)
        RATINGS_[team1] = r1_new
        RATINGS_[team2] = r2_new
        ASI_[team1] = asi1_new
        ASI_[team2] = asi2_new
        DWI_[team1] = dwi1_new
        DWI_[team2] = dwi2_new

    def scoreboard(self):
        team1 = str(self.side1_.get_team())
        team2 = str(self.side2_.get_team())
        if self.must_win_ or not self.is_draw():
            if self.Winner.get_team() == team1:
                team1 += '*'
            else:
                team2 += '*'

        a, b = self.get_score()
        res = f'{team1} {a}-{b} {team2}'
        if self.et_:
            res += ' (a.e.t)'

        print(res)
                
        
    def play(self):
        self._play_ft()

        if self.must_win_ and self.is_draw():
            self._play_et()

        if self.must_win_ and self.is_draw():
            self._play_pk()

        if self.must_win_ or not self.is_draw():
            self._set_winner()
        
        self._update_dynamic()


class Group:
    def __init__(self, side1: Side, side2: Side, side3: Side, side4: Side):
        self.sides_ = [side1, side2, side3, side4]
        self.sides_sorted_ = self.sides_.copy()

        self.matches_ = [side1 * side2, side3 * side4, 
                         side1 * side3, side2 * side4,
                         side1 * side4, side2 * side3]

        self.name_ = ''
        self._gg_ = 'poisson'
        self._rho_ = 0
        self.played_ = False

        for match in self.matches_:
            match.gg_ = self.gg_
            match.rho_ = self.rho_

        self.table_ = adict()
        self.table_[side1.get_team()] = adict(int, {'P': 0, 'W': 0, 'D': 0, 'GS': 0, 'GC': 0, 'Pts': 0})
        self.table_[side2.get_team()] = adict(int, {'P': 0, 'W': 0, 'D': 0, 'GS': 0, 'GC': 0, 'Pts': 0})
        self.table_[side3.get_team()] = adict(int, {'P': 0, 'W': 0, 'D': 0, 'GS': 0, 'GC': 0, 'Pts': 0})
        self.table_[side4.get_team()] = adict(int, {'P': 0, 'W': 0, 'D': 0, 'GS': 0, 'GC': 0, 'Pts': 0})

        self.Winner = Side()
        self.Runnerup = Side()
        self.Third = Side()
        self.third_uncertain_ = Side()

        self.Winner.coming_from_ = self
        self.Runnerup.coming_from_ = self
        self.Third.coming_from_ = self

        self.third_qualified_ = False

    @property
    def gg_(self):
        return self._gg_

    @gg_.setter
    def gg_(self, value):
        for match in self.matches_:
            match.gg_ = value
        self._gg_ = value

    @property
    def rho_(self):
        return self._rho_

    @rho_.setter
    def rho_(self, value):
        for match in self.matches_:
            match.rho_ = value
        self._rho_ = value

    def get_teams(self):
        return {side.get_team() for side in self.sides_}

    def _order_key(self, side):
        row = self.table_[side.get_team()]
        return (row['Pts'],
                row['GS'] - row['GC'],
                row['GS'],
                RANDOM_.random())

    def update_table(self, team, a, b): # scored a and conceded b
        pts, w, d = 0, 0, 0
        if a > b:
            pts = 3
            w = 1
        elif a == b:
            pts = 1
            d = 1
            
        self.table_[team]['P'] += 1
        self.table_[team]['W'] += w
        self.table_[team]['D'] += d
        self.table_[team]['Pts'] += pts
        self.table_[team]['GS'] += a
        self.table_[team]['GC'] += b

    def reorder(self):
        self.sides_sorted_.sort(key=lambda x: self._order_key(x), reverse=True)

    def qualify_third(self):
        self.third_qualified_ = True
        self.Third.set_team(self.third_uncertain_.get_team())

    def finish(self):
        self.reorder()
        self.played_ = True
        self.Winner.set_team(self.sides_sorted_[0].get_team())
        self.Runnerup.set_team(self.sides_sorted_[1].get_team())
        self.third_uncertain_.set_team(self.sides_sorted_[2].get_team())

    def play(self):
        for match in self.matches_:
            match.play()
            a, b = match.get_score()
            team1 = match.side1_.get_team()
            team2 = match.side2_.get_team()
            self.update_table(team1, a, b)
            self.update_table(team2, b, a)
        self.finish()

    def get_df(self):
        df = []
        for i, side in enumerate(self.sides_sorted_):
            Pos = i + 1
            team = side.get_team()
            row = self.table_[team]
            P = row['P']
            W = row['W']
            D = row['D']
            L = P - W - D
            GS = row['GS']
            GC = row['GC']
            GD = GS - GC
            Pts = row['Pts']
            df.append({'Pos': Pos, 'Team': team,
                       'P': P, 'W': W, 'D': D, 'L': L,
                       'GS': GS, 'GC': GC, 'GD': GD,
                       'Pts': Pts})
        return pd.DataFrame(df)

    def plot(self, ax):
        df = self.get_df()
        df = df.set_index('Pos')
        df.Team = df.Team.apply(lambda x: FLAGS[x])

        for col in df.columns:
            if is_float_dtype(df[col]):
                df[col] = df[col].round(2)

        col_defs = [
                ColumnDefinition(name="Pos", textprops={"ha": "center"}),
                ColumnDefinition(name="Team", plot_fn=image, textprops={"ha": "center"}),
                ColumnDefinition(name="P", textprops={"ha": "center"}),
                ColumnDefinition(name="W", textprops={"ha": "center"}),
                ColumnDefinition(name="D", textprops={"ha": "center"}),
                ColumnDefinition(name="L", textprops={"ha": "center"}),
                ColumnDefinition(name="GS", textprops={"ha": "center"}),
                ColumnDefinition(name="GC", textprops={"ha": "center"}),
                ColumnDefinition(name="GD", textprops={"ha": "center"}),
                ColumnDefinition(name="Pts", textprops={"weight": "bold", "ha": "center"})
        ]

        tab = Table(df, column_definitions=col_defs, ax=ax)
        tab.rows[0].set_facecolor('lightgreen')
        tab.rows[1].set_facecolor('lightgreen')
        if self.third_qualified_:
            tab.rows[2].set_facecolor('lightgreen')

        ax.set_title(f'Group {self.name_}', loc='left', pad=10)
        # plt.show()
# alias
G = Group


class GroupStage:
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def __init__(self, best_thirds=0, third_combinations=None):
        self._groups_ = []
        self._gg_ = 'poisson'
        self._rho_ = 0
        self.best_thirds_ = best_thirds
        self.BestThirds = {}
        self.third_combinations_ = third_combinations

    @property
    def groups_(self):
        return self._groups_

    @groups_.setter
    def groups_(self, groups):
        for ch, group in zip(self.ALPHABET, groups):
            setattr(self, ch, group)
            group.name_ = ch
            group.gg_ = self.gg_
            group.rho_ = self.rho_
            self.BestThirds[f'toFace1{ch}'] = Side()
        self._groups_ = groups

    @property
    def matches_(self):
        matches = []
        for group in self.groups_:
            matches.extend(group.matches_)
        return matches      

    @property
    def gg_(self):
        return self._gg_

    @gg_.setter
    def gg_(self, value):
        for group in self.groups_:
            group.gg_ = value
        self._gg_ = value

    @property
    def rho_(self):
        return self._rho_

    @rho_.setter
    def rho_(self, value):
        for group in self.groups_:
            group.rho_ = value
        self._rho_ = value

    def get_teams(self):
        teams = set()
        for group in self.groups_:
            teams.update(group.get_teams())
        return teams

    def _order_key(self, row):
        return (row['Pts'],
                row['GS'] - row['GC'],
                row['GS'],
                RANDOM_.random())  
    
    def _decide_best_thirds(self):
        if not self.best_thirds_:
            return
            
        thirds = []
        for group in self.groups_:
            third = group.third_uncertain_
            row = group.table_[third.get_team()]
            thirds.append((group, row))
        thirds.sort(key=lambda x: self._order_key(x[1]), reverse=True)

        for i in range(self.best_thirds_):
            thirds[i][0].qualify_third()

    def _decide_best_thirds_opponents(self):
        if not self.best_thirds_:
            return
        qualified_thirds = [g.name_ for g in self.groups_ if g.third_qualified_]
        qualified_thirds = ''.join(qualified_thirds)
        opp_dict = self.third_combinations_[self.third_combinations_.qualified_thirds == qualified_thirds].iloc[0].to_dict()
        for group in self.groups_:
            ch_winner = group.name_
            if f'1{ch_winner}' not in opp_dict.keys():
                continue
            ch_third = opp_dict[f'1{ch_winner}'][-1]
            self.BestThirds[f'toFace1{ch_winner}'].set_team(getattr(self, ch_third).Third.get_team())
            self.BestThirds[f'toFace1{ch_winner}'].coming_from_ = getattr(self, ch_third)

    def play(self):
        for group in self.groups_:
            group.play()
        self._decide_best_thirds()
        self._decide_best_thirds_opponents()

    def plot_tables(self, buffer=False):
        fig, axs = plt.subplots(4, 3, figsize=(20, 20))
        axs = axs.flatten()
        for i, (group, ax) in enumerate(zip(self.groups_, axs)):
            group.plot(ax)
        [fig.delaxes(ax) for ax in axs[i + 1:]]

        if buffer:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            return buffer

        plt.show()
# alias
GS = GroupStage


class KnockoutRound:
    def __init__(self):
        self._matches_ = []
        self._gg_ = 'poisson'
        self._rho_ = 0

    @property
    def matches_(self):
        return self._matches_

    @matches_.setter
    def matches_(self, matches):
        for i, match in enumerate(matches):
            setattr(self, f'M{i + 1}', match)
            match.must_win_ = True
            match.rho_ = self.rho_
            match.gg_ = self.gg_
        self._matches_ = matches

    @property
    def gg_(self):
        return self._gg_

    @gg_.setter
    def gg_(self, value):
        for match in self.matches_:
            match.gg_ = value
        self._gg_ = value

    @property
    def rho_(self):
        return self._rho_

    @rho_.setter
    def rho_(self, value):
        for match in self.matches_:
            match.rho_ = value
        self._rho_ = value

    def get_teams(self):
        teams = set()
        for match in self.matches_:
            teams.update(match.get_teams())
        return teams
    
    def play(self):
        for match in self.matches_:
            match.play()

    def scoreboard(self):
        for match in self.matches_:
            match.scoreboard()

    def key(self):
        # Tuple of teams sorted alphabetically
        return tuple(sorted([str(team) for team in self.get_teams()]))
# alias
KO = KnockoutRound


class Tournament:
    def __init__(self, group_stages=[], knockout_rounds=[], best_thirds=[], third_combinations=[],
                 rho=0, gg='poisson'):
        if not best_thirds:
            best_thirds = [0] * len(group_stages)
        if not third_combinations:
            third_combinations = [None] * len(group_stages)
        self.group_stages_ = group_stages
        self.knockout_rounds_ = knockout_rounds
        self.rho_ = rho
        self.gg_ = gg

        for gs, n, df in zip(group_stages, best_thirds, third_combinations):
            setattr(self, gs, GS(best_thirds=n, third_combinations=df))
            gs_ = getattr(self, gs)
            gs_.rho_ = rho
            gs_.gg_ = gg

        for ko in knockout_rounds:
            setattr(self, ko, KO())
            ko_ = getattr(self, ko)
            ko_.rho_ = rho
            ko_.gg_ = gg

    def __setattr__(self, name, value):
        if name.endswith('Groups'):
            gs = name.split('Groups')[0]
            getattr(self, gs).groups_ = value

        if name.endswith('Matches'):
            ko = name.split('Matches')[0]
            getattr(self, ko).matches_ = value

        super().__setattr__(name, value)

    @property
    def matches_(self):
        matches = []
        for rnd in self.group_stages_ + self.knockout_rounds_:
            matches.extend(getattr(self, rnd).matches_)
        return matches

    def get_teams(self):
        return getattr(self, self.group_stages_[0]).get_teams()

    def play(self):
        for gs in self.group_stages_:
            getattr(self, gs).play()
        for ko in self.knockout_rounds_:
            getattr(self, ko).play()

    def plot(self):
        for gs in self.group_stages_:
            getattr(self, gs).plot_tables()
        for ko in self.knockout_rounds_:
           getattr(self, ko).scoreboard()
           print('-' * 30)

    def _collect_bracket_matches(self):
        """DFS (pre-order, side1 before side2) to collect knockout
        matches in top-to-bottom display order, grouped by round depth.
        Depth 0 = earliest knockout round (e.g. R32), max depth = Final."""
        from collections import defaultdict
 
        if not self.knockout_rounds_:
            return {}
 
        final_ko  = getattr(self, self.knockout_rounds_[-1])
        final_match = final_ko.matches_[0]
        max_depth = len(self.knockout_rounds_) - 1
 
        rounds  = defaultdict(list)
        visited = set()
 
        def dfs(src, depth):
            if not isinstance(src, Match):
                return
            mid = id(src)
            if mid in visited:
                return
            visited.add(mid)
            rounds[depth].append(src)
            dfs(src.side1_.coming_from_, depth - 1)
            dfs(src.side2_.coming_from_, depth - 1)
 
        dfs(final_match, max_depth)
        return dict(rounds)
 
    def _match_to_dict(self, match):
        """Return a plain dict with display data for one match."""
        s1, s2 = match.get_score()
        t1      = str(match.side1_.get_team())
        t2      = str(match.side2_.get_team())
        winner  = str(match.Winner.get_team())
        suffix  = ''
        if match.pk_ > 0:
            suffix = 'a.p.'
        elif match.et_:
            suffix = 'a.e.t.'
        return dict(team1=t1, team2=t2, score1=s1, score2=s2,
                    winner=winner, suffix=suffix)
 
    def _match_card_html(self, m):
        """Inline-styled HTML card for one match (dark theme)."""
        t1w = m['winner'] == m['team1']
        t2w = m['winner'] == m['team2']
        # winner row: dark-green tint + bright text; loser: dim bg + muted text
        s1 = ('background:#0d2e1a;color:#00e676;font-weight:700;' if t1w else
              'background:#0c1a22;color:#7a9aaa;font-weight:400;')
        s2 = ('background:#0d2e1a;color:#00e676;font-weight:700;' if t2w else
              'background:#0c1a22;color:#7a9aaa;font-weight:400;')
        sfx = ''
        if m['suffix']:
            sfx = (f'<div style="font-size:10px;color:#4a6a7a;text-align:center;'
                   f'padding:2px 6px;background:#060d13;">{m["suffix"]}</div>')
        base = ('border:1px solid rgba(255,255,255,.1);border-radius:5px;'
                'overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.4);'
                'min-width:155px;')
        row  = ('display:flex;justify-content:space-between;align-items:center;'
                'padding:5px 9px;border-bottom:1px solid rgba(255,255,255,.07);')
        last = ('display:flex;justify-content:space-between;align-items:center;'
                'padding:5px 9px;')
        nm   = 'flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
        sc   = 'font-weight:700;min-width:18px;text-align:right;margin-left:6px;'
        return (
            f'<div style="{base}">'
            f'<div style="{row}{s1}">'
            f'<span style="{nm}">{m["team1"]}</span>'
            f'<span style="{sc}">{m["score1"]}</span>'
            f'</div>'
            f'<div style="{last}{s2}">'
            f'<span style="{nm}">{m["team2"]}</span>'
            f'<span style="{sc}">{m["score2"]}</span>'
            f'</div>'
            f'{sfx}'
            f'</div>'
        )
 
    def _render_bracket_html(self, all_rounds, round_names):
        """Build the full bracket HTML from a list-of-rounds (each round is a
        list of match-dicts in top-to-bottom display order)."""
        n_rounds = len(all_rounds)
        first_n  = len(all_rounds[0])
        SLOT_H   = 74          # px per match slot in the first (deepest) round
        TOTAL_H  = first_n * SLOT_H
        HDR_H    = 44          # px for the round header row
        C        = '#2a5060'   # connector line colour
 
        wrap = (
            'display:flex;font-family:"Segoe UI",Arial,sans-serif;font-size:12px;'
            'padding:20px;background:#060d13;overflow-x:auto;'
            'min-width:max-content;border-radius:10px;'
        )
        parts = [f'<div style="{wrap}">']
 
        for ri, (name, matches) in enumerate(zip(round_names, all_rounds)):
            n       = len(matches)
            slot_h  = TOTAL_H // n
            is_last = ri == n_rounds - 1
 
            # ── round column ──────────────────────────────────────
            parts.append('<div style="display:flex;flex-direction:column;">')
 
            # header badge
            parts.append(
                f'<div style="height:{HDR_H}px;display:flex;align-items:center;'
                f'justify-content:center;padding:0 8px;">'
                f'<div style="font-weight:700;padding:5px 13px;'
                f'background:#1e3a5f;color:#fff;border-radius:5px;'
                f'font-size:11px;white-space:nowrap;letter-spacing:.6px;">'
                f'{name.upper()}</div></div>'
            )
 
            # matches area
            parts.append(
                f'<div style="height:{TOTAL_H}px;display:flex;'
                f'flex-direction:column;min-width:182px;">'
            )
 
            for mi, m in enumerate(matches):
                is_top = (mi % 2 == 0)
 
                parts.append(
                    f'<div style="height:{slot_h}px;display:flex;'
                    f'align-items:center;padding:2px 6px;">'
                )
                parts.append('<div style="flex:1;min-width:0;">')
                parts.append(self._match_card_html(m))
                parts.append('</div>')
 
                # right connector arm (not on the Final column)
                if not is_last:
                    if is_top:
                        # ─┐  (horizontal at match centre, then down)
                        parts.append(
                            f'<div style="width:18px;height:100%;'
                            f'display:flex;flex-direction:column;">'
                            f'<div style="flex:1;"></div>'
                            f'<div style="flex:1;border-top:2px solid {C};'
                            f'border-right:2px solid {C};"></div>'
                            f'</div>'
                        )
                    else:
                        # ─┘  (up from bottom, then horizontal at match centre)
                        parts.append(
                            f'<div style="width:18px;height:100%;'
                            f'display:flex;flex-direction:column;">'
                            f'<div style="flex:1;border-bottom:2px solid {C};'
                            f'border-right:2px solid {C};"></div>'
                            f'<div style="flex:1;"></div>'
                            f'</div>'
                        )
 
                parts.append('</div>')   # close match slot
 
            parts.append('</div>')       # close matches area
            parts.append('</div>')       # close round column
 
            # ── inter-round horizontal connector ──────────────────
            if not is_last:
                next_n      = len(all_rounds[ri + 1])
                next_slot_h = TOTAL_H // next_n
 
                parts.append('<div style="display:flex;flex-direction:column;">')
                parts.append(f'<div style="height:{HDR_H}px;"></div>')
                parts.append(
                    f'<div style="height:{TOTAL_H}px;width:14px;'
                    f'display:flex;flex-direction:column;">'
                )
                for _ in range(next_n):
                    parts.append(
                        f'<div style="height:{next_slot_h}px;'
                        f'display:flex;align-items:center;">'
                        f'<div style="width:100%;height:2px;'
                        f'background:{C};"></div></div>'
                    )
                parts.append('</div>')
                parts.append('</div>')  # inter-round connector col
 
        parts.append('</div>')  # outer wrapper
        return '\n'.join(parts)

    def plot_bracket(self):
        """Return an HTML string containing the full knockout bracket."""
        rounds = self._collect_bracket_matches()
        if not rounds:
            return '<p>No bracket data available.</p>'
 
        sorted_keys = sorted(rounds.keys())
        n = len(sorted_keys)
        NAMES = ['Round of 32', 'Round of 16', 'Quarter-finals',
                 'Semi-finals', 'Final']
        round_names = NAMES[max(0, 5 - n):][:n]
 
        all_rounds = [
            [self._match_to_dict(m) for m in rounds[k]]
            for k in sorted_keys
        ]
        return self._render_bracket_html(all_rounds, round_names)
 
    def get_plot(self):
        result = {}
        buffer = self.GS.plot_tables(buffer=True)
        result['group_stage_image'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
 
        result['bracket_html'] = self.plot_bracket()
        return result


class TournamentReport(adict):  # adict is ArithmeticDict, created in utils (Section 1.3)
    _ATTRIBUTES_INT = ['avg_goals_per_match', 'draw_ratio', 'oo_ratio',
                       'heavy_score_ratio', 'goal_party_ratio']
    _ATTRIBUTES_ADICT = ['rank_table', 'gs_table', 'sf_record', 'f_record', 'champion_record']

    def __init__(self, wc=None):
        super().__init__()
        for attr in self._ATTRIBUTES_INT:
            self[attr] = 0
        for attr in self._ATTRIBUTES_ADICT:
            self[attr] = adict(int)
        self['gs_table'] = adict(lambda: adict(lambda: adict()))
        self.wc = wc
        self.wc_rounds = None
        if self.wc is not None:
            self.wc_rounds = [getattr(self.wc, rnd) for rnd in self.wc.group_stages_ + self.wc.knockout_rounds_]

    def update_rank_table(self):
        for team in self.wc.GS.get_teams():
            self['rank_table'][team] = 0 # Just to be there, it's 0 by default anyway
        for rnd in self.wc_rounds[1:]:
            for team in rnd.get_teams():
                self['rank_table'][team] += 1
        self['rank_table'][self.wc.Champion.get_team()] += 1

    def update_data_per_match(self, heavy_score_threshold=3, goal_party_threshold=5):
        n = 0
        for rnd in self.wc_rounds:
            for match in rnd.matches_:
                n += 1
                a, b = match.get_score()
                self['avg_goals_per_match'] += a + b
                self['draw_ratio'] += (a == b)
                self['oo_ratio'] += (a + b == 0)
                self['goal_party_ratio'] += (a + b >= goal_party_threshold)
                self['heavy_score_ratio'] += (abs(a - b) >= heavy_score_threshold)

        self['avg_goals_per_match'] /= n
        self['draw_ratio'] /= n
        self['oo_ratio'] /= n
        self['goal_party_ratio'] /= n
        self['heavy_score_ratio'] /= n

    def update_gs_table(self):
        self['gs_table'] = adict(int, {group.name_: group.table_ for group in self.wc.GS.groups_})

    def update_records(self):
        self['sf_record'][self.wc.SF.key()] += 1
        self['f_record'][self.wc.F.key()] += 1
        self['champion_record'][self.wc.Champion.get_team()] += 1

    def build(self):
        self.update_rank_table()
        self.update_data_per_match()
        self.update_gs_table()
        self.update_records()

    def rbmse(self, rbe):
        rbse = (self['rank_table'] - rbe) ** 2
        return sum(rbse.values()) / len(rbse.values())
# alias
TR = TournamentReport


class SimulationManager:
    def __init__(self, n=1000, rho=0.5, mu=0.5, year=2026,
                 wc_builder=lambda: None):
        global GG_, to_keep, RANDOM_, PXGBR_, VENUE_
        self.year_ = year
        self.n_ = n
        self.rho_ = rho
        self.mu_ = mu
        self.build = wc_builder
        self.trs_ = []
        PXGBR_ = XGBRegressor(**joblib.load("models/params.joblib"))
        PXGBR_.load_model("models/model.json")
        GG_['1x2'] = Poisson1X2Generator(mu=self.mu_)
        RANDOM_ = random.Random(int(time()))
        VENUE_ = venue_weight_26

    def run(self, verbose=0):
        global ASI_, DWI_, RATINGS_
        teams_data = pd.read_csv('data/teams_data.csv')
        tr_sum = TR()
        iterable = tqdm(range(self.n_), desc='Running Simulation') if verbose else range(self.n_)
        for _ in iterable:
            RATINGS_ = {team: rate for team, rate in zip(teams_data.team, teams_data.rate)}
            ASI_ = {team: asi for team, asi in zip(teams_data.team, teams_data.asi)}
            DWI_ ={team: dwi for team, dwi in zip(teams_data.team, teams_data.dwi)}
            wc = self.build(rho=self.rho_)
            wc.play()
            tr = TR(wc=wc)
            tr.build()
            tr_sum += tr
            self.trs_.append(tr)
        self.report_ = tr_sum
        self.report_ /= self.n_
# alias
SM = SimulationManager

Albania = Team(name='Albania', flag='data/flags/AL.png', continent='UEFA', nwcs=0); FLAGS[Albania] = Albania.flag_
Algeria = Team(name='Algeria', flag='data/flags/DZ.png', continent='CAF', nwcs=0); FLAGS[Algeria] = Algeria.flag_
Argentina = Team(name='Argentina', flag='data/flags/AR.png', continent='CONMEBOL', nwcs=3); FLAGS[Argentina] = Argentina.flag_
Australia = Team(name='Australia', flag='data/flags/AU.png', continent='AFC', nwcs=0); FLAGS[Australia] = Australia.flag_
Austria = Team(name='Austria', flag='data/flags/AT.png', continent='UEFA', nwcs=0); FLAGS[Austria] = Austria.flag_
Belgium = Team(name='Belgium', flag='data/flags/BE.png', continent='UEFA', nwcs=0); FLAGS[Belgium] = Belgium.flag_
Bolivia = Team(name='Bolivia', flag='data/flags/BO.png', continent='CONMEBOL', nwcs=0); FLAGS[Bolivia] = Bolivia.flag_
Bosnia = Team(name='Bosnia', flag='data/flags/BA.png', continent='UEFA', nwcs=0); FLAGS[Bosnia] = Bosnia.flag_
Brazil = Team(name='Brazil', flag='data/flags/BR.png', continent='CONMEBOL', nwcs=5); FLAGS[Brazil] = Brazil.flag_
CIV = Team(name='CIV', flag='data/flags/CI.png', continent='CAF', nwcs=0); FLAGS[CIV] = CIV.flag_
CPV = Team(name='CPV', flag='data/flags/CV.png', continent='CAF', nwcs=0); FLAGS[CPV] = CPV.flag_
CR = Team(name='CR', flag='data/flags/CR.png', continent='CONCACAF', nwcs=0); FLAGS[CR] = CR.flag_
Cameroon = Team(name='Cameroon', flag='data/flags/CM.png', continent='CAF', nwcs=0); FLAGS[Cameroon] = Cameroon.flag_
Canada = Team(name='Canada', flag='data/flags/CA.png', continent='CONCACAF', nwcs=0); FLAGS[Canada] = Canada.flag_
Chile = Team(name='Chile', flag='data/flags/CL.png', continent='CONMEBOL', nwcs=0); FLAGS[Chile] = Chile.flag_
Colombia = Team(name='Colombia', flag='data/flags/CO.png', continent='CONMEBOL', nwcs=0); FLAGS[Colombia] = Colombia.flag_
Croatia = Team(name='Croatia', flag='data/flags/HR.png', continent='UEFA', nwcs=0); FLAGS[Croatia] = Croatia.flag_
Curacao = Team(name='Curacao', flag='data/flags/CUR.png', continent='CONCACAF', nwcs=0); FLAGS[Curacao] = Curacao.flag_
Czechia = Team(name='Czechia', flag='data/flags/CZ.png', continent='UEFA', nwcs=0); FLAGS[Czechia] = Czechia.flag_
DRC = Team(name='DRC', flag='data/flags/CD.png', continent='CAF', nwcs=0); FLAGS[DRC] = DRC.flag_
Denmark = Team(name='Denmark', flag='data/flags/DK.png', continent='UEFA', nwcs=0); FLAGS[Denmark] = Denmark.flag_
Ecuador = Team(name='Ecuador', flag='data/flags/EC.png', continent='CONMEBOL', nwcs=0); FLAGS[Ecuador] = Ecuador.flag_
Egypt = Team(name='Egypt', flag='data/flags/EG.png', continent='CAF', nwcs=0); FLAGS[Egypt] = Egypt.flag_
England = Team(name='England', flag='data/flags/ENG.png', continent='UEFA', nwcs=1); FLAGS[England] = England.flag_
France = Team(name='France', flag='data/flags/FR.png', continent='UEFA', nwcs=2); FLAGS[France] = France.flag_
Germany = Team(name='Germany', flag='data/flags/DE.png', continent='UEFA', nwcs=4); FLAGS[Germany] = Germany.flag_
Ghana = Team(name='Ghana', flag='data/flags/GH.png', continent='CAF', nwcs=0); FLAGS[Ghana] = Ghana.flag_
Greece = Team(name='Greece', flag='data/flags/GR.png', continent='UEFA', nwcs=0); FLAGS[Greece] = Greece.flag_
Haiti = Team(name='Haiti', flag='data/flags/HT.png', continent='CONCACAF', nwcs=0); FLAGS[Haiti] = Haiti.flag_
Honduras = Team(name='Honduras', flag='data/flags/HN.png', continent='CONCACAF', nwcs=0); FLAGS[Honduras] = Honduras.flag_
Iceland = Team(name='Iceland', flag='data/flags/IS.png', continent='UEFA', nwcs=0); FLAGS[Iceland] = Iceland.flag_
Iran = Team(name='Iran', flag='data/flags/IR.png', continent='AFC', nwcs=0); FLAGS[Iran] = Iran.flag_
Iraq = Team(name='Iraq', flag='data/flags/IQ.png', continent='AFC', nwcs=0); FLAGS[Iraq] = Iraq.flag_
Ireland = Team(name='Ireland', flag='data/flags/IE.png', continent='UEFA', nwcs=0); FLAGS[Ireland] = Ireland.flag_
Italy = Team(name='Italy', flag='data/flags/IT.png', continent='UEFA', nwcs=4); FLAGS[Italy] = Italy.flag_
Jamaica = Team(name='Jamaica', flag='data/flags/JM.png', continent='CONCACAF', nwcs=0); FLAGS[Jamaica] = Jamaica.flag_
Japan = Team(name='Japan', flag='data/flags/JP.png', continent='AFC', nwcs=0); FLAGS[Japan] = Japan.flag_
Jordan = Team(name='Jordan', flag='data/flags/JO.png', continent='AFC', nwcs=0); FLAGS[Jordan] = Jordan.flag_
KSA = Team(name='KSA', flag='data/flags/SA.png', continent='AFC', nwcs=0); FLAGS[KSA] = KSA.flag_
Korea = Team(name='Korea', flag='data/flags/KR.png', continent='AFC', nwcs=0); FLAGS[Korea] = Korea.flag_
Kosovo = Team(name='Kosovo', flag='data/flags/RS.png', continent='UEFA', nwcs=0); FLAGS[Kosovo] = Kosovo.flag_
Macedonia = Team(name='Macedonia', flag='data/flags/MK.png', continent='UEFA', nwcs=0); FLAGS[Macedonia] = Macedonia.flag_
Mexico = Team(name='Mexico', flag='data/flags/MX.png', continent='CONCACAF', nwcs=0); FLAGS[Mexico] = Mexico.flag_
Morocco = Team(name='Morocco', flag='data/flags/MA.png', continent='CAF', nwcs=0); FLAGS[Morocco] = Morocco.flag_
NC = Team(name='NC', flag='data/flags/NC.png', continent='OFC', nwcs=0); FLAGS[NC] = NC.flag_
NIR = Team(name='NIR', flag='data/flags/NIR.png', continent='UEFA', nwcs=0); FLAGS[NIR] = NIR.flag_
NK = Team(name='NK', flag='data/flags/KP.png', continent='AFC', nwcs=0); FLAGS[NK] = NK.flag_
NZ = Team(name='NZ', flag='data/flags/NZ.png', continent='OFC', nwcs=0); FLAGS[NZ] = NZ.flag_
Netherlands = Team(name='Netherlands', flag='data/flags/NL.png', continent='UEFA', nwcs=0); FLAGS[Netherlands] = Netherlands.flag_
Nigeria = Team(name='Nigeria', flag='data/flags/NG.png', continent='CAF', nwcs=0); FLAGS[Nigeria] = Nigeria.flag_
Norway = Team(name='Norway', flag='data/flags/NO.png', continent='UEFA', nwcs=0); FLAGS[Norway] = Norway.flag_
Panama = Team(name='Panama', flag='data/flags/PA.png', continent='CONCACAF', nwcs=0); FLAGS[Panama] = Panama.flag_
Paraguay = Team(name='Paraguay', flag='data/flags/PY.png', continent='CONMEBOL', nwcs=0); FLAGS[Paraguay] = Paraguay.flag_
Peru = Team(name='Peru', flag='data/flags/PE.png', continent='CONMEBOL', nwcs=0); FLAGS[Peru] = Peru.flag_
Poland = Team(name='Poland', flag='data/flags/PL.png', continent='UEFA', nwcs=0); FLAGS[Poland] = Poland.flag_
Portugal = Team(name='Portugal', flag='data/flags/PT.png', continent='UEFA', nwcs=0); FLAGS[Portugal] = Portugal.flag_
Qatar = Team(name='Qatar', flag='data/flags/QA.png', continent='AFC', nwcs=0); FLAGS[Qatar] = Qatar.flag_
RSA = Team(name='RSA', flag='data/flags/ZA.png', continent='CAF', nwcs=0); FLAGS[RSA] = RSA.flag_
Romania = Team(name='Romania', flag='data/flags/RO.png', continent='UEFA', nwcs=0); FLAGS[Romania] = Romania.flag_
Russia = Team(name='Russia', flag='data/flags/RU.png', continent='UEFA', nwcs=0); FLAGS[Russia] = Russia.flag_
Scotland = Team(name='Scotland', flag='data/flags/SCO.png', continent='UEFA', nwcs=0); FLAGS[Scotland] = Scotland.flag_
Senegal = Team(name='Senegal', flag='data/flags/SN.png', continent='CAF', nwcs=0); FLAGS[Senegal] = Senegal.flag_
Serbia = Team(name='Serbia', flag='data/flags/RS.png', continent='UEFA', nwcs=0); FLAGS[Serbia] = Serbia.flag_
Slovakia = Team(name='Slovakia', flag='data/flags/SK.png', continent='UEFA', nwcs=0); FLAGS[Slovakia] = Slovakia.flag_
Slovenia = Team(name='Slovenia', flag='data/flags/SI.png', continent='UEFA', nwcs=0); FLAGS[Slovenia] = Slovenia.flag_
Spain = Team(name='Spain', flag='data/flags/ES.png', continent='UEFA', nwcs=1); FLAGS[Spain] = Spain.flag_
Suriname = Team(name='Suriname', flag='data/flags/SR.png', continent='CONCACAF', nwcs=0); FLAGS[Suriname] = Suriname.flag_
Sweden = Team(name='Sweden', flag='data/flags/SE.png', continent='UEFA', nwcs=0); FLAGS[Sweden] = Sweden.flag_
Switzerland = Team(name='Switzerland', flag='data/flags/CH.png', continent='UEFA', nwcs=0); FLAGS[Switzerland] = Switzerland.flag_
Tunisia = Team(name='Tunisia', flag='data/flags/TN.png', continent='CAF', nwcs=0); FLAGS[Tunisia] = Tunisia.flag_
Turkey = Team(name='Turkey', flag='data/flags/TR.png', continent='UEFA', nwcs=0); FLAGS[Turkey] = Turkey.flag_
USA = Team(name='USA', flag='data/flags/US.png', continent='CONCACAF', nwcs=0); FLAGS[USA] = USA.flag_
Ukraine = Team(name='Ukraine', flag='data/flags/UA.png', continent='UEFA', nwcs=0); FLAGS[Ukraine] = Ukraine.flag_
Uruguay = Team(name='Uruguay', flag='data/flags/UY.png', continent='CONMEBOL', nwcs=2); FLAGS[Uruguay] = Uruguay.flag_
Uzbekistan = Team(name='Uzbekistan', flag='data/flags/UZ.png', continent='AFC', nwcs=0); FLAGS[Uzbekistan] = Uzbekistan.flag_
Wales = Team(name='Wales', flag='data/flags/WLS.png', continent='UEFA', nwcs=0); FLAGS[Wales] = Wales.flag_


df = pd.read_csv('data/best_thirds_fwc26.csv')
def wc26_builder(rho=1):
    wc26 = Tournament(group_stages=['GS'], 
                  knockout_rounds=['R32', 'R16', 'QF', 'SF', 'F'],
                  best_thirds=[8],
                  third_combinations=[df], rho=rho, gg='1x2')

    wc26.GSGroups = [
        G(~Mexico, ~RSA, ~Korea, ~Czechia),
        G(~Canada, ~Bosnia, ~Qatar, ~Switzerland),
        G(~Brazil, ~Morocco, ~Haiti, ~Scotland),
        G(~USA, ~Paraguay, ~Australia, ~Turkey),
        G(~Germany, ~Curacao, ~CIV, ~Ecuador),
        G(~Netherlands, ~Japan, ~Sweden, ~Tunisia),
        G(~Belgium, ~Egypt, ~Iran, ~NZ),
        G(~Spain, ~CPV, ~KSA, ~Uruguay),
        G(~France, ~Senegal, ~Iraq, ~Norway),
        G(~Argentina, ~Algeria, ~Austria, ~Jordan),
        G(~Portugal, ~DRC, ~Uzbekistan, ~Colombia),
        G(~England, ~Croatia, ~Ghana, ~Panama)
    ]
    
    wc26.R32Matches = [
        wc26.GS.A.Runnerup * wc26.GS.B.Runnerup,  # 73 (1)
        wc26.GS.E.Winner * wc26.GS.BestThirds['toFace1E'],  # 74 (2)
        wc26.GS.F.Winner * wc26.GS.C.Runnerup,  # 75 (3)
        wc26.GS.C.Winner * wc26.GS.F.Runnerup,  # 76 (4)
        wc26.GS.I.Winner * wc26.GS.BestThirds['toFace1I'],  # 77 (5)
        wc26.GS.E.Runnerup * wc26.GS.I.Runnerup, # 78 (6)
        wc26.GS.A.Winner * wc26.GS.BestThirds['toFace1A'],  # 79 (7)
        wc26.GS.L.Winner * wc26.GS.BestThirds['toFace1L'],  # 80 (8)
        wc26.GS.D.Winner * wc26.GS.BestThirds['toFace1D'],  # 81 (9)
        wc26.GS.G.Winner * wc26.GS.BestThirds['toFace1G'],  # 82 (10)
        wc26.GS.K.Runnerup * wc26.GS.L.Runnerup,  # 83 (11)
        wc26.GS.H.Winner * wc26.GS.J.Runnerup,  # 84 (12)
        wc26.GS.B.Winner * wc26.GS.BestThirds['toFace1B'],  # 85 (13)
        wc26.GS.J.Winner * wc26.GS.H.Runnerup,  # 86 (14)
        wc26.GS.K.Winner * wc26.GS.BestThirds['toFace1K'],  # 87 (15)
        wc26.GS.D.Runnerup * wc26.GS.G.Runnerup  # 88 (16)
    ]
    
    wc26.R16Matches = [
        wc26.R32.M2.Winner * wc26.R32.M5.Winner,
        wc26.R32.M1.Winner * wc26.R32.M3.Winner,
        wc26.R32.M11.Winner * wc26.R32.M12.Winner,
        wc26.R32.M9.Winner * wc26.R32.M10.Winner,
        wc26.R32.M4.Winner * wc26.R32.M6.Winner,
        wc26.R32.M7.Winner * wc26.R32.M8.Winner,
        wc26.R32.M14.Winner * wc26.R32.M16.Winner,
        wc26.R32.M13.Winner * wc26.R32.M15.Winner
    ]
    
    wc26.QFMatches = [
        wc26.R16.M1.Winner * wc26.R16.M2.Winner,
        wc26.R16.M3.Winner * wc26.R16.M4.Winner,
        wc26.R16.M5.Winner * wc26.R16.M6.Winner,
        wc26.R16.M7.Winner * wc26.R16.M8.Winner
    ]
    
    wc26.SFMatches = [
        wc26.QF.M1.Winner * wc26.QF.M2.Winner,
        wc26.QF.M3.Winner * wc26.QF.M4.Winner
    ]
    
    wc26.FMatches = [
        wc26.SF.M1.Winner * wc26.SF.M2.Winner
    ]
    
    wc26.Champion = wc26.F.M1.Winner

    return wc26


def run_sim(rho=0.5, mu=0.5):
    sm = SM(n=1, year=2026, wc_builder=wc26_builder, rho=rho, mu=mu)
    sm.run()
    wc = sm.trs_[0].wc
    return wc.get_plot()
