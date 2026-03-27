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


VENUE_ = None
RATINGS_ = None
ASI_ = None
DWI_ = None
GG_ = None
PXGBR_ = None
RANDOM_ = None
FLAGS = None


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
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def fit(self, lambdas_in_pairs, scores):
        pass

    def predict(self, lambdas_in_pairs):
        raise NotImplementedError


class Poisson1X2Generator(GoalGenerator):
    def __init__(self, seed=42, mu=0.5, n=10):
        self.original_rng = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.shuffle_rng = random.Random(seed)
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
            if np.issubdtype(df[col].dtype, np.floating):
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

    def plot_tables(self):
        fig, axs = plt.subplots(4, 3, figsize=(20, 20))
        axs = axs.flatten()
        for i, (group, ax) in enumerate(zip(self.groups_, axs)):
            group.plot(ax)
        [fig.delaxes(ax) for ax in axs[i + 1:]]
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
                 wc_builder=lambda: None, seed=42):
        global GG_, to_keep, RANDOM_, PXGBR_, venues, VENUE_, pxgbrs
        self.year_ = year
        self.n_ = n
        self.rho_ = rho
        self.mu_ = mu
        self.build = wc_builder
        self.trs_ = []
        PXGBR_ = pxgbrs[year]
        GG_['1x2'] = Poisson1X2Generator(mu=self.mu_)
        RANDOM_ = random.Random(seed)
        VENUE_ = venues[year]

    def run(self, verbose=0):
        global ASI_, DWI_, RATINGS_, wcs
        tr_sum = TR()
        iterable = tqdm(range(self.n_), desc='Running Simulation') if verbose else range(self.n_)
        for _ in iterable:
            RATINGS_ = wcs[self.year_]['rate'].copy()
            ASI_ = wcs[self.year_]['asi'].copy()
            DWI_ = wcs[self.year_]['dwi'].copy()
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
