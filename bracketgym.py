import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from bracketology_rl import Bracket, Game, Team
import warnings
import gym
import numpy as np
from tourney_tools import * 
warnings.filterwarnings('ignore')
from IPython.display import display



class TournamentEnv(gym.Env):
    '''
    This is a open aigym simulator to allow for reinforcement leaning agents simulate the March Madness College Basketball Tournament 
    built using the bracketology simulation environment


    '''

    def __init__(self, team_data,season=None,discrete=False,weight_scores=False,verbose=False):
        '''
        parameters:
        Data - Dataframe
            Any data you would like to use to allow the model to predict the outcome of the game.
            By default It uses the sumbitable format from the 2022 kaggle march madness comeption
            A combination of all possible games bewteen all possible teams in the tournament.


        '''
        super().__init__()
        # team_data=map_team_names(team_data)
        self.data=team_data
        self.weight_scores=weight_scores
        self._favoritewins=0
        self._udogwins=0
        self._n_games=0
        self.current_round=0
        self.n_simulations=0
        self.current_pred=None
        self.action=None
        self.verbose=verbose
        valid_years=[y for y in range(2009,2022)]
        valid_years.remove(2020)
        self.track_wins=False
        if season:
            self.season=season
        else:
            ## valid years up to 2021 and not 2020 because the Marchmadness tournament was not played
            self.season=np.random.choice(valid_years)
        self.bracket=Bracket(self.season)
        self.pca=PCA(n_components=8000)
        self.games_bar=tqdm(range(120000))
        self.games_bar.set_description(f'Reset! Sim {self.n_simulations} Bracket for year {self.season}')

        
        ## mappings to turn roun names and funtioncs into ints so we can step through them
        self.round_functions={
            1:'run_first_round',
            2:'run_second_round',
            3:'run_sweet_sixteen',
            4:'run_elite_eight',
            5:'run_final_four',
            6:'run_championship',
        }
        
        self.rounds={
            1:'round1',
            2:'round2',
            3:'round3',
            4:'round4',
            5:'round5',
            6:'round6',
        }

        self.round_regions={
            1:['East','Midwest','South','West'],
            2:['East','Midwest','South','West'],
            3:['East','Midwest','South','West'],
            4:['East','Midwest','South','West'],
            5:['Finals'],
            6:['Finals'],
        }
        self.tracking_frames={team.name:pd.DataFrame(index=[team.name for team in self.bracket.round1],
                                columns =[i for i in range(1,7)]).fillna(0) for team in self.bracket.round1}
        self.current_matchups=self.get_round1_matchups()
        self.round_template=pd.DataFrame(index=self.current_matchups.index)
        self.done=False

        self.current_data=self.make_round_data(round_frame=self.current_matchups)
        cat_columns=self.current_data.select_dtypes(object).columns
        
        num_columns=self.current_data.select_dtypes(np.number).columns

        self.processor=make_feature_processor(self.current_data,cat_features=cat_columns)
        self.processor.fit(self.current_data)
        obs=self.processor.transform(self.current_data)
        # obs=self.pca.fit_transform(obs.T).T
        obs=obs.flatten()
        self.last_obs=obs

        self.observation_space = gym.spaces.Box(
                                                low=-float('Inf'), 
                                                high=float('Inf'),
                                                shape=obs.shape, 
                                                dtype=np.float32
                                                )
        
        if discrete:
            self.action_space = gym.spaces.Discrete(self.current_data.shape[0]*2)
        else:
            self.action_space = gym.spaces.Box(
                                    low=0., 
                                    high=1.,
                                    shape= (self.current_data.shape[0]*2,), ## pick between all teams in the round
                                    dtype=np.float32
                                                )
        
    def impute_columns(self,data):
        cols=data.select_dtypes(np.number).columns
        for col in cols:
            if data[col].isna().sum()>0:
                m=data[col].mean()
                data[col]=data[col].fillna(m)
        
        cols=data.select_dtypes(object).columns
        for col in cols:
            if data[col].isna().sum()>0:
                data[col]=data[col].fillna('Unknown')

        return data

    def fill_nogame_columns(self,data):
        cols=data.select_dtypes(np.number).columns
        for col in cols:
            if data[col].isna().sum()>0:
                m=data[col].mean()
                data[col]=data[col].fillna(0)
        
        cols=data.select_dtypes(object).columns
        for col in cols:
            if data[col].isna().sum()>0:
                data[col]=data[col].fillna('Nogame')

        return data

    def make_round_data(self,round_frame):
        season_data=self.data[self.data.Season==self.season]
        tour_data=pd.merge(round_frame,season_data,left_on=['bottom_team'],right_on=['TeamName'],how='left')
        tour_data=pd.merge(tour_data,season_data,left_on=['top_team'],right_on=['TeamName'],how='left')
        tour_data['Round']=self.current_round
        tour_data=tour_data.drop(tour_data.filter(like='Seed').columns,axis=1)
        tour_data=self.impute_columns(tour_data)
        tour_data=tour_data.reindex(index=self.round_template.index)
        round_data=self.fill_nogame_columns(tour_data)


        ## fix the column forder for easy viewing
        name_cols=[c for c in round_data.filter(like='TeamName',axis=1).columns]
        other_columns = [c for c in round_data.columns if c not in name_cols]
        col_order=name_cols+other_columns
        round_data=round_data[col_order]
        cat_columns=round_data.select_dtypes(object).columns.to_list()
        keep=['bottom_team', 'top_team','bottom_team_seed','top_team_seed']
        drop=[c for c in cat_columns if c not in keep]
        round_data=round_data.drop(drop,axis=1)
        round_data[['top_team_seed','bottom_team_seed']]=round_data[['top_team_seed','bottom_team_seed']].astype(int)

        return round_data

    def update_spaces(self,obs):

        self.observation_space = gym.spaces.Box(
                                                low=-float('Inf'), 
                                                high=float('Inf'),
                                                shape=obs.shape, 
                                                dtype=np.float32
                                                )

        self.action_space = gym.spaces.Box(
                        low=0., 
                        high=1.,
                        shape= (self.current_data.shape[0]*2,), ## pick between all teams in the round
                        dtype=np.float32
                                    )

    def get_round1_matchups(self):
        regions=['East','Midwest','South','West']
        bracket_dir=dir(self.bracket)
        backet_regions=[getattr(self.bracket,attr) for attr in bracket_dir if attr in regions ]
        allgames_dict={}
        all_rounds=[]
        i=0
        for region in backet_regions:
            region_dir=dir(region)
            games=[getattr(region,attr) for attr in region_dir if 'Game' in attr]
            for game in games:
                game_dir=dir(game)
                game_dict={d:getattr(game,d) for d in game_dir if '__' not in d}
                game_dict2={d:game_dict[d].name for d in game_dict if not isinstance(game_dict[d],int)}
                seed_dict={f'{d}_seed':game_dict[d].seed for d in game_dict if not isinstance(game_dict[d],int)}
                round=np.array([game_dict[d] for d in game_dict if isinstance(game_dict[d],int)])
                all_rounds.append(round)
                # print(round)
                # print(seed_dict)
                allgames_dict[i]=game_dict2|seed_dict
                i+=1
        r1_frame=pd.DataFrame(allgames_dict).T
        r1_frame[['top_team_seed','bottom_team_seed']]=r1_frame[['top_team_seed','bottom_team_seed']].astype(int)
        r1_frame['Round']=np.array(all_rounds)
        return r1_frame

    def get_matchup_frame(self):
        if self.current_round<6:
            round_get=self.current_round+1
        else:
            round_get=self.current_round

        regions=self.round_regions[round_get]
        bracket_dir=dir(self.bracket)
        backet_regions=[getattr(self.bracket,attr) for attr in bracket_dir if attr in regions ]
        allgames_dict={}
        all_rounds=[]
        i=0
        for region in backet_regions:
            games=getattr(region,self.rounds[round_get])
            for game in games:
                game_dir=dir(game)
                game_dict={d:getattr(game,d) for d in game_dir if '__' not in d}
                game_dict2={d:game_dict[d].name for d in game_dict if not isinstance(game_dict[d],int)}
                seed_dict={f'{d}_seed':game_dict[d].seed for d in game_dict if not isinstance(game_dict[d],int)}
                round=np.array([game_dict[d] for d in game_dict if isinstance(game_dict[d],int)])
                all_rounds.append(round)
                # print(round)
                # print(seed_dict)
                allgames_dict[i]=game_dict2|seed_dict
                i+=1
        r2_frame=pd.DataFrame(allgames_dict).T
        r2_frame[['top_team_seed','bottom_team_seed']]=r2_frame[['top_team_seed','bottom_team_seed']].astype(int)

        r2_frame['Round']=np.array(all_rounds)
        return r2_frame

    def get_actions(self ,action):
            #2 actions for each game pick the top or bottom team
            actions=action.reshape(self.current_data.shape[0],2)
            actions=pd.DataFrame(actions)
            return actions

    def step(self,action=None):

        self.update_spaces(self.last_obs)
        done=False
        self.current_round+=1
        self.games_bar.update(1)
        self.action=self.get_actions(action)

        if self.current_round>=6:
            done=True

        # print('-----------------------------------')
        bracket_round=self.round_functions[self.current_round].split('_')[1:]

        bracket_round=' '.join(bracket_round)
        # print(f"Run Round {self.current_round}:",bracket_round)
        round_func=getattr(self.bracket,self.round_functions[self.current_round])
        round_func(self.choose_winner)

        score,num_correct=self.bracket.score_round(verbose=self.verbose)

        if self.weight_scores:
            score=score*(self.current_round+1)
        self.current_matchups=self.get_matchup_frame()
        self.current_data=self.make_round_data(round_frame=self.current_matchups)
        obs=self.processor.transform(self.current_data)
        # obs=self.pca.transform(obs.T).T
        obs=obs.flatten().astype(np.float32)
        # display(self.current_data)
        # self.update_spaces(obs)
        self.last_obs=obs

        info= {'round':bracket_round,
                'score':score,
                'num_correct':num_correct
        }
        if np.isnan(obs).sum()>0:
            print(np.isnan(obs))
        return obs,score,done, info

    def choose_random_winner(self,game):
        # Extract Teams from Game
        top_team = game.top_team
        bottom_team = game.bottom_team

        if top_team.seed>=bottom_team.seed:
            high_team=top_team
            low_team=bottom_team

        else:
            high_team=bottom_team
            low_team=top_team

        
        # game_data=self.data.loc[(self.data.LowTeamName == low_team) & (self.data.HighTeamName == high_team)]
        # game_data=self.feature_processor.transform(game_data)
        # self._n_games+=1

        # winner_proba=self.model.predict(game_data)
        # winner =np.argmax(winner_proba)
        teams = [high_team,low_team]
        winning_team=low_team
        self._n_games+=1

        return winning_team

    def choose_winner(self,game):
        # Extract Teams from Game
        actions=self.action
        data=self.current_data
        top_team = game.top_team
        bottom_team = game.bottom_team
        # round=game.round
        # print(dir(game))
        round=game.round_number

        # print(game)

       
        poz=data[(data.top_team==top_team.name)&(data.bottom_team==bottom_team.name)].index

        
        actions=actions.loc[poz].values[0].flatten()
        action=np.argmax(actions)
        op_action=np.argmin(actions)

        
        # game_data=self.feature_processor.transform(game_data)
        # self._n_games+=1

        # winner_proba=self.model.predict(game_data)
        # winner =np.argmax(winner_proba)
        teams = [top_team,bottom_team]
        team_names=[top_team,bottom_team]
        
        winning_team=teams[action]
        losing_team=teams[op_action]
        if self.track_wins:
            self.tracking_frames[winning_team.name].loc[losing_team.name,round]=\
            self.tracking_frames[winning_team.name].loc[losing_team.name,round]+1

        self._n_games+=1

        return winning_team
    
    def choose_team(self,game):
        # Extract Teams from Game
        actions=self.action
        data=self.current_data
        top_team = game.top_team
        bottom_team = game.bottom_team
        # round=game.round
        # print(dir(game))
        round=game.round_number 
        poz=data[(data.top_team==top_team.name)&(data.bottom_team==bottom_team.name)].index
        poz_values=data[(data.top_team==top_team.name)&(data.bottom_team==bottom_team.name)].values
        
        actions=actions.loc[poz].values[0].flatten()
        action=np.argmax(actions)
        op_action=np.argmin(actions)

        
        # game_data=self.feature_processor.transform(game_data)
        # self._n_games+=1

        # winner_proba=self.model.predict(game_data)
        # winner =np.argmax(winner_proba)
        teams = [top_team,bottom_team]
        team_names=[top_team,bottom_team]
        
        winning_team=teams[action]
        losing_team=teams[op_action]
        if self.track_wins:
            self.tracking_frames[winning_team.name].loc[losing_team.name,round]=\
            self.tracking_frames[winning_team.name].loc[losing_team.name,round]+1

        self._n_games+=1

        return winning_team

    def tracked_winner(self,game):

        # Extract Teams from Game
        actions=self.action
        data=self.current_data
        top_team = game.top_team
        bottom_team = game.bottom_team
        round=game.round_number
        # print(game)

        top_team_wins=self.tracking_frames[top_team.name].loc[bottom_team.name,round]
        bottom_team_wins=self.tracking_frames[bottom_team.name].loc[top_team.name,round]

        poz=data[(data.top_team==top_team.name)&(data.bottom_team==bottom_team.name)].index

        
        actions=[top_team_wins,bottom_team_wins]
        action=np.argmax(actions)
        op_action=np.argmin(actions)

        
        # game_data=self.feature_processor.transform(game_data)
        # self._n_games+=1

        # winner_proba=self.model.predict(game_data)
        # winner =np.argmax(winner_proba)
        teams = [top_team,bottom_team]
    
        
        winning_team=teams[action]

        self._n_games+=1
        return winning_team

    def run_final_bracket(self,season):
        self.season=season
        self.bracket=Bracket(self.season)
        self.bracket.sim(self.tracked_winner)

    def reset(self,season=None,weight_scores=True,track_wins=False):
        if season!= None:
            self.season = season
        else:             
            valid_years=[y for y in range(2009,2022)]
            valid_years.remove(2020)
            self.season = np.random.choice(valid_years)
        self.track_wins=track_wins
        self.current_round=0
        self.n_simulations+=1
        self.games_bar.update(1)

        self.weight_scores=weight_scores
        self.games_bar.set_description(f'Reset! Sim {self.n_simulations} Bracket for year {self.season}')

        # print(f'Bracket for year {self.season}')
        self.bracket=Bracket(self.season)

        self.current_matchups=self.get_round1_matchups()
        self.current_data=self.make_round_data(self.current_matchups)

        cat_columns=self.current_data.select_dtypes(object).columns
        num_columns=self.current_data.select_dtypes(np.number).columns

        self.processor=make_feature_processor(self.current_data,cat_features=cat_columns)
        
        self.processor.fit(self.current_data)

        obs=self.processor.transform(self.current_data)
        # obs=self.pca.fit_transform(obs.T).T
        
        obs=obs.flatten()
        return obs.astype(np.float32)
    
    def render():
        pass




