import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import bracketology as br
import warnings
import gym
import numpy as np
from tourney_tools import * 
warnings.filterwarnings('ignore')



class TournamentEnv(gym.Env):
    '''
    This is a open aigym simulator to allow for reinforcement leaning agents simulate the March Madness College Basketball Tournament 
    built using the bracketology simulation environment


    '''

    def __init__(self, team_data,discrete=False):
        '''
        parameters:
        Data - Dataframe
            Any data you would like to use to allow the model to predict the outcome of the game.
            By default It uses the sumbitable format from the 2022 kaggle march madness comeption
            A combination of all possible games bewteen all possible teams in the tournament.

        Model, 
            Pass a model with a predict method


        '''
        super().__init__()

        self.data=team_data
        self._favoritewins=0
        self._udogwins=0
        self._n_games=0
        self.current_round=0
        
        self.season=np.random.choice(range(2009,2019))
        self.bracket=Bracket(self.season)

        self.round1_frame=self.get_round1_frame()
        self.round2_frame=pd.DataFrame(columns=self.round1_frame.columns)
        self.sweet6_frame=pd.DataFrame(columns=self.round1_frame.columns)
        self.elite8_frame=pd.DataFrame(columns=self.round1_frame.columns)
        self.final4_frame=pd.DataFrame(columns=self.round1_frame.columns)
        self.champ_frame=pd.DataFrame(columns=self.round1_frame.columns)


        self.round1_obs=self.make_round_data(round_frame=self.round1_frame)

        cat_columns=self.round1_obs.select_dtypes(object).columns
        num_columns=self.round1_obs.select_dtypes(np.number).columns

        self.processor=make_feature_processor(self.round1_obs,cat_features=cat_columns)
        self.processor.fit(self.round1_obs)

        self.observation_space = gym.spaces.Box(
                                                low=-float('Inf'), 
                                                high=float('Inf'),
                                                shape=(1,(self.round1_obs.shape[-1])), 
                                                dtype=np.float64
                                                )
        
        if discrete:
            self.action_space = gym.spaces.Discrete(2)
        else:
            self.action_space = gym.spaces.Box(
                                    low=0., 
                                    high=1.,
                                    shape=(1, 2), ## pick between 2 teams 
                                    dtype=np.float64
                                                )
                    
    def step(self,action):
        self.current_round=1
        
    def fill_na_columns(self,data):
        cols=data.select_dtypes(np.number).columns
        for col in cols:
            if data[col].isna().sum()>0:
                m=data[col].mean()
                data[col]=data[col].fillna(m)
        
        cols=data.select_dtypes(object).columns
        for col in cols:
            if data[col].isna().sum()>0:

                data[col]=data[col].fillna(method='ffill')
        return data

    def make_round_data(self,round_frame):
        season_data=self.data[self.data.Season==self.season]
        tour_data=pd.merge(round_frame,season_data,left_on=['bottom_team'],right_on=['TeamName'],how='left')
        tour_data=pd.merge(tour_data,season_data,left_on=['top_team'],right_on=['TeamName'],how='left')
        tour_data['Round']=self.current_round
        tour_data=tour_data.drop(tour_data.filter(like='Seed').columns,axis=1)
        tour_data=tour_data.fillna(method='ffill').fillna(method='bfill')
        return tour_data

    def get_round1_frame(self):
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
                i+=1
                game_dir=dir(game)

                game_dict={d:getattr(game,d) for d in game_dir if '__' not in d}
                game_dict2={d:game_dict[d].name for d in game_dict if not isinstance(game_dict[d],int)}
                seed_dict={f'{d}_seed':game_dict[d].seed for d in game_dict if not isinstance(game_dict[d],int)}
                round=np.array([game_dict[d] for d in game_dict if isinstance(game_dict[d],int)])
                all_rounds.append(round)
                # print(round)
                # print(seed_dict)
                allgames_dict[i]=game_dict2|seed_dict

        r1_frame=pd.DataFrame(allgames_dict).T
        r1_frame['Round']=np.array(all_rounds)
        return r1_frame

    def get_round2_frame(self):
        regions=['East','Midwest','South','West']
        bracket_dir=dir(self.bracket)
        backet_regions=[getattr(self.bracket,attr) for attr in bracket_dir if attr in regions ]
        region_names=[attr for attr in bracket_dir if attr in regions ]
        print('region_names:',region_names)
        allgames_dict={}
        all_rounds=[]
        i=0
        for region in backet_regions:
            print(region.round2)
            

            # rounds=[getattr(region,attr) for attr in region_dir if 'round2' in attr]
            # round_names=[attr for attr in region_dir if 'round2' in attr]
            # print('Rounds:',round_names)
            # print('Rounds',rounds)
            # print(region_dir)


        #     for game in games:
        #         i+=1
        #         game_dir=dir(game)
        #         game_dict={d:getattr(game,d) for d in game_dir if '__' not in d}
        #         game_dict2={d:game_dict[d].name for d in game_dict if not isinstance(game_dict[d],int)}
        #         seed_dict={f'{d}_seed':game_dict[d].seed for d in game_dict if not isinstance(game_dict[d],int)}
        #         round=np.array([game_dict[d] for d in game_dict if isinstance(game_dict[d],int)])
        #         all_rounds.append(round)
        #         # print(round)
        #         # print(seed_dict)


        #         allgames_dict[i]=game_dict2|seed_dict

        # r1_frame=pd.DataFrame(allgames_dict).T
        # r1_frame['Round']=np.array(all_rounds)

    def choose_game_winner(self,game):
        # Extract Teams from Game
        top_team = game.top_team
        bottom_team = game.bottom_team


        if top_team.seed>=bottom_team.seed:
            high_team=top_team
            low_team=bottom_team
            winner=1
        else:
            high_team=bottom_team
            low_team=top_team
            winner=0
        
        # game_data=self.data.loc[(self.data.LowTeamName == low_team) & (self.data.HighTeamName == high_team)]
        # game_data=self.feature_processor.transform(game_data)
        # self._n_games+=1

        # winner_proba=self.model.predict(game_data)
        # winner =np.argmax(winner_proba)
        teams = [high_team,low_team]
        winning_team=teams[winner]
        self._n_games+=1

        return winning_team

    def get_upset_pct(self):
        fav_win_pct = round(self._favoritewins/self._n_games, 4) * 100
        udog_win_pct = round(self._udogwins/self._n_games, 4) * 100

    def reset(self):
        self.current_round=0
        self.season=np.random.choice(range(2009,2019))
        self.bracket=Bracket(self.season)

        self.round1_frame=self.get_round1_frame()
        self.round1_obs=self.make_round_data(self.round1_frame)

        cat_columns=self.round1_obs.select_dtypes(object).columns
        num_columns=self.round1_obs.select_dtypes(np.number).columns

        self.processor=make_feature_processor(self.round1_obs,cat_features=cat_columns)
        self.processor.fit(self.round1_obs)
        obs=self.processor.transform(self.round1_obs)
        obs=obs.toarray()

        return obs.flatten()
