import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from bracketology_rl import Bracket, Game, Team
import warnings
import gymnasium as gym
import numpy as np
from tourney_tools import * 
warnings.filterwarnings('ignore')
from IPython.display import display
# import graphviz




class TournamentEnv(gym.Env):
    '''
    This is a open aigym simulator to allow for reinforcement leaning agents simulate the March Madness College Basketball Tournament 
    built using the bracketology simulation environment


    '''

    def __init__(self, team_data,season=None,
                                discrete=False,
                                loading_bar=True,
                                weight_scores=False,
                                verbose=False,
                                bust_stop=True,
                                num_simulations=100000
                                ):
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
        # team_data=map_team_names(team_data)
        self.data=team_data
        self.weight_scores=weight_scores
        self._favoritewins=0
        self._udogwins=0
        self._n_games=0
        self.current_round=1
        self.last_round=1
        self.current_step=0
        self.n_simulations=0
        self.num_correct=0
        self.total_score=0
        self.round_score=0
        self.bust_stop=bust_stop
        self.current_pred=None
        self.action=None
        self.verbose=verbose
        valid_years=[y for y in range(2009,2022)]
        valid_years.remove(2020)
        valid_years.remove(2021)
        self.track_wins=False
        if season:
            self.season=season
        else:
            ## valid years up to 2021 and not 2020 because the Marchmadness tournament was not played
            self.season=np.random.choice(valid_years)
        self.bracket=Bracket(self.season)
        self.pca=PCA(n_components=8000)
        if loading_bar:
            self.games_bar=tqdm(range(num_simulations))
            self.games_bar.set_description(f'Reset! Sim {self.n_simulations} Bracket for year {self.season}')
        self.target_simulations=num_simulations
        self.loading_bar=loading_bar
        ## get mappings for the step-> round, step->score, and round-> data frame index
        result_dict,round_step_dir,round_index=self.get_step_dict(self.bracket)
        score_dict=self.get_scoring_dict(self.bracket)

        self.score_dict=score_dict
        self.result_dict=result_dict
        self.round_step_dict=round_step_dir
        self.round_index=round_index
        

        ## mappings to turn round names and funtions into ints so we can step through them
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
            1:['W','X','Y','Z'],
            2:['W','X','Y','Z'],
            3:['W','X','Y','Z'],
            4:['W','X','Y','Z'],
            5:['Finals'],
            6:['Finals'],
        }

        self.current_matchups=self.get_round1_matchups()

        self.done=False

        self.current_data=self.make_round_data(matchup_frame=self.current_matchups)

        self.fit_data=self.stack_cat_columns()
        self.action_data=pd.DataFrame(columns=['winning_team','losing_team','round'])
        self.tracking_frames=self.action_data.copy()

        self.cat_columns=self.current_data.select_dtypes(object).columns
        num_columns=self.current_data.select_dtypes(np.number).columns
        self.cat_columns=[]

        self.processor=build_feature_processor(num_features=num_columns,cat_features=self.cat_columns)
        self.processor.fit(self.fit_data)
        obs=self.processor.transform(self.current_data.loc[[0]])


        try:obs=obs.flatten().astype(np.float32)
        except:obs=obs.toarray()
        finally:obs=obs.flatten().astype(np.float32)

        
        self.last_obs=obs

        self.observation_space = gym.spaces.Box(
                                                low=-float('Inf'), 
                                                high=float('Inf'),
                                                shape=obs.shape, 
                                                dtype=np.float32
                                                )
        self.discrete=discrete
        if discrete:
            self.action_space = gym.spaces.Discrete(2)## pick between two teams in the round
        else:
            self.action_space = gym.spaces.Box(
                                    low=0., 
                                    high=1.,
                                    shape= (2,), ## pick between two teams in the round
                                    dtype=np.float32
                                                )


    def get_step_dict(self,bracket):
        results=bracket.result
        step_dir={}
        round_step_dir={}
        round_indexs={}
        i=0 # index 0
        last_i=0 # start index
        r=0
        ## get the length of teams in the round
        for round in results:
            #round rumber
            r+=1
            n_teams=int(len(results[round]))
            n_games=int(n_teams/2)
            #update the last index for after the first round
            last_i+=n_games

            if round=='winner':
                step_dir[last_i]=[results[round]['Team']]
                round_step_dir[last_i]=r

            else:
                for j in range(i,last_i):
                    step_dir[j]=[team['Team'] for team in results[round]]
                    round_step_dir[j]=r
                
                round_indexs[r]=n_games

            i= last_i
        round_step_dir[i-1]=r
        round_indexs[r]=[last_i-2]
        ## step->team_list for scoring,step-> round for steping,  round->step_index list for updating matchup data indexs      
        return step_dir,round_step_dir,round_indexs

    def get_scoring_dict(self,bracket):
            score_dir={}
            results=bracket.result
            i=0
            r=0
            for round in results:
                r+=1
                if i <=1:
                    i+=1
                else:
                    i=i*2
                score_dir[r]=i
            ## ste -> scoreing so we know how much to score for each round
            return score_dir

    def stack_cat_columns(self):

        data=self.current_data.rename(columns={'top_team':'bottom_team','bottom_team':'top_team'})
        data=pd.concat([self.current_data,data],axis=0)
        return data

    def make_round_data(self,matchup_frame):
        ## get the data you want to use to predict
        season_data=self.data[self.data.Season==self.season]
        ## merge the 2 teams with the matchup frame
        tour_data=pd.merge(matchup_frame,season_data,left_on=['bottom_team'],right_on=['TeamName'],how='left')
        tour_data=pd.merge(tour_data,season_data,left_on=['top_team'],right_on=['TeamName'],how='left')

        ## add the current round
        tour_data['Round']=self.current_round
        tour_data=tour_data.drop(tour_data.filter(like='Seed').columns,axis=1)

        ## get the predetermined step index.... maybe change to indexing as we go
        idx=self.current_matchups.index

        ## update the index to match the steps
        try:
            tour_data.index=idx
            round_data=tour_data

        except: 
            print('Something is wrong with your indexing')
            display('index:',idx)
            display('index map',self.round_index[self.current_round])
            display('Matchups')
            display(self.current_matchups)
            display(tour_data)

        #tour_data.index=idx
        # self.current_matchups.index=idx
        

        ## fix the column forder for easy viewing
        name_cols=[c for c in round_data.filter(like='TeamName',axis=1).columns]
        other_columns = [c for c in round_data.columns if c not in name_cols]
        col_order=name_cols+other_columns
        round_data=round_data[col_order]
        #drop extra Columns
        cat_columns=round_data.select_dtypes(object).columns.to_list()
        keep=['bottom_team', 'top_team','bottom_team_seed','top_team_seed']
        drop=[c for c in cat_columns if c not in keep]
        round_data=round_data.drop(drop,axis=1)
        round_data[['top_team_seed','bottom_team_seed']]=round_data[['top_team_seed','bottom_team_seed']].astype(int)

        return round_data

    def get_round1_matchups(self):
        regions=['W','X','Y','Z']
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
                allgames_dict[i]['Region']=region.region
                i+=1
        r1_frame=pd.DataFrame(allgames_dict).T
        r1_frame[['top_team_seed','bottom_team_seed']]=r1_frame[['top_team_seed','bottom_team_seed']].astype(int)
        r1_frame['Round']=np.array(all_rounds)
        return r1_frame
    
    def generate_matchup_index(self,matchups):

        current_step=self.current_step
        num_games=len(matchups)
        next_index_list=[i for i in range(current_step,current_step+num_games)]

        return next_index_list

    def update_round(self):
        step=self.current_step
        index=self.current_matchups.index
        if step>index[-1]:
            self.current_round+=1
        return self.current_round

    def generate_matchups(self):
        bracket_round=self.round_functions[self.last_round].split('_')[1:]
        bracket_round=' '.join(bracket_round)
        ## run the round in the bracket to generate the next matchups
        round_func=getattr(self.bracket,self.round_functions[self.last_round])
        round_func(self.choose_winners)

    def get_matchup_frame(self):
        ## get the regions associated with the Current Round
        round_get=self.current_round
        regions=self.round_regions[round_get]

        ## loop through the bracket and get the teams and seeds from the game from the bracket 
        bracket_dir=dir(self.bracket)
        bracket_regions=[getattr(self.bracket,attr) for attr in bracket_dir if attr in regions ]
        allgames_dict={}
        all_rounds=[]
        i=0
        for region in bracket_regions:
            games=getattr(region,self.rounds[round_get])
            for game in games:
                game_dir=dir(game)
                game_dict={d:getattr(game,d) for d in game_dir if '__' not in d}
                game_dict2={d:game_dict[d].name for d in game_dict if not isinstance(game_dict[d],int)}
                seed_dict={f'{d}_seed':game_dict[d].seed for d in game_dict if not isinstance(game_dict[d],int)}

                round=np.array([game_dict[d] for d in game_dict if isinstance(game_dict[d],int)])
                all_rounds.append(round)
                allgames_dict[i]=game_dict2|seed_dict
                try:allgames_dict[i]['Region']=region.region
                except:allgames_dict[i]['Region']=self.rounds[round_get]
                i+=1
        matchup_frame=pd.DataFrame(allgames_dict).T
        matchup_frame[['top_team_seed','bottom_team_seed']]=matchup_frame[['top_team_seed','bottom_team_seed']].astype(int)

        matchup_frame['Round']=np.array(all_rounds)
        matchup_frame.index=self.generate_matchup_index(matchups=matchup_frame)

        return matchup_frame

    def get_actions(self ,action):
        #2 actions for each game pick the top or bottom team
        if self.discrete:
            actions=np.zeros(2)
            actions[action]=1
            action=actions
        actions=action.flatten()
        return actions

    def step(self,action=None):

        ## start by  getting the score from the action
        done=False
        truncated=False
        self.action=self.get_actions(action)
        ## chose the winning team
        winning_team=self.choose_team(self.action)
        
        #get your score
        score=self.score_game(team=winning_team)
        #step forward

        self.current_step+=1
        self.current_round=self.update_round()
        
        if self.current_round>6:
            ## done if youre on round 7
            done=True
            obs=self.last_obs

        if not done:
            if self.current_round!=self.last_round:
                ## if you didnt get any points in the last round your bracket is busted and youre done
                if self.round_score==0 and self.bust_stop:
                    
                    done=True
                    truncated=True
                    game=winning_team

                ## reset the round score at the end of each round
                self.round_score=0
                ## run the bracket simulate function
                self.generate_matchups()
                self.current_matchups=self.get_matchup_frame()
                self.current_data=self.make_round_data(matchup_frame=self.current_matchups)
            ## get next game obs
            game=self.current_data.loc[[self.current_step]]
            obs=self.processor.transform(game)
            ## this is because the feature tran sometimes returna a sparse array
            try:obs=obs.flatten().astype(np.float32)
            except:obs=obs.toarray()
            finally:obs=obs.flatten().astype(np.float32)

            self.last_obs=obs
            self.last_round=self.current_round
        else: game=winning_team
        info= {'round':self.current_round,
                'game_num':self.current_step,
                'score':score,
                'num_correct':self.num_correct,
                'round_score':self.round_score,
                'total_score':self.total_score,
                'espisode_start':self.total_score,
                'match_up':game
        }

        return obs,score,done,truncated, info

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

        teams = [high_team,low_team]
        winning_team=low_team
        self._n_games+=1

        return winning_team

    def choose_winners(self,game):
        # Extract Teams from Game
        actions=self.action

        ## data is the accumulated Action winners from the steps
        data=self.action_data
        top_team = game.top_team
        bottom_team = game.bottom_team
        round=game.round_number

        # get the actions from the previously chosen teams
        act=data[(data.top_team==top_team.name)&(data.bottom_team==bottom_team.name)].index
        action=data.loc[act,['top_team_proba','bottom_team_proba']].values
        # Use the max of the two values to decide the winner
        action=np.argmax(action)
        teams = [top_team,bottom_team]
        winning_team=teams[action]

        self._n_games+=1

        return winning_team
    
    def choose_team(self,action):
        # Extract Teams from Game
        actions=action

        ## data is current data so we only have to make the data for each round once
        data=self.current_data
        top_team = data.loc[self.current_step].top_team
        top_seed = data.loc[self.current_step].top_team_seed
        bottom_team = data.loc[self.current_step].bottom_team
        bottom_seed = data.loc[self.current_step].bottom_team_seed
        region = self.current_matchups.loc[self.current_step].Region


        ### save the action into a ronud for rendering and analysis later
        self.action_data.loc[self.current_step,'round']=self.current_round
        self.action_data.loc[self.current_step,'top_team_proba']=actions[0]
        self.action_data.loc[self.current_step,'bottom_team_proba']=actions[1]
        self.action_data.loc[self.current_step,'top_team']=top_team
        self.action_data.loc[self.current_step,'top_seed']=top_seed
        self.action_data.loc[self.current_step,'bottom_team']=bottom_team
        self.action_data.loc[self.current_step,'bottom_seed']=bottom_seed
        self.action_data.loc[self.current_step,'region']=region 

        ## grab the action so we can predict the team
        action=np.argmax(actions)
        actmax=np.max(actions)
        actmin=np.max(actions)
        op_action=np.argmin(actions)

        teams = [top_team,bottom_team]
        seeds = [top_seed,bottom_seed]

        
        winning_team=teams[action]
        winning_seed=seeds[action]
        losing_team=teams[op_action]
        losing_seed=seeds[op_action]

        ## save the upset and probablilties
        self.action_data.loc[self.current_step,'winning_team']=winning_team
        self.action_data.loc[self.current_step,'winning_seed']=winning_seed
        self.action_data.loc[self.current_step,'losing_team']=losing_team
        self.action_data.loc[self.current_step,'losing_seed']=losing_seed
        self.action_data.loc[self.current_step,'upset']=int(winning_seed>losing_seed)


        return winning_team

    def score_game(self,team):
        possible_score=self.score_dict[self.current_round]
        score=0
        if team in self.result_dict[self.current_step]:
            score=possible_score
            self.num_correct+=1
        else:
            if self.bust_stop:
                score=0
            else:score=0
        ## update the Round Score and Total Scores
        ## only update the round score if the score is positive otherwise real score would be 0 instead of the -penalty score
        if score>0:
            self.total_score+=score
            self.round_score+=score
        if self.weight_scores:
            score=score**self.current_round
        return score

    def tracked_winner(self,game):

        # Extract Teams from Game
        actions=self.action
        data=self.current_data
        top_team = game.top_team
        bottom_team = game.bottom_team
        round=game.round_number
        # print(game)
        df=self.tracking_frames
        win_probas=df[(df.top_team==top_team.name) & (df.bottom_team==bottom_team.name)][['top_team_proba','bottom_team_proba']].values
        idx=df[(df.top_team==top_team.name) & (df.bottom_team==bottom_team.name)].index
        win_probas=df[(df.top_team==top_team.name) & (df.bottom_team==bottom_team.name)][['top_team_proba','bottom_team_proba']].values
        upset_proba=df.loc[idx,'upset']


        actions=win_probas
        action=np.argmax(actions)
        op_action=np.argmin(actions)

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
            valid_years.remove(2021)
            self.season = np.random.choice(valid_years)
        self.track_wins=track_wins
        self.num_correct=0
        self.total_score=0
        self.current_step=0
        self.current_round=self.round_step_dict[self.current_step]
        self.last_round=self.current_round
        self.n_simulations+=1
        if self.loading_bar:
            self.games_bar.update(1)
            self.games_bar.set_description(f'Reset! Sim {self.n_simulations} Bracket for year {self.season}')

        self.weight_scores=weight_scores


        ## get the new bracket
        self.bracket=Bracket(self.season)

        ## update the step dict for the new season
        result_dict,round_step_dir,round_index=self.get_step_dict(self.bracket)
        score_dict=self.get_scoring_dict(self.bracket)
        self.score_dict=score_dict
        self.result_dict=result_dict
        self.round_step_dict=round_step_dir
        self.round_index=round_index

        ## get the first round matchups
        self.current_matchups=self.get_round1_matchups()
        self.current_data=self.make_round_data(matchup_frame=self.current_matchups)
        self.fit_data=self.stack_cat_columns()

        ## remeber that team simulation and save it
        if self.track_wins:
            self.tracking_frames=pd.concat([self.tracking_frames,self.action_data],axis=0)
            try:
                self.tracking_frames=self.tracking_frames.groupby(['top_team','bottom_team']).agg({'top_team_proba': 'mean', 'bottom_team_proba': 'mean','upset':'mean'})
                self.tracking_frames=self.tracking_frames.reset_index()
            except Exception as e:
                print(e)


        self.action_data=pd.DataFrame(columns=['winning_team','losing_team','round'])


        # cat_columns=self.current_data.select_dtypes(object).columns
        # cat_columns=[]
        num_columns=self.current_data.select_dtypes(np.number).columns


        ## scale the data
        self.processor=build_feature_processor(num_features=num_columns,cat_features=self.cat_columns)
        self.processor.fit(self.fit_data)
        game=self.current_data.loc[[self.current_step]]

        obs=self.processor.transform(game)
        ## this is because sometimes the feature processor will return a sparse array
        try:obs=obs.flatten().astype(np.float32)
        except:obs=obs.toarray()
        finally:obs=obs.flatten().astype(np.float32)
        

        return obs
    
    def render(self,score=None):

        # Create a Graphviz graph
        tournament=self.action_data
        year=self.season

        dot = graphviz.Digraph(f'MM Bracket {year} Simulation {self.n_simulations}',
                                 comment=f'March Madness Bracket {year} Simulation {self.n_simulations}',
                                 format='png' ,
                                 engine="dot")
        # Set Graphviz attributes
        dot.attr(label=f'March Madness Bracket {year} {score} Simulation {self.n_simulations}')
        dot.attr(rankdir='RL', nodesep='.2', ranksep='.3', splines='line',size='40,40')

        dot.attr('node', shape='plaintext', fixedsize='false', width='1',style='filled')
        dot.graph_attr.update(landscape="false")

        dot.edge_attr.update(arrowhead='none', penwidth='1')
        ## loop through the data and create the selctions
        for i,row in tournament.iterrows():

            win_team=row['winning_team']
            win_seed=row['winning_seed']
            
            team_a=row['top_team']
            team_b=row['bottom_team']

            seed_a=row['top_seed']
            seed_b=row['bottom_seed']

            rank=str(row['round'])
            next_rank=str((row['round']+1))
            region=row['region']

            ## create two team nodes
            dot.node(f'{team_a}{rank}',label=f'{team_a} {int(seed_a)}',rank=rank,color=color_dict[region],group=region)
            dot.node(f'{team_b}{rank}',label=f'{team_b} {int(seed_b)}',rank=rank,color=color_dict[region],group=region)
            ## create the winner Node
            dot.node(f'{win_team}{next_rank}',label=f'{win_team} {int(win_seed)}',rank=next_rank,color=color_dict[region])
            
            ## cluster by regions but dont cluster final 4 games
            cust_name=f'cluster_{region}'if region not in ['round5','round6'] else region
            with dot.subgraph(name=cust_name) as c:
                c.attr(label=f'{region}')
                ## link left to right for Visualization like espn
                if region in ['W','X']:
                    c.edge(f'{team_a}{rank}',f'{win_team}{next_rank}',constraint='true',samehead='true')
                    c.edge(f'{team_b}{rank}',f'{win_team}{next_rank}',constraint='true',samehead='true')
                ## link opopsites link espn
                elif region in ['round5','round6']:
                    c.attr(rankdir='LR' ,nodesep='.5', ranksep='.5')

                    c.edge(f'{team_b}{rank}',f'{win_team}{next_rank}',constraint='true',)
                    c.edge(f'{win_team}{next_rank}',f'{team_a}{rank}',constraint='true',)

                ## link Right to left for Visualization like espn
                else:
                    c.edge(f'{win_team}{next_rank}',f'{team_a}{rank}',constraint='true',samehead='true')
                    c.edge(f'{win_team}{next_rank}',f'{team_b}{rank}',constraint='true',samehead='true')
        ## save data and image for analysis
        tournament.to_excel(f'Renders/MM Bracket {year} {score} Sim {self.n_simulations}.xlsx')
        dot.render(f'Renders/MM Bracket {year} {score} Sim {self.n_simulations}', view=True)
        return



color_dict={
    'round6':'green', 
    'round5':'blue',
    'W':'yellow', 
    'X':'purple', 
    'Y':'orange', 
    'Z':'red'
}