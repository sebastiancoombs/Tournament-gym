import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
# from bracketology_rl import Bracket
# from bracketgym import TournamentEnv

import gymnasium as gym
from pprint import pprint
import warnings
import numpy as np
from tourney_tools import * 
import re
warnings.filterwarnings('ignore')
from IPython.display import display
import random
from tqdm.autonotebook import tqdm
import traceback
import copy
copy.deepcopy
import graphviz
class Team():

    """
    A team that is in a Game or a Bracket
    
    Attributes
    ----------
    name : str
        Name of the school (or abbreviation)
    seed : int
        Seed of the team in the tournament (1-16)
    stats : (dict)
        A dictionary with other information about the team, like season stats
    """
    def __init__(self, name, seed,stats=None):
        """
        Parameters
        ----------
        name : str
            The name of the school for the team.
        seed : int
            The seed of the team in the bracket.
        """
        if type(name) != str: 
            name=str(name)

        if type(seed) != int: 
            seed=int(re.sub(r'[A-Za-z]','',seed))
            
        self.name = name
        self.seed = seed
        self.stats = stats
        self.slot = None
        self.n_wins=0
        
    
    def update_stats(self,stats):
        self.stats=stats

    def __repr__(self):
        return f"<{self.slot}: {self.seed} {self.name}>"
    
class TournmentNode():
    "Class to represent a Game has a slot for the high team and low team to fill the bracket"
    def __init__(self,slot,round,display_true=False):
        ## dictionary pairing the slot the team came from and Team object
        self.display_true=display_true
        self.slot=slot
        self.round=round

        ### simulation chosen_winners and matchups
        self.winner=None
        self.teams={}

        # if round==0:
            # self._starting_node=True
        # if you have the real_ reults we save what actually happened
        self._true_winner=None
        self._true_teams={}
        self.next_node=None
        self.next_slot=None
        self.previous_nodes={}
        self.team_positions={}
        return 
    
    def advance_team(self,team_name):
        team=self.teams[team_name]
        team.slot=self.slot
        team.n_wins+=1
        self.winner=team
        self.next_node.add_team(team)

    def add_result(self,team_ob,matchup=None):
        self._true_winner=team_ob
        if matchup:
            for m in matchup:
                self._true_teams[m.name]=m
        
    def add_team(self,team):
        self.teams[team.name]=team
        self.team_positions[team.slot]=team.name
        self.update_game()
    
    def link_previous_node(self,previous_node ,verbose=False):

        previous_node.next_node=self
        previous_node.next_slot=self.slot
        self.previous_nodes[previous_node.slot]=previous_node
        if verbose:
            print(f'link {previous_node.slot} -> {self.slot}')
        if len(self.previous_nodes)>2:
            print(previous_node.slot,self.slot)
            print(self.previous_nodes)
            raise ValueError('You cant have more than two slots feeding a node')
        
    def update_game(self):
        ## after a winner has been choosen update the teams or fill with the starting teams
        if len(self.previous_nodes)==0:
            self.teams=self._true_teams
            self.team_positions={str(t.seed):t.name for t in self._true_teams.values()}
        
        elif len(self.previous_nodes)==1 and self.round==1 and self._true_teams:
            previous_teams=[]
            for game in self.previous_nodes.values():
                for name,team in game.teams.items():
                    previous_teams.append(name)
            
            other_team=[t for t in self._true_teams if t not in previous_teams][0]
            self.teams[other_team]=self._true_teams[other_team]

        else:
            for slot,node in self.previous_nodes.items():
                winner=node.winner
                if winner:
                    self.teams[winner.name]=winner
                    self.team_positions[winner.slot]=winner.name

        if self.round in ['1','0',0,1]:
            sub_slots=['a','b']
            for i,team in enumerate(self.teams.values()):
                if team.slot==None:
                    team.slot=self.slot+sub_slots[i]
                    self.team_positions[team.slot]=team.name
                    
                elif team.slot in self.team_positions:
                        team.slot=self.slot+sub_slots[i]
                        self.team_positions[team.slot]=team.name
                else:
                    self.team_positions[team.slot]=team.name

    def reset_game(self):
        self.winner=None
        if len(self.previous_nodes)==1 and self.round==1:
            previous_teams=[]
            for game in self.previous_nodes.values():
                for name,team in game.teams.items():
                    previous_teams.append(name)
            
            other_team=[t for t in self._true_teams if t not in previous_teams][0]
            unk_team=[t for t in self._true_teams if t not in previous_teams][0]

            self.teams[other_team]=self._true_teams[other_team]
            self.teams.pop(unk_team)
        else:
            if self.round!=0:
                for team in self.teams():
                    self.teams.pop(team)
            
    def game_info(self):
        self.update_game()
        match_up=self.get_matchup()
        top_team=self.teams[match_up[0]]
        bottom_team=self.teams[match_up[1]]

        info={'top_team_name':top_team.name,
        'bottom_team_name':bottom_team.name,
        'top_team_slot':top_team.slot,
        'bottom_team_slot':bottom_team.slot,
        'top_team_seed':top_team.seed,
        'bottom_team_seed':bottom_team.seed,}

        return info

    def get_matchup(self):
        'returns a asorted list of team names based on the slot the team was in'
        if self.round<=1:
            match_up=sorted([t for t in self.teams])

        else:
            positions=sorted([pos for pos in self.team_positions])
            match_up=[self.team_positions[pos] for pos in positions]
        return match_up

    def predict_winner(self,action):
        self.update_game()
        if action not in [0,1]:
            raise ValueError('actions must be binary 0 or 1')
        
        match_up=self.get_matchup()

        winner_name=match_up[action]
        self.advance_team(winner_name)

        if self.winner.name==self._true_winner.name:
            score=self.round**2
            if score==0:
                score=.5
        else:
            score=0
        return float(score)
    
    def get_winning_action(self):
        self.update_game()
        match_up=self.get_matchup()
        if self._true_winner.name==match_up[0]:
            action=0
        else: action=1
        return action

    def advance_winner(self):
        self.advance_team(self._true_winner.name)
        score=self.round**2
        return score
    
    def get_game_stats(self):
        self.update_game()
        stats=[]
        ## this step is crucial for setting the top and bottom side of the bracket
        match_up=self.get_matchup()
        ## grab the stats
        for team_name in match_up:
            team=self.teams[team_name]
            team_stats=team.stats.select_dtypes(np.number).values.flatten()
            stats.append(team_stats)
        stats=np.array(stats).flatten()
        return stats

    def __repr__(self):

        rep_string=[f"<{self.slot}: "]

        if self.display_true:

            match = ' vs. '.join([t.__repr__() for t in self._true_teams.values()])
            rep_string.append( match + f': {self._true_winner.name}')

        elif len(self.teams)>1:

            match = ' vs. '.join([t.__repr__() for t in self.teams])
            rep_string.append(match +f': {self.winner}')

        elif len(self.teams)==1:

            match = ' vs. '.join([t.__repr__() for t in self.teams]+['TBD'])
            rep_string.append(match +f': {self.winner}')        

        elif len (self.previous_nodes.keys())>0:
            rep_string.append(f'{[c for c in self.previous_nodes.keys()] }')
        
        else:
            rep_string.append('|')

        if self.next_slot:
            end_str=f" -> {self.next_slot}>"
            rep_string.append(end_str)

        final=' '.join(rep_string)
        return final

class TournamentEnv(gym.Env):

    def __init__(self,season,
                 team_stats=None,
                 verbose=False,
                 display_results=False,
                 discrete=True,
                 shuffle=True,
                 MW='M',
                 loading_bar=False,
                 reward_on_round_end=True,
                 exclude_seasons=[],
                 historypath=None):
        self.valid_years = [y for y in range(2003, 2025)]
        self.valid_years.remove(2020)
        self.reward_on_round_end=reward_on_round_end
        self.shuffle=shuffle
        if exclude_seasons:
            for s in exclude_seasons:
                if s in self.valid_years:
                    self.valid_years.remove(s)
        self.verbose=verbose
        self.discrete=discrete
        self.MW=MW
        self.display_results=display_results
        self.season=season


        if historypath==None:
            self.historypath = f'bracketology_rl/data/{MW}better_brackets2024.json'
        else:
            self.historypath = historypath

        
        self.check_season(season)
        with open(self.historypath, 'r') as f:
            brackets_dict = json.load(f)
        tournament=copy.deepcopy(brackets_dict[str(season)])
        self.slots=tournament.get('structure')
        self.team_map=tournament.get('teams')
        self.results=tournament.get('results')
        self.matchups=tournament.get('matchups')
        self.team_data=team_stats         
        self.winner=None
        self.bracket={}
        self.game_step_map={}
        self.n_rounds=None

        self.current_step=0
        self.current_round=0
        self.last_round=0
        self.score=0
        self.round_score=0
        self.n_correct=0
        
        self.steps=None
        self.max_step=None
        self.n_simulations=0

        if loading_bar==True:
            self.games_bar=tqdm(range(10000))
            self.games_bar.set_description(f'Reset! Sim {self.n_simulations} {self.__repr__()}')
        else:self.games_bar=None
        self.loading_bar=loading_bar

        obs,info=self.reset(season=season)

        self.observation_space = gym.spaces.Box(
                                                low=-float('Inf'), 
                                                high=float('Inf'),
                                                shape=obs.shape, 
                                                dtype=np.float32
                                                )
        if self.discrete:
            self.action_space = gym.spaces.Discrete(2)## pick between two teams in the round
        else:
            self.action_space = gym.spaces.Box(0,1,
                                               shape=(2,)
                                               )
        return 


    def step(self,action):
        terminated=False
        truncated=False

        round, slot=self.steps.loc[self.current_step]
        game=self.bracket[round][slot]
        action=self.check_action(action)
        reward=game.predict_winner(action)
        self.score+=reward
        self.round_score+=reward
        if reward>0:

            self.n_correct+=1
        
        self.last_round=int(round)
        ## step to the next round 
        self.current_step+=1
        terminated,truncated=self.check_done()
        next_round, next_slot=self.steps.loc[self.current_step]
        if not terminated:
            self.current_round=int(next_round) 

            next_game=self.bracket[next_round][next_slot]
            observation=next_game.get_game_stats()
            if self. reward_on_round_end:
                if round != next_round:
                    reward=self.round_score
                else:
                    reward=0.1
            

        else:
            observation=np.zeros(self.observation_space.shape)


        info=self.get_info(done=terminated)

        

        return observation,reward,terminated,truncated,info
        
    def reset(self,seed=None,new_season=True,season=None):
        if new_season:
            if not season:
                season=random.choice(self.valid_years)
            else:
                self.check_season(season)
            
        self.season=season
        with open(self.historypath, 'r') as f:
            brackets_dict = json.load(f)
        tournament=copy.deepcopy(brackets_dict[str(season)])
        self.slots=tournament.get('structure')
        self.team_map=tournament.get('teams')
        self.results=tournament.get('results')
        self.matchups=tournament.get('matchups')

        self.team_stats=self.team_data[self.team_data['Season']==season]     
        self.max_step=0
        self.build_bracket()
        self.current_step=0
        
        self.score=0
        self.round_score=0
        self.n_correct=0
        
        round, slot=self.steps.loc[self.current_step]
        self.current_round=round
        self.last_round=self.current_round
        game=self.bracket[round][slot]
        observation=game.get_game_stats()
        info=self.get_info()

        self.n_simulations+=1

        if self.loading_bar==True:
            self.games_bar.update(1)
            self.games_bar.set_description(f'Reset! Sim {self.n_simulations} Bracket for year {self.season}')

        return observation,info
    
    def get_info(self,done=False):
        if not done:
            round, slot=self.steps.loc[self.current_step]

            game=self.bracket[round][slot]
            game_info=game.game_info()
            
        else:
            round, slot=self.steps.loc[self.current_step-1]
            Champ=self.bracket[round][slot]
            game_info={
                'True_Champ':Champ._true_winner.name,
                'True_Champ_seed':Champ._true_winner.seed,
                'Chosen_Champ':Champ.winner.name,
                'Chosen_Champ_seed':Champ.winner.seed,
                }
        info={
                'season':self.season,
                'step':self.current_step,
                'max step':self.max_step,
                'num_correct':self.n_correct,
                'round':round,
                'slot':slot,
                'score':self.score,
                'round_score':self.round_score,

                }
        
        info.update(game_info)

        return info

    def check_action(self ,action):
        if self.discrete:
            return action
        #2 actions for each game pick the top or bottom team
        else:
            return int(np.argmax(action))

    def cheat_action(self):
        round, slot=self.steps.loc[self.current_step]
        game=self.bracket[round][slot]
        correct_action=game.get_winning_action()
        if not self.discrete:
            actions=np.zeros(self.action_space.shape)
            actions[correct_action]=1
            correct_action=actions
            
        return correct_action

    def check_season(self,season):
        if season not in self.valid_years:
            raise ValueError(f"Picked season:{season} Season must be between 2003 and 2023, the 2020 tournament was not played ")
        
    def check_done(self):
        term=False
        if self.current_step>self.max_step:
            term=True
        
        trunc=self.check_bracket_busted()
        if trunc:
            term=True

        return term,trunc
    
    def check_bracket_busted(self):
        trunc=False
        ## on each round scheck you score
        if int(self.last_round)!=int(self.current_round):
            # print(type(self.round_score),type(self.max_round),type(self.current_round))
            if (self.round_score==0) and (int(self.max_round)>int(self.current_round)) and (int(self.current_round)>1):
                trunc=True
            else:
                self.round_score=0
        return trunc

    def get_n_rounds(self):
        teams=self.n_teams
        n_rounds=1
        while teams>1:
            self.rounds[n_rounds]={}
            teams=teams/2
            n_rounds+=1
        return n_rounds
    
    def build_bracket(self,add_results=True):
        step_id=0
        self.bracket={}
        self.game_step_map={}
        add_results=False if self.season==2024 else True
        for r,round in self.slots.items():
            if r != '7':
                self.bracket[r]={}
                for slot, next_slot in round.items():


                    if next_slot in self.bracket[r]:
                        node=self.bracket[r][next_slot]                    

                    else:
                        node=TournmentNode(next_slot,round=int(r))
                        self.bracket[r][next_slot]=node
                        self.game_step_map[step_id]=[r,next_slot]
                        step_id+=1

        
                    if add_results:
                        match=self.matchups[r][next_slot]
                        winner_name=self.results[r][next_slot]
                        loser_name=[c for c in match if c !=winner_name][0]
                        winner_ob=Team(name=winner_name,seed=self.team_map[winner_name])
                        if loser_name in self.team_map:
                            loser_ob=Team(name=loser_name,seed=self.team_map[loser_name])
                        else:
                            loser_ob=Team(name=loser_name,seed=12)

                        node.add_result(winner_ob,[winner_ob,loser_ob])
        self.max_step=step_id

        self.link_nodes()
        self.update_bracket()
        self.add_team_stats()
        self.add_winning_node()
        self.shuffle_rounds()

    def add_winning_node(self):
        last_round=max(self.bracket)
        champ_round=int(max(self.bracket))+1

        self.max_round=champ_round

        champ_game=[c for c in self.bracket[last_round].values()][0]
        winner_node=TournmentNode(slot='CHAMP',round=champ_round)
        winner_node.link_previous_node(champ_game)
        if self.results!=None:
            winner_name=self.results.get('7')[0]
            if winner_name:
                
                self.winner=winner_name
                winner_ob=Team(name=winner_name,seed=self.team_map[winner_name])
                winner_node.add_result(winner_ob)
            winner_ob.slot='CHAMP'
            self.bracket[str(champ_round)]=winner_ob
            self.max_step=max(self.game_step_map)
            self.game_step_map[self.max_step+1]='Champ'

    def link_nodes(self):
        for r_num, round  in self.bracket.items():
            next_r_num=str(int(r_num)+1)
            if next_r_num in self.bracket and int(next_r_num)<= int(max(self.bracket)):
                next_slots=self.slots[next_r_num]
                next_round= self.bracket[next_r_num]

                for slot, node in round.items():
                    next_slot=next_slots[slot]
                    next_node=next_round[next_slot]
                    next_node.link_previous_node(node,verbose=self.verbose)
                
    def add_team_stats(self):
        data=self.team_stats
        for r_num,round in self.bracket.items():
            if r_num in ['0','1']:
                for game in round.values():
                    for team in game.teams.values():
                        team_data=data[data.TeamName==team.name]
                        team.update_stats(team_data)
                        if self.verbose:
                            print(team.name)
                            display(team_data)
        return

    def update_bracket(self):
        for r_num,round in self.bracket.items():
            if r_num!='7':
                for slot,game in round.items():
                    game.update_game()
                
    def reset_bracket(self):
        for round in self.bracket.values():
            for game in round.values():
                game.reset_game()

    def shuffle_rounds(self):
        steps=pd.DataFrame().from_dict(self.game_step_map).T
        
        steps.columns=['Round','Slot']
        self.steps=steps
        shuffler=[]
        for round,data in steps.groupby('Round'):
            ids=[c for c in data.index]
            random.shuffle(ids)
            [shuffler.append(i) for i in ids]
        if self.shuffle:
            self.steps=steps.loc[shuffler].reset_index(drop=True)

    def render(self):
        graph = graphviz.Digraph(f'March Madness Bracket {self.season}',
                                    engine='dot',
                                    node_attr={'shape': 'rounded',
                                            'color': 'lightblue2',
                                            'fixedsize':'false', 
                                            'width':'1',
                                            'style':'filled',

                                            })
        # Set Graphviz attributes
        graph.attr(label=f'March Madness Bracket {self.season}')
        # graph.attr(layout="neato",)
        graph.attr(rankdir='RL', nodesep='.2', ranksep='.3', splines='line',size='40,40')
        graph.graph_attr.update(landscape="false")   
        def bracket_waterfall(game,color):
            slot=game.slot
            finals= [4,5,6,7]
            round=game.round

            region=slot[:3]
            
            cust_name=f'Cluster_{region}'
            rotate_dict={
                    'R6C':'TB', 
                    'R5':'TB',
                    'W':'RL', 
                    'Y':'RL', 
                    'X':'LR', 
                    'Z':'LR'
                    }
            
            with graph.subgraph(name=cust_name,) as c:
                    c.graph_attr.update(label= region )
                    # c.graph_attr.update(orientation= rotate_dict[region])
                    game_params={
                        'label':f'{game.winner.seed} {game.winner.name}',
                        'color':color,
                        'rank':f'{game.round}',
                            'group':region
                            }
                    c.node(slot, **game_params)
                    c.edge(slot,game.next_slot,constraint='true',samehead='true')
                    # print(game)
                    if round>1:
                            for prev_game in game.previous_nodes.values():
                                    if game. winner.name==prev_game.winner.name:
                                            color='green'
                                    else:
                                            color='red'
                                    bracket_waterfall(prev_game,color)
                            
                    else:
                            for team in game.teams.values():
                                    if game. winner.name==team.name:
                                            color='green'
                                    else:
                                            color='red'
                                    team_params={
                                                    'label':f'{team.seed} {team.name}',
                                                    'color':color,
                                                    'rank':f'{game.round-1}',
                                                    'group':region

                                                    }
                                    
                                    c.node(f'{team.name}{slot}',**team_params )
                                    c.edge(f'{team.name}{slot}',slot,)


        final_game=self.bracket['6']['R6CH']
        bracket_waterfall(final_game,color='green')
        graph.render(f'march_madness_bracket {self.season}', view=True)

    def __repr__(self):

        return f"<NCAA {self.MW} March Madness {self.season} {self.winner}>"

