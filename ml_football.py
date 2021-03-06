import pandas as pd
import numpy as np
import collections
from sklearn.linear_model import LinearRegression

data = pd.read_csv("RPL.csv", encoding = 'UTF-8', delimiter=';')
data.head()


RPL_2018_2019 = pd.read_csv('Team Name 2018 2019.csv', encoding = 'cp1251')

teamList = RPL_2018_2019['Team Name'].tolist()

deleteTeam = [x for x in pd.unique(data['Team']) if x not in teamList]
for name in deleteTeam:
    data = data[data['Team'] != name]
    data = data[data['Rival'] != name]
data = data.reset_index(drop=True)

def GetSeasonTeamStat(team, season):
    goalScored = 0
    goalAllowed = 0

    gameWin = 0
    gameDraw = 0
    gameLost = 0

    totalScore = 0

    matches = 0
    
    xG = 0
    
    shot = 0
    shotOnTarget = 0
    
    cross = 0
    accurateCross = 0
    
    totalHandle = 0
    averageHandle = 0
    
    Pass = 0
    accuratePass = 0
    
    PPDA = 0

    for i in range(len(data)):
        if (((data['Year'][i] == season) and (data['Team'][i] == team) and (data['Part'][i] == 2)) or ((data['Year'][i] == season-1) and (data['Team'][i] == team) and (data['Part'][i] == 1))):
            matches += 1
                
            goalScored += data['Scored'][i]
            goalAllowed += data['Conceded'][i]

            if (data['Scored'][i] > data['Conceded'][i]):
                totalScore += 3
                gameWin += 1
            elif (data['Scored'][i] < data['Conceded'][i]):
                gameLost +=1
            else:
                totalScore += 1
                gameDraw += 1
            
            xG += data['xG'][i]
            
            shot += data['Shoots'][i]
            shotOnTarget += data['Shoots on target'][i]
            
            Pass += data['Passes'][i]
            accuratePass += data['Accurate passes'][i]
            
            totalHandle += data['Possesion'][i]
            
            cross += data['Crosses'][i]
            accurateCross += data['Accurate crosses'][i]
            
            PPDA += data['PPDA'][i]
            
    averageHandle = round(totalHandle/matches, 3)
    
    return [gameWin, gameDraw, gameLost, 
            goalScored, goalAllowed, totalScore, 
            round(xG, 3), round(PPDA, 3),
            shot, shotOnTarget, 
            Pass, accuratePass,
            cross, accurateCross,
            round(averageHandle, 3)]

returnNames = ["Wins", "Draws", "Losses",
               "\nScored", "Against", "\nPoints",
               "\nxG (season)", "PPDA (season)",
               "\nShoots", "Shoots on target", 
               "\nPasses", "Accurate passes",
               "\nCrosses", "Accurate crosses",
                "\nPossesion (average)"]

for i, n in zip(returnNames, GetSeasonTeamStat("Спартак", 2018)):
        print(i, n)

def GetSeasonAllTeamStat(season):
    annual = collections.defaultdict(list)
    for team in teamList:
        team_vector = GetSeasonTeamStat(team, season)
        annual[team] = team_vector
    return annual

def GetTrainingData(seasons):
    totalNumGames = 0
    for season in seasons:
        annual = data[data['Year'] == season]
        totalNumGames += len(annual.index)
    numFeatures = len(GetSeasonTeamStat('Зенит', 2016))
    xTrain = np.zeros(( totalNumGames, numFeatures))
    yTrain = np.zeros(( totalNumGames ))
    indexCounter = 0
    for season in seasons:
        team_vectors = GetSeasonAllTeamStat(season)
        annual = data[data['Year'] == season]
        numGamesInYear = len(annual.index)
        xTrainAnnual = np.zeros(( numGamesInYear, numFeatures))
        yTrainAnnual = np.zeros(( numGamesInYear ))
        counter = 0
        for index, row in annual.iterrows():
            team = row['Team']
            t_vector = team_vectors[team]
            rivals = row['Rival']
            r_vector = team_vectors[rivals]
           
            diff = [a - b for a, b in zip(t_vector, r_vector)]
            
            if len(diff) != 0:
                xTrainAnnual[counter] = diff
            if team == row['Winner']:
                yTrainAnnual[counter] = 1
            else: 
                yTrainAnnual[counter] = 0
            counter += 1   
        xTrain[indexCounter:numGamesInYear+indexCounter] = xTrainAnnual
        yTrain[indexCounter:numGamesInYear+indexCounter] = yTrainAnnual
        indexCounter += numGamesInYear
    return xTrain, yTrain

years = range(2016,2019)
xTrain, yTrain = GetTrainingData(years)


model = LinearRegression()
model.fit(xTrain, yTrain)

def createGamePrediction(team1_vector, team2_vector):
    diff = [[a - b for a, b in zip(team1_vector, team2_vector)]]
    predictions = model.predict(diff)
    return predictions



team1_name = "Зенит"
team2_name = "Спартак"

team1_vector = GetSeasonTeamStat(team1_name, 2019)
team2_vector = GetSeasonTeamStat(team2_name, 2019)

print ('Possibility that ' + team1_name + ' will win:', createGamePrediction(team1_vector, team2_vector))
print ('Possibility that ' + team2_name + ' will win:', createGamePrediction(team2_vector, team1_vector))
