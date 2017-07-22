# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import sys,random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
        self.pos_list = []


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        self.pos_list.insert(0, gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition())
        if len(self.pos_list)>5:
            self.pos_list.pop()

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentPos = currentGameState.getPacmanPosition()
        newfoodPos = newFood.asList()
        curfoodPos = currentGameState.getFood().asList()
        curghostPos = currentGameState.getGhostPositions()
        newghostPos = successorGameState.getGhostPositions()
        
        #distList =[]
        dist =0;
        if currentPos == newPos:
            return -1000

        if len(newfoodPos) == 0:
            return 1000

        if len(curfoodPos) ==0:
            return 1000
        
        foodDistList =[]
        
        for f in curfoodPos:
            foodDist = manhattanDistance(currentPos,f)
            foodDistList.append(foodDist)
        
        for f in newfoodPos:
            foodDist = manhattanDistance(newPos,f)
            foodDistList.append(foodDist)
        
        closestFoodDist = min(foodDistList)
        dist = 0
        if newPos in curfoodPos:
            dist += 1
        if newPos in newfoodPos:
            dist += 1
        if newPos in currentGameState.getCapsules():
            dist += 1
        if newPos in self.pos_list:
            dist -=1
        # dist -=1
        
        for index,g in enumerate(newghostPos):
            ghostDist = manhattanDistance(newPos,g)
            if ghostDist == 0:
                ghostDist = -10000
            if ghostDist < newScaredTimes[index]:
                dist += (1/ghostDist)
            elif ghostDist < 2:
                dist -= (1/ghostDist)
            elif closestFoodDist != 0:
                dist += (1/closestFoodDist)
            #distList.append(dist)
        #print successorGameState.getScore()
        return dist
        #+ 1/sum(foodDistList)
        #return max(distList)
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # MAX NODE - PACMAN TURN
        
        def maxPlayer(gameState,depth):

            actionsList = gameState.getLegalActions(0)
            bestScore = -sys.maxint
            bestAction = None
            
            #if len(actionsList) == 0:
             #   return (self.evaluationFunction(gameState))
            #if depth == 0:
             #   return (self.evaluationFunction(gameState))
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState))        

            for action in actionsList:
                successorState = gameState.generateSuccessor(0,action)
                newScore = minPlayer(successorState,1,depth)
                #if(newScore > bestScore):
                  #  bestScore= newScore
                bestScore = max(bestScore,newScore)
            return bestScore

        # MIN NODE - GHOSTS TURN
        def minPlayer(gameState, agentIndex, depth):
            actionsList = gameState.getLegalActions(agentIndex)
            bestScore = sys.maxint
            bestAction = None
            numAgents = gameState.getNumAgents()
            
            #if len(actionsList) == 0:
             #   return (self.evaluationFunction(gameState))
            #if depth == 0:
             #   return (self.evaluationFunction(gameState))            
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState))
            
            for action in actionsList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex == numAgents-1):
                    newScore = maxPlayer(successorState,depth-1)
                else:
                    newScore = minPlayer(successorState,agentIndex+1,depth)

                bestScore = min(bestScore,newScore)
            return bestScore

        actionsList = gameState.getLegalActions(0)
        bestScore = -sys.maxint
        bestAction = None
        for action in actionsList:
            successorState = gameState.generateSuccessor(0, action)
            newScore = minPlayer(successorState,1,self.depth)
            oldScore = bestScore
            bestScore = max(bestScore,newScore)
            if(bestScore > oldScore):
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # MAX NODE - PACMAN TURN
        
        def maxPlayer(gameState,depth,alpha,beta):

            actionsList = gameState.getLegalActions(0)
            bestScore = -sys.maxint
            bestAction = None
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState))        

            for action in actionsList:
                if alpha > beta:
                    return bestScore
                successorState = gameState.generateSuccessor(0,action)
                newScore = minPlayer(successorState,1,depth,alpha,beta)
                bestScore = max(bestScore,newScore)
                alpha = max(alpha,newScore)
            return bestScore

        # MIN NODE - GHOSTS TURN
        def minPlayer(gameState, agentIndex, depth,alpha,beta):
            actionsList = gameState.getLegalActions(agentIndex)
            bestScore = sys.maxint
            bestAction = None
            numAgents = gameState.getNumAgents()
            
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState))
            
            for action in actionsList:
                if alpha > beta:
                    return bestScore
                successorState = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex == numAgents-1):
                    newScore = maxPlayer(successorState,depth-1,alpha,beta)
                else:
                    newScore = minPlayer(successorState,agentIndex+1,depth,alpha,beta)

                bestScore = min(bestScore,newScore)
                beta = min(beta,newScore)
            return bestScore

        actionsList = gameState.getLegalActions(0)
        bestScore = -sys.maxint
        bestAction = None
        alpha = -sys.maxint
        beta = sys.maxint
        for action in actionsList:
            successorState = gameState.generateSuccessor(0, action)
            newScore = minPlayer(successorState,1,self.depth,alpha,beta)
            oldScore = bestScore
            bestScore = max(bestScore,newScore)
            if(bestScore > oldScore):
                bestAction = action
            if bestScore < beta:
                alpha = max(alpha,bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # CHANCE PLAYER - PROBABILITY NODE
        def chancePlayer(gameState, agentIndex, depth):
            actionsList = gameState.getLegalActions(agentIndex)
            totalScore = 0
            bestAction = None
            numAgents = gameState.getNumAgents()

            if len(actionsList) == 0:
                return (self.evaluationFunction(gameState))
    
            prob = 1.0/len(actionsList)
            
            for action in actionsList:
                newState = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex == numAgents - 1):
                    newScore = maxPlayer(newState, depth - 1)
                else:
                    newScore = chancePlayer(newState, agentIndex + 1, depth)
                totalScore += prob * newScore
            return totalScore
        # MAX - NODE - EXPECTIMAX
        
        def maxPlayer(gameState,depth):

            actionsList = gameState.getLegalActions(0)
            bestScore = -sys.maxint
            bestAction = None
            
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState))        

            for action in actionsList:
                successorState = gameState.generateSuccessor(0,action)
                newScore = chancePlayer(successorState,1,depth)
                bestScore = max(bestScore,newScore)
            return bestScore

        actionsList = gameState.getLegalActions(0)
        bestScore = -sys.maxint
        bestAction = None
        for action in actionsList:
            successorState = gameState.generateSuccessor(0, action)
            newScore = chancePlayer(successorState,1,self.depth)
            oldScore = bestScore
            bestScore = max(bestScore,newScore)
            if(bestScore > oldScore):
                bestAction = action

        return bestAction        

    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      
      For better evaluation function I have considered the following:
      
      ghostDist: This is the closest ghost distance to the pacman. This is added to the score only if the ghost is less than 3 moves away from pacman. As this distance effects the total score only in this case and when ghost is away from pacman there is no need to consider it.
      
      closestfood: This is the closest food distance to the pacman. This is subtracted from the score as this gives more weightage to further distanced foods thus lowering the overall score.

      total num of food: len(foodList) - This is also taken into account as there seems to be a case where the pacman doesnot eat food as it would make the food distance higher in next turn.

      num of capsules: len(capsules) - This is just an extra parameter to increase the score. I gave less weightage to this as it has effect only when it is near the capsule.

      I adjusted the weights according to the pacman game. These weights work good with the autograder.

    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
      return sys.maxint
    if currentGameState.isLose():
      return -sys.maxint
    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostPositions()
    pacPos = currentGameState.getPacmanPosition()
    closestFood = sys.maxint
    ghostDist = sys.maxint
    score =0
    score = currentGameState.getScore()
    for foodPos in foodList:
        foodDist = util.manhattanDistance(foodPos,pacPos)
        closestFood = min(foodDist,closestFood)

    for ghostPos in ghostList:
        gd = util.manhattanDistance(ghostPos,pacPos)
        ghostDist = min(gd,ghostDist)
    
    capsules = currentGameState.getCapsules()
    score -= closestFood*2
    score -= len(foodList)*100
    score -= len(capsules)*10
    if ghostDist < 3:
        score += ghostDist*4
    return score
    
# Abbreviation
better = betterEvaluationFunction

