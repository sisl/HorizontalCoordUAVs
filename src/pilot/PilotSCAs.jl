# TODO: consider linked list to avoid concatenation costs

module PilotSCAs

export SCA
export numStates, numActions
export reward, nextStates

export ind2a, state2gridState, gridState2state, index2state, ind2x

export State, Action

using GridInterpolations, DiscreteMDPs

import DiscreteMDPs.DiscreteMDP
import DiscreteMDPs.reward
import DiscreteMDPs.nextStates
import DiscreteMDPs.numStates
import DiscreteMDPs.numActions

using PilotSCAConst

type SCA <: DiscreteMDP
    nStates::Int64
    nActions::Int64
    actions::Vector{Symbol}
    grid::RectangleGrid
    function SCA()
        grid = RectangleGrid(Ranges, Thetas, Bearings, Speeds, Speeds, Responses, Responses)
        return new(NStates, NActions, Actions, grid)
    end # function SCA
end # type SCA

type State
    r::Float64
    th::Float64
    bearing::Float64
    speedOwnship::Float64
    speedIntruder::Float64
    clearOfConflict::Bool
    respondingOwnship::Bool
    respondingIntruder::Bool
end # type State

type Action
    ownship::Symbol
    intruder::Symbol
end # type Action

function numStates(mdp::SCA)    
    return mdp.nStates
end # function numStates

function numActions(mdp::SCA)
    return mdp.nActions
end # function numActions

function reward(mdp::SCA, istate::Int64, iaction::Int64)
    state = State(
        0.0,  # range
        0.0,  # theta 
        0.0,  # bearing
        10.0,  # speedOwnship
        10.0,  # speedIntruder
        true,  # clearOfConflict
        false,  # respondingOwnship
        false)  # respondingIntruder

    if istate < mdp.nStates
        state = gridState2state(ind2x(mdp.grid, istate))
    end # if

    action = ind2a(mdp.actions, iaction)
    return reward(mdp, state, action)
end # function reward


function reward(mdp::SCA, state::State, action::Action)
    reward = 0.0
    if action.ownship != :clearOfConflict
        reward -= PenConflict
    end # if
    
    if action.intruder != :clearOfConflict
        reward -= PenConflict
    end # if
   
    #turnOwnship = getTurnAngle(action.ownship, state.respondingOwnship)
    #turnIntruder = getTurnAngle(action.intruder, state.respondingIntruder)
    turnOwnship = getTurnAngleReward(action.ownship, state.respondingOwnship)
    turnIntruder = getTurnAngleReward(action.intruder, state.respondingIntruder)
    reward -= PenAction * (turnOwnship^2 + turnIntruder^2) 
 
    if !state.clearOfConflict
        minSepSq = getSepSq(state)
        minSepSq = Inf
        rs = []
        for ti = 1:DT / DTI
            minSepSq = min(minSepSq, getSepSq(state))
            state = getNextState(state, action, 0.0, 0.0, DTI)

        end # for ti

        if minSepSq < MinSepSq
            reward -= PenMinSep
        end # if
     
        reward -= PenCloseness * exp(-minSepSq * InvVar)
    end # if

    return reward
end # function reward

function ind2a(actions::Vector{Symbol}, iaction::Int64)
    iOwnship = iaction % length(actions)
    if iOwnship == 0
        iOwnship = 6
    end # if
    
    iIntruder = (iaction - iOwnship) / length(actions) + 1
    return Action(actions[round(Int,iOwnship)], actions[round(Int,iIntruder)])
end # function ind2a

# Returns turn angle corresponding to action in degrees.
function getTurnAngleReward(action::Symbol, responding::Bool)
    if action == :straight
        return 0.0
    elseif !responding
        return 0.0
    elseif action == :clearOfConflict
        return 0.0
    elseif action == :left10
        return 6.
    elseif action == :right10
        return -6.
    elseif action == :left20
        return 12.
    elseif action == :right20
        return -12.
    else
        throw(ArgumentError("illegal action symbol"))
    end # if
end # function getTurnAngleReward

# Returns turn angle corresponding to action in degrees.
function getTurnAngle(action::Symbol, responding::Bool)    
    if !responding
        return 0.0
    elseif action == :clearOfConflict
        return 0.0
    elseif action == :straight
        return 0.0
    elseif action == :left10
        return 6.
    elseif action == :right10
        return -6.
    elseif action == :left20
        return 12.
    elseif action == :right20
        return -12.
    else
        throw(ArgumentError("illegal action symbol"))
    end # if    
end # function getTurnAngle

function getSepSq(state::State)
    if state.clearOfConflict
        return Inf
    else
        return state.r^2
    end # if 
end # function getSepSq

function getNextState(
    state::State,
    action::Action,
    sigmaTurnOwnship::Float64 = 0.0,
    sigmaTurnIntruder::Float64 = 0.0,
    dt::Float64 = DT)

   
    #Initialize new state
    newState = State(
    state.r,
    state.th,
    state.bearing,
    state.speedOwnship,
    state.speedIntruder,
    state.clearOfConflict,
    state.respondingOwnship,
    state.respondingIntruder)

    if !state.clearOfConflict

       #Get Turn Angles
       turnOwnship = deg2rad(getTurnAngle(action.ownship, state.respondingOwnship)) + sigmaTurnOwnship
       turnIntruder = deg2rad(getTurnAngle(action.intruder, state.respondingIntruder)) + sigmaTurnIntruder

       #Initialize New State Variables
       newBearing = state.bearing
       newX = state.r*cos(state.th)
       newY = state.r*sin(state.th)

       #Initialize the straight path cases for intruder and ownship
       bearingChangeOwnship = 0.0
       bearingChangeIntruder = 0.0
       OwnPos = [state.speedOwnship*dt,0.0] #With respect to ownship frame
       IntPos = [state.speedIntruder*dt,0.0] #With respect to intruder frame

       #Rotate intruder change in position into ownship frame
       rotMatInt = [cos(newBearing) -sin(newBearing); sin(newBearing) cos(newBearing)]
       IntPos = rotMatInt*IntPos
       
       if abs(turnIntruder) > 0.00001  # intruder is turning
           gtanInt = G * tan(turnIntruder)
           bearingChangeIntruder = dt * gtanInt / state.speedIntruder #Angle of turn
           radiusIntruder = abs(state.speedIntruder^2 / gtanInt)  # Radius of turn

           #Recalculate change in intruder position with respect to ownship frame
           xInt_local = radiusIntruder * sin(bearingChangeIntruder)    *sign(bearingChangeIntruder)
           yInt_local = radiusIntruder *(1-cos(bearingChangeIntruder)) *sign(bearingChangeIntruder)
           IntPos = rotMatInt * [xInt_local,yInt_local]
       end
       
       if abs(turnOwnship)  > 0.00001  #ownship is turning
           gtanOwn = G * tan(turnOwnship)
           bearingChangeOwnship = dt * gtanOwn / state.speedOwnship #Angle of turn
           radiusOwnship = abs(state.speedOwnship^2 / gtanOwn)  #Radius of turn

           #Recalculate change in ownship position with respect to ownship frame
           xOwn_local = radiusOwnship * sin(bearingChangeOwnship)    *sign(bearingChangeOwnship)
           yOwn_local = radiusOwnship *(1-cos(bearingChangeOwnship)) *sign(bearingChangeOwnship)
           OwnPos = [xOwn_local,yOwn_local]
       end

       #Update new state values
       newX += IntPos[1] - OwnPos[1]
       newY += IntPos[2] - OwnPos[2]
       newBearing += bearingChangeIntruder-bearingChangeOwnship
       newBearing = norm_angle(newBearing)

       #Check if Clear of Conflict conditions are met, or else update the new state
       if newX < Xmin || newX > Xmax || newY < Ymin || newY > Ymax
           newState.clearOfConflict = true
       else
           newState.r = sqrt(newX^2 + newY^2)
           newState.bearing = newBearing

           #Must write theta in terms of rotated coordinate frame!
           newState.th = norm_angle(atan2(newY, newX)-bearingChangeOwnship, true)
       end # if
   end # if
   return newState
end # function getNextState

#=
This function norms an angle to between 0 and 360 unless the 
flag neg_pi_to_pi is set to true, in which case it norms it
to between -180 and 180
=#
function norm_angle(angle::Float64, neg_pi_to_pi::Bool = false)
    new_angle = ((angle % (2 * pi)) + 2 * pi) % (2 * pi)
    if neg_pi_to_pi
        if new_angle > pi
            new_angle -= 2 * pi
        end
    end
    return new_angle
end # function norm_angle

# Returns next states and associated transition probabilities.
function nextStates(mdp::SCA, istate::Int64, iaction::Int64)
    if istate == mdp.nStates
        return [mdp.nStates], [1.0]
    end # if

    state = gridState2state(ind2x(mdp.grid, istate))
    action = ind2a(mdp.actions, iaction)

    if action.ownship == :clearOfConflict && action.intruder == :clearOfConflict
        return sigmaSample(mdp, state, action)
    elseif action.ownship != :clearOfConflict && action.intruder == :clearOfConflict
        if state.respondingOwnship
            return sigmaSample(mdp, state, action)
        else  # !state.respondingOwnship
        
            # case 1: ownship doesn't respond
            nonresponsiveIndices, nonresponsiveProbs = sigmaSample(mdp, state, action)

            # case 2: ownship responds
            state.respondingOwnship = true
            responsiveIndices, responsiveProbs = sigmaSample(mdp, state, action)
            
            return [nonresponsiveIndices; responsiveIndices],
                   [nonresponsiveProbs * NonRespondingProb; responsiveProbs * RespondingProb]
        end # if
    elseif action.ownship == :clearOfConflict && action.intruder != :clearOfConflict

        if state.respondingIntruder
            return sigmaSample(mdp, state, action)
        else  # !state.respondingIntruder

            # case 1: intruder doesn't respond
            nonresponsiveIndices, nonresponsiveProbs = sigmaSample(mdp, state, action)

            # case 2: intruder responds
            state.respondingIntruder = true
            responsiveIndices, responsiveProbs = sigmaSample(mdp, state, action)
            
            return [nonresponsiveIndices; responsiveIndices],
                   [nonresponsiveProbs * NonRespondingProb; responsiveProbs * RespondingProb]
        end # if    
    else  # action.ownship != :clearOfConflict && action.intruder != :clearOfConflict

        if state.respondingOwnship && state.respondingIntruder
            return sigmaSample(mdp, state, action)
        elseif !state.respondingOwnship && state.respondingIntruder

            # case 1: ownship doesn't respond
            nonresponsiveIndices, nonresponsiveProbs = sigmaSample(mdp, state, action)

            # case 2: ownship responds
            state.respondingOwnship = true
            responsiveIndices, responsiveProbs = sigmaSample(mdp, state, action)
            
            return [nonresponsiveIndices; responsiveIndices],
                   [nonresponsiveProbs * NonRespondingProb; responsiveProbs * RespondingProb]
        elseif state.respondingOwnship && !state.respondingIntruder

            # case 1: intruder doesn't respond
            nonresponsiveIndices, nonresponsiveProbs = sigmaSample(mdp, state, action)

            # case 2: intruder responds
            state.respondingIntruder = true
            responsiveIndices, responsiveProbs = sigmaSample(mdp, state, action)
            
            return [nonresponsiveIndices; responsiveIndices],
                   [nonresponsiveProbs * NonRespondingProb; responsiveProbs * RespondingProb]        
        else  # !state.respondingOwnship && !state.respondingOwnship

            # case 1: both ownship and intruder don't respond
            noresponseIndices, noresponseProbs = sigmaSample(mdp, state, action)

            # case 2: ownship responds but not intruder
            state.respondingOwnship = true
            ownshipResponseIndices, ownshipResponseProbs = sigmaSample(mdp, state, action)

            # case 3: both ownship and intruder respond
            state.respondingIntruder = true
            bothResponseIndices, bothResponseProbs = sigmaSample(mdp, state, action)

            # case 4: intruder responds but not ownship
            state.respondingOwnship = false
            intruderResponseIndices, intruderResponseProbs = sigmaSample(mdp, state, action)

            return [
                noresponseIndices;
                ownshipResponseIndices;
                bothResponseIndices;
                intruderResponseIndices], 
                [
                noresponseProbs * NonRespondingProb * NonRespondingProb;
                ownshipResponseProbs * RespondingProb * NonRespondingProb;
                bothResponseProbs * RespondingProb * RespondingProb;
                intruderResponseProbs * NonRespondingProb * RespondingProb]
        end # if
    end # if
end # function nextStates

function sigmaSample(mdp::SCA, state::State, action::Action)
    nominalIndices, nominalProbs = nextStatesSigma(mdp, state, action)
    speedIndices, speedProbs = sigmaSpeed(mdp, state, action)
    bankIndices, bankProbs = sigmaBank(mdp, state, action)
    return [
            nominalIndices;
            speedIndices;
            bankIndices], 
        [
            nominalProbs * SigmaWeightNominal;
            speedProbs * SigmaWeightOffNominal;
            bankProbs * SigmaWeightOffNominal]
end # function sigmaSample

function sigmaSpeed(mdp::SCA, state::State, action::Action)
    # negative sigma
    state.speedOwnship -= SigmaSpeed
    negIndicesOwnship, negProbsOwnship = nextStatesSigma(mdp, state, action)

    # positive sigma
    state.speedOwnship += 2 * SigmaSpeed
    posIndicesOwnship, posProbsOwnship = nextStatesSigma(mdp, state, action)

    # restore original
    state.speedOwnship -= SigmaSpeed

    # negative sigma
    state.speedIntruder -= SigmaSpeed
    negIndicesIntruder, negProbsIntruder = nextStatesSigma(mdp, state, action)

    # positive sigma
    state.speedIntruder += 2 * SigmaSpeed
    posIndicesIntruder, posProbsIntruder = nextStatesSigma(mdp, state, action)

    # restore original
    state.speedIntruder -= SigmaSpeed

    return [negIndicesOwnship; posIndicesOwnship; negIndicesIntruder; posIndicesIntruder],
           [negProbsOwnship; posProbsOwnship; negProbsIntruder; posProbsIntruder]
end # function sigmaSpeed

function sigmaBank(mdp::SCA, state::State, action::Action)
    sigmaBankVal = SigmaBank
    
    if !state.respondingOwnship || action.ownship == :clearOfConflict
        sigmaBankVal = SigmaBankCOC
    end # if

    # negative sigma
    negIndicesOwnship, negProbsOwnship = nextStatesSigma(mdp, state, action, -sigmaBankVal, 0.0)

    # positive sigma
    posIndicesOwnship, posProbsOwnship = nextStatesSigma(mdp, state, action, sigmaBankVal, 0.0)

    if !state.respondingIntruder || action.intruder == :clearOfConflict
        sigmaBankVal = SigmaBankCOC
    end # if

    # negative sigma
    negIndicesIntruder, negProbsIntruder = nextStatesSigma(mdp, state, action, 0.0, -sigmaBankVal)

    # positive sigma
    posIndicesIntruder, posProbsIntruder = nextStatesSigma(mdp, state, action, 0.0, sigmaBankVal)

    return [negIndicesOwnship; posIndicesOwnship; negIndicesIntruder; posIndicesIntruder], 
           [negProbsOwnship; posProbsOwnship; negProbsIntruder; posProbsIntruder]
end # function sigmaBank

function nextStatesSigma(
        mdp::SCA,
        state::State,
        action::Action,
        sigmaTurnOwnship::Float64 = 0.0,
        sigmaTurnIntruder::Float64 = 0.0)
    
    trueNextState = getNextState(state, action, sigmaTurnOwnship, sigmaTurnIntruder)
    gridNextState = state2gridState(trueNextState)
    
    if trueNextState.clearOfConflict
        return [mdp.nStates], [1.0]
    else
        return interpolants(mdp.grid, gridNextState)
    end # if
end # function nextStatesSigma

function state2gridState(state::State)
    # indicator values for responsiveness: false == 0, true == 1
    respondingOwnship = 0.0
    respondingIntruder = 0.0
    
    if state.respondingOwnship
        respondingOwnship = 1.0
    end # if

    if state.respondingIntruder
        respondingIntruder = 1.0
    end # if

    return [
        state.r,
        state.th,
        #state.x,
        #state.y,
        state.bearing,
        state.speedOwnship,
        state.speedIntruder,
        respondingOwnship,
        respondingIntruder]
end # function state2gridState

function index2state(mdp::SCA, stateIndices::Vector{Int64})
    states = Array(State, length(stateIndices))
    
    for index = 1:length(stateIndices)
        states[index] = gridState2state(ind2x(mdp.grid, index))
    end # for index
    
    return states   
end # function index2state

function gridState2state(gridState::Vector{Float64})    
    # indicator values for responsiveness: false == 0, true == 1
    respondingOwnship = false
    respondingIntruder = false
    
    if gridState[6] == 1.0
        respondingOwnship = true
    end # if

    if gridState[7] == 1.0
        respondingIntruder = true
    end # if

    return State(
        gridState[1],  # range # x
        gridState[2],  # thetat # y
        gridState[3],  # bearing
        gridState[4],  # speedOwnship
        gridState[5],  # speedIntruder
        false,  # clearOfConflict
        respondingOwnship,  # respondingOwnship
        respondingIntruder)  # respondingIntruder
end # function gridState2state

end # module PilotSCAs
