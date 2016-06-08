module PilotSCAConst

export PenConflict, PenMinSep, PenCloseness, PenAction, StdDist, MinSepSq, InvVar
export DT, DTI, G, Xmin, Xmax, Ymin, Ymax, Bearingmin, Bearingmax, Speedmin, Speedmax
export RangeDim, ThetaMin, ThetaMax, ThetaDim, RangeMin, RangeMax
export Bearingdim, Speeddim, COCdim, NStates, NActions
export Xs, Ys, Bearings, Speeds, Actions, Thetas, Ranges
export SigmaSpeed, SigmaBank, SigmaBankCOC
export SigmaDim, SigmaWeightNominal, SigmaWeightOffNominal
export Responses, MeanResponseTime, RespondingProb, NonRespondingProb


const PenConflict = 10.0
const PenMinSep = 1000.0
const PenCloseness = 10.0
const PenAction = 0.02

const StdDist = 153.0  # [m]
const MinSepSq = StdDist^2  # [m^2]
const InvVar = 1 / MinSepSq  # [1/m^2]

const DT = 5.0  # [s]
const DTI = 1.0  # [s]
const G = 9.8  # [m/s^2]

const Xmin = -12800.0  # [m]
const Xmax = 12800.0  # [m]
const Ymin = -12800.0  # [m]
const Ymax = 12800.0  # [m]

const RangeMin = 0.
const RangeMax = 12800.
const Ranges = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 12800]
const RangeDim = length(Ranges)

const ThetaMin = -pi  # [rad]
const ThetaMax = pi  # [rad]
const ThetaDim = 37

const Bearingmin = 0.0  # [rad]
const Bearingmax = 2 * pi  # [rad]
const Bearingdim = 37

const Speedmin = 35.0  # [m/s]
const Speedmax = 45.0  # [m/s]
const Speeddim = 3

const Responsedim = 2

const NStates = RangeDim * ThetaDim * Bearingdim * Speeddim^2 * Responsedim^2 + 1
const NActions = 36

const Thetas = collect(linspace(ThetaMin, ThetaMax, ThetaDim))
const Bearings = collect(linspace(Bearingmin, Bearingmax, Bearingdim))
const Speeds = collect(linspace(Speedmin, Speedmax, Speeddim))

const Responses = [0.0, 1.0]  # indicator values: false == 0, true == 1

const MeanResponseTime = 5.0  # [s]
const RespondingProb = DT / (DT + MeanResponseTime)
const NonRespondingProb = 1.0 - RespondingProb

const Actions = [:right20, :right10, :straight, :left10, :left20, :clearOfConflict]

const SigmaSpeed = 4.0  # [m/s]
const SigmaBank = deg2rad(2.0)  # [rad]
const SigmaBankCOC = deg2rad(5.0)  # [rad]

const SigmaDim = 4  # number of dimensions for sigma-point sampling
const SigmaWeightNominal = 1 / 3
const SigmaWeightOffNominal = (1 - SigmaWeightNominal) / (2 * SigmaDim)

end # module PilotSCAConst
