import * as ort from 'onnxruntime-web';

import { SolverMove, SolverResult } from './BfidaSolver';
import { BoardShape, shapes } from '../shapes';

const MODEL_PATH = '/static/models/dqn_kongming_cross.onnx';
const MAX_DURATION_MS = 60000;
const ORT_WASM_BASE_PATH = '/static/onnx';

type ActionDef = {
  index: number;
  fromIndex: number;
  toIndex: number;
  jumpIndex: number;
  move: SolverMove;
};

type ActionEval = {
  def: ActionDef;
  nextState: bigint;
  qValue: number;
  used: boolean;
  queued: boolean;
};

type CandidateEntry = {
  action: ActionEval;
  pathMoves: SolverMove[];
  stateHistory: bigint[];
};

export type AiSolverOptions = {
  maxDurationMs?: number;
  abortSignal?: AbortSignal;
};

const crossShape: BoardShape = (() => {
  const shape = shapes.find(entry => entry.id === 'cross');
  if (!shape) {
    throw new Error('Cross shape definition missing for AI solver.');
  }
  return shape;
})();

const holeToIndex = new Map<string, number>();
const bitMasks: bigint[] = [];
crossShape.holes.forEach((hole, index) => {
  holeToIndex.set(hole, index);
  bitMasks.push(1n << BigInt(index));
});

const actionDefs: ActionDef[] = (() => {
  const defs: ActionDef[] = [];
  let actionIndex = 0;
  crossShape.allowedMoves.forEach((targets, from) => {
    const fromIndex = holeToIndex.get(from);
    if (fromIndex === undefined) return;
    targets.forEach((jump, to) => {
      const toIndex = holeToIndex.get(to);
      const jumpIndex = holeToIndex.get(jump);
      if (toIndex === undefined || jumpIndex === undefined) {
        return;
      }
      defs.push({
        index: actionIndex++,
        fromIndex,
        toIndex,
        jumpIndex,
        move: { from, to, jump },
      });
    });
  });
  return defs;
})();

const centerIndex = holeToIndex.get(crossShape.empty) ?? 0;

let sessionPromise: Promise<ort.InferenceSession> | null = null;

ort.env.wasm.wasmPaths = {
  wasm: `${ORT_WASM_BASE_PATH}/ort-wasm-simd-threaded.jsep.wasm`,
  mjs: `${ORT_WASM_BASE_PATH}/ort-wasm-simd-threaded.jsep.mjs`,
};

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
  }
  return sessionPromise;
}

function toStateBits(pegs: Set<string>): bigint {
  let state = 0n;
  pegs.forEach(cell => {
    const index = holeToIndex.get(cell);
    if (index !== undefined) {
      state |= bitMasks[index];
    }
  });
  return state;
}

function hasPeg(state: bigint, index: number): boolean {
  return (state & bitMasks[index]) !== 0n;
}

function applyAction(state: bigint, action: ActionDef): bigint {
  let next = state;
  next &= ~bitMasks[action.fromIndex];
  next &= ~bitMasks[action.jumpIndex];
  next |= bitMasks[action.toIndex];
  return next;
}

function isSolved(state: bigint): boolean {
  return popCount(state) === 1 && hasPeg(state, centerIndex);
}

function popCount(value: bigint): number {
  let count = 0;
  let current = value;
  while (current) {
    current &= current - 1n;
    count += 1;
  }
  return count;
}

function isLegal(state: bigint, action: ActionDef): boolean {
  return hasPeg(state, action.fromIndex) && hasPeg(state, action.jumpIndex) && !hasPeg(state, action.toIndex);
}

function historyIncludes(history: bigint[], target: bigint): boolean {
  return history.some(entry => entry === target);
}

function stateToTensor(state: bigint): ort.Tensor {
  const buffer = new Float32Array(crossShape.holes.length);
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] = hasPeg(state, i) ? 1 : 0;
  }
  return new ort.Tensor('float32', buffer, [1, buffer.length]);
}

class CandidateQueue {
  private entries: CandidateEntry[] = [];

  public push(entry: CandidateEntry): void {
    this.entries.push(entry);
  }

  public pop(): CandidateEntry | undefined {
    if (!this.entries.length) return undefined;
    let bestIndex = 0;
    let bestValue = this.entries[0].action.qValue;
    for (let i = 1; i < this.entries.length; i++) {
      if (this.entries[i].action.qValue > bestValue) {
        bestValue = this.entries[i].action.qValue;
        bestIndex = i;
      }
    }
    const [entry] = this.entries.splice(bestIndex, 1);
    return entry;
  }

  public clear(): void {
    this.entries = [];
  }
}

export class AiSolverAbortError extends Error {
  constructor() {
    super('AI solver aborted');
    this.name = 'AiSolverAbortError';
  }
}

export async function solveCrossWithAi(
  pegPositions: Set<string>,
  options: AiSolverOptions = {},
): Promise<SolverResult> {
  if (!crossShape) {
    throw new Error('Cross shape unavailable for AI solver.');
  }
  const signal = options.abortSignal;
  const startTime = performance.now();
  const deadline = startTime + (options.maxDurationMs ?? MAX_DURATION_MS);
  const session = await getSession();
  const stateCache = new Map<bigint, ActionEval[]>();
  const queue = new CandidateQueue();
  const initialState = toStateBits(pegPositions);
  const initialHistory: bigint[] = [initialState];
  const initialPath: SolverMove[] = [];
  let bestPath: SolverMove[] = [];
  let bestPegCount = popCount(initialState);
  let nodesExplored = 0;

  const throwIfAborted = () => {
    if (signal?.aborted) {
      throw new AiSolverAbortError();
    }
  };

  const loadActions = async (state: bigint): Promise<ActionEval[]> => {
    let cached = stateCache.get(state);
    if (cached) {
      return cached;
    }
    throwIfAborted();
    nodesExplored += 1;
    const tensor = stateToTensor(state);
    const output = await session.run({ state: tensor });
    const tensorOutput = output.q_values as ort.Tensor;
    const data = tensorOutput.data as Float32Array;
    const evals: ActionEval[] = [];
    actionDefs.forEach(action => {
      if (!isLegal(state, action)) return;
      const nextState = applyAction(state, action);
      evals.push({
        def: action,
        nextState,
        qValue: data[action.index] ?? Number.NEGATIVE_INFINITY,
        used: false,
        queued: false,
      });
    });
    evals.sort((a, b) => b.qValue - a.qValue);
    stateCache.set(state, evals);
    return evals;
  };

  const enqueueAlternatives = (
    actions: ActionEval[],
    pathMoves: SolverMove[],
    history: bigint[],
  ): void => {
    actions.forEach(action => {
      if (action.used || action.queued) return;
      action.queued = true;
      queue.push({
        action,
        pathMoves: [...pathMoves],
        stateHistory: [...history],
      });
    });
  };

  const pickBestAction = async (
    state: bigint,
    pathMoves: SolverMove[],
    history: bigint[],
  ): Promise<ActionEval | null> => {
    const actions = await loadActions(state);
    const available = actions.filter(action => !action.used && !historyIncludes(history, action.nextState));
    if (!available.length) {
      return null;
    }
    const [best, ...rest] = available;
    best.used = true;
    enqueueAlternatives(rest, pathMoves, history);
    return best;
  };

  const pursueGreedy = async (
    startState: bigint,
    pathMoves: SolverMove[],
    history: bigint[],
  ): Promise<{ solved: boolean; moves: SolverMove[]; finalState: bigint; timedOut: boolean }> => {
    let currentState = startState;
    let currentMoves = [...pathMoves];
    let stateHistory = [...history];
    // Track immediate improvement for starting state.
    const startPegCount = popCount(currentState);
    if (startPegCount < bestPegCount) {
      bestPegCount = startPegCount;
      bestPath = [...currentMoves];
    }
    while (true) {
      throwIfAborted();
      if (performance.now() > deadline) {
        return { solved: false, moves: currentMoves, finalState: currentState, timedOut: true };
      }
      if (isSolved(currentState)) {
        bestPath = [...currentMoves];
        return { solved: true, moves: currentMoves, finalState: currentState, timedOut: false };
      }
      const action = await pickBestAction(currentState, currentMoves, stateHistory);
      if (!action) {
        return { solved: false, moves: currentMoves, finalState: currentState, timedOut: false };
      }
      currentMoves = [...currentMoves, action.def.move];
      const nextState = action.nextState;
      stateHistory = [...stateHistory, nextState];
      currentState = nextState;
      const pegCount = popCount(currentState);
      if (pegCount < bestPegCount) {
        bestPegCount = pegCount;
        bestPath = [...currentMoves];
      }
    }
  };

  let lastResult = await pursueGreedy(initialState, initialPath, initialHistory);
  if (lastResult.solved) {
    return buildResult(true, lastResult.moves, nodesExplored, startTime, 'solved');
  }
  if (lastResult.timedOut) {
    return buildResult(false, bestPath, nodesExplored, startTime, 'timeout');
  }

  let timedOut = false;
  while (true) {
    throwIfAborted();
    if (performance.now() > deadline) {
      timedOut = true;
      break;
    }
    const nextCandidate = queue.pop();
    if (!nextCandidate) {
      break;
    }
    const { action, pathMoves, stateHistory } = nextCandidate;
    if (action.used) {
      continue;
    }
    action.used = true;
    const nextMoves = [...pathMoves, action.def.move];
    const nextHistory = [...stateHistory, action.nextState];
    const pegCount = popCount(action.nextState);
    if (pegCount < bestPegCount) {
      bestPegCount = pegCount;
      bestPath = [...nextMoves];
    }
    lastResult = await pursueGreedy(action.nextState, nextMoves, nextHistory);
    if (lastResult.solved) {
      return buildResult(true, lastResult.moves, nodesExplored, startTime, 'solved');
    }
    if (lastResult.timedOut) {
      timedOut = true;
      break;
    }
  }

  return buildResult(false, bestPath, nodesExplored, startTime, timedOut ? 'timeout' : 'exhausted');
}

function buildResult(
  solved: boolean,
  moves: SolverMove[],
  nodesExplored: number,
  startTime: number,
  reason: SolverResult['reason'],
): SolverResult {
  return {
    solved,
    moves: [...moves],
    bestMoves: [...moves],
    nodesExplored,
    durationMs: Math.max(0, performance.now() - startTime),
    reason,
  };
}
