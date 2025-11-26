import * as ort from 'onnxruntime-web';

import { SolverMove, SolverResult } from './BfidaSolver';
import { BoardShape } from '../shapes';

const buildModelPath = (shapeId: string) => `/static/models/peg_solitaire_dqn_${shapeId}.onnx`;
const MAX_DURATION_MS = 60000;
const MAX_BATCHED_STATES = 16;
const YIELD_INTERVAL_MS = 200;
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

async function yieldToUi(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0));
}

export type AiSolverProgress = {
  bestMoves: SolverMove[];
  nodesExplored: number;
};

export type AiSolverOptions = {
  maxDurationMs?: number;
  abortSignal?: AbortSignal;
  onProgress?: (progress: AiSolverProgress) => void;
  guessMode?: boolean;
  guessThreshold?: number;
};

type ShapeSolverData = {
  shape: BoardShape;
  holeToIndex: Map<string, number>;
  bitMasks: bigint[];
  actionDefs: ActionDef[];
  centerIndex: number;
};

const solverDataByShape = new Map<string, ShapeSolverData>();

function buildSolverData(shape: BoardShape): ShapeSolverData {
  const holeToIndex = new Map<string, number>();
  const bitMasks: bigint[] = [];
  shape.holes.forEach((hole, index) => {
    holeToIndex.set(hole, index);
    bitMasks.push(1n << BigInt(index));
  });

  const actionDefs: ActionDef[] = [];
  let actionIndex = 0;
  shape.allowedMoves.forEach((targets, from) => {
    const fromIndex = holeToIndex.get(from);
    if (fromIndex === undefined) return;
    targets.forEach((jump, to) => {
      const toIndex = holeToIndex.get(to);
      const jumpIndex = holeToIndex.get(jump);
      if (toIndex === undefined || jumpIndex === undefined) {
        return;
      }
      actionDefs.push({
        index: actionIndex++,
        fromIndex,
        toIndex,
        jumpIndex,
        move: { from, to, jump },
      });
    });
  });

  const centerIndex = holeToIndex.get(shape.empty) ?? 0;
  return {
    shape,
    holeToIndex,
    bitMasks,
    actionDefs,
    centerIndex,
  };
}

function getSolverData(shape: BoardShape): ShapeSolverData {
  const cached = solverDataByShape.get(shape.id);
  if (cached) return cached;
  const data = buildSolverData(shape);
  solverDataByShape.set(shape.id, data);
  return data;
}

const sessionPromises = new Map<string, Promise<ort.InferenceSession>>();

ort.env.wasm.wasmPaths = {
  wasm: `${ORT_WASM_BASE_PATH}/ort-wasm-simd-threaded.jsep.wasm`,
  mjs: `${ORT_WASM_BASE_PATH}/ort-wasm-simd-threaded.jsep.mjs`,
};

async function getSession(shape: BoardShape): Promise<ort.InferenceSession> {
  const modelPath = buildModelPath(shape.id);
  let promise = sessionPromises.get(modelPath);
  if (!promise) {
    promise = ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });
    sessionPromises.set(modelPath, promise);
  }
  try {
    return await promise;
  } catch (error) {
    sessionPromises.delete(modelPath);
    throw error;
  }
}

function toStateBits(pegs: Set<string>, data: ShapeSolverData): bigint {
  let state = 0n;
  pegs.forEach(cell => {
    const index = data.holeToIndex.get(cell);
    if (index !== undefined) {
      state |= data.bitMasks[index];
    }
  });
  return state;
}

function hasPeg(state: bigint, index: number, data: ShapeSolverData): boolean {
  return (state & data.bitMasks[index]) !== 0n;
}

function applyAction(state: bigint, action: ActionDef, data: ShapeSolverData): bigint {
  let next = state;
  next &= ~data.bitMasks[action.fromIndex];
  next &= ~data.bitMasks[action.jumpIndex];
  next |= data.bitMasks[action.toIndex];
  return next;
}

function isSolved(state: bigint, data: ShapeSolverData): boolean {
  return popCount(state) === 1 && hasPeg(state, data.centerIndex, data);
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

function isLegal(state: bigint, action: ActionDef, data: ShapeSolverData): boolean {
  return (
    hasPeg(state, action.fromIndex, data) &&
    hasPeg(state, action.jumpIndex, data) &&
    !hasPeg(state, action.toIndex, data)
  );
}

function historyIncludes(history: bigint[], target: bigint): boolean {
  return history.some(entry => entry === target);
}

function applyMoveSequence(state: bigint, moves: SolverMove[], data: ShapeSolverData): bigint {
  let current = state;
  for (const move of moves) {
    const fromIndex = data.holeToIndex.get(move.from);
    const toIndex = data.holeToIndex.get(move.to);
    const jumpIndex = data.holeToIndex.get(move.jump);
    if (
      fromIndex === undefined ||
      toIndex === undefined ||
      jumpIndex === undefined
    ) {
      continue;
    }
    current &= ~data.bitMasks[fromIndex];
    current &= ~data.bitMasks[jumpIndex];
    current |= data.bitMasks[toIndex];
  }
  return current;
}

function writeStateToBuffer(
  state: bigint,
  data: ShapeSolverData,
  target: Float32Array,
  offset: number,
): void {
  for (let i = 0; i < data.bitMasks.length; i++) {
    target[offset + i] = hasPeg(state, i, data) ? 1 : 0;
  }
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

type PendingEvaluation = {
  state: bigint;
  resolve: (values: Float32Array) => void;
  reject: (error: unknown) => void;
};

class BatchedEvaluator {
  private pending: PendingEvaluation[] = [];
  private flushing = false;
  private scheduled = false;

  constructor(
    private session: ort.InferenceSession,
    private solverData: ShapeSolverData,
    private maxBatchSize: number = MAX_BATCHED_STATES,
  ) { }

  public evaluate(state: bigint): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      this.pending.push({ state, resolve, reject });
      this.scheduleFlush();
    });
  }

  private scheduleFlush(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => this.flush());
  }

  private async flush(): Promise<void> {
    if (this.flushing) return;
    this.flushing = true;
    this.scheduled = false;
    try {
      while (this.pending.length) {
        const batch = this.pending.splice(0, this.maxBatchSize);
        await this.runBatch(batch);
      }
    } finally {
      this.flushing = false;
      if (this.pending.length) {
        this.scheduleFlush();
      }
    }
  }

  private async runBatch(batch: PendingEvaluation[]): Promise<void> {
    const batchSize = batch.length;
    const stateSize = this.solverData.shape.holes.length;
    const actionSize = this.solverData.actionDefs.length;
    const buffer = new Float32Array(batchSize * stateSize);
    batch.forEach((entry, idx) => {
      writeStateToBuffer(entry.state, this.solverData, buffer, idx * stateSize);
    });
    try {
      const tensor = new ort.Tensor('float32', buffer, [batchSize, stateSize]);
      const output = await this.session.run({ state: tensor });
      const qValues = output.q_values.data as Float32Array;
      batch.forEach((entry, idx) => {
        const start = idx * actionSize;
        const slice = qValues.subarray(start, start + actionSize);
        entry.resolve(new Float32Array(slice));
      });
    } catch (error) {
      batch.forEach(entry => entry.reject(error));
    }
  }
}

export class AiSolverAbortError extends Error {
  constructor() {
    super('AI solver aborted');
    this.name = 'AiSolverAbortError';
  }
}

export async function solveWithAi(
  shape: BoardShape,
  pegPositions: Set<string>,
  options: AiSolverOptions = {},
): Promise<SolverResult> {
  const solverData = getSolverData(shape);
  const signal = options.abortSignal;
  const startTime = performance.now();
  const deadline = startTime + (options.maxDurationMs ?? MAX_DURATION_MS);
  const guessMode = options.guessMode ?? true;
  const guessThreshold = options.guessThreshold ?? 400;
  const session = await getSession(shape);
  const evaluator = new BatchedEvaluator(session, solverData);
  const initialState = toStateBits(pegPositions, solverData);
  let totalNodesExplored = 0;
  let committedMoves: SolverMove[] = [];
  let committedState = initialState;

  const throwIfAborted = () => {
    if (signal?.aborted) {
      throw new AiSolverAbortError();
    }
  };

  type ChunkOutcome = {
    solved: boolean;
    solution: SolverMove[];
    bestPath: SolverMove[];
    nodesExplored: number;
    reason: 'solved' | 'timeout' | 'exhausted' | 'budget';
  };

  const runChunk = async (rootState: bigint): Promise<ChunkOutcome> => {
    const stateCache = new Map<bigint, ActionEval[]>();
    const queue = new CandidateQueue();
    let bestPath: SolverMove[] = [];
    let bestPegCount = popCount(rootState);
    let nodesExplored = 0;
    let lastYieldTime = performance.now();
    let latestPath: SolverMove[] = [];
    const maxBranchExpansions = guessMode ? Math.max(0, guessThreshold - 1) : Number.POSITIVE_INFINITY;
    let branchExpansions = 0;

    const notifyProgress = () => {
      if (options.onProgress) {
        const combined = committedMoves.length
          ? [...committedMoves, ...bestPath]
          : [...bestPath];
        options.onProgress({
          bestMoves: combined,
          nodesExplored: totalNodesExplored + nodesExplored,
        });
      }
    };

    const maybeYield = async (): Promise<void> => {
      if (performance.now() - lastYieldTime >= YIELD_INTERVAL_MS) {
        lastYieldTime = performance.now();
        await yieldToUi();
      }
    };

    const loadActions = async (state: bigint): Promise<ActionEval[]> => {
      const cached = stateCache.get(state);
      if (cached) {
        return cached;
      }
      throwIfAborted();
      nodesExplored += 1;
      const qValues = await evaluator.evaluate(state);
      const evals: ActionEval[] = [];
      solverData.actionDefs.forEach(action => {
        if (!isLegal(state, action, solverData)) return;
        const nextState = applyAction(state, action, solverData);
        evals.push({
          def: action,
          nextState,
          qValue: qValues[action.index] ?? Number.NEGATIVE_INFINITY,
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
    ): Promise<{ solved: boolean; moves: SolverMove[]; finalState: bigint; reason: 'solved' | 'timeout' | 'deadend' }> => {
      let currentState = startState;
      let currentMoves = [...pathMoves];
      let stateHistory = [...history];
      const startPegCount = popCount(currentState);
      if (startPegCount < bestPegCount) {
        bestPegCount = startPegCount;
        bestPath = [...currentMoves];
        notifyProgress();
      } else if (!bestPath.length) {
        latestPath = [...currentMoves];
      }
      while (true) {
        await maybeYield();
        throwIfAborted();
        if (performance.now() > deadline) {
          return { solved: false, moves: currentMoves, finalState: currentState, reason: 'timeout' };
        }
        if (isSolved(currentState, solverData)) {
          bestPath = [...currentMoves];
          notifyProgress();
          return { solved: true, moves: currentMoves, finalState: currentState, reason: 'solved' };
        }
        const action = await pickBestAction(currentState, currentMoves, stateHistory);
        if (!action) {
          return { solved: false, moves: currentMoves, finalState: currentState, reason: 'deadend' };
        }
        currentMoves = [...currentMoves, action.def.move];
        const nextState = action.nextState;
        stateHistory = [...stateHistory, nextState];
        currentState = nextState;
        const pegCount = popCount(currentState);
        if (pegCount < bestPegCount) {
          bestPegCount = pegCount;
          bestPath = [...currentMoves];
          notifyProgress();
        } else if (!bestPath.length) {
          latestPath = [...currentMoves];
        }
      }
    };

    let greedyResult = await pursueGreedy(rootState, [], [rootState]);
    if (greedyResult.reason === 'solved') {
      return {
        solved: true,
        solution: greedyResult.moves,
        bestPath: greedyResult.moves,
        nodesExplored,
        reason: 'solved',
      };
    }
    if (greedyResult.reason === 'timeout') {
      return {
        solved: false,
        solution: greedyResult.moves,
        bestPath: bestPath.length ? bestPath : latestPath,
        nodesExplored,
        reason: 'timeout',
      };
    }
    // Greedy search never hits a chunk budget; continue into queued alternatives.

    let timedOut = false;
    while (true) {
      await maybeYield();
      throwIfAborted();
      if (performance.now() > deadline) {
        timedOut = true;
        break;
      }
      if (branchExpansions >= maxBranchExpansions) {
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
      greedyResult = await pursueGreedy(action.nextState, nextMoves, nextHistory);
      if (greedyResult.reason === 'solved') {
        return {
          solved: true,
          solution: greedyResult.moves,
          bestPath: greedyResult.moves,
          nodesExplored,
          reason: 'solved',
        };
      }
      if (greedyResult.reason === 'timeout') {
        return {
          solved: false,
          solution: greedyResult.moves,
          bestPath: bestPath.length ? bestPath : latestPath,
          nodesExplored,
          reason: 'timeout',
        };
      }
      branchExpansions += 1;
      // Continue exploring queued alternatives until limits or queue exhaustion.
    }

    return {
      solved: false,
      solution: [],
      bestPath: bestPath.length ? bestPath : latestPath,
      nodesExplored,
      reason: timedOut ? 'timeout' : 'budget',
    };
  };

  while (true) {
    const outcome = await runChunk(committedState);
    totalNodesExplored += outcome.nodesExplored;
    if (outcome.solved) {
      const finalMoves = [...committedMoves, ...outcome.solution];
      return buildResult(true, finalMoves, totalNodesExplored, startTime, 'solved');
    }
    if (guessMode && outcome.bestPath.length) {
      const nextMove = outcome.bestPath[0];
      if (nextMove) {
        console.log('[AI] Committing partial plan move', {
          committedDepth: committedMoves.length + 1,
          nodesExplored: totalNodesExplored,
          bestPathRemaining: outcome.bestPath.length,
          move: nextMove,
        });
        committedMoves = [...committedMoves, nextMove];
        committedState = applyMoveSequence(committedState, [nextMove], solverData);
        continue;
      }
    }
    const combinedBest = committedMoves.length
      ? [...committedMoves, ...outcome.bestPath]
      : [...outcome.bestPath];
    return buildResult(
      false,
      combinedBest,
      totalNodesExplored,
      startTime,
      outcome.reason === 'timeout' ? 'timeout' : 'exhausted',
    );
  }
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
