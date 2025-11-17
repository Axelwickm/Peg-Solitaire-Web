export interface SolverMove {
  from: string;
  to: string;
  jump: string;
}

export interface SolverResult {
  solved: boolean;
  moves: SolverMove[];
  nodesExplored: number;
  durationMs: number;
  bestMoves: SolverMove[];
}

export interface SolverSessionProgress {
  done: boolean;
  solved: boolean;
  nodesExplored: number;
  currentPath: SolverMove[];
  bestMoves: SolverMove[];
}

interface Frame {
  pegs: Set<string>;
  key: string;
  neighbors: SolverMove[];
  idx: number;
  g: number;
  move: SolverMove | null;
}

export class IdaStarSession {
  private bound: number;
  private nextBound: number;
  private stack: Frame[] = [];
  private seen = new Set<string>();
  private path: SolverMove[] = [];
  private nodes = 0;
  private solvedPath: SolverMove[] | null = null;
  private done = false;
  private solved = false;
  private bestPath: SolverMove[] = [];
  private bestPegs = Infinity;
  private initial: Set<string>;
  private target: string;

  constructor(private allowedMoves: Map<string, Map<string, string>>, initial: Set<string>, target: string) {
    this.initial = new Set(initial);
    this.target = target;
    this.bound = this.heuristic(this.initial);
    this.nextBound = Infinity;
    this.resetStack();
  }

  public runChunk(maxMs: number): SolverSessionProgress {
    const deadline = performance.now() + maxMs;
    while (performance.now() < deadline && !this.done) {
      this.step();
    }
    return {
      done: this.done,
      solved: this.solved,
      nodesExplored: this.nodes,
      currentPath: [...this.path],
      bestMoves: this.solved
        ? [...(this.solvedPath ?? this.path)]
        : this.bestPath.length
        ? [...this.bestPath]
        : [],
    };
  }

  private step(): void {
    if (this.stack.length === 0) {
      if (this.nextBound === Infinity) {
        this.done = true;
        return;
      }
      this.bound = this.nextBound;
      this.nextBound = Infinity;
      this.resetStack();
      return;
    }

    const frame = this.stack[this.stack.length - 1];
    this.nodes += 1;
    const h = this.heuristic(frame.pegs);
    const f = frame.g + h;
    if (f > this.bound) {
      this.nextBound = Math.min(this.nextBound, f);
      this.popFrame();
      return;
    }
    if (frame.pegs.size === 1 && frame.pegs.has(this.target)) {
      this.done = true;
      this.solved = true;
      this.solvedPath = [...this.path];
      this.bestPath = [...this.path];
      return;
    }
    if (frame.pegs.size < this.bestPegs) {
      this.bestPegs = frame.pegs.size;
      this.bestPath = [...this.path];
    }
    if (frame.idx >= frame.neighbors.length) {
      this.popFrame();
      return;
    }
    const move = frame.neighbors[frame.idx++];
    const next = new Set(frame.pegs);
    next.delete(move.from);
    next.delete(move.jump);
    next.add(move.to);
    const key = this.serialize(next);
    if (this.seen.has(key)) {
      return;
    }
    this.seen.add(key);
    this.path.push(move);
    this.pushFrame(next, key, frame.g + 1, move);
  }

  private resetStack(): void {
    this.stack = [];
    this.seen.clear();
    this.path = [];
    const key = this.serialize(this.initial);
    this.stack.push({
      pegs: new Set(this.initial),
      key,
      neighbors: this.generateMoves(this.initial),
      idx: 0,
      g: 0,
      move: null,
    });
    this.seen.add(key);
  }

  private pushFrame(pegs: Set<string>, key: string, g: number, move: SolverMove): void {
    this.stack.push({
      pegs,
      key,
      neighbors: this.generateMoves(pegs),
      idx: 0,
      g,
      move,
    });
  }

  private popFrame(): void {
    const frame = this.stack.pop();
    if (!frame) return;
    this.seen.delete(frame.key);
    if (frame.move) {
      this.path.pop();
    }
  }

  private generateMoves(pegs: Set<string>): SolverMove[] {
    const moves: SolverMove[] = [];
    for (const peg of pegs) {
      const destinations = this.allowedMoves.get(peg);
      if (!destinations) continue;
      for (const [to, jump] of destinations.entries()) {
        if (pegs.has(to)) continue;
        if (!pegs.has(jump)) continue;
        moves.push({ from: peg, to, jump });
      }
    }
    return moves;
  }

  private heuristic(pegs: Set<string>): number {
    return Math.max(0, pegs.size - 1);
  }

  private serialize(pegs: Set<string>): string {
    return [...pegs].sort().join(';');
  }
}

export class IdaStarSolver {
  constructor(private allowedMoves: Map<string, Map<string, string>>) {}

  public createSession(initial: Set<string>, target: string): IdaStarSession {
    return new IdaStarSession(this.allowedMoves, initial, target);
  }

  public solve(initial: Set<string>, target: string): SolverResult {
    const session = this.createSession(initial, target);
    const startTime = performance.now();
    let progress: SolverSessionProgress;
    do {
      progress = session.runChunk(Infinity);
    } while (!progress.done);
    return {
      solved: progress.solved,
      moves: progress.bestMoves,
      bestMoves: progress.bestMoves,
      nodesExplored: progress.nodesExplored,
      durationMs: Math.max(0, performance.now() - startTime),
    };
  }
}
