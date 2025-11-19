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

export interface SolverOptions {
  enforceTargetTypeInvariant?: boolean;
}

type BitState = bigint;

type PegTypeCounts = [number, number, number, number];

type Direction = 'forward' | 'backward';

interface Transition {
  from: number;
  over: number;
  to: number;
}

interface SearchNode {
  state: BitState;
  boardKey: string;
  key: string;
  g: number;
  lastPeg: number;
  heuristic?: number;
  typeCounts?: PegTypeCounts;
  pagoda?: number;
}

interface ParentInfo {
  parentKey: string | null;
  move: Transition | null;
}

interface FrontierEntry {
  boardKey: string;
  nodeKey: string;
  state: BitState;
  g: number;
  typeCounts: PegTypeCounts;
  pagoda: number;
}

interface FrontierSummary {
  minTypeCounts: PegTypeCounts;
  maxTypeCounts: PegTypeCounts;
  minPagoda: number;
  maxPagoda: number;
}

interface SessionDerivedData {
  cornerMaskWithoutTarget: BitState;
  goalTypeCounts: PegTypeCounts;
  targetIndex: number;
  targetType: number;
  goalStateBit: BitState;
  pagodaWeights: number[] | null;
  goalPagodaValue: number;
  startPagodaValue: number;
  pagodaEnabled: boolean;
  enforceTargetTypeInvariant: boolean;
}

function cloneTypeCounts(counts: PegTypeCounts): PegTypeCounts {
  return [counts[0], counts[1], counts[2], counts[3]];
}

const enum PegParityClass {
  Same = 0,
  Different = 1,
}

const IDLE_PEG = -1;
const PAGODA_TOLERANCE = 1e-9;

class BoardContext {
  public readonly cells: string[];
  public readonly cellToIndex: Map<string, number>;
  public readonly transitionsForward: Transition[];
  public readonly transitionsBackward: Transition[];
  public readonly forwardMovesByFrom: Map<number, Transition[]>;
  public readonly backwardMovesByFrom: Map<number, Transition[]>;
  public readonly cellTypes: number[];
  public readonly typeMasks: BitState[];
  public readonly pegClassByType: PegParityClass[];
  public readonly cornerMask: BitState;
  public readonly cornerCaptureTypes: Set<number>;
  public readonly regionMasks: BitState[];
  public readonly maxCapturesPerType: number[];
  public readonly positionClassMasks: BitState[];
  private readonly pagodaCache: Map<number, number[] | null>;

  private readonly rowColByIndex: Array<{ row: number; col: number }> = [];
  private readonly adjacency: Map<number, number[]> = new Map();
  private readonly bitForIndex: BitState[];

  constructor(private allowedMoves: Map<string, Map<string, string>>) {
    this.cells = this.collectCells();
    this.cellToIndex = new Map(this.cells.map((cell, index) => [cell, index]));
    this.rowColByIndex = this.cells.map(cell => {
      const [row, col] = cell.split(',').map(Number);
      return { row, col };
    });
    this.bitForIndex = this.cells.map((_cell, idx) => 1n << BigInt(idx));
    this.transitionsForward = this.buildTransitions(false);
    this.transitionsBackward = this.buildTransitions(true);
    this.forwardMovesByFrom = this.groupTransitions(this.transitionsForward);
    this.backwardMovesByFrom = this.groupTransitions(this.transitionsBackward);
    this.cellTypes = this.cells.map(cell => this.computePegType(cell));
    this.typeMasks = this.buildTypeMasks();
    this.pegClassByType = this.cellTypes.map(type => this.computePegClass(type));
    this.cornerMask = this.computeCornerMask();
    this.cornerCaptureTypes = this.computeCornerCaptureTypes();
    this.buildAdjacency();
    this.regionMasks = this.computeMersonRegions();
    this.maxCapturesPerType = this.computeMaxCapturesPerType();
    this.positionClassMasks = this.computePositionClassMasks();
    this.pagodaCache = new Map();
  }

  public stateFromSet(pegs: Set<string>): BitState {
    let state = 0n;
    pegs.forEach(cell => {
      const idx = this.cellToIndex.get(cell);
      if (idx !== undefined) {
        state |= this.bitForIndex[idx];
      }
    });
    return state;
  }

  public stateKey(state: BitState): string {
    return state.toString();
  }

  public popCount(state: BitState): number {
    let count = 0;
    let value = state;
    while (value) {
      count += Number(value & 1n);
      value >>= 1n;
    }
    return count;
  }

  public typeCounts(state: BitState): PegTypeCounts {
    const counts: PegTypeCounts = [0, 0, 0, 0];
    this.typeMasks.forEach((mask, type) => {
      counts[type as 0 | 1 | 2 | 3] = this.popCount(state & mask);
    });
    return counts;
  }

  public isCorner(index: number): boolean {
    return (this.cornerMask & this.bitForIndex[index]) !== 0n;
  }

  public regionCountFilled(state: BitState): number {
    let filled = 0;
    for (const mask of this.regionMasks) {
      if (mask !== 0n && (state & mask) === mask) {
        filled += 1;
      }
    }
    return filled;
  }

  public pegType(index: number): number {
    return this.cellTypes[index];
  }

  public bit(index: number): BitState {
    return this.bitForIndex[index];
  }

  public pegIndices(state: BitState): number[] {
    const indices: number[] = [];
    for (let i = 0; i < this.cells.length; i++) {
      if (state & this.bitForIndex[i]) {
        indices.push(i);
      }
    }
    return indices;
  }

  public pagodaWeights(targetIndex: number): number[] | null {
    let weights = this.pagodaCache.get(targetIndex);
    if (!weights) {
      weights = this.computePagodaWeights(targetIndex);
      this.pagodaCache.set(targetIndex, weights);
    }
    return weights;
  }

  public pagodaValue(state: BitState, weights: number[] | null): number {
    if (!weights) return 0;
    let value = 0;
    for (let i = 0; i < this.cells.length; i++) {
      if (state & this.bitForIndex[i]) {
        value += weights[i];
      }
    }
    return value;
  }

  public positionClassKey(state: BitState): string | null {
    if (!this.positionClassMasks.length) {
      return null;
    }
    return this.positionClassMasks
      .map(mask => ((this.popCount(state & mask) & 1) === 1 ? '1' : '0'))
      .join('');
  }

  public haveMatchingPositionClasses(a: BitState, b: BitState): boolean {
    if (!this.positionClassMasks.length) {
      return true;
    }
    return this.positionClassKey(a) === this.positionClassKey(b);
  }

  private computePositionClassMasks(): BitState[] {
    if (!this.transitionsForward.length) {
      return [];
    }
    const numCols = this.cells.length;
    const matrix = this.transitionsForward.map(
      move => this.bit(move.from) | this.bit(move.over) | this.bit(move.to),
    );
    const pivotForCol = new Array<number>(numCols).fill(-1);
    let currentRow = 0;
    for (let col = 0; col < numCols && currentRow < matrix.length; col++) {
      let pivotRow = -1;
      for (let row = currentRow; row < matrix.length; row++) {
        if (((matrix[row] >> BigInt(col)) & 1n) !== 0n) {
          pivotRow = row;
          break;
        }
      }
      if (pivotRow === -1) continue;
      [matrix[currentRow], matrix[pivotRow]] = [matrix[pivotRow], matrix[currentRow]];
      pivotForCol[col] = currentRow;
      for (let row = 0; row < matrix.length; row++) {
        if (row !== currentRow && ((matrix[row] >> BigInt(col)) & 1n) !== 0n) {
          matrix[row] ^= matrix[currentRow];
        }
      }
      currentRow += 1;
    }
    const masks: BitState[] = [];
    for (let col = 0; col < numCols; col++) {
      if (pivotForCol[col] !== -1) continue;
      let vector = 1n << BigInt(col);
      for (let pivotCol = 0; pivotCol < numCols; pivotCol++) {
        const rowIndex = pivotForCol[pivotCol];
        if (rowIndex === -1) continue;
        if (((matrix[rowIndex] >> BigInt(col)) & 1n) !== 0n) {
          vector |= 1n << BigInt(pivotCol);
        }
      }
      if (vector !== 0n) {
        masks.push(vector);
      }
    }
    return masks;
  }

  private collectCells(): string[] {
    const cells = new Set<string>();
    this.allowedMoves.forEach((destinations, from) => {
      cells.add(from);
      destinations.forEach((jump, to) => {
        cells.add(jump);
        cells.add(to);
      });
    });
    return Array.from(cells).sort((a, b) => {
      const [ar, ac] = a.split(',').map(Number);
      const [br, bc] = b.split(',').map(Number);
      return ar === br ? ac - bc : ar - br;
    });
  }

  private buildTransitions(reverse: boolean): Transition[] {
    const list: Transition[] = [];
    this.allowedMoves.forEach((destinations, fromCell) => {
      const fromIdx = this.cellToIndex.get(fromCell);
      if (fromIdx === undefined) return;
      destinations.forEach((jumpCell, toCell) => {
        const toIdx = this.cellToIndex.get(toCell);
        const jumpIdx = this.cellToIndex.get(jumpCell);
        if (toIdx === undefined || jumpIdx === undefined) return;
        if (reverse) {
          list.push({ from: toIdx, over: jumpIdx, to: fromIdx });
        } else {
          list.push({ from: fromIdx, over: jumpIdx, to: toIdx });
        }
      });
    });
    return list;
  }

  private groupTransitions(transitions: Transition[]): Map<number, Transition[]> {
    const map = new Map<number, Transition[]>();
    transitions.forEach(move => {
      if (!map.has(move.from)) {
        map.set(move.from, []);
      }
      map.get(move.from)!.push(move);
    });
    return map;
  }

  private computePegType(cell: string): number {
    const [row, col] = cell.split(',').map(Number);
    const rowParity = row & 1;
    const colParity = col & 1;
    return rowParity * 2 + colParity;
  }

  private computePegClass(type: number): PegParityClass {
    const rowParity = type >> 1;
    const colParity = type & 1;
    return rowParity === colParity ? PegParityClass.Same : PegParityClass.Different;
  }

  private buildTypeMasks(): BitState[] {
    const masks: BitState[] = [0n, 0n, 0n, 0n];
    this.cellTypes.forEach((type, idx) => {
      masks[type] |= this.bitForIndex[idx];
    });
    return masks;
  }

  private computeCornerMask(): BitState {
    const jumpCells = new Set<number>();
    this.transitionsForward.forEach(move => jumpCells.add(move.over));
    let mask = 0n;
    this.cells.forEach((cell, idx) => {
      if (!jumpCells.has(idx)) {
        mask |= this.bitForIndex[idx];
      }
    });
    return mask;
  }

  private computeCornerCaptureTypes(): Set<number> {
    const types = new Set<number>();
    this.forwardMovesByFrom.forEach((_moves, from) => {
      if (!this.isCorner(from)) return;
      const moves = this.forwardMovesByFrom.get(from) ?? [];
      moves.forEach(move => {
        types.add(this.cellTypes[move.over]);
      });
    });
    return types;
  }

  private buildAdjacency(): void {
    for (let i = 0; i < this.cells.length; i++) {
      this.adjacency.set(i, []);
    }
    const addEdge = (a: number, b: number): void => {
      if (!this.adjacency.get(a)!.includes(b)) {
        this.adjacency.get(a)!.push(b);
      }
    };
    this.transitionsForward.forEach(move => {
      addEdge(move.from, move.over);
      addEdge(move.over, move.from);
      addEdge(move.over, move.to);
      addEdge(move.to, move.over);
      addEdge(move.from, move.to);
      addEdge(move.to, move.from);
    });
  }

  private computeMersonRegions(): BitState[] {
    const graph = new Map<number, Set<number>>();
    this.transitionsForward.forEach(move => {
      if (!graph.has(move.over)) {
        graph.set(move.over, new Set());
      }
      graph.get(move.over)!.add(move.from);
    });
    const sccs = this.findSccs(graph);
    const sinks = this.findSinkComponents(sccs, graph);
    const regions: BitState[] = [];
    sinks.forEach(component => {
      const withoutCorners = component.filter(idx => !this.isCorner(idx));
      const visited = new Set<number>();
      withoutCorners.forEach(idx => {
        if (visited.has(idx)) return;
        const connected = this.collectAdjacent(idx, withoutCorners, visited);
        if (connected.length) {
          let mask = 0n;
          connected.forEach(cellIdx => {
            mask |= this.bitForIndex[cellIdx];
          });
          if (mask !== 0n) {
            regions.push(mask);
          }
        }
      });
    });
    return regions;
  }

  private findSccs(graph: Map<number, Set<number>>): number[][] {
    const nodes = this.cells.map((_cell, idx) => idx);
    const indexMap = new Map<number, number>();
    const lowLink = new Map<number, number>();
    const stack: number[] = [];
    const inStack = new Set<number>();
    const sccs: number[][] = [];
    let index = 0;

    const strongConnect = (v: number): void => {
      indexMap.set(v, index);
      lowLink.set(v, index);
      index += 1;
      stack.push(v);
      inStack.add(v);
      const neighbors = graph.get(v) ?? new Set<number>();
      neighbors.forEach(w => {
        if (!indexMap.has(w)) {
          strongConnect(w);
          lowLink.set(v, Math.min(lowLink.get(v) ?? 0, lowLink.get(w) ?? 0));
        } else if (inStack.has(w)) {
          lowLink.set(v, Math.min(lowLink.get(v) ?? 0, indexMap.get(w) ?? 0));
        }
      });
      if (lowLink.get(v) === indexMap.get(v)) {
        const component: number[] = [];
        while (true) {
          const w = stack.pop();
          if (w === undefined) break;
          inStack.delete(w);
          component.push(w);
          if (w === v) break;
        }
        if (component.length) {
          sccs.push(component);
        }
      }
    };

    nodes.forEach(v => {
      if (!indexMap.has(v)) {
        strongConnect(v);
      }
    });
    return sccs;
  }

  private findSinkComponents(sccs: number[][], graph: Map<number, Set<number>>): number[][] {
    const componentIndex = new Map<number, number>();
    sccs.forEach((component, idx) => {
      component.forEach(node => componentIndex.set(node, idx));
    });
    const outDegrees = new Array(sccs.length).fill(0);
    graph.forEach((neighbors, node) => {
      const fromComp = componentIndex.get(node);
      neighbors.forEach(neighbor => {
        const toComp = componentIndex.get(neighbor);
        if (fromComp !== undefined && toComp !== undefined && fromComp !== toComp) {
          outDegrees[fromComp] += 1;
        }
      });
    });
    return sccs.filter((_comp, idx) => outDegrees[idx] === 0);
  }

  private collectAdjacent(start: number, allowed: number[], visited: Set<number>): number[] {
    const allowedSet = new Set(allowed);
    const queue: number[] = [start];
    const result: number[] = [];
    visited.add(start);
    while (queue.length) {
      const current = queue.shift();
      if (current === undefined) break;
      result.push(current);
      const neighbors = this.adjacency.get(current) ?? [];
      neighbors.forEach(neighbor => {
        if (allowedSet.has(neighbor) && !visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      });
    }
    return result;
  }

  private computeMaxCapturesPerType(): number[] {
    const maxCaptures = [1, 1, 1, 1];
    const usedJumpMask = new Set<number>();

    const dfs = (current: number, counts: number[]): void => {
      const moves = this.forwardMovesByFrom.get(current) ?? [];
      for (const move of moves) {
        if (usedJumpMask.has(move.over)) continue;
        usedJumpMask.add(move.over);
        const capturedType = this.cellTypes[move.over];
        counts[capturedType] += 1;
        for (let i = 0; i < 4; i++) {
          maxCaptures[i] = Math.max(maxCaptures[i], counts[i]);
        }
        dfs(move.to, counts);
        counts[capturedType] -= 1;
        usedJumpMask.delete(move.over);
      }
    };

    this.cells.forEach((_cell, idx) => {
      dfs(idx, [0, 0, 0, 0]);
    });

    return maxCaptures.map(value => Math.max(1, value));
  }

  private computePagodaWeights(targetIndex: number): number[] | null {
    const solver = new SimplexSolver(this.cells.length);
    this.transitionsForward.forEach(move => {
      const coeffs = new Array(this.cells.length).fill(0);
      coeffs[move.to] += 1;
      coeffs[move.from] -= 1;
      coeffs[move.over] -= 1;
      solver.addConstraint(coeffs, '<=', 0);
    });
    const targetConstraint = new Array(this.cells.length).fill(0);
    targetConstraint[targetIndex] = 1;
    solver.addConstraint(targetConstraint, '=', 1);
    solver.setObjective(new Array(this.cells.length).fill(1));
    const solution = solver.solve();
    if (!solution) {
      console.warn('[Solver] Failed to compute pagoda weights; falling back to no pagoda pruning.');
      return null;
    }
    const normalized = solution.map(value => (value < PAGODA_TOLERANCE ? 0 : value));
    const targetWeight = normalized[targetIndex];
    if (targetWeight <= PAGODA_TOLERANCE) {
      console.warn('[Solver] Pagoda solver produced zero weight for target; disabling pagoda pruning.');
      return null;
    }
    for (let i = 0; i < normalized.length; i++) {
      normalized[i] /= targetWeight;
    }
    if (!this.validatePagodaWeights(normalized)) {
      console.warn('[Solver] Pagoda weights failed validation; disabling pagoda pruning.');
      return null;
    }
    return normalized;
  }

  private validatePagodaWeights(weights: number[]): boolean {
    let positive = false;
    for (const weight of weights) {
      if (weight > PAGODA_TOLERANCE) {
        positive = true;
        break;
      }
    }
    if (!positive) {
      return false;
    }
    for (const move of this.transitionsForward) {
      const delta = weights[move.to] - weights[move.from] - weights[move.over];
      if (delta > PAGODA_TOLERANCE) {
        return false;
      }
    }
    return true;
  }
}

type ConstraintType = '<=' | '>=' | '=';

interface LinearConstraint {
  coeffs: number[];
  rhs: number;
  type: ConstraintType;
}

class SimplexSolver {
  private readonly constraints: LinearConstraint[] = [];
  private objective: number[] | null = null;
  private static readonly EPS = 1e-9;

  constructor(private readonly variableCount: number) {}

  public addConstraint(coeffs: number[], type: ConstraintType, rhs: number): void {
    if (coeffs.length !== this.variableCount) {
      throw new Error('Coefficient length mismatch for constraint.');
    }
    this.constraints.push({ coeffs: [...coeffs], rhs, type });
  }

  public setObjective(coeffs: number[]): void {
    if (coeffs.length !== this.variableCount) {
      throw new Error('Objective length mismatch.');
    }
    this.objective = [...coeffs];
  }

  public solve(): number[] | null {
    if (!this.objective) {
      throw new Error('Objective not set.');
    }
    const processed = this.constraints.map(constraint => this.normalizeConstraint(constraint));
    let slackCount = 0;
    let artificialCount = 0;
    processed.forEach(constraint => {
      if (constraint.type === '<=') {
        slackCount += 1;
      } else if (constraint.type === '>=') {
        slackCount += 1;
        artificialCount += 1;
      } else {
        artificialCount += 1;
      }
    });
    const rows = processed.length;
    const cols = this.variableCount + slackCount + artificialCount;
    const tableau = Array.from({ length: rows + 1 }, () => new Array(cols + 1).fill(0));
    const basis = new Array<number>(rows).fill(-1);
    const artificialColumns = new Set<number>();
    let slackIndex = this.variableCount;
    let artificialIndex = this.variableCount + slackCount;

    processed.forEach((constraint, rowIndex) => {
      const row = tableau[rowIndex];
      for (let i = 0; i < this.variableCount; i++) {
        row[i] = constraint.coeffs[i];
      }
      row[cols] = constraint.rhs;
      if (constraint.type === '<=') {
        row[slackIndex] = 1;
        basis[rowIndex] = slackIndex;
        slackIndex += 1;
      } else if (constraint.type === '>=') {
        row[slackIndex] = -1;
        row[artificialIndex] = 1;
        artificialColumns.add(artificialIndex);
        basis[rowIndex] = artificialIndex;
        slackIndex += 1;
        artificialIndex += 1;
      } else {
        row[artificialIndex] = 1;
        artificialColumns.add(artificialIndex);
        basis[rowIndex] = artificialIndex;
        artificialIndex += 1;
      }
    });

    const phaseObjective = new Array(cols).fill(0);
    artificialColumns.forEach(col => {
      phaseObjective[col] = -1;
    });
    this.setObjectiveRow(tableau, basis, rows, cols, phaseObjective);
    if (!this.performSimplex(tableau, basis, rows, cols)) {
      return null;
    }
    if (tableau[rows][cols] < -SimplexSolver.EPS) {
      return null;
    }

    const mainObjective = new Array(cols).fill(0);
    for (let i = 0; i < this.variableCount; i++) {
      mainObjective[i] = this.objective[i];
    }
    this.setObjectiveRow(tableau, basis, rows, cols, mainObjective);
    if (!this.performSimplex(tableau, basis, rows, cols)) {
      return null;
    }

    const solution = new Array(this.variableCount).fill(0);
    for (let i = 0; i < rows; i++) {
      const basic = basis[i];
      if (basic >= 0 && basic < this.variableCount) {
        solution[basic] = tableau[i][cols];
      }
    }
    return solution;
  }

  private normalizeConstraint(constraint: LinearConstraint): LinearConstraint {
    let { coeffs, rhs, type } = constraint;
    if (rhs < 0) {
      coeffs = coeffs.map(value => -value);
      rhs = -rhs;
      if (type === '<=') {
        type = '>=';
      } else if (type === '>=') {
        type = '<=';
      }
    }
    return { coeffs, rhs, type };
  }

  private setObjectiveRow(
    tableau: number[][],
    basis: number[],
    objectiveRowIndex: number,
    cols: number,
    coefficients: number[],
  ): void {
    for (let col = 0; col < cols; col++) {
      tableau[objectiveRowIndex][col] = coefficients[col] ?? 0;
    }
    tableau[objectiveRowIndex][cols] = 0;
    for (let row = 0; row < basis.length; row++) {
      const basic = basis[row];
      if (basic >= 0) {
        const coeff = coefficients[basic] ?? 0;
        if (Math.abs(coeff) > SimplexSolver.EPS) {
          for (let col = 0; col <= cols; col++) {
            tableau[objectiveRowIndex][col] -= coeff * tableau[row][col];
          }
        }
      }
    }
  }

  private performSimplex(
    tableau: number[][],
    basis: number[],
    objectiveRowIndex: number,
    cols: number,
  ): boolean {
    while (true) {
      let entering = -1;
      let best = SimplexSolver.EPS;
      for (let col = 0; col < cols; col++) {
        const value = tableau[objectiveRowIndex][col];
        if (value > best) {
          best = value;
          entering = col;
        }
      }
      if (entering === -1) {
        return true;
      }
      let pivotRow = -1;
      let bestRatio = Infinity;
      for (let row = 0; row < objectiveRowIndex; row++) {
        const coefficient = tableau[row][entering];
        if (coefficient > SimplexSolver.EPS) {
          const ratio = tableau[row][cols] / coefficient;
          if (ratio < bestRatio - SimplexSolver.EPS) {
            bestRatio = ratio;
            pivotRow = row;
          }
        }
      }
      if (pivotRow === -1) {
        return false;
      }
      this.pivot(tableau, pivotRow, entering, cols);
      basis[pivotRow] = entering;
    }
  }

  private pivot(tableau: number[][], pivotRow: number, pivotCol: number, cols: number): void {
    const pivotValue = tableau[pivotRow][pivotCol];
    if (Math.abs(pivotValue) < SimplexSolver.EPS) {
      throw new Error('Degenerate pivot encountered.');
    }
    for (let col = 0; col <= cols; col++) {
      tableau[pivotRow][col] /= pivotValue;
    }
    for (let row = 0; row < tableau.length; row++) {
      if (row === pivotRow) continue;
      const factor = tableau[row][pivotCol];
      if (Math.abs(factor) < SimplexSolver.EPS) continue;
      for (let col = 0; col <= cols; col++) {
        tableau[row][col] -= factor * tableau[pivotRow][col];
      }
    }
  }
}

class DirectionState {
  public bound = 0;
  public nextBound = Infinity;
  public exhausted = false;
  public lastIterationExpansions = 0;

  private readonly transitions: Map<number, Transition[]>;
  private readonly direction: Direction;
  private readonly sessionData: SessionDerivedData;
  private readonly context: BoardContext;
  private readonly rootState: BitState;
  private boundInitialized = false;

  private currentLayer: SearchNode[] = [];
  private nextLayer: SearchNode[] = [];
  private currentIndex = 0;
  private pendingParents = new Map<string, ParentInfo>();
  private parents = new Map<string, ParentInfo>();
  private frontier = new Map<string, FrontierEntry>();
  private iterationExpansions = 0;
  private totalExpansions = 0;
  private summary: FrontierSummary | null = null;
  private rootKey: string;

  constructor(
    context: BoardContext,
    sessionData: SessionDerivedData,
    initialState: BitState,
    direction: Direction,
  ) {
    this.context = context;
    this.sessionData = sessionData;
    this.rootState = initialState;
    this.direction = direction;
    this.transitions = direction === 'forward' ? context.forwardMovesByFrom : context.backwardMovesByFrom;
    const boardKey = this.context.stateKey(initialState);
    this.rootKey = `${boardKey}|idle`;
    this.parents.set(this.rootKey, { parentKey: null, move: null });
  }

  public get isForward(): boolean {
    return this.direction === 'forward';
  }

  public getFrontier(): Map<string, FrontierEntry> {
    return this.frontier;
  }

  public getSummary(): FrontierSummary | null {
    return this.summary;
  }

  public getTotalNodes(): number {
    return this.totalExpansions;
  }

  public getBound(): number {
    return this.bound;
  }

  public getNextBound(): number {
    return this.nextBound;
  }

  public getLastIterationExpansions(): number {
    return this.lastIterationExpansions;
  }

  public startIteration(initialBound: number | null = null): void {
    if (this.exhausted) return;
    const rootCounts = this.context.typeCounts(this.rootState);
    const rootHeuristic = this.computeHeuristic(this.rootState, rootCounts);
    const rootPagoda = this.sessionData.pagodaEnabled
      ? this.context.pagodaValue(this.rootState, this.sessionData.pagodaWeights)
      : 0;
    if (!this.boundInitialized) {
      this.bound = rootHeuristic;
      this.boundInitialized = true;
    }
    this.currentLayer = [
      this.createNode(
        this.rootState,
        IDLE_PEG,
        0,
        rootHeuristic,
        rootCounts,
        this.sessionData.pagodaEnabled ? rootPagoda : undefined,
      ),
    ];
    this.currentIndex = 0;
    this.nextLayer = [];
    this.pendingParents = new Map();
    this.frontier = new Map();
    this.summary = null;
    this.iterationExpansions = 0;
    this.nextBound = Infinity;
    if (initialBound !== null) {
      this.bound = initialBound;
    }
  }

  public step(
    opposingFrontier: Map<string, FrontierEntry> | null,
    opposingSummary: FrontierSummary | null,
    session: BidirectionalBfidaSession,
    deadline: number,
  ): boolean {
    if (this.exhausted) {
      return true;
    }
    while (performance.now() < deadline) {
      if (this.currentIndex >= this.currentLayer.length) {
        if (!this.prepareNextLayer()) {
          this.finishIteration();
          return true;
        }
        continue;
      }
      const node = this.currentLayer[this.currentIndex++];
      const typeCounts = node.typeCounts ?? this.context.typeCounts(node.state);
      if (
        this.sessionData.enforceTargetTypeInvariant &&
        typeCounts[this.sessionData.targetType] === 0 &&
        node.state !== this.sessionData.goalStateBit
      ) {
        continue;
      }
      const pagodaEnabled = this.sessionData.pagodaEnabled;
      const pagodaValue = pagodaEnabled
        ? node.pagoda ?? this.context.pagodaValue(node.state, this.sessionData.pagodaWeights)
        : 0;
      if (pagodaEnabled) {
        if (this.isForward) {
          if (pagodaValue + PAGODA_TOLERANCE < this.sessionData.goalPagodaValue) {
            continue;
          }
          if (opposingSummary && pagodaValue - PAGODA_TOLERANCE > opposingSummary.maxPagoda) {
            continue;
          }
        } else {
          if (pagodaValue - PAGODA_TOLERANCE > this.sessionData.startPagodaValue) {
            continue;
          }
          if (opposingSummary && pagodaValue + PAGODA_TOLERANCE < opposingSummary.minPagoda) {
            continue;
          }
        }
      }
      if (opposingSummary) {
        if (this.isForward) {
          if (!this.meetsMinCounts(typeCounts, opposingSummary.minTypeCounts)) {
            continue;
          }
        } else {
          if (!this.meetsMaxCounts(typeCounts, opposingSummary.maxTypeCounts)) {
            continue;
          }
        }
      }
      const heuristicValue = node.heuristic ?? this.computeHeuristic(node.state, typeCounts);
      const isIdle = node.lastPeg === IDLE_PEG;
      const adjustedH = Math.max(0, heuristicValue - (isIdle ? 0 : 1));
      const f = node.g + adjustedH;
      if (isIdle) {
        if (opposingFrontier) {
          const match = opposingFrontier.get(node.boardKey);
          if (match) {
            session.tryUpdateSolution(this, node, match);
          }
        }
        session.considerPreviewPath(this, node, f);
      }
      if (f > this.bound) {
        this.nextBound = Math.min(this.nextBound, f);
        continue;
      }
      this.iterationExpansions += 1;
      this.totalExpansions += 1;
      const hadPrunedChild = this.expandNode(
        node,
        typeCounts,
        heuristicValue,
        pagodaEnabled ? pagodaValue : null,
      );
      if (isIdle && hadPrunedChild) {
        this.recordFrontier(node, typeCounts, pagodaEnabled ? pagodaValue : 0);
      }
    }
    return false;
  }

  private meetsMinCounts(counts: PegTypeCounts, limits: PegTypeCounts): boolean {
    for (let i = 0; i < limits.length; i++) {
      if (counts[i] < limits[i]) {
        return false;
      }
    }
    return true;
  }

  private meetsMaxCounts(counts: PegTypeCounts, limits: PegTypeCounts): boolean {
    for (let i = 0; i < limits.length; i++) {
      if (counts[i] > limits[i]) {
        return false;
      }
    }
    return true;
  }

  private recordFrontier(node: SearchNode, typeCounts: PegTypeCounts, pagodaValue: number): void {
    const existing = this.frontier.get(node.boardKey);
    if (!existing || node.g < existing.g) {
      this.frontier.set(node.boardKey, {
        boardKey: node.boardKey,
        nodeKey: node.key,
        state: node.state,
        g: node.g,
        typeCounts: [...typeCounts] as PegTypeCounts,
        pagoda: pagodaValue,
      });
    }
  }

  private expandNode(
    node: SearchNode,
    parentTypeCounts: PegTypeCounts,
    nodeHeuristic: number,
    parentPagoda: number | null,
  ): boolean {
    let prunedChild = false;
    if (node.lastPeg !== IDLE_PEG) {
      const restNode = this.createNode(
        node.state,
        IDLE_PEG,
        node.g,
        nodeHeuristic,
        cloneTypeCounts(parentTypeCounts),
        this.sessionData.pagodaEnabled && parentPagoda !== null ? parentPagoda : undefined,
      );
      const restKey = restNode.key;
      if (!this.pendingParents.has(restKey)) {
        this.pendingParents.set(restKey, { parentKey: node.key, move: null });
        this.nextLayer.push(restNode);
      }
    }
    const candidates = node.lastPeg === IDLE_PEG ? this.context.pegIndices(node.state) : [node.lastPeg];
    candidates.forEach(fromIdx => {
      const moves = this.transitions.get(fromIdx) ?? [];
      for (const move of moves) {
        const nextState = this.applyMove(node.state, move);
        if (nextState === null) continue;
        const childCounts = this.computeChildTypeCounts(parentTypeCounts, move);
        const childHeuristic = this.computeHeuristic(nextState, childCounts);
        const childAdjustedH = Math.max(0, childHeuristic - 1);
        const continuing = node.lastPeg === move.from;
        const childG = node.g + (continuing ? 0 : 1);
        const childPagoda =
          this.sessionData.pagodaEnabled && parentPagoda !== null
            ? this.computeChildPagodaValue(parentPagoda, move)
            : null;
        if (this.sessionData.pagodaEnabled && childPagoda !== null) {
          if (this.isForward) {
            if (childPagoda + PAGODA_TOLERANCE < this.sessionData.goalPagodaValue) {
              prunedChild = true;
              continue;
            }
          } else if (childPagoda - PAGODA_TOLERANCE > this.sessionData.startPagodaValue) {
            prunedChild = true;
            continue;
          }
        }
        const childF = childG + childAdjustedH;
        if (childF > this.bound) {
          prunedChild = true;
          this.nextBound = Math.min(this.nextBound, childF);
          continue;
        }
        const childNode = this.createNode(
          nextState,
          move.to,
          childG,
          childHeuristic,
          childCounts,
          this.sessionData.pagodaEnabled && childPagoda !== null ? childPagoda : undefined,
        );
        this.pendingParents.set(childNode.key, { parentKey: node.key, move });
        this.nextLayer.push(childNode);
      }
    });
    return prunedChild;
  }

  private computeChildPagodaValue(parentPagoda: number, move: Transition): number {
    const weights = this.sessionData.pagodaWeights;
    if (!weights) {
      return parentPagoda;
    }
    if (this.isForward) {
      return parentPagoda - weights[move.from] - weights[move.over] + weights[move.to];
    }
    return parentPagoda - weights[move.from] + weights[move.over] + weights[move.to];
  }

  private computeChildTypeCounts(parentCounts: PegTypeCounts, move: Transition): PegTypeCounts {
    const counts = cloneTypeCounts(parentCounts);
    const fromType = this.context.pegType(move.from);
    const overType = this.context.pegType(move.over);
    const toType = this.context.pegType(move.to);
    if (this.isForward) {
      counts[fromType] -= 1;
      counts[overType] -= 1;
      counts[toType] += 1;
    } else {
      counts[fromType] -= 1;
      counts[overType] += 1;
      counts[toType] += 1;
    }
    return counts;
  }

  private applyMove(state: BitState, move: Transition): BitState | null {
    if (this.isForward) {
      if (!(state & this.context.bit(move.from))) return null;
      if (!(state & this.context.bit(move.over))) return null;
      if (state & this.context.bit(move.to)) return null;
      return state - this.context.bit(move.from) - this.context.bit(move.over) + this.context.bit(move.to);
    }
    if (!(state & this.context.bit(move.from))) return null;
    if (state & this.context.bit(move.over)) return null;
    if (state & this.context.bit(move.to)) return null;
    return state + this.context.bit(move.over) + this.context.bit(move.to) - this.context.bit(move.from);
  }

  private createNode(
    state: BitState,
    lastPeg: number,
    g: number,
    heuristic?: number,
    typeCounts?: PegTypeCounts,
    pagoda?: number,
  ): SearchNode {
    const boardKey = this.context.stateKey(state);
    const pegKey = lastPeg === IDLE_PEG ? 'idle' : String(lastPeg);
    return {
      state,
      boardKey,
      key: `${boardKey}|${pegKey}`,
      g,
      lastPeg,
      heuristic,
      typeCounts,
      pagoda,
    };
  }

  private prepareNextLayer(): boolean {
    if (!this.nextLayer.length) {
      this.currentLayer = [];
      this.currentIndex = 0;
      return false;
    }
    const dedup = new Map<string, SearchNode>();
    this.nextLayer.forEach(node => {
      const existing = dedup.get(node.key);
      if (!existing || node.g < existing.g) {
        dedup.set(node.key, node);
      }
    });
    dedup.forEach((_node, key) => {
      const parent = this.pendingParents.get(key);
      if (parent) {
        this.parents.set(key, parent);
      }
    });
    this.pendingParents.clear();
    this.currentLayer = Array.from(dedup.values());
    this.currentIndex = 0;
    this.nextLayer = [];
    return true;
  }

  private finishIteration(): void {
    this.lastIterationExpansions = this.iterationExpansions;
    if (this.nextBound === Infinity) {
      this.exhausted = true;
    } else {
      this.bound = this.nextBound;
    }
    this.pruneParentMap();
    this.summary = this.buildSummary();
  }

  private pruneParentMap(): void {
    const requiredKeys = new Set<string>();
    this.frontier.forEach(entry => {
      requiredKeys.add(entry.nodeKey);
    });
    const queue = Array.from(requiredKeys);
    while (queue.length) {
      const key = queue.pop();
      if (key === undefined) continue;
      const parent = this.parents.get(key);
      if (parent && parent.parentKey && !requiredKeys.has(parent.parentKey)) {
        requiredKeys.add(parent.parentKey);
        queue.push(parent.parentKey);
      }
    }
    const pruned = new Map<string, ParentInfo>();
    requiredKeys.forEach(key => {
      const info = this.parents.get(key);
      if (info) {
        pruned.set(key, info);
      }
    });
    if (this.parents.has(this.rootKey)) {
      pruned.set(this.rootKey, this.parents.get(this.rootKey)!);
    }
    this.parents = pruned;
  }

  private buildSummary(): FrontierSummary | null {
    if (!this.frontier.size) {
      return null;
    }
    const minCounts: PegTypeCounts = [Infinity, Infinity, Infinity, Infinity];
    const maxCounts: PegTypeCounts = [0, 0, 0, 0];
    const enforcePagoda = this.sessionData.pagodaEnabled;
    let minPagoda = enforcePagoda ? Infinity : -Infinity;
    let maxPagoda = enforcePagoda ? -Infinity : Infinity;
    this.frontier.forEach(entry => {
      entry.typeCounts.forEach((value, idx) => {
        minCounts[idx] = Math.min(minCounts[idx], value);
        maxCounts[idx] = Math.max(maxCounts[idx], value);
      });
      if (enforcePagoda) {
        minPagoda = Math.min(minPagoda, entry.pagoda);
        maxPagoda = Math.max(maxPagoda, entry.pagoda);
      }
    });
    minCounts.forEach((value, idx) => {
      if (!Number.isFinite(value)) {
        minCounts[idx] = 0;
      }
    });
    if (!enforcePagoda) {
      minPagoda = -Infinity;
      maxPagoda = Infinity;
    } else {
      if (!Number.isFinite(minPagoda)) {
        minPagoda = 0;
      }
      if (!Number.isFinite(maxPagoda)) {
        maxPagoda = 0;
      }
    }
    return {
      minTypeCounts: minCounts,
      maxTypeCounts: maxCounts,
      minPagoda,
      maxPagoda,
    };
  }

  private computeHeuristic(state: BitState, typeCounts: PegTypeCounts): number {
    const hc = this.context.popCount(state & this.sessionData.cornerMaskWithoutTarget);
    let ht = 0;
    for (let type = 0; type < 4; type++) {
      if (this.context.cornerCaptureTypes.has(type)) continue;
      const goalCount = this.sessionData.goalTypeCounts[type];
      const excess = Math.max(0, typeCounts[type] - goalCount);
      if (excess === 0) continue;
      const capacity = this.context.maxCapturesPerType[type];
      ht = Math.max(ht, Math.ceil(excess / capacity));
    }
    const hm = 0; // Temporarily disable inferred Merson regions to protect heuristic consistency.
    return hc + Math.max(ht, hm);
  }

  public buildMoves(nodeKey: string): SolverMove[] {
    const moves: Transition[] = [];
    let key: string | null = nodeKey;
    while (key && key !== this.rootKey) {
      const parent = this.parents.get(key);
      if (!parent) break;
      if (parent.move) {
        moves.push(parent.move);
      }
      key = parent.parentKey;
    }
    moves.reverse();
    return moves.map(move => this.toSolverMove(move));
  }

  private toSolverMove(move: Transition): SolverMove {
    const from = this.context.cells[move.from];
    const to = this.context.cells[move.to];
    const jump = this.context.cells[move.over];
    if (this.isForward) {
      return { from, to, jump };
    }
    return { from, to, jump };
  }
}

export class BidirectionalBfidaSession {
  private readonly context: BoardContext;
  private readonly sessionData: SessionDerivedData;
  private readonly forward: DirectionState;
  private readonly backward: DirectionState;
  private activeDirection: DirectionState | null = null;
  private forwardFrontier: Map<string, FrontierEntry> | null = null;
  private backwardFrontier: Map<string, FrontierEntry> | null = null;
  private forwardSummary: FrontierSummary | null = null;
  private backwardSummary: FrontierSummary | null = null;
  private bestMoves: SolverMove[] = [];
  private bestCost = Infinity;
  private solved = false;
  private done = false;
  private reportedCompletion = false;
  private previewMoves: SolverMove[] = [];
  private previewEstimate = Infinity;

  constructor(
    context: BoardContext,
    sessionData: SessionDerivedData,
    initialState: BitState,
    goalState: BitState,
    targetCell: string,
  ) {
    this.context = context;
    this.sessionData = sessionData;
    this.forward = new DirectionState(this.context, this.sessionData, initialState, 'forward');
    this.backward = new DirectionState(this.context, this.sessionData, goalState, 'backward');
    console.log('[Solver] Initialized BFIDA session', {
      cells: this.context.cells.length,
      initialPegCount: this.context.popCount(initialState),
      target: targetCell,
    });
  }

  public runChunk(maxMs: number): SolverSessionProgress {
    const deadline = performance.now() + maxMs;
    while (performance.now() < deadline && !this.done) {
      const direction = this.chooseDirection();
      if (!direction) break;
      const opposingFrontier = direction.isForward ? this.backwardFrontier : this.forwardFrontier;
      const opposingSummary = direction.isForward ? this.backwardSummary : this.forwardSummary;
      const finished = direction.step(opposingFrontier, opposingSummary, this, deadline);
      if (finished) {
        const summary = direction.getSummary();
        if (direction.isForward) {
          this.forwardFrontier = direction.getFrontier();
          this.forwardSummary = summary;
        } else {
          this.backwardFrontier = direction.getFrontier();
          this.backwardSummary = summary;
        }
        console.log('[Solver] Completed iteration', {
          direction: direction.isForward ? 'forward' : 'backward',
          bound: direction.getBound(),
          nextBound: direction.getNextBound(),
          expansions: direction.getLastIterationExpansions(),
        });
        if (direction.exhausted && this.bestCost === Infinity) {
          this.done = true;
        }
        this.activeDirection = null;
        this.checkForCompletion();
      }
    }
    return {
      done: this.done,
      solved: this.done && this.solved,
      nodesExplored: this.forward.getTotalNodes() + this.backward.getTotalNodes(),
      currentPath: this.previewMoves.length ? [...this.previewMoves] : [...this.bestMoves],
      bestMoves: [...this.bestMoves],
    };
  }

  public tryUpdateSolution(direction: DirectionState, node: SearchNode, match: FrontierEntry): void {
    const totalCost = node.g + match.g;
    if (totalCost >= this.bestCost) {
      return;
    }
    const forwardDir = direction.isForward ? direction : this.forward;
    const backwardDir = direction.isForward ? this.backward : direction;
    const forwardKey = direction.isForward ? node.key : match.nodeKey;
    const backwardKey = direction.isForward ? match.nodeKey : node.key;
    const forwardMoves = forwardDir.buildMoves(forwardKey);
    const backwardMoves = backwardDir.buildMoves(backwardKey);
    const convertedBackward = this.convertBackwardSegment(backwardMoves);
    this.bestMoves = direction.isForward
      ? [...forwardMoves, ...convertedBackward]
      : [...forwardMoves, ...convertedBackward];
    this.bestCost = totalCost;
    this.previewMoves = [...this.bestMoves];
    this.previewEstimate = totalCost;
    console.log('[Solver] Found better solution candidate', {
      cost: totalCost,
      nodesExplored: this.forward.getTotalNodes() + this.backward.getTotalNodes(),
    });
    if (this.forward.bound >= totalCost && this.backward.bound >= totalCost) {
      this.done = true;
      this.solved = true;
    }
  }

  public considerPreviewPath(direction: DirectionState, node: SearchNode, estimate: number): void {
    if (!direction.isForward) return;
    const preview = direction.buildMoves(node.key);
    const betterEstimate = estimate < this.previewEstimate;
    const deeper = preview.length > this.previewMoves.length;
    if (!betterEstimate && !deeper) {
      return;
    }
    this.previewMoves = preview;
    this.previewEstimate = estimate;
    console.log('[Solver] Preview path updated', {
      estimate,
      length: preview.length,
    });
  }

  private convertBackwardSegment(moves: SolverMove[]): SolverMove[] {
    const result: SolverMove[] = [];
    for (let i = moves.length - 1; i >= 0; i--) {
      const move = moves[i];
      result.push({ from: move.to, to: move.from, jump: move.jump });
    }
    return result;
  }

  private chooseDirection(): DirectionState | null {
    if (this.done) return null;
    if (this.activeDirection) return this.activeDirection;
    const candidates = [this.forward, this.backward].filter(dir => !dir.exhausted);
    if (!candidates.length) {
      this.done = true;
      return null;
    }
    let chosen = candidates[0];
    if (candidates.length === 2) {
      chosen = candidates[0].lastIterationExpansions <= candidates[1].lastIterationExpansions ? candidates[0] : candidates[1];
    }
    chosen.startIteration();
    console.log('[Solver] Starting iteration', {
      direction: chosen.isForward ? 'forward' : 'backward',
      bound: chosen.getBound(),
    });
    this.activeDirection = chosen;
    return chosen;
  }

  private checkForCompletion(): void {
    if (this.bestCost !== Infinity) {
      const forwardReady = this.forward.bound >= this.bestCost || this.forward.exhausted;
      const backwardReady = this.backward.bound >= this.bestCost || this.backward.exhausted;
      if (forwardReady && backwardReady) {
        this.done = true;
        this.solved = true;
      }
    }
    if (this.done && !this.reportedCompletion) {
      this.reportedCompletion = true;
      console.log('[Solver] Session completed', {
        solved: this.solved,
        cost: this.bestCost,
        nodesExplored: this.forward.getTotalNodes() + this.backward.getTotalNodes(),
      });
    }
  }
}

export class BidirectionalBfidaSolver {
  private readonly context: BoardContext;
  private readonly enforceTargetTypeInvariant: boolean;

  constructor(allowedMoves: Map<string, Map<string, string>>, options?: SolverOptions) {
    this.context = new BoardContext(allowedMoves);
    this.enforceTargetTypeInvariant = options?.enforceTargetTypeInvariant ?? true;
  }

  public createSession(initial: Set<string>, target: string): BidirectionalBfidaSession | null {
    const targetIndex = this.context.cellToIndex.get(target);
    if (targetIndex === undefined) {
      throw new Error('Target cell is not part of the board.');
    }
    const initialState = this.context.stateFromSet(initial);
    const goalState = this.context.bit(targetIndex);
    const initClass = this.context.positionClassKey(initialState);
    const goalClass = this.context.positionClassKey(goalState);
    if (!this.context.haveMatchingPositionClasses(initialState, goalState)) {
      console.warn('[Solver] Position class mismatch; instance unsolvable.', {
        target,
        initialClass: initClass,
        goalClass: goalClass,
      });
      return null;
    }
    const targetType = this.context.pegType(targetIndex);
    const goalTypeCounts: PegTypeCounts = [0, 0, 0, 0];
    goalTypeCounts[targetType] = 1;
    const pagodaWeights = this.context.pagodaWeights(targetIndex);
    const pagodaEnabled = !!pagodaWeights;
    const startPagodaValue = pagodaEnabled ? this.context.pagodaValue(initialState, pagodaWeights) : 0;
    const goalPagodaValue = pagodaEnabled ? this.context.pagodaValue(goalState, pagodaWeights) : 0;
    const sessionData: SessionDerivedData = {
      cornerMaskWithoutTarget: this.context.cornerMask & ~this.context.bit(targetIndex),
      goalTypeCounts,
      targetIndex,
      targetType,
      goalStateBit: goalState,
      pagodaWeights,
      goalPagodaValue,
      startPagodaValue,
      pagodaEnabled,
      enforceTargetTypeInvariant: this.enforceTargetTypeInvariant,
    };
    return new BidirectionalBfidaSession(this.context, sessionData, initialState, goalState, target);
  }

  public solve(initial: Set<string>, target: string): SolverResult {
    const start = performance.now();
    const session = this.createSession(initial, target);
    if (!session) {
      return {
        solved: false,
        moves: [],
        bestMoves: [],
        nodesExplored: 0,
        durationMs: Math.max(0, performance.now() - start),
      };
    }
    let progress: SolverSessionProgress;
    do {
      progress = session.runChunk(10);
    } while (!progress.done);
    return {
      solved: progress.solved,
      moves: progress.bestMoves,
      bestMoves: progress.bestMoves,
      nodesExplored: progress.nodesExplored,
      durationMs: Math.max(0, performance.now() - start),
    };
  }
}
