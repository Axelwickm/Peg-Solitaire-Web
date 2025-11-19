export interface BoardShape {
  id: string;
  name: string;
  width: number;
  height: number;
  holes: string[];
  empty: string;
  layout?: 'grid' | 'triangle';
  allowedMoves: Map<string, Map<string, string>>;
  finalTargetDescription: string;
}

const crossHoles: string[] = [];
for (let r = 0; r < 7; r++) {
  for (let c = 0; c < 7; c++) {
    const inCross = (r >= 2 && r <= 4) || (c >= 2 && c <= 4);
    if (inCross) {
      crossHoles.push(`${r},${c}`);
    }
  }
}

const triangleWidth = 9;
const triangleHeight = 5;
const triangleRows = buildTriangleRows(triangleWidth, triangleHeight);
const triangleHoles = triangleRows.flat();
const europeanHoles = (() => {
  const base = new Set(crossHoles);
  ['1,1', '1,5', '5,1', '5,5'].forEach(cell => base.add(cell));
  return Array.from(base).sort((a, b) => {
    const [ar, ac] = a.split(',').map(Number);
    const [br, bc] = b.split(',').map(Number);
    return ar === br ? ac - bc : ar - br;
  });
})();
const germanHoles = (() => {
  const base = new Set<string>();
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const inCross = (r >= 3 && r <= 5) || (c >= 3 && c <= 5);
      if (inCross) {
        base.add(`${r},${c}`);
      }
    }
  }
  return Array.from(base).sort((a, b) => {
    const [ar, ac] = a.split(',').map(Number);
    const [br, bc] = b.split(',').map(Number);
    return ar === br ? ac - bc : ar - br;
  });
})();
const asymCrossHoles = (() => {
  const base = new Set<string>();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 9; c++) {
      const verticalArm = c >= 3 && c <= 5;
      const horizontalArm = r >= 3 && r <= 5 && c >= 1 && c <= 8;
      if (verticalArm || horizontalArm) {
        base.add(`${r},${c}`);
      }
    }
  }
  return Array.from(base).sort((a, b) => {
    const [ar, ac] = a.split(',').map(Number);
    const [br, bc] = b.split(',').map(Number);
    return ar === br ? ac - bc : ar - br;
  });
})();
const diamondHoles = (() => {
  const base = new Set<string>();
  const size = 9;
  const center = Math.floor(size / 2);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (Math.abs(r - center) + Math.abs(c - center) <= 4) {
        base.add(`${r},${c}`);
      }
    }
  }
  return Array.from(base).sort((a, b) => {
    const [ar, ac] = a.split(',').map(Number);
    const [br, bc] = b.split(',').map(Number);
    return ar === br ? ac - bc : ar - br;
  });
})();

type Axes = string[][];

const crossAxes = buildGridAxes(crossHoles, 7, 7);
const europeanAxes = buildGridAxes(europeanHoles, 7, 7);
const germanAxes = buildGridAxes(germanHoles, 9, 9);
const asymCrossAxes = buildGridAxes(asymCrossHoles, 9, 8);
const diamondAxes = buildGridAxes(diamondHoles, 9, 9);
const triangleAxes = [
  ...triangleRows,
  ...collectTriangleAxes(triangleRows, 1, 0),
  ...collectTriangleAxes(triangleRows, 1, 1),
];

const crossAllowedMoves = buildAllowedMovesFromAxes(crossAxes);
const europeanAllowedMoves = buildAllowedMovesFromAxes(europeanAxes);
const germanAllowedMoves = buildAllowedMovesFromAxes(germanAxes);
const asymCrossAllowedMoves = buildAllowedMovesFromAxes(asymCrossAxes);
const diamondAllowedMoves = buildAllowedMovesFromAxes(diamondAxes);
const triangleAllowedMoves = buildAllowedMovesFromAxes(triangleAxes);

export const shapes: BoardShape[] = [
  {
    id: 'cross',
    name: 'Kongming Cross',
    width: 7,
    height: 7,
    holes: crossHoles,
    empty: '3,3',
    layout: 'grid',
    allowedMoves: crossAllowedMoves,
    finalTargetDescription: 'center',
  },
  {
    id: 'triangle',
    name: 'Triangle 5x5',
    width: triangleWidth,
    height: triangleHeight,
    holes: triangleHoles,
    empty: '0,4',
    layout: 'triangle',
    allowedMoves: triangleAllowedMoves,
    finalTargetDescription: 'tip of the triangle',
  },
  {
    id: 'european',
    name: 'European Board',
    width: 7,
    height: 7,
    holes: europeanHoles,
    empty: '2,3',
    layout: 'grid',
    allowedMoves: europeanAllowedMoves,
    finalTargetDescription: 'spot above the center',
  },
  {
    id: 'diamond',
    name: 'Diamond 41',
    width: 9,
    height: 9,
    holes: diamondHoles,
    empty: '4,4',
    layout: 'grid',
    allowedMoves: diamondAllowedMoves,
    finalTargetDescription: 'center',
  },
  {
    id: 'german',
    name: 'German Board',
    width: 9,
    height: 9,
    holes: germanHoles,
    empty: '4,4',
    layout: 'grid',
    allowedMoves: germanAllowedMoves,
    finalTargetDescription: 'center',
  },
  {
    id: 'asym-cross',
    name: 'Asymmetrical Cross',
    width: 9,
    height: 8,
    holes: asymCrossHoles,
    empty: '4,4',
    layout: 'grid',
    allowedMoves: asymCrossAllowedMoves,
    finalTargetDescription: 'center',
  },
];

function buildGridAxes(holes: string[], width: number, height: number): Axes {
  const holeSet = new Set(holes);
  const axes: Axes = [];
  for (let row = 0; row < height; row++) {
    const rowAxis: string[] = [];
    for (let col = 0; col < width; col++) {
      const key = `${row},${col}`;
      if (holeSet.has(key)) {
        rowAxis.push(key);
      }
    }
    if (rowAxis.length) {
      axes.push(rowAxis);
    }
  }
  for (let col = 0; col < width; col++) {
    const colAxis: string[] = [];
    for (let row = 0; row < height; row++) {
      const key = `${row},${col}`;
      if (holeSet.has(key)) {
        colAxis.push(key);
      }
    }
    if (colAxis.length) {
      axes.push(colAxis);
    }
  }
  return axes;
}

function buildTriangleRows(width: number, height: number): string[][] {
  const rows: string[][] = [];
  for (let r = 0; r < height; r++) {
    const rowLength = r + 1;
    const offset = Math.floor((width - rowLength) / 2);
    const rowAxis: string[] = [];
    for (let c = 0; c < rowLength; c++) {
      rowAxis.push(`${r},${offset + c}`);
    }
    rows.push(rowAxis);
  }
  return rows;
}

function collectTriangleAxes(rows: string[][], dr: number, dIndex: number): Axes {
  const axes: Axes = [];
  const getKey = (row: number, index: number): string | undefined => {
    const rowList = rows[row];
    return rowList ? rowList[index] : undefined;
  };
  rows.forEach((rowList, row) => {
    rowList.forEach((_cell, index) => {
      if (getKey(row - dr, index - dIndex)) {
        return;
      }
      const axis: string[] = [];
      let currentRow = row;
      let currentIndex = index;
      while (true) {
        const key = getKey(currentRow, currentIndex);
        if (!key) break;
        axis.push(key);
        currentRow += dr;
        currentIndex += dIndex;
      }
      if (axis.length) {
        axes.push(axis);
      }
    });
  });
  return axes;
}

function buildAllowedMovesFromAxes(axes: Axes): Map<string, Map<string, string>> {
  const moves = new Map<string, Map<string, string>>();
  const addMove = (from: string, to: string, jump: string): void => {
    if (!moves.has(from)) {
      moves.set(from, new Map());
    }
    moves.get(from)?.set(to, jump);
  };
  axes.forEach(axis => {
    for (let i = 0; i <= axis.length - 3; i++) {
      const from = axis[i];
      const jump = axis[i + 1];
      const to = axis[i + 2];
      addMove(from, to, jump);
      addMove(to, from, jump);
    }
  });
  return moves;
}
