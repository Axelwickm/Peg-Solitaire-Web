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

type Axes = string[][];

const crossAxes = buildGridAxes(crossHoles, 7, 7);
const triangleAxes = [
  ...triangleRows,
  ...collectTriangleAxes(triangleRows, 1, 0),
  ...collectTriangleAxes(triangleRows, 1, 1),
];

const crossAllowedMoves = buildAllowedMovesFromAxes(crossAxes);
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
