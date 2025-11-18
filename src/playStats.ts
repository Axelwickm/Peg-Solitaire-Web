import { shapes } from './shapes';
import type { KongmingGame, PlayLogEntry } from './Game';

const GRAPH_WIDTH = 320;
const GRAPH_HEIGHT = 160;
const GRAPH_PADDING = 22;

export interface PlayStatsController {
  refreshShapeFilter(shapeId: string): void;
}

export function initPlayStats(game: KongmingGame): PlayStatsController | null {
  const playStatsEl = document.getElementById('play-stats');
  const playTimerEl = document.getElementById('play-timer');
  const playTimerButton = document.getElementById('play-timer-button');
  const playStatsPopup = document.getElementById('play-stats-popup');
  const statsGraphEl = document.getElementById('stats-graph') as SVGSVGElement | null;
  const statsCountEl = document.getElementById('play-stats-count');
  const statsShapeFilter = document.getElementById('stats-shape-filter') as HTMLSelectElement | null;
  const exportStatsButton = document.getElementById('export-stats');

  game.setPlayStatsElements(playStatsEl, playTimerEl);

  if (
    !playTimerButton ||
    !playStatsPopup ||
    !statsGraphEl ||
    !statsCountEl ||
    !statsShapeFilter ||
    !exportStatsButton
  ) {
    return null;
  }

  let statsPopupVisible = false;
  let statsEntriesCache: PlayLogEntry[] | null = null;
  const timerButton = playTimerButton!;
  const popup = playStatsPopup!;
  const graph = statsGraphEl!;
  const countEl = statsCountEl!;
  const shapeFilter = statsShapeFilter!;
  const exportButton = exportStatsButton!;

  function computeScore(entry: PlayLogEntry): number {
    const base = 120000 / Math.max(entry.durationMs, 1);
    const capped = Math.min(100, Math.max(10, Math.round(base)));
    return capped;
  }

  function ensureStatsEntries(forceReload = false): PlayLogEntry[] {
    if (forceReload || statsEntriesCache === null) {
      statsEntriesCache = game.getPlayLogEntries();
    }
    return statsEntriesCache;
  }

  function filterEntries(entries: PlayLogEntry[]): PlayLogEntry[] {
    const selected = shapeFilter.value;
    if (!selected) {
      return entries;
    }
    return entries.filter(entry => entry.shapeId === selected);
  }

  function formatDurationLabel(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  function renderStatsGraph(entries: PlayLogEntry[]): void {
    const svg = graph;
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    const samples = entries.slice(-20);
    if (!samples.length) {
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.textContent = 'No runs yet';
      text.setAttribute('x', (GRAPH_WIDTH / 2).toString());
      text.setAttribute('y', (GRAPH_HEIGHT / 2).toString());
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', 'currentColor');
      text.setAttribute('font-size', '0.8rem');
      svg.appendChild(text);
      return;
    }
    const graphWidth = GRAPH_WIDTH - GRAPH_PADDING * 2;
    const graphHeight = GRAPH_HEIGHT - GRAPH_PADDING * 2;
    const durations = samples.map(entry => entry.durationMs);
    const maxDuration = Math.max(...durations, 1);
    const minDuration = Math.min(...durations);
    const scores = samples.map(computeScore);
    const maxScore = Math.max(...scores, 1);
    const minScore = Math.min(...scores);
    const xStep = samples.length > 1 ? graphWidth / (samples.length - 1) : graphWidth / 2;
    const formatPoint = (value: number, maxValue: number, index: number): string => {
      const ratio = maxValue === 0 ? 0 : value / maxValue;
      const x = GRAPH_PADDING + (samples.length === 1 ? graphWidth / 2 : index * xStep);
      const y = GRAPH_PADDING + graphHeight - ratio * graphHeight;
      return `${x},${y}`;
    };
    const axisLine = (x1: number, y1: number, x2: number, y2: number, color: string): SVGLineElement => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1.toString());
      line.setAttribute('y1', y1.toString());
      line.setAttribute('x2', x2.toString());
      line.setAttribute('y2', y2.toString());
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', '1');
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('opacity', '0.3');
      return line;
    };
    svg.appendChild(
      axisLine(
        GRAPH_PADDING,
        GRAPH_PADDING + graphHeight,
        GRAPH_PADDING + graphWidth,
        GRAPH_PADDING + graphHeight,
        'currentColor',
      ),
    );
    svg.appendChild(
      axisLine(
        GRAPH_PADDING + graphWidth,
        GRAPH_PADDING,
        GRAPH_PADDING + graphWidth,
        GRAPH_PADDING + graphHeight,
        '#ffcc57',
      ),
    );
    svg.appendChild(
      axisLine(
        GRAPH_PADDING,
        GRAPH_PADDING,
        GRAPH_PADDING,
        GRAPH_PADDING + graphHeight,
        '#76b7ff',
      ),
    );
    const addLabel = (x: number, y: number, text: string, color: string, anchor = 'end'): void => {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.textContent = text;
      label.setAttribute('x', x.toString());
      label.setAttribute('y', y.toString());
      label.setAttribute('fill', color);
      label.setAttribute('text-anchor', anchor);
      label.setAttribute('font-size', '0.5rem');
      svg.appendChild(label);
    };
    addLabel(GRAPH_PADDING - 6, GRAPH_PADDING + 6, formatDurationLabel(maxDuration), '#76b7ff');
    addLabel(
      GRAPH_PADDING - 6,
      GRAPH_PADDING + graphHeight,
      formatDurationLabel(minDuration),
      '#76b7ff',
    );
    addLabel(
      GRAPH_PADDING + graphWidth + 6,
      GRAPH_PADDING + 6,
      maxScore.toString(),
      '#ffcc57',
      'start',
    );
    addLabel(
      GRAPH_PADDING + graphWidth + 6,
      GRAPH_PADDING + graphHeight,
      minScore.toString(),
      '#ffcc57',
      'start',
    );
    for (let i = 0; i < 3; i += 1) {
      const y = GRAPH_PADDING + (graphHeight / 2) * i;
      svg.appendChild(
        axisLine(GRAPH_PADDING, y, GRAPH_PADDING + graphWidth, y, 'currentColor'),
      );
    }
    const timePoints = samples
      .map((entry, index) => formatPoint(entry.durationMs, maxDuration, index))
      .join(' ');
    const scorePoints = samples
      .map((entry, index) => formatPoint(computeScore(entry), maxScore, index))
      .join(' ');
    const createPolyline = (points: string, color: string): SVGPolylineElement => {
      const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      polyline.setAttribute('points', points);
      polyline.setAttribute('fill', 'none');
      polyline.setAttribute('stroke', color);
      polyline.setAttribute('stroke-width', '2.5');
      polyline.setAttribute('stroke-linecap', 'round');
      polyline.setAttribute('stroke-linejoin', 'round');
      return polyline;
    };
    svg.appendChild(createPolyline(timePoints, '#76b7ff'));
    svg.appendChild(createPolyline(scorePoints, '#ffcc57'));
    const lastIndex = samples.length - 1;
    const lastEntry = samples[lastIndex];
    if (lastEntry) {
      const x = GRAPH_PADDING + (samples.length === 1 ? graphWidth / 2 : lastIndex * xStep);
      const timeCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      timeCircle.setAttribute('cx', x.toString());
      timeCircle.setAttribute(
        'cy',
        (
          GRAPH_PADDING +
          graphHeight -
          (lastEntry.durationMs / maxDuration) * graphHeight
        ).toString(),
      );
      timeCircle.setAttribute('r', '4');
      timeCircle.setAttribute('fill', '#76b7ff');
      timeCircle.setAttribute('stroke', '#fff');
      timeCircle.setAttribute('stroke-width', '1.5');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `Time ${formatDurationLabel(lastEntry.durationMs)}`;
      timeCircle.appendChild(title);
      svg.appendChild(timeCircle);
      const scoreCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      scoreCircle.setAttribute('cx', x.toString());
      scoreCircle.setAttribute(
        'cy',
        (
          GRAPH_PADDING +
          graphHeight -
          (computeScore(lastEntry) / maxScore) * graphHeight
        ).toString(),
      );
      scoreCircle.setAttribute('r', '3.5');
      scoreCircle.setAttribute('fill', '#ffcc57');
      scoreCircle.setAttribute('stroke', '#fff');
      scoreCircle.setAttribute('stroke-width', '1');
      const scoreTitle = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      scoreTitle.textContent = `Score ${computeScore(lastEntry)}`;
      scoreCircle.appendChild(scoreTitle);
      svg.appendChild(scoreCircle);
    }
  }

  function populateShapeFilter(): void {
    shapeFilter.innerHTML = '';
    const allOption = document.createElement('option');
    allOption.value = '';
    allOption.textContent = 'All shapes';
    shapeFilter.appendChild(allOption);
    shapes.forEach(shape => {
      const option = document.createElement('option');
      option.value = shape.id;
      option.textContent = shape.name;
    shapeFilter.appendChild(option);
    });
    refreshShapeFilter(game.getCurrentShape().id);
  }

  function refreshShapeFilter(shapeId: string): void {
    shapeFilter.value = shapeId;
  }

  function updateStatsPopup(): void {
    const baseEntries = ensureStatsEntries();
    const filteredEntries = filterEntries(baseEntries);
    const summary =
      filteredEntries.length === baseEntries.length
        ? `${filteredEntries.length}`
        : `${filteredEntries.length} (${baseEntries.length} total)`;
    countEl.textContent = `Games played: ${summary}`;
    renderStatsGraph(filteredEntries);
  }

  function toggleStatsPopup(force?: boolean): void {
    const desired = typeof force === 'boolean' ? force : !statsPopupVisible;
    if (desired === statsPopupVisible) return;
    statsPopupVisible = desired;
    popup.classList.toggle('visible', desired);
    popup.setAttribute('aria-hidden', desired ? 'false' : 'true');
    if (desired) {
      statsEntriesCache = null;
      refreshShapeFilter(game.getCurrentShape().id);
      updateStatsPopup();
    }
  }

  function exportFilteredEntries(): void {
    const entries = filterEntries(ensureStatsEntries());
    if (!entries.length) return;
    const header = [
      'timestamp',
      'duration_ms',
      'duration_label',
      'shape_id',
      'perfect',
      'pegs_left',
    ];
    const rows = entries.map(entry =>
      [
        entry.timestamp,
        entry.durationMs.toString(),
        formatDurationLabel(entry.durationMs),
        entry.shapeId,
        entry.perfect ? 'perfect' : 'regular',
        entry.pegsLeft.toString(),
      ].join(','),
    );
    const csv = [header.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `kongming-stats-${new Date().toISOString().slice(0, 10)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  populateShapeFilter();

  timerButton.addEventListener('click', event => {
    event.stopPropagation();
    toggleStatsPopup();
  });
  popup.addEventListener('click', event => {
    event.stopPropagation();
  });
  document.addEventListener('click', () => {
    if (statsPopupVisible) {
      toggleStatsPopup(false);
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && statsPopupVisible) {
      toggleStatsPopup(false);
    }
  });
  shapeFilter.addEventListener('change', () => {
    updateStatsPopup();
  });
  document.addEventListener('playlog:updated', () => {
    statsEntriesCache = null;
    if (statsPopupVisible) {
      updateStatsPopup();
    }
  });
  exportButton.addEventListener('click', exportFilteredEntries);

  return {
    refreshShapeFilter,
  };
}
