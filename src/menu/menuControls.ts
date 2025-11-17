import { MenuControl } from './MenuControl';

export function createMenuControls(): MenuControl[] {
  const panels = Array.from(document.querySelectorAll('.menu-panel'));
  return panels.map(panel => new MenuControl(panel));
}
