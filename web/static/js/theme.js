/** Explicit themes keep chart and page contrast consistent. */
const saved = (() => { try { return localStorage.getItem('spectra-theme'); } catch { return null; } })();
let theme = ['dark', 'light'].includes(saved) ? saved : matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
function apply() {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme === 'light' ? 'only light' : 'dark';
  const button = document.getElementById('btn-theme');
  if (button) {
    button.textContent = `Theme: ${theme}`;
    button.setAttribute('aria-label', `Use ${theme === 'dark' ? 'light' : 'dark'} theme`);
  }
}
apply();
export function initializeTheme() {
  apply();
  document.getElementById('btn-theme').addEventListener('click', () => {
    theme = theme === 'dark' ? 'light' : 'dark';
    apply();
    try { localStorage.setItem('spectra-theme', theme); } catch { /* Storage is optional. */ }
    document.dispatchEvent(new Event('themechange'));
  });
}
