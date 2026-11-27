// Screenshot of the "Kinerja model AI" panel (thesis Figure 4.15c): officer -> Dashboard -> Detail teknis.
//   APP_URL=http://localhost:8000 node model-panel.shot.mjs
// Writes screenshots/figure-4-15c-model-panel.png (panel only, 2x) and ...-page.png (the whole screen).
// The viewport is tall (430 x 3600) so the whole panel renders above the tab bar: the app scrolls inside its own
// container, so a phone-height page would cut the panel off at the bottom of the screen.
import fs from 'node:fs';
import { chromium } from 'playwright';

const APP = process.env.APP_URL ?? 'http://localhost:8081';
const OUT = new URL('./screenshots/', import.meta.url).pathname;
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const ctx = await browser.newContext({ viewport: { width: 430, height: 3600 }, deviceScaleFactor: 2, locale: 'id-ID' });
const page = await ctx.newPage();
await page.goto(`${APP}/login`);
await page.waitForTimeout(2000);
await page.getByText('✉️ Email', { exact: true }).last().click();
await page.getByLabel('✉️ Email').last().fill('officer@nutrisense.id');
await page.getByLabel('🔑 Kata sandi').last().fill('Demo1234!');
await page.getByText('Masuk', { exact: true }).last().click();
await page.waitForURL(/\/(home|dashboard)$/, { timeout: 15000 });
await page.goto(`${APP}/dashboard`);
await page.waitForTimeout(2500);
await page.getByText('Detail teknis', { exact: false }).last().click();
await page.waitForTimeout(1200);
const title = page.getByText('Kinerja model AI', { exact: true }).last();
await title.scrollIntoViewIfNeeded();
await page.waitForTimeout(500);
// The panel is the card that holds the title: walk up to the first ancestor that also holds the limitations box.
const card = title.locator('xpath=ancestor::div[.//*[contains(text(), "⚠️")]][1]');
await card.screenshot({ path: `${OUT}figure-4-15c-model-panel.png` });
// The whole screen, cropped to where the content ends (above the tab bar pinned to the bottom of the tall viewport).
const bottom = await page.evaluate((vh) => {
  let b = 0;
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (!r.height || r.bottom > vh - 160) continue;
    if (el.childElementCount === 0 && (el.textContent.trim() || el.tagName === 'svg' || el.tagName === 'IMG')) b = Math.max(b, r.bottom);
  }
  return b;
}, 3600);
await page.screenshot({ path: `${OUT}figure-4-15c-model-panel-page.png`, clip: { x: 0, y: 0, width: 430, height: Math.min(3600, Math.ceil(bottom) + 40) } });
console.log(`saved ${OUT}figure-4-15c-model-panel.png`);
await browser.close();
