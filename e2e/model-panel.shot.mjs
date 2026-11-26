// Screenshot of the "Kinerja model AI" panel (thesis Figure 4.15c): officer -> Dashboard -> Detail teknis.
//   APP_URL=http://localhost:8000 node model-panel.shot.mjs
// Writes screenshots/figure-4-15c-model-panel.png (panel only, 2x) and ...-page.png (the whole screen).
import fs from 'node:fs';
import { chromium } from 'playwright';

const APP = process.env.APP_URL ?? 'http://localhost:8081';
const OUT = new URL('./screenshots/', import.meta.url).pathname;
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
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
await page.screenshot({ path: `${OUT}figure-4-15c-model-panel-page.png` });
console.log(`saved ${OUT}figure-4-15c-model-panel.png`);
await browser.close();
