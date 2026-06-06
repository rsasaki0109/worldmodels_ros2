// Render a sequence of frames from a render_*.html page and screenshot each.
//   node capture.js <html> <data.json> <outDir> <numFrames>
// Uses the system Chrome via playwright-core (no browser download).
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const [htmlPath, dataPath, outDir, nStr] = process.argv.slice(2);
  if (!htmlPath || !dataPath || !outDir || !nStr) {
    console.error('usage: node capture.js <html> <data.json> <outDir> <numFrames>');
    process.exit(2);
  }
  const N = parseInt(nStr, 10);
  const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({ channel: 'chrome', args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 760, height: 420 }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.resolve(htmlPath));
  await page.waitForFunction('window.__ready === true');
  await page.evaluate((d) => { if (window.setData) window.setData(d); else window.DATA = d; }, data);
  await page.waitForFunction('window.imagesReady === undefined || window.imagesReady === true');

  const canvas = await page.$('#c');
  for (let i = 0; i < N; i++) {
    await page.evaluate((k) => window.renderFrame(k), i);
    await page.waitForTimeout(15);
    await canvas.screenshot({ path: path.join(outDir, `f${String(i).padStart(3, '0')}.png`) });
  }
  await browser.close();
  console.log(`captured ${N} frames -> ${outDir}`);
})().catch((e) => { console.error(e); process.exit(1); });
