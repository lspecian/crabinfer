/**
 * CrabInfer Electron E2E Smoke Test
 *
 * Uses Playwright's Electron support to:
 *  1. Launch the app
 *  2. Load a model via the sidebar
 *  3. Send a chat message
 *  4. Verify streamed response appears (tokens flowing)
 *  5. Stop inference and check no crashes
 *
 * Run:  npm run test:e2e
 * Or:   node e2e/smoke.mjs
 */

import { _electron as electron } from 'playwright'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(__dirname, '..')

// ── Config ───────────────────────────────────────────────────────────────────

const TIMEOUT_MODEL_LOAD = 60_000
const TIMEOUT_STREAMING_DETECT = 30_000
const TIMEOUT_STOP = 10_000
const TEST_MESSAGE = 'Say hello in exactly 5 words.'

let app
let consoleErrors = []
let passed = 0
let failed = 0

function log(msg) {
  console.log(`[e2e] ${msg}`)
}

function fail(msg) {
  console.error(`  ❌ FAIL: ${msg}`)
  failed++
}

function pass(msg) {
  console.log(`  ✅ PASS: ${msg}`)
  passed++
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  log('Launching Electron app...')

  app = await electron.launch({
    args: [projectRoot],
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: '',
      ELECTRON_NO_ATTACH_CONSOLE: '',
      NODE_ENV: 'test',
    },
  })

  log('Electron launched, waiting for first window...')

  const window = await app.firstWindow()
  log(`Window opened: "${await window.title()}"`)

  // Collect console errors
  window.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text())
    }
  })

  // Capture main process stderr for Rust panics
  app.process().stderr?.on('data', (data) => {
    const text = data.toString()
    if (text.includes('panicked') || text.includes('fatal runtime error')) {
      consoleErrors.push(`[RUST PANIC] ${text}`)
    }
  })

  // ── Test 1: Window loads ────────────────────────────────────────────────
  try {
    await window.waitForSelector('textarea', { timeout: 15_000 })
    pass('Window loaded with chat UI')
  } catch (e) {
    fail(`Window did not load chat UI: ${e.message}`)
    await cleanup()
    return
  }

  // ── Test 2: Load a model via sidebar ────────────────────────────────────
  try {
    // Wait for model catalog to populate
    await window.waitForFunction(
      () => document.body.innerText.includes('Click to load') || document.body.innerText.includes('Loaded'),
      undefined,
      { timeout: 10_000 }
    )

    // Check if a model is already loaded
    const alreadyLoaded = await window.evaluate(() => {
      const ta = document.querySelector('textarea')
      return ta && !ta.disabled
    })

    if (alreadyLoaded) {
      pass('Model already loaded')
    } else {
      log('Found downloaded model in sidebar, clicking to load...')
      // Click the first model card that shows "Click to load"
      await window.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'))
        const btn = buttons.find((b) => b.textContent?.includes('Click to load'))
        if (btn) btn.click()
      })

      // Wait for textarea to become enabled
      await window.waitForFunction(
        () => {
          const ta = document.querySelector('textarea')
          return ta && !ta.disabled
        },
        undefined,
        { timeout: TIMEOUT_MODEL_LOAD }
      )
      pass('Model loaded successfully')
    }
  } catch (e) {
    fail(`Could not load model: ${e.message}`)
    try {
      await window.screenshot({ path: resolve(projectRoot, 'e2e/fail-model-load.png') })
      log('Screenshot saved to e2e/fail-model-load.png')
    } catch {}
    await cleanup()
    return
  }

  // ── Test 3: Send a message ──────────────────────────────────────────────
  try {
    const textarea = window.locator('textarea')
    await textarea.fill(TEST_MESSAGE)
    log(`Typed: "${TEST_MESSAGE}"`)

    // Click Send
    await window.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'))
      const send = buttons.find((b) => b.textContent?.trim() === 'Send')
      if (send) send.click()
    })
    log('Clicked Send')

    // Verify user message appears
    await window.waitForFunction(
      (msg) => document.body.innerText.includes(msg),
      TEST_MESSAGE,
      { timeout: 5_000 }
    )
    pass('User message appears in chat')
  } catch (e) {
    fail(`Could not send message: ${e.message}`)
    try {
      await window.screenshot({ path: resolve(projectRoot, 'e2e/fail-send.png') })
    } catch {}
    await cleanup()
    return
  }

  // ── Test 4: Verify tokens are streaming ─────────────────────────────────
  try {
    // Wait for streaming to start — Stop button appears
    await window.waitForFunction(
      () => {
        const buttons = Array.from(document.querySelectorAll('button'))
        return buttons.some((b) => b.textContent?.includes('Stop'))
      },
      undefined,
      { timeout: TIMEOUT_STREAMING_DETECT }
    )
    log('Streaming started (Stop button visible)')

    // Wait a few seconds for tokens to accumulate, then snapshot the text length
    await new Promise((r) => setTimeout(r, 3000))
    const len1 = await window.evaluate(() => document.body.innerText.length)
    await new Promise((r) => setTimeout(r, 2000))
    const len2 = await window.evaluate(() => document.body.innerText.length)

    if (len2 > len1) {
      pass(`Tokens are flowing (text grew from ${len1} to ${len2} chars)`)
    } else {
      fail(`Tokens are not flowing (text stayed at ${len1} chars)`)
    }

    // Check for crash/panic text in the response
    const bodyText = await window.evaluate(() => document.body.innerText)
    if (bodyText.includes('byte index') && bodyText.includes('char boundary')) {
      fail('UTF-8 boundary panic text leaked into response')
    } else {
      pass('No UTF-8 panics in response text')
    }
  } catch (e) {
    fail(`Streaming did not start: ${e.message}`)
    try {
      await window.screenshot({ path: resolve(projectRoot, 'e2e/fail-streaming.png') })
    } catch {}
  }

  // ── Test 5: Stop inference gracefully ───────────────────────────────────
  try {
    // Click Stop button
    await window.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'))
      const stop = buttons.find((b) => b.textContent?.includes('Stop'))
      if (stop) stop.click()
    })
    log('Clicked Stop')

    // Wait for Send button to reappear
    await window.waitForFunction(
      () => {
        const buttons = Array.from(document.querySelectorAll('button'))
        return buttons.some((b) => b.textContent?.trim() === 'Send')
      },
      undefined,
      { timeout: TIMEOUT_STOP }
    )
    pass('Inference stopped gracefully (Send button reappeared)')
  } catch (e) {
    fail(`Stop did not work: ${e.message}`)
  }

  // ── Test 6: No Rust panics ──────────────────────────────────────────────
  const panics = consoleErrors.filter(
    (e) => e.includes('RUST PANIC') || e.includes('panicked') || e.includes('char boundary')
  )
  if (panics.length > 0) {
    fail(`Rust panics detected:\n${panics.join('\n')}`)
  } else {
    pass('No Rust panics detected')
  }

  // ── Summary ─────────────────────────────────────────────────────────────
  await cleanup()

  if (consoleErrors.length > 0) {
    console.log('\n─── Console errors ───')
    consoleErrors.forEach((e) => console.log(`  • ${e}`))
  }

  console.log(`\n─── Results: ${passed} passed, ${failed} failed ───`)
  if (failed > 0) {
    console.log('💥 Some tests failed.')
    process.exitCode = 1
  } else {
    console.log('🦀 All tests passed!')
  }
}

async function cleanup() {
  if (app) {
    log('Closing app...')
    try {
      await app.close()
    } catch {}
  }
}

main().catch((e) => {
  console.error('Unhandled error:', e)
  process.exitCode = 1
  cleanup()
})
