/*
  Headless smoke-test for the Evolve web dashboard.
  1. Verifies the HTTP endpoint (default http://localhost:9000) returns 200.
  2. Opens a WebSocket to the simulation (ws://localhost:8080) and waits up to
     5 s for a JSON SimulationState message.
  3. Dumps a small report to stdout and exits 0 on success, 1 on failure.
*/

import { WebSocket } from 'ws'; // npm i ws  (dev-only)
import fetch from 'node-fetch'; // node18 has global fetch but import for <18

const HTTP_URL = process.env.DASH_URL || 'http://localhost:9000/';
const WS_URL = process.env.SIM_WS || 'ws://localhost:8080/ws';

(async () => {
  try {
    const res = await fetch(HTTP_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    console.log(`[ok] HTTP respond ${res.status}`);
  } catch (e) {
    console.error(`[fail] dashboard HTTP not reachable: ${e.message}`);
    process.exit(1);
  }

  let received = false;
  const ws = new WebSocket(WS_URL);
  const timeout = setTimeout(() => {
    if (!received) {
      console.error('[fail] No SimulationState frame within 5s');
      process.exit(1);
    }
  }, 5000);

  ws.on('message', (data) => {
    try {
      const txt = data.toString();
      const obj = JSON.parse(txt);
      if (obj && typeof obj.current_tick === 'number') {
        received = true;
        clearTimeout(timeout);
        console.log('[ok] Received SimulationState frame');
        ws.close();
        console.log(JSON.stringify(obj, null, 2).slice(0, 300) + '...');
        process.exit(0);
      }
    } catch (_) {
      /* ignore */
    }
  });

  ws.on('error', (err) => {
    console.error(`[fail] WebSocket error: ${err.message}`);
    process.exit(1);
  });
})();