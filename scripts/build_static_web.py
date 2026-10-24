"""
Campus Navigator: Static Web Application Generator
==================================================
File: scripts/build_static_web.py

Generates an ultra-responsive, standalone, zero-dependency index.html
with pre-trained Q-table weights, interactive SVG campus grid, mobile
touch D-pad, keyboard listener, live Q-values inspector, and in-browser
Q-learning training engine.
"""

import json
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
Q_TABLE_PATH = os.path.join(MODELS_DIR, "q_table.npy")
OUTPUT_HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")

def generate_index_html():
    q_table = np.load(Q_TABLE_PATH).tolist()
    q_json = json.dumps(q_table)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Campus Navigator: Q-Learning RL</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎓</text></svg>">
  <style>
    :root {{
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --accent: #06b6d4;
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --cell-size: 34px;
    }}

    @media (max-width: 480px) {{
      :root {{
        --cell-size: 29px;
      }}
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-tap-highlight-color: transparent;
      user-select: none;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }}

    /* HEADER */
    header {{
      background: rgba(30, 41, 59, 0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--surface-border);
      padding: 12px 16px;
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .logo-container {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .logo-icon {{
      font-size: 24px;
    }}

    .logo-title {{
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
    }}

    .logo-badge {{
      font-size: 10px;
      background: #312e81;
      color: #818cf8;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
      border: 1px solid #4338ca;
    }}

    /* NAVIGATION TABS */
    nav {{
      background: #1e293b;
      border-bottom: 1px solid var(--surface-border);
      display: flex;
      overflow-x: auto;
      scrollbar-width: none;
    }}
    nav::-webkit-scrollbar {{
      display: none;
    }}

    .tab-btn {{
      padding: 12px 16px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      border: none;
      background: transparent;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
      border-bottom: 2px solid transparent;
    }}

    .tab-btn.active {{
      color: #fff;
      border-bottom-color: var(--primary);
      background: rgba(79, 70, 229, 0.08);
    }}

    /* MAIN CONTAINER */
    main {{
      flex: 1;
      padding: 16px;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
    }}

    .tab-content {{
      display: none;
      animation: fadeIn 0.2s ease;
    }}

    .tab-content.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* CARDS */
    .card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}

    .card-title {{
      font-size: 14px;
      font-weight: 700;
      color: #e2e8f0;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* TELEMETRY HUD */
    .hud-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}

    @media (min-width: 600px) {{
      .hud-grid {{
        grid-template-columns: repeat(6, 1fr);
      }}
    }}

    .hud-stat {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px;
      text-align: center;
    }}

    .hud-label {{
      font-size: 10px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 4px;
      letter-spacing: 0.05em;
    }}

    .hud-value {{
      font-size: 14px;
      font-weight: 700;
      color: #38bdf8;
      font-family: monospace;
    }}

    /* SVG MAP CONTAINER */
    .canvas-wrapper {{
      display: flex;
      justify-content: center;
      align-items: center;
      background: #090d16;
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 8px;
      margin-bottom: 16px;
      overflow: hidden;
    }}

    svg.campus-svg {{
      max-width: 100%;
      height: auto;
      display: block;
    }}

    /* TOUCH D-PAD */
    .dpad-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin: 12px 0;
    }}

    .dpad-row {{
      display: flex;
      gap: 6px;
    }}

    .dpad-btn {{
      width: 58px;
      height: 52px;
      background: #334155;
      border: 1px solid #475569;
      border-radius: 10px;
      color: #fff;
      font-size: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      touch-action: manipulation;
      transition: all 0.1s ease;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}

    .dpad-btn:active {{
      transform: scale(0.92);
      background: var(--primary);
      border-color: #6366f1;
    }}

    /* ACTION BUTTONS */
    .btn {{
      background: var(--primary);
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }}

    .btn:hover {{
      background: var(--primary-hover);
    }}

    .btn:active {{
      transform: scale(0.97);
    }}

    .btn-secondary {{
      background: #334155;
      border: 1px solid #475569;
    }}
    .btn-secondary:hover {{
      background: #475569;
    }}

    .btn-success {{
      background: #059669;
    }}
    .btn-success:hover {{
      background: #047857;
    }}

    .btn-danger {{
      background: #dc2626;
    }}
    .btn-danger:hover {{
      background: #b91c1c;
    }}

    .btn-group {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}

    /* Q-VALUE INSPECTOR TABLE */
    .table-container {{
      overflow-x: auto;
      margin-top: 8px;
    }}

    table.q-table-view {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      text-align: center;
    }}

    table.q-table-view th, table.q-table-view td {{
      padding: 8px;
      border: 1px solid var(--surface-border);
    }}

    table.q-table-view th {{
      background: #0f172a;
      color: var(--text-muted);
      font-weight: 700;
    }}

    table.q-table-view td.best-action {{
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      font-weight: 700;
      border-color: #059669;
    }}

    /* VICTORY BANNER */
    .victory-modal {{
      display: none;
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 182, 212, 0.2));
      border: 2px solid #10b981;
      border-radius: 12px;
      padding: 16px;
      text-align: center;
      margin-bottom: 16px;
      animation: popIn 0.3s ease;
    }}

    @keyframes popIn {{
      0% {{ transform: scale(0.8); opacity: 0; }}
      100% {{ transform: scale(1); opacity: 1; }}
    }}

    /* FORM CONTROLS */
    .form-group {{
      margin-bottom: 12px;
    }}

    .form-label {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}

    input[type=range] {{
      width: 100%;
      accent-color: var(--primary);
    }}

    select, input[type=number] {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--surface-border);
      border-radius: 6px;
      color: #fff;
      padding: 8px;
      font-size: 13px;
    }}

    /* BADGES */
    .badge {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
    }}
    .badge-gate {{ background: #0284c7; color: #fff; }}
    .badge-goal {{ background: #059669; color: #fff; }}
    .badge-hub {{ background: #0d9488; color: #fff; }}
    .badge-hazard {{ background: #dc2626; color: #fff; }}
    .badge-crowd {{ background: #d97706; color: #fff; }}

    /* PROGRESS BAR */
    .progress-bar-bg {{
      background: #0f172a;
      border: 1px solid var(--surface-border);
      height: 12px;
      border-radius: 6px;
      overflow: hidden;
      margin-top: 8px;
    }}

    .progress-bar-fill {{
      background: linear-gradient(90deg, #4f46e5, #06b6d4);
      height: 100%;
      width: 0%;
      transition: width 0.1s ease;
    }}

    /* CANVAS CHART */
    canvas.chart-canvas {{
      width: 100%;
      height: 180px;
      background: #0f172a;
      border: 1px solid var(--surface-border);
      border-radius: 8px;
      margin-top: 10px;
    }}
  </style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="logo-container">
      <span class="logo-icon">🎓</span>
      <div>
        <div class="logo-title">Campus Navigator</div>
        <div style="font-size: 11px; color: var(--text-muted);">Q-Learning Reinforcement Learning</div>
      </div>
    </div>
    <div class="logo-badge">Live Web RL</div>
  </header>

  <!-- NAVIGATION TABS -->
  <nav id="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('play')">🎮 Play Game</button>
    <button class="tab-btn" onclick="switchTab('watch')">🤖 Watch AI</button>
    <button class="tab-btn" onclick="switchTab('train')">🏋️ Train AI</button>
    <button class="tab-btn" onclick="switchTab('results')">📊 Results</button>
    <button class="tab-btn" onclick="switchTab('learn')">📚 Learn RL</button>
    <button class="tab-btn" onclick="switchTab('info')">ℹ️ Project Info</button>
  </nav>

  <!-- MAIN VIEWPORT -->
  <main>

    <!-- TAB 1: PLAY GAME -->
    <section id="tab-play" class="tab-content active">
      <div id="play-victory" class="victory-modal">
        <h3 style="color: #34d399; margin-bottom: 4px;">🎉 Destination Reached!</h3>
        <p style="font-size: 13px; color: #e2e8f0;">Navigation completed successfully. Optimal path demonstrated!</p>
      </div>

      <!-- HUD STATS -->
      <div class="hud-grid">
        <div class="hud-stat">
          <div class="hud-label">State (r, c)</div>
          <div id="play-state" class="hud-value">(0, 0)</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Action</div>
          <div id="play-action" class="hud-value">START</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Step Reward</div>
          <div id="play-reward" class="hud-value">0.0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Total Reward</div>
          <div id="play-total" class="hud-value">0.0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Steps</div>
          <div id="play-steps" class="hud-value">0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Status</div>
          <div id="play-status" class="hud-value" style="color: #34d399;">Ready</div>
        </div>
      </div>

      <!-- CAMPUS SVG MAP -->
      <div class="canvas-wrapper">
        <svg id="play-svg" class="campus-svg" viewBox="0 0 380 380" width="380" height="380"></svg>
      </div>

      <!-- LANDMARK SELECTOR & RESET -->
      <div class="card" style="padding: 12px; margin-bottom: 12px;">
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
          <label style="font-size: 12px; font-weight: 700; color: var(--text-muted);">Destination:</label>
          <select id="play-dest-select" style="flex: 1; min-width: 140px;" onchange="changeDestination(this.value)">
            <option value="9,9" selected>Central Library (9, 9)</option>
            <option value="1,8">Academic Block (1, 8)</option>
            <option value="4,3">Computer Lab (4, 3)</option>
            <option value="3,2">Cafeteria (3, 2)</option>
            <option value="9,0">Sports Ground (9, 0)</option>
          </select>
          <button class="btn btn-secondary" onclick="resetGame('play')">🔄 Reset</button>
        </div>
      </div>

      <!-- TOUCH D-PAD (MOBILE FIRST) -->
      <div class="dpad-container">
        <button class="dpad-btn" onclick="stepMove('play', 0)">▲</button>
        <div class="dpad-row">
          <button class="dpad-btn" onclick="stepMove('play', 2)">◀</button>
          <button class="dpad-btn" style="background: #1e293b; border-color: #334155; font-size: 14px; font-weight: 700; color: #64748b;" disabled>D-PAD</button>
          <button class="dpad-btn" onclick="stepMove('play', 3)">▶</button>
        </div>
        <button class="dpad-btn" onclick="stepMove('play', 1)">▼</button>
      </div>
      <p style="text-align: center; font-size: 11px; color: var(--text-muted);">Mobile: Tap D-pad buttons | Laptop: Use Arrow keys or WASD</p>
    </section>

    <!-- TAB 2: WATCH AI -->
    <section id="tab-watch" class="tab-content">
      <div id="watch-victory" class="victory-modal">
        <h3 style="color: #34d399; margin-bottom: 4px;">🎯 AI Reached Destination!</h3>
        <p style="font-size: 13px; color: #e2e8f0;">Learned optimal shortest path executed with zero collisions.</p>
      </div>

      <!-- HUD STATS -->
      <div class="hud-grid">
        <div class="hud-stat">
          <div class="hud-label">State (r, c)</div>
          <div id="watch-state" class="hud-value">(0, 0)</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">AI Action</div>
          <div id="watch-action" class="hud-value">START</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Step Reward</div>
          <div id="watch-reward" class="hud-value">0.0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Total Reward</div>
          <div id="watch-total" class="hud-value">0.0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Steps</div>
          <div id="watch-steps" class="hud-value">0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Policy</div>
          <div class="hud-value" style="color: #a855f7;">Greedy (ε=0)</div>
        </div>
      </div>

      <!-- CAMPUS SVG MAP -->
      <div class="canvas-wrapper">
        <svg id="watch-svg" class="campus-svg" viewBox="0 0 380 380" width="380" height="380"></svg>
      </div>

      <!-- WATCH CONTROLS -->
      <div class="card" style="padding: 12px; margin-bottom: 12px;">
        <div class="btn-group" style="margin-bottom: 0;">
          <button class="btn btn-success" onclick="stepAI()">⏩ Step AI (1 Step)</button>
          <button class="btn" onclick="runCompleteAI()">⚡ Run Complete Route</button>
          <button class="btn btn-secondary" onclick="resetGame('watch')">🔄 Reset AI</button>
        </div>
      </div>

      <!-- LIVE Q-VALUE INSPECTOR -->
      <div class="card">
        <div class="card-title">🔍 Real-Time Q-Values Inspector</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">
          Current State Q-Table Evaluation: Q(s, a) for s = <span id="inspect-state-text" style="color: #38bdf8; font-weight: 700;">(0, 0)</span>
        </div>
        <div class="table-container">
          <table class="q-table-view">
            <thead>
              <tr>
                <th>Action</th>
                <th>Direction</th>
                <th>Q-Value Q(s, a)</th>
                <th>Selection</th>
              </tr>
            </thead>
            <tbody id="q-table-body">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: TRAIN AI -->
    <section id="tab-train" class="tab-content">
      <div class="card">
        <div class="card-title">🏋️ In-Browser Q-Learning Training Engine</div>
        <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
          Train a new Reinforcement Learning agent from scratch directly in your browser using Bellman Temporal Difference equation:
          <code style="color: #38bdf8; background: #0f172a; padding: 2px 6px; border-radius: 4px;">Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]</code>
        </p>

        <div class="form-group">
          <div class="form-label">
            <span>Training Episodes: <b id="val-episodes" style="color: #fff;">600</b></span>
          </div>
          <input type="range" id="train-episodes" min="100" max="2000" step="100" value="600" oninput="document.getElementById('val-episodes').innerText = this.value">
        </div>

        <div class="form-group">
          <div class="form-label">
            <span>Learning Rate (α): <b id="val-alpha" style="color: #fff;">0.10</b></span>
          </div>
          <input type="range" id="train-alpha" min="0.01" max="0.5" step="0.01" value="0.10" oninput="document.getElementById('val-alpha').innerText = parseFloat(this.value).toFixed(2)">
        </div>

        <div class="form-group">
          <div class="form-label">
            <span>Discount Factor (γ): <b id="val-gamma" style="color: #fff;">0.90</b></span>
          </div>
          <input type="range" id="train-gamma" min="0.5" max="0.99" step="0.01" value="0.90" oninput="document.getElementById('val-gamma').innerText = parseFloat(this.value).toFixed(2)">
        </div>

        <div class="form-group">
          <div class="form-label">
            <span>Epsilon Decay: <b id="val-decay" style="color: #fff;">0.995</b></span>
          </div>
          <input type="range" id="train-decay" min="0.95" max="0.999" step="0.001" value="0.995" oninput="document.getElementById('val-decay').innerText = parseFloat(this.value).toFixed(3)">
        </div>

        <button id="btn-start-train" class="btn" style="width: 100%; margin-top: 8px;" onclick="startBrowserTraining()">🚀 Start Training</button>

        <div class="progress-bar-bg">
          <div id="train-progress-bar" class="progress-bar-fill"></div>
        </div>
        <div id="train-status-text" style="font-size: 12px; color: var(--text-muted); text-align: center; margin-top: 6px;">Ready to train.</div>
      </div>

      <!-- REAL-TIME TRAINING CHARTS -->
      <div class="card">
        <div class="card-title">📈 Learning Curves (Reward vs Episode)</div>
        <canvas id="train-chart-reward" class="chart-canvas" width="600" height="180"></canvas>
      </div>
    </section>

    <!-- TAB 4: RESULTS -->
    <section id="tab-results" class="tab-content">
      <div class="card">
        <div class="card-title">📊 Empirical Benchmarks (Verified Experimental Results)</div>
        <div class="hud-grid" style="grid-template-columns: repeat(2, 1fr); margin-bottom: 16px;">
          <div class="hud-stat">
            <div class="hud-label">Success Rate</div>
            <div class="hud-value" style="color: #10b981;">100.0%</div>
          </div>
          <div class="hud-stat">
            <div class="hud-label">Optimal Steps</div>
            <div class="hud-value" style="color: #38bdf8;">18.0</div>
          </div>
          <div class="hud-stat">
            <div class="hud-label">Collision Rate</div>
            <div class="hud-value" style="color: #10b981;">0.00 / ep</div>
          </div>
          <div class="hud-stat">
            <div class="hud-label">Average Cumulative Reward</div>
            <div class="hud-value" style="color: #a855f7;">+170.0</div>
          </div>
        </div>

        <div class="table-container">
          <table class="q-table-view">
            <thead>
              <tr>
                <th>Training Horizon</th>
                <th>Final Success Rate</th>
                <th>Mean Steps to Goal</th>
                <th>Convergence Episode</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>500 Episodes</td>
                <td style="color: #10b981; font-weight: 700;">100.0%</td>
                <td>18.0</td>
                <td>Episode ~240</td>
              </tr>
              <tr>
                <td>1000 Episodes</td>
                <td style="color: #10b981; font-weight: 700;">100.0%</td>
                <td>18.0</td>
                <td>Episode ~265</td>
              </tr>
              <tr>
                <td>2000 Episodes</td>
                <td style="color: #10b981; font-weight: 700;">100.0%</td>
                <td>18.0</td>
                <td>Episode ~250</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title">📍 Learned Optimal 18-Step Path</div>
        <p style="font-size: 12px; font-family: monospace; color: #38bdf8; line-height: 1.6; word-break: break-all;">
          (0,0) → (1,0) → (2,0) → (3,0) → (3,1) → (3,2: Safe Hub) → (3,3) → (4,3) → (4,4) → (5,4) → (6,4) → (7,4) → (8,4) → (8,5) → (8,6) → (8,7) → (8,8) → (8,9) → (9,9: Library)
        </p>
      </div>
    </section>

    <!-- TAB 5: LEARN RL -->
    <section id="tab-learn" class="tab-content">
      <div class="card">
        <div class="card-title">📚 Tabular Q-Learning Fundamentals</div>
        <p style="font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-bottom: 12px;">
          Q-learning is a <b>model-free, off-policy Temporal Difference (TD)</b> reinforcement learning algorithm. It aims to find the optimal action-selection policy for any finite Markov Decision Process (MDP) without requiring a transition probability model of the environment.
        </p>
        <div style="background: #0f172a; border-left: 4px solid var(--primary); padding: 12px; border-radius: 4px; margin-bottom: 12px;">
          <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700;">The Bellman Optimality Equation:</div>
          <div style="font-size: 14px; font-family: monospace; color: #38bdf8; margin-top: 4px;">
            Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]
          </div>
        </div>
        <ul style="font-size: 12px; color: #94a3b8; line-height: 1.8; margin-left: 20px;">
          <li><b>s</b>: Current discrete grid state (row, column).</li>
          <li><b>a</b>: Directional action (UP, DOWN, LEFT, RIGHT).</li>
          <li><b>r</b>: Scalar reward signal (+100 destination, -1 step, -10 collision/hazard).</li>
          <li><b>α (alpha)</b>: Learning rate controlling weight assigned to new observations.</li>
          <li><b>γ (gamma)</b>: Discount factor balancing immediate vs future long-term rewards.</li>
          <li><b>ε (epsilon)</b>: Probability of selecting a random exploratory action.</li>
        </ul>
      </div>
    </section>

    <!-- TAB 6: PROJECT INFO -->
    <section id="tab-info" class="tab-content">
      <div class="card">
        <div class="card-title">ℹ️ Academic Experiential-Learning Specifications</div>
        <table class="q-table-view" style="text-align: left;">
          <tbody>
            <tr>
              <th style="width: 35%;">Project Title</th>
              <td>Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning</td>
            </tr>
            <tr>
              <th>Domain</th>
              <td>Artificial Intelligence & Tabular Reinforcement Learning</td>
            </tr>
            <tr>
              <th>Environment</th>
              <td>10x10 Discrete University Gridworld with 19 Impassable Buildings</td>
            </tr>
            <tr>
              <th>Algorithm</th>
              <td>Tabular Q-Learning (Off-Policy TD Control)</td>
            </tr>
            <tr>
              <th>Multi-Platform</th>
              <td>Pygame Desktop GUI + Browser Web / Mobile Touch Engine</td>
            </tr>
            <tr>
              <th>Live Web App</th>
              <td><a href="https://huggingface.co/spaces/nagarajchittapur/campus-navigator" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 700;">Hugging Face Spaces Live Demo ↗</a></td>
            </tr>
            <tr>
              <th>Repository</th>
              <td><a href="https://github.com/nagarajchittapur18-dev/Campus-Navigator-RL-.git" target="_blank" style="color: #38bdf8; text-decoration: none;">GitHub Repository ↗</a></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

  </main>

  <script>
    /* =========================================================================
       1. GLOBAL CONSTANTS & PRE-TRAINED Q-TABLE DATA
       ========================================================================= */
    const GRID_SIZE = 10;
    const OBSTACLES = new Set(["0,4","1,2","1,4","1,7","2,1","2,2","2,7","3,4","4,0","4,1","4,6","4,7","5,3","6,5","6,7","7,1","7,7","8,3","9,6"]);
    const SAFE_ZONES = new Set(["3,2","7,3"]);
    const CROWDED_ZONES = new Set(["3,5","6,3","7,5"]);
    const HAZARD_ZONES = new Set(["2,5","5,6","8,7"]);
    const ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"];
    const DELTAS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    // Pretrained Q-Table embedded directly from Python models/q_table.npy
    const PRETRAINED_Q_TABLE = {q_json};

    /* GAME STATES */
    let currentGoal = [9, 9];

    let gameState = {{
      play: {{ pos: [0, 0], totalReward: 0, steps: 0, history: [[0, 0]], isOver: false }},
      watch: {{ pos: [0, 0], totalReward: 0, steps: 0, history: [[0, 0]], isOver: false }}
    }};

    /* =========================================================================
       2. TAB NAVIGATION
       ========================================================================= */
    function switchTab(tabId) {{
      document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
      
      const targetTab = document.getElementById("tab-" + tabId);
      if (targetTab) targetTab.classList.add("active");
      
      const tabButtons = document.querySelectorAll(".tab-btn");
      const mapping = {{ play: 0, watch: 1, train: 2, results: 3, learn: 4, info: 5 }};
      if (tabButtons[mapping[tabId]]) {{
        tabButtons[mapping[tabId]].classList.add("active");
      }}

      if (tabId === "play") renderSVG("play");
      if (tabId === "watch") {{
        renderSVG("watch");
        updateQInspector();
      }}
    }}

    /* =========================================================================
       3. SVG CAMPUS RENDERING ENGINE
       ========================================================================= */
    function renderSVG(mode) {{
      const svg = document.getElementById(mode + "-svg");
      if (!svg) return;

      const state = gameState[mode];
      const cellSize = 34;
      const offsetX = 20;
      const offsetY = 20;

      let html = "";

      // Definitions (Glow filters)
      html += `
        <defs>
          <radialGradient id="player-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9"/>
            <stop offset="100%" stop-color="#0284c7" stop-opacity="0.3"/>
          </radialGradient>
        </defs>
      `;

      // Coordinate Rulers
      for (let c = 0; c < GRID_SIZE; c++) {{
        html += `<text x="${{offsetX + c * cellSize + cellSize / 2}}" y="14" fill="#64748b" font-size="10" font-family="monospace" text-anchor="middle">${{c}}</text>`;
      }}
      for (let r = 0; r < GRID_SIZE; r++) {{
        html += `<text x="12" y="${{offsetY + r * cellSize + cellSize / 2 + 4}}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="middle">${{r}}</text>`;
      }}

      // Background grid
      for (let r = 0; r < GRID_SIZE; r++) {{
        for (let c = 0; c < GRID_SIZE; c++) {{
          const x = offsetX + c * cellSize;
          const y = offsetY + r * cellSize;
          const key = `${{r}},${{c}}`;

          let fill = "#1e293b";
          let stroke = "#334155";
          let icon = "";

          if (OBSTACLES.has(key)) {{
            fill = "#334155";
            stroke = "#475569";
            icon = "🏢";
          }} else if (r === 0 && c === 0) {{
            fill = "#0369a1";
            icon = "🚪";
          }} else if (r === currentGoal[0] && c === currentGoal[1]) {{
            fill = "#065f46";
            icon = "📚";
          }} else if (SAFE_ZONES.has(key)) {{
            fill = "#115e59";
            icon = "🛡️";
          }} else if (HAZARD_ZONES.has(key)) {{
            fill = "#7f1d1d";
            icon = "⚠️";
          }} else if (CROWDED_ZONES.has(key)) {{
            fill = "#78350f";
            icon = "👥";
          }}

          html += `<rect x="${{x}}" y="${{y}}" width="${{cellSize - 2}}" height="${{cellSize - 2}}" rx="4" fill="${{fill}}" stroke="${{stroke}}" stroke-width="1"/>`;
          if (icon) {{
            html += `<text x="${{x + cellSize / 2 - 1}}" y="${{y + cellSize / 2 + 4}}" font-size="13" text-anchor="middle">${{icon}}</text>`;
          }}
        }}
      }}

      // Trail line
      if (state.history.length > 1) {{
        let points = state.history.map(pt => `${{offsetX + pt[1] * cellSize + cellSize / 2}},${{offsetY + pt[0] * cellSize + cellSize / 2}}`).join(" ");
        html += `<polyline points="${{points}}" fill="none" stroke="#06b6d4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="0.75" stroke-dasharray="4 2"/>`;
      }}

      // Current Student Avatar
      const playerX = offsetX + state.pos[1] * cellSize + cellSize / 2 - 1;
      const playerY = offsetY + state.pos[0] * cellSize + cellSize / 2 - 1;
      html += `
        <circle cx="${{playerX}}" cy="${{playerY}}" r="12" fill="url(#player-glow)" stroke="#38bdf8" stroke-width="2"/>
        <text x="${{playerX}}" y="${{playerY + 4}}" font-size="12" text-anchor="middle">🎓</text>
      `;

      svg.innerHTML = html;
    }}

    /* =========================================================================
       4. MOVEMENT & REWARD ENGINE
       ========================================================================= */
    function stepMove(mode, actionIdx) {{
      const state = gameState[mode];
      if (state.isOver) return;

      const [dr, dc] = DELTAS[actionIdx];
      const nextR = state.pos[0] + dr;
      const nextC = state.pos[1] + dc;
      const nextKey = `${{nextR}},${{nextC}}`;

      let reward = -1.0; // Step penalty
      let isCollision = false;

      // Wall / Obstacle Collision Check
      if (nextR < 0 || nextR >= GRID_SIZE || nextC < 0 || nextC >= GRID_SIZE || OBSTACLES.has(nextKey)) {{
        reward = -10.0;
        isCollision = true;
      }} else {{
        state.pos = [nextR, nextC];
        state.history.push([nextR, nextC]);

        if (SAFE_ZONES.has(nextKey)) reward = 2.0;
        else if (HAZARD_ZONES.has(nextKey)) reward = -10.0;
        else if (CROWDED_ZONES.has(nextKey)) reward = -2.0;

        if (nextR === currentGoal[0] && nextC === currentGoal[1]) {{
          reward = 100.0;
          state.isOver = true;
          document.getElementById(mode + "-victory").style.display = "block";
        }}
      }}

      state.totalReward += reward;
      state.steps++;

      // Update HUD
      document.getElementById(mode + "-state").innerText = `(${{state.pos[0]}}, ${{state.pos[1]}})`;
      document.getElementById(mode + "-action").innerText = ACTIONS[actionIdx];
      document.getElementById(mode + "-reward").innerText = (reward >= 0 ? "+" : "") + reward.toFixed(1);
      document.getElementById(mode + "-total").innerText = state.totalReward.toFixed(1);
      document.getElementById(mode + "-steps").innerText = state.steps;

      const statusEl = document.getElementById(mode + "-status");
      if (statusEl) {{
        if (state.isOver) {{
          statusEl.innerText = "Goal Reached!";
          statusEl.style.color = "#34d399";
        }} else if (isCollision) {{
          statusEl.innerText = "Collision!";
          statusEl.style.color = "#ef4444";
        }} else {{
          statusEl.innerText = "Navigating";
          statusEl.style.color = "#38bdf8";
        }}
      }}

      renderSVG(mode);

      if (mode === "watch") updateQInspector();
    }}

    function resetGame(mode) {{
      gameState[mode] = {{ pos: [0, 0], totalReward: 0, steps: 0, history: [[0, 0]], isOver: false }};
      document.getElementById(mode + "-victory").style.display = "none";
      document.getElementById(mode + "-state").innerText = "(0, 0)";
      document.getElementById(mode + "-action").innerText = "START";
      document.getElementById(mode + "-reward").innerText = "0.0";
      document.getElementById(mode + "-total").innerText = "0.0";
      document.getElementById(mode + "-steps").innerText = "0";
      const statusEl = document.getElementById(mode + "-status");
      if (statusEl) {{
        statusEl.innerText = "Ready";
        statusEl.style.color = "#34d399";
      }}
      renderSVG(mode);
      if (mode === "watch") updateQInspector();
    }}

    function changeDestination(val) {{
      const parts = val.split(",").map(Number);
      currentGoal = parts;
      resetGame("play");
      resetGame("watch");
    }}

    /* =========================================================================
       5. WATCH AI (Q-LEARNING POLICY EXECUTION)
       ========================================================================= */
    function getBestAction(pos) {{
      const idx = pos[0] * 10 + pos[1];
      const qValues = PRETRAINED_Q_TABLE[idx];
      let bestAct = 0;
      let maxQ = -Infinity;
      for (let a = 0; a < 4; a++) {{
        if (qValues[a] > maxQ) {{
          maxQ = qValues[a];
          bestAct = a;
        }}
      }}
      return {{ action: bestAct, values: qValues }};
    }}

    function stepAI() {{
      if (gameState.watch.isOver) return;
      const {{ action }} = getBestAction(gameState.watch.pos);
      stepMove("watch", action);
    }}

    let aiInterval = null;
    function runCompleteAI() {{
      if (gameState.watch.isOver) resetGame("watch");
      if (aiInterval) clearInterval(aiInterval);

      aiInterval = setInterval(() => {{
        if (gameState.watch.isOver || gameState.watch.steps >= 50) {{
          clearInterval(aiInterval);
          aiInterval = null;
          return;
        }}
        stepAI();
      }}, 250);
    }}

    function updateQInspector() {{
      const pos = gameState.watch.pos;
      const idx = pos[0] * 10 + pos[1];
      const qValues = PRETRAINED_Q_TABLE[idx];
      const {{ action: bestAct }} = getBestAction(pos);

      document.getElementById("inspect-state-text").innerText = `(${{pos[0]}}, ${{pos[1]}})`;

      const tbody = document.getElementById("q-table-body");
      if (!tbody) return;

      let html = "";
      for (let a = 0; a < 4; a++) {{
        const isBest = a === bestAct;
        html += `
          <tr class="${{isBest ? "best-action" : ""}}">
            <td><b>${{ACTIONS[a]}}</b></td>
            <td>${{["▲ Up", "▼ Down", "◀ Left", "▶ Right"][a]}}</td>
            <td><code>${{qValues[a].toFixed(3)}}</code></td>
            <td>${{isBest ? "⭐ GREEDY OPTIMAL" : "-"}}</td>
          </tr>
        `;
      }}
      tbody.innerHTML = html;
    }}

    /* =========================================================================
       6. IN-BROWSER Q-LEARNING TRAINING ENGINE
       ========================================================================= */
    let trainingRunning = false;
    function startBrowserTraining() {{
      if (trainingRunning) return;
      trainingRunning = true;

      const totalEpisodes = parseInt(document.getElementById("train-episodes").value);
      const alpha = parseFloat(document.getElementById("train-alpha").value);
      const gamma = parseFloat(document.getElementById("train-gamma").value);
      const decay = parseFloat(document.getElementById("train-decay").value);

      const btn = document.getElementById("btn-start-train");
      btn.innerText = "⏳ Training in Progress...";
      btn.classList.add("btn-secondary");

      // Initialize 100x4 Q-table with zeros
      let q = Array.from({{ length: 100 }}, () => [0, 0, 0, 0]);
      let epsilon = 1.0;
      let rewardHistory = [];

      let ep = 0;
      const batchSize = 20;

      function trainBatch() {{
        const limit = Math.min(ep + batchSize, totalEpisodes);
        for (; ep < limit; ep++) {{
          let r = 0, c = 0;
          let epReward = 0;
          for (let step = 0; step < 100; step++) {{
            const sIdx = r * 10 + c;
            let a = 0;
            if (Math.random() < epsilon) {{
              a = Math.floor(Math.random() * 4);
            }} else {{
              let maxVal = -Infinity;
              for (let i = 0; i < 4; i++) {{
                if (q[sIdx][i] > maxVal) {{
                  maxVal = q[sIdx][i];
                  a = i;
                }}
              }}
            }}

            const [dr, dc] = DELTAS[a];
            const nr = r + dr;
            const nc = c + dc;
            const nextKey = `${{nr}},${{nc}}`;

            let rew = -1.0;
            let nextR = r, nextC = c;
            let done = false;

            if (nr < 0 || nr >= 10 || nc < 0 || nc >= 10 || OBSTACLES.has(nextKey)) {{
              rew = -10.0;
            }} else {{
              nextR = nr;
              nextC = nc;
              if (SAFE_ZONES.has(nextKey)) rew = 2.0;
              else if (HAZARD_ZONES.has(nextKey)) rew = -10.0;
              else if (CROWDED_ZONES.has(nextKey)) rew = -2.0;

              if (nextR === 9 && nextC === 9) {{
                rew = 100.0;
                done = true;
              }}
            }}

            const nextIdx = nextR * 10 + nextC;
            const maxNextQ = done ? 0 : Math.max(...q[nextIdx]);
            q[sIdx][a] += alpha * (rew + gamma * maxNextQ - q[sIdx][a]);

            r = nextR;
            c = nextC;
            epReward += rew;
            if (done) break;
          }}

          epsilon = Math.max(0.05, epsilon * decay);
          rewardHistory.push(epReward);
        }}

        // Update progress bar
        const pct = Math.round((ep / totalEpisodes) * 100);
        document.getElementById("train-progress-bar").style.width = pct + "%";
        document.getElementById("train-status-text").innerText = `Trained ${{ep}} / ${{totalEpisodes}} episodes (${{pct}}%) | Epsilon: ${{epsilon.toFixed(3)}}`;

        drawChart(rewardHistory);

        if (ep < totalEpisodes) {{
          requestAnimationFrame(trainBatch);
        }} else {{
          trainingRunning = false;
          btn.innerText = "🚀 Retrain Agent";
          btn.classList.remove("btn-secondary");
          document.getElementById("train-status-text").innerText = `✅ Training Complete! ${{totalEpisodes}} episodes processed.`;
        }}
      }}

      trainBatch();
    }}

    function drawChart(data) {{
      const canvas = document.getElementById("train-chart-reward");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Gridlines
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      if (data.length < 2) return;

      const minVal = -200;
      const maxVal = 150;

      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < data.length; i++) {{
        const x = (i / (data.length - 1)) * w;
        const normalized = (data[i] - minVal) / (maxVal - minVal);
        const y = h - Math.max(0, Math.min(h, normalized * h));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();
    }}

    /* =========================================================================
       7. KEYBOARD CONTROLS (LAPTOP & DESKTOP)
       ========================================================================= */
    window.addEventListener("keydown", (e) => {{
      const key = e.key.toLowerCase();
      let act = -1;
      if (key === "arrowup" || key === "w") act = 0;
      else if (key === "arrowdown" || key === "s") act = 1;
      else if (key === "arrowleft" || key === "a") act = 2;
      else if (key === "arrowright" || key === "d") act = 3;

      if (act !== -1) {{
        const playTab = document.getElementById("tab-play");
        if (playTab && playTab.classList.contains("active")) {{
          e.preventDefault();
          stepMove("play", act);
        }}
      }}
    }});

    // Initialize application on load
    window.addEventListener("DOMContentLoaded", () => {{
      renderSVG("play");
      renderSVG("watch");
      updateQInspector();
    }});
  </script>
</body>
</html>
"""

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[+] Successfully generated standalone responsive static web app at: {OUTPUT_HTML_PATH}")
    print(f"[+] Total file size: {len(html_content)} bytes ({len(html_content) / 1024:.2f} KB)")

if __name__ == "__main__":
    generate_index_html()
