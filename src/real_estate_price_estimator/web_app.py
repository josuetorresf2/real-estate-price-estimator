from __future__ import annotations

import argparse
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .market_data import ensure_zhvi_csv, latest_zhvi_for_zip, market_calibrated_estimate
from .pipeline import load_model, load_training_data, predict_price, save_model, train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "price_pipeline.joblib"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_housing.csv"
DEFAULT_ZHVI_PATH = PROJECT_ROOT / "data" / "zillow_zhvi_zip.csv"
STATIC_ROOT = PROJECT_ROOT / "static"

DEFAULT_FORM_VALUES = {
    "address": "1201 Market Signal Dr",
    "city": "Austin",
    "neighborhood": "North Loop",
    "zip_code": "78751",
    "square_feet": "1850",
    "bedrooms": "3",
    "bathrooms": "2",
    "lot_size": "0.18",
    "year_built": "1998",
    "school_rating": "8.6",
    "distance_to_city_center_miles": "4.2",
    "crime_index": "31",
}

FIELD_LABELS = {
    "address": "Address",
    "city": "City",
    "neighborhood": "Neighborhood",
    "zip_code": "ZIP code",
    "square_feet": "Square feet",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "lot_size": "Lot size (acres)",
    "year_built": "Year built",
    "school_rating": "School rating",
    "distance_to_city_center_miles": "Miles to city center",
    "crime_index": "Crime index",
}

NUMERIC_FIELDS = {
    "square_feet": float,
    "bedrooms": float,
    "bathrooms": float,
    "lot_size": float,
    "year_built": int,
    "school_rating": float,
    "distance_to_city_center_miles": float,
    "crime_index": float,
}


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH, data_path: Path = DEFAULT_DATA_PATH):
    if model_path.exists():
        return load_model(model_path)
    data = load_training_data(data_path)
    model, _ = train(data)
    save_model(model, model_path)
    return model


def parse_form(body: str) -> tuple[dict[str, object], dict[str, str], list[str]]:
    parsed = parse_qs(body, keep_blank_values=True)
    values = {key: parsed.get(key, [""])[0].strip() for key in DEFAULT_FORM_VALUES}
    errors = []
    features: dict[str, object] = {}

    for field, value in values.items():
        if field == "address":
            continue
        if not value:
            errors.append(f"{FIELD_LABELS[field]} is required.")
            continue
        if field in NUMERIC_FIELDS:
            try:
                features[field] = NUMERIC_FIELDS[field](value)
            except ValueError:
                errors.append(f"{FIELD_LABELS[field]} must be a number.")
        else:
            features[field] = value

    return features, values, errors


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_page(
    values: dict[str, str] | None = None,
    *,
    prediction: float | None = None,
    model_prediction: float | None = None,
    market_signal: object | None = None,
    errors: list[str] | None = None,
) -> str:
    form_values = values or DEFAULT_FORM_VALUES
    error_items = "".join(f"<li>{html.escape(error)}</li>" for error in errors or [])
    result = ""
    if prediction is not None:
        market_note = ""
        if market_signal is not None:
            market_note = f"""
            <dl class="market-card">
              <div><dt>Zillow ZIP signal</dt><dd>{format_currency(market_signal.typical_home_value)}</dd></div>
              <div><dt>Market</dt><dd>{html.escape(market_signal.city)}, {html.escape(market_signal.state)} {html.escape(market_signal.zip_code)}</dd></div>
              <div><dt>Data month</dt><dd>{html.escape(market_signal.date)}</dd></div>
            </dl>
            <p class="source-note">Market signal: Zillow Research ZHVI. This is ZIP-level typical home value data, not an address-level Zestimate.</p>
            """
        elif model_prediction is not None:
            market_note = """
            <p class="source-note">No Zillow Research ZIP signal was found for this ZIP, so this result uses the trained feature model only.</p>
            """
        result = f"""
        <section class="result" aria-live="polite">
          <span>Market-calibrated estimate</span>
          <strong>{format_currency(prediction)}</strong>
          {f'<p class="model-note">Feature model: {format_currency(model_prediction)}</p>' if model_prediction is not None else ''}
          {market_note}
        </section>
        """
    else:
        result = """
        <section class="empty-result" aria-live="polite">
          <strong>Live market estimate</strong>
          <p class="note">Enter property signals, then blend the trained model with Zillow Research ZIP-level ZHVI when available.</p>
        </section>
        """

    fields = "\n".join(
        f"""
        <label class="{field}">
          <span>{html.escape(label)}</span>
          <input name="{field}" value="{html.escape(form_values.get(field, ''))}" {input_attributes(field)}>
        </label>
        """
        for field, label in FIELD_LABELS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real Estate Price Estimator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {{
      color-scheme: dark;
      --ink: #d7e2ea;
      --muted: rgba(215, 226, 234, 0.64);
      --soft: rgba(215, 226, 234, 0.1);
      --line: rgba(215, 226, 234, 0.16);
      --field: rgba(255, 255, 255, 0.08);
      --accent: #b600a8;
      --accent-strong: #be4c00;
      --warn: #ffb36b;
      --bg: #0c0c0c;
      --panel: rgba(12, 12, 12, 0.72);
      --shadow: rgba(0, 0, 0, 0.46);
    }}
    * {{ box-sizing: border-box; }}
    html {{ min-height: 100%; background: var(--bg); }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Kanit", ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(123deg, rgba(24, 1, 31, 0.86) 7%, rgba(182, 0, 168, 0.22) 37%, rgba(118, 33, 176, 0.18) 72%, rgba(190, 76, 0, 0.14) 100%),
        #0c0c0c;
      overflow-x: hidden;
    }}
    body.loading main {{
      filter: blur(10px);
      opacity: 0.38;
    }}
    .loader {{
      position: fixed;
      inset: 0;
      z-index: 10;
      display: grid;
      place-items: center;
      background: rgba(12, 12, 12, 0.82);
      transition: opacity 450ms ease, visibility 450ms ease;
    }}
    body:not(.loading) .loader {{
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }}
    .loader-orbit {{
      width: 190px;
      height: 190px;
      border: 1px solid rgba(215, 226, 234, 0.18);
      border-radius: 50%;
      display: grid;
      place-items: center;
      position: relative;
      animation: spin 1.8s linear infinite;
    }}
    .loader-orbit::before,
    .loader-orbit::after {{
      content: "";
      position: absolute;
      inset: 28px;
      border-radius: 50%;
      border: 1px solid rgba(182, 0, 168, 0.52);
      transform: rotate(48deg) scaleY(0.56);
    }}
    .loader-orbit::after {{
      border-color: rgba(190, 76, 0, 0.7);
      transform: rotate(-32deg) scaleY(0.48);
    }}
    .loader-core {{
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: linear-gradient(123deg, #18011f 7%, #b600a8 37%, #7621b0 72%, #be4c00 100%);
      box-shadow: 0 0 46px rgba(182, 0, 168, 0.7);
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    #cityScene {{
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      display: block;
      z-index: 0;
    }}
    .scene-vignette {{
      position: fixed;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      background:
        linear-gradient(90deg, rgba(12, 12, 12, 0.92), rgba(12, 12, 12, 0.28) 48%, rgba(12, 12, 12, 0.74)),
        linear-gradient(180deg, rgba(12, 12, 12, 0.08), rgba(12, 12, 12, 0.92));
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: linear-gradient(180deg, black, transparent 78%);
      z-index: 2;
    }}
    .hero-heading {{
      background: linear-gradient(180deg, #646973 0%, #bbccd7 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      min-height: 100vh;
      padding: 34px 0;
      position: relative;
      z-index: 3;
      display: grid;
      align-content: center;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      margin-bottom: 20px;
    }}
    nav {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 28px;
      color: var(--ink);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    nav a {{
      color: inherit;
      text-decoration: none;
      transition: opacity 200ms ease;
    }}
    nav a:hover {{ opacity: 0.7; }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      margin-bottom: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.07);
      color: var(--muted);
      padding: 0 12px;
      font-size: 0.82rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      backdrop-filter: blur(18px) saturate(150%);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(3rem, 10vw, 8.9rem);
      line-height: 0.82;
      letter-spacing: 0;
      max-width: 940px;
      text-wrap: balance;
      text-transform: uppercase;
      font-weight: 900;
    }}
    header p {{
      margin: 0;
      max-width: 330px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 1rem;
    }}
    form {{
      background: transparent;
    }}
    .surface {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(330px, 0.42fr);
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.1), rgba(215, 226, 234, 0.035)),
        var(--panel);
      padding: 12px;
      box-shadow: 0 36px 120px var(--shadow);
      backdrop-filter: blur(32px) saturate(145%);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.085), rgba(215, 226, 234, 0.035));
      padding: 12px;
    }}
    label.address {{
      grid-column: span 3;
    }}
    label {{
      display: grid;
      gap: 8px;
      min-width: 0;
      border: 1px solid rgba(215, 226, 234, 0.11);
      border-radius: 8px;
      background: rgba(215, 226, 234, 0.055);
      padding: 12px;
    }}
    label:focus-within {{
      border-color: rgba(182, 0, 168, 0.58);
      background: rgba(215, 226, 234, 0.085);
    }}
    label span {{
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    input {{
      width: 100%;
      min-height: 34px;
      border: 0;
      border-bottom: 1px solid rgba(215, 226, 234, 0.22);
      border-radius: 0;
      background: transparent;
      color: var(--ink);
      padding: 2px 0 8px;
      font: inherit;
      font-size: 1.05rem;
      font-weight: 650;
    }}
    input:focus {{
      outline: 0;
      border-color: var(--accent);
    }}
    .actions {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 18px;
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.12), transparent 38%),
        linear-gradient(145deg, rgba(24, 1, 31, 0.86), rgba(12, 12, 12, 0.98));
      color: white;
      padding: 22px;
      min-height: 100%;
      border: 1px solid rgba(215, 226, 234, 0.12);
      position: relative;
      overflow: hidden;
    }}
    .actions::before {{
      content: "";
      position: absolute;
      inset: 14px 14px auto auto;
      width: 96px;
      height: 96px;
      border-top: 1px solid rgba(190, 76, 0, 0.58);
      border-right: 1px solid rgba(182, 0, 168, 0.42);
    }}
    .actions > * {{
      position: relative;
    }}
    button {{
      min-height: 46px;
      border: 0;
      border-radius: 8px;
      background: linear-gradient(123deg, #18011f 7%, #b600a8 37%, #7621b0 72%, #be4c00 100%);
      color: white;
      padding: 0 24px;
      font: inherit;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      cursor: pointer;
      outline: 2px solid white;
      outline-offset: -3px;
      box-shadow: 0 4px 4px rgba(181, 1, 167, 0.25), 4px 4px 12px #7721b1 inset;
    }}
    button:hover {{ filter: brightness(1.06); }}
    .result {{
      display: grid;
      gap: 12px;
      align-content: center;
      text-align: left;
      min-height: 300px;
    }}
    .market-card {{
      display: grid;
      gap: 10px;
      margin: 8px 0 0;
    }}
    .market-card div {{
      border-top: 1px solid rgba(215, 226, 234, 0.12);
      padding-top: 10px;
    }}
    .market-card dt,
    .source-note,
    .model-note {{
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }}
    .market-card dd {{
      margin: 3px 0 0;
      font-weight: 700;
      color: white;
    }}
    .result span {{
      color: var(--muted);
      font-weight: 700;
      font-size: 0.9rem;
    }}
    .result strong {{
      color: white;
      font-size: clamp(2.6rem, 5vw, 4.6rem);
      line-height: 0.88;
      letter-spacing: 0;
      font-weight: 900;
    }}
    .empty-result {{
      display: grid;
      gap: 10px;
      align-content: center;
      min-height: 260px;
      color: var(--muted);
    }}
    .empty-result strong {{
      color: white;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.1;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .errors {{
      margin: 0 0 10px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fff7ed;
      color: var(--warn);
      padding: 12px 16px;
    }}
    .note {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.4;
    }}
    .scene {{
      height: 150px;
      margin-bottom: 12px;
      border: 1px solid rgba(215, 226, 234, 0.12);
      border-radius: 8px;
      background:
        linear-gradient(123deg, rgba(24, 1, 31, 0.95) 7%, rgba(182, 0, 168, 0.35) 37%, rgba(118, 33, 176, 0.26) 72%, rgba(190, 76, 0, 0.22) 100%),
        linear-gradient(180deg, rgba(215, 226, 234, 0.08), rgba(215, 226, 234, 0.02));
      overflow: hidden;
      position: relative;
    }}
    .scene::before {{
      content: "";
      position: absolute;
      inset: auto 8% 0;
      height: 78%;
      background:
        linear-gradient(90deg, transparent 0 6%, rgba(215, 226, 234, 0.22) 6% 7%, transparent 7% 14%, rgba(215, 226, 234, 0.16) 14% 15%, transparent 15% 24%, rgba(215, 226, 234, 0.2) 24% 25%, transparent 25% 100%),
        linear-gradient(180deg, transparent 0 18%, rgba(215, 226, 234, 0.18) 18% 19%, transparent 19% 42%, rgba(215, 226, 234, 0.12) 42% 43%, transparent 43% 100%);
      clip-path: polygon(0 72%, 8% 48%, 20% 58%, 30% 18%, 44% 42%, 58% 10%, 70% 46%, 82% 32%, 100% 62%, 100% 100%, 0 100%);
      opacity: 0.9;
    }}
    .scene::after {{
      content: "";
      position: absolute;
      left: 9%;
      right: 9%;
      bottom: 18px;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(182, 0, 168, 0.85), rgba(190, 76, 0, 0.75), transparent);
    }}
    @media (max-width: 820px) {{
      header {{ align-items: stretch; flex-direction: column; }}
      header p, .result {{ max-width: none; text-align: left; }}
      .surface {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      label.address {{ grid-column: span 2; }}
      button {{ width: 100%; }}
      main {{ align-content: start; }}
    }}
    @media (max-width: 520px) {{
      main {{ width: min(100% - 20px, 1080px); padding: 18px 0; }}
      h1 {{ font-size: clamp(2.45rem, 15vw, 4.1rem); }}
      header p {{ font-size: 0.94rem; }}
      .grid {{ grid-template-columns: 1fr; }}
      label.address {{ grid-column: auto; }}
      .scene {{ height: 120px; }}
      .result, .empty-result {{ min-height: 190px; }}
    }}
  </style>
</head>
<body class="loading">
  <div class="loader" aria-hidden="true"><div class="loader-orbit"><div class="loader-core"></div></div></div>
  <canvas id="cityScene" aria-hidden="true"></canvas>
  <div class="scene-vignette" aria-hidden="true"></div>
  <main>
    <nav aria-label="Primary">
      <a href="#estimate">Estimate</a>
      <a href="https://www.zillow.com/research/data/" target="_blank" rel="noreferrer">Zillow Data</a>
      <a href="#market">Market Signal</a>
      <a href="#contact">Contact</a>
    </nav>
    <header>
      <div>
        <span class="eyebrow">Cosmos interface</span>
        <h1 class="hero-heading">Real Estate Price Estimator</h1>
      </div>
      <p>Navigate property signals through a premium 3D pricing interface.</p>
    </header>
    <form id="estimate" method="post" action="/predict">
      {f'<ul class="errors">{error_items}</ul>' if error_items else ''}
      <div class="surface">
        <div class="grid">
          {fields}
        </div>
        <div class="actions">
          <div class="scene" aria-hidden="true"></div>
          {result}
          <button type="submit">Estimate price</button>
        </div>
      </div>
    </form>
  </main>
  <script type="module">
    import * as THREE from "/static/vendor/three.module.js";

    window.addEventListener("load", () => {{
      setTimeout(() => document.body.classList.remove("loading"), 450);
    }});
    document.body.classList.add("loading");
    document.querySelector("form").addEventListener("submit", (event) => {{
      const button = event.currentTarget.querySelector("button");
      button.disabled = true;
      button.textContent = "Loading market data";
      document.body.classList.add("loading");
    }});

    const canvas = document.getElementById("cityScene");
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0c0c0c, 10, 36);

    const camera = new THREE.PerspectiveCamera(44, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 4.6, 13);
    camera.lookAt(0, 0.4, 0);

    const ambient = new THREE.HemisphereLight(0xbbccd7, 0x18011f, 1.15);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xbe4c00, 2.2);
    key.position.set(4, 9, 6);
    scene.add(key);
    const rim = new THREE.PointLight(0xb600a8, 26, 34);
    rim.position.set(-6, 3, -5);
    scene.add(rim);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(38, 28, 18, 18),
      new THREE.MeshStandardMaterial({{ color: 0x0c0c0c, metalness: 0.28, roughness: 0.78 }})
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.3;
    scene.add(ground);

    const grid = new THREE.GridHelper(42, 42, 0xb600a8, 0x252733);
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    grid.position.y = -1.28;
    scene.add(grid);

    const group = new THREE.Group();
    const coreMaterial = new THREE.MeshStandardMaterial({{
      color: 0x15151a,
      metalness: 0.78,
      roughness: 0.18,
      emissive: 0x4d124f,
      emissiveIntensity: 0.38
    }});
    const glassMaterial = new THREE.MeshStandardMaterial({{
      color: 0xbbccd7,
      metalness: 0.24,
      roughness: 0.08,
      transparent: true,
      opacity: 0.34,
      emissive: 0x223348,
      emissiveIntensity: 0.28
    }});

    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.36, 2), coreMaterial);
    core.position.set(2.7, 1.35, -2.2);
    group.add(core);

    const glassShell = new THREE.Mesh(new THREE.IcosahedronGeometry(1.78, 1), glassMaterial);
    glassShell.position.copy(core.position);
    group.add(glassShell);

    const ringMaterial = new THREE.MeshBasicMaterial({{ color: 0xbbccd7, transparent: true, opacity: 0.34 }});
    const hotRingMaterial = new THREE.MeshBasicMaterial({{ color: 0xb600a8, transparent: true, opacity: 0.62 }});
    const rings = [];
    [2.9, 4.2, 5.7].forEach((radius, index) => {{
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.012 + index * 0.004, 8, 160),
        index === 1 ? hotRingMaterial : ringMaterial
      );
      ring.position.copy(core.position);
      ring.rotation.x = Math.PI / 2.3 + index * 0.32;
      ring.rotation.y = index * 0.45;
      rings.push(ring);
      group.add(ring);
    }});

    const markerMaterial = new THREE.MeshStandardMaterial({{
      color: 0xbe4c00,
      metalness: 0.6,
      roughness: 0.24,
      emissive: 0xbe4c00,
      emissiveIntensity: 0.5
    }});
    const markers = [];
    for (let i = 0; i < 18; i += 1) {{
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055 + (i % 3) * 0.025, 16, 16), markerMaterial);
      const angle = (i / 18) * Math.PI * 2;
      const radius = 2.9 + (i % 3) * 0.95;
      marker.userData = {{ angle, radius, speed: 0.22 + (i % 4) * 0.045, y: (i % 5) * 0.13 - 0.26 }};
      markers.push(marker);
      group.add(marker);
    }}

    const starGeometry = new THREE.BufferGeometry();
    const starPositions = [];
    for (let i = 0; i < 420; i += 1) {{
      starPositions.push(
        (Math.random() - 0.5) * 42,
        Math.random() * 18 - 2,
        (Math.random() - 0.5) * 34
      );
    }}
    starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(
      starGeometry,
      new THREE.PointsMaterial({{ color: 0xbbccd7, size: 0.035, transparent: true, opacity: 0.7 }})
    );
    scene.add(stars);

    scene.add(group);

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function resize() {{
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener("resize", resize);

    function animate(time) {{
      const t = time * 0.001;
      if (!reducedMotion) {{
        group.rotation.y = Math.sin(t * 0.26) * 0.12 - 0.18;
        core.rotation.x = t * 0.28;
        core.rotation.y = t * 0.4;
        glassShell.rotation.y = -t * 0.22;
        rings.forEach((ring, index) => {{
          ring.rotation.z = t * (0.08 + index * 0.035);
          ring.rotation.y += 0.0015 + index * 0.0005;
        }});
        markers.forEach((marker) => {{
          const next = marker.userData.angle + t * marker.userData.speed;
          marker.position.set(
            core.position.x + Math.cos(next) * marker.userData.radius,
            core.position.y + marker.userData.y + Math.sin(next * 1.7) * 0.25,
            core.position.z + Math.sin(next) * marker.userData.radius
          );
        }});
        stars.rotation.y = t * 0.018;
        camera.position.x = Math.sin(t * 0.18) * 0.55;
        camera.lookAt(0.9, 0.6, -1.6);
      }}
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate(0);
  </script>
</body>
</html>"""


def input_attributes(field: str) -> str:
    if field == "year_built":
        return 'type="number" min="1800" max="2026" step="1"'
    if field in {"address", "city", "neighborhood", "zip_code"}:
        return 'type="text"'
    return 'type="number" step="0.01"'


class AppHandler(BaseHTTPRequestHandler):
    model = None
    zhvi_path = DEFAULT_ZHVI_PATH

    def do_GET(self) -> None:
        if self.path == "/health":
            self.respond_json({"status": "ok"})
            return
        if self.path.startswith("/static/"):
            self.respond_static(self.path.removeprefix("/static/"))
            return
        if self.path not in {"/", "/predict"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.respond_html(render_page())

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        features, values, errors = parse_form(body)
        prediction = None
        model_prediction = None
        market_signal = None
        if not errors:
            try:
                model_prediction = predict_price(self.model, features)
                market_signal = self.lookup_market_signal(str(values.get("zip_code", "")))
                prediction = market_calibrated_estimate(model_prediction, market_signal)
            except ValueError as exc:
                errors.append(str(exc))
            except OSError:
                model_prediction = predict_price(self.model, features)
                prediction = model_prediction
                market_signal = None
        self.respond_html(
            render_page(
                values,
                prediction=prediction,
                model_prediction=model_prediction,
                market_signal=market_signal,
                errors=errors,
            )
        )

    def lookup_market_signal(self, zip_code: str):
        zhvi_path = ensure_zhvi_csv(self.zhvi_path)
        return latest_zhvi_for_zip(zip_code, zhvi_path)

    def respond_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_json(self, payload: dict[str, str]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_static(self, relative_path: str) -> None:
        requested = (STATIC_ROOT / relative_path).resolve()
        if not requested.is_file() or STATIC_ROOT.resolve() not in requested.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real estate price estimator web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--zhvi-cache", default=DEFAULT_ZHVI_PATH, type=Path)
    args = parser.parse_args()

    AppHandler.model = ensure_model(args.model, args.data)
    AppHandler.zhvi_path = args.zhvi_cache
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving real estate estimator at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
