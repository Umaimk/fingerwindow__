import { HandLandmarker, FaceLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const video = $("src");
const aiVideo = $("ai");
const canvas = $("out");
const ctx = canvas.getContext("2d");
const statusEl = $("status");

// Offscreen canvas for the stylised layer
const small = document.createElement("canvas");
const sctx = small.getContext("2d", { willReadFrequently: true });

let landmarker = null;
let faceLandmarker = null;
const faceSm = new Float32Array(478 * 2); // smoothed face landmarks (normalised)
let faceHas = false;
let lastFaceTime = -1e9;
let faceTick = 0;
const noiseTab = new Uint8Array(65536).map(() => (Math.random() * 256) | 0);
let usingWebcam = false;
let hasAi = false;
let rect = null; // current smoothed window, normalised 0..1 in DISPLAY space
let wasOpen = false;
let openFlash = 0;
let lastVideoTime = -1;
let lastResult = null;
let recorder = null;
let chunks = [];

// reusable buffers (avoid per-frame allocation)
let grayBuf = null;
let prevGray = null;
let copyBuf = null;
const glitch = { until: 0, next: 0, slices: [] };

const ui = {
  mode: $("mode"),
  mirror: $("mirror"),
  border: $("border"),
  levels: $("levels"),
  edge: $("edge"),
  smooth: $("smooth"),
  style: $("style"),
  quality: $("quality"),
  glow: $("glow"),
  fx: $("fx"),
};

function setStatus(msg) { statusEl.textContent = msg; }

// ---------- Setup ----------
async function loadModel() {
  if (landmarker) return;
  setStatus("Loading hand-tracking model…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
    });
  } catch (e) {
    faceLandmarker = null;
    setStatus("Face tracking unavailable: " + e.message);
  }
}

function sizeCanvases() {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const sw = Math.min(+ui.quality.value, w);
  const sh = Math.round((sw * h) / w);
  if (small.width !== sw || small.height !== sh) {
    small.width = sw;
    small.height = sh;
    grayBuf = new Float32Array(sw * sh);
    prevGray = null;
    copyBuf = null;
  }
}

$("btnCam").addEventListener("click", async () => {
  try {
    await loadModel();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    usingWebcam = true;
    ui.mirror.checked = true;
    await video.play();
    $("btnRec").disabled = false;
    setStatus("Raise both hands and frame a window between them.");
    requestAnimationFrame(frame);
  } catch (e) {
    setStatus("Camera error: " + e.message);
  }
});

$("fileSrc").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  await loadModel();
  usingWebcam = false;
  video.srcObject = null;
  video.src = URL.createObjectURL(f);
  video.loop = true;
  ui.mirror.checked = false;
  await video.play();
  if (hasAi) { aiVideo.currentTime = 0; aiVideo.play(); }
  $("btnRec").disabled = false;
  setStatus("Video loaded.");
  requestAnimationFrame(frame);
});

$("fileAi").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  aiVideo.src = URL.createObjectURL(f);
  aiVideo.loop = true;
  hasAi = true;
  ui.mode.value = "ai";
  await aiVideo.play().catch(() => {});
  setStatus("AI clip loaded. Load/restart your real video so they stay in sync.");
});

// ---------- Window geometry from two hands ----------
function computeTarget(hands) {
  if (!hands || hands.length < 2) return null;
  const mirror = ui.mirror.checked;
  const boxes = hands.map((lm) => {
    let minX = 1, maxX = 0, minY = 1, maxY = 0, sx = 0;
    for (const p of lm) {
      const x = mirror ? 1 - p.x : p.x;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      sx += x;
    }
    return { minX, maxX, minY, maxY, cx: sx / lm.length };
  });
  boxes.sort((a, b) => a.cx - b.cx);
  const [L, R] = boxes;
  const x1 = L.maxX, x2 = R.minX;
  if (x2 - x1 < 0.06) return null; // hands too close → window closed
  return {
    x1, x2,
    y1: Math.min(L.minY, R.minY),
    y2: Math.max(L.maxY, R.maxY),
  };
}

function updateRect(target) {
  const k = 1 - ui.smooth.value / 100; // higher smoothing slider = slower follow
  if (target) {
    rect = rect
      ? {
          x1: rect.x1 + (target.x1 - rect.x1) * k,
          x2: rect.x2 + (target.x2 - rect.x2) * k,
          y1: rect.y1 + (target.y1 - rect.y1) * k,
          y2: rect.y2 + (target.y2 - rect.y2) * k,
        }
      : { ...target, x2: target.x1 }; // grow open from nothing
  } else if (rect) {
    const cx = (rect.x1 + rect.x2) / 2;
    rect = {
      x1: rect.x1 + (cx - rect.x1) * 0.3,
      x2: rect.x2 + (cx - rect.x2) * 0.3,
      y1: rect.y1, y2: rect.y2,
    };
    if (rect.x2 - rect.x1 < 0.004) rect = null; // fully closed
  }
}

// ---------- Stylised layer ----------
function getGlitch(now, sw, sh, fx) {
  if (now > glitch.next) {
    glitch.until = now + 80 + Math.random() * 140;
    glitch.next = now + 900 + Math.random() * 2400;
    const count = 2 + ((Math.random() * 3) | 0);
    glitch.slices = [];
    for (let k = 0; k < count; k++) {
      const y0 = (Math.random() * sh * 0.9) | 0;
      const hgt = 3 + ((Math.random() * sh * 0.08) | 0);
      glitch.slices.push({
        y0, y1: y0 + hgt,
        dx: Math.round((Math.random() - 0.5) * sw * 0.12 * fx),
      });
    }
  }
  return now < glitch.until ? glitch.slices : [];
}

// colour split, glitch slices, rolling bright bar and flicker (cyberpunk only)
function applyCyberFx(d, sw, sh, now) {
  const fx = ui.fx.value / 100;
  const flick = 0.97 + 0.03 * Math.sin(now / 37) - (Math.random() < 0.015 * fx ? 0.15 : 0);
  const barY = ((now / 3500) % 1) * (sh + 60) - 30;
  const barH = sh * 0.03;
  const ab = Math.round((fx * sw) / 160); // chromatic aberration in px
  const slices = fx > 0 ? getGlitch(now, sw, sh, fx) : [];

  if (!copyBuf || copyBuf.length !== d.length) copyBuf = new Uint8ClampedArray(d.length);
  copyBuf.set(d);

  for (let y = 0; y < sh; y++) {
    let dx = 0;
    for (const s of slices) if (y >= s.y0 && y < s.y1) dx = s.dx;
    const gain = flick * (Math.abs(y - barY) < barH ? 1.3 : 1);
    const row = y * sw;
    for (let x = 0; x < sw; x++) {
      const sx = Math.min(sw - 1, Math.max(0, x - dx));
      const o = (row + x) * 4;
      const r = copyBuf[(row + Math.max(0, sx - ab)) * 4];
      const g = copyBuf[(row + sx) * 4 + 1];
      const b = copyBuf[(row + Math.min(sw - 1, sx + ab)) * 4 + 2];
      d[o] = r * gain;
      d[o + 1] = g * gain;
      d[o + 2] = b * gain;
    }
  }
}

function comicFrame(now) {
  const sw = small.width, sh = small.height;
  sctx.save();
  if (ui.mirror.checked) { sctx.translate(sw, 0); sctx.scale(-1, 1); }
  // light blur (scaled to resolution) removes camera noise so bands and lines stay clean
  sctx.filter = `blur(${(sw / 340).toFixed(2)}px) saturate(1.3) contrast(1.08)`;
  sctx.drawImage(video, 0, 0, sw, sh);
  sctx.restore();

  const img = sctx.getImageData(0, 0, sw, sh);
  const d = img.data;
  const n = sw * sh;
  const gray = grayBuf;

  // temporal smoothing of brightness = far less band flicker; skipped where things move fast
  const hadPrev = prevGray !== null;
  if (!hadPrev) prevGray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const cur = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    const g = hadPrev && Math.abs(cur - prevGray[i]) < 35 ? prevGray[i] * 0.4 + cur * 0.6 : cur;
    gray[i] = g;
    prevGray[i] = g;
  }

  const style = ui.style.value;
  const levels = +ui.levels.value;
  const threshold = +ui.edge.value;
  const softness = threshold * 0.6;
  const norm = sw / 480;                          // keep edge strength resolution-independent
  const dotP = Math.max(4, Math.round(sw / 120)); // halftone cell size
  const half = dotP / 2;
  const sp = Math.max(2, Math.round(sw / 200));   // scanline spacing
  const noiseSeed = (Math.random() * 65536) | 0;
  const scanShift = Math.floor(now / 70) % sp;    // scanlines crawl

  for (let y = 1; y < sh - 1; y++) {
    const rowScan = (y + scanShift) % sp === 0 ? 0.72 : 1;
    for (let x = 1; x < sw - 1; x++) {
      const i = y * sw + x;
      const gx =
        -gray[i - sw - 1] + gray[i - sw + 1] -
        2 * gray[i - 1] + 2 * gray[i + 1] -
        gray[i + sw - 1] + gray[i + sw + 1];
      const gy =
        -gray[i - sw - 1] - 2 * gray[i - sw] - gray[i - sw + 1] +
        gray[i + sw - 1] + 2 * gray[i + sw] + gray[i + sw + 1];
      const mag = Math.sqrt(gx * gx + gy * gy) * norm;
      const o = i * 4;
      const lum = gray[i];

      if (style === "sketch") {
        const line = Math.min(1, Math.max(0, (mag - threshold * 0.3) / (threshold * 1.4)));
        const shade = 0.82 + 0.18 * (lum / 255);
        const v = 255 * shade * (1 - 0.9 * line);
        d[o] = v; d[o + 1] = v * 0.97; d[o + 2] = v * 0.92;
        continue;
      }

      if (style === "anime") {
        // flat anime shading with a neon palette; a little of the real colour is kept
        const bands = Math.max(3, levels);
        const t = Math.round((lum / 255) * (bands - 1)) / (bands - 1);
        let r, g, b;
        if (t < 0.34) {
          const k = t / 0.34;
          r = 14 + k * (96 - 14); g = 8 + k * (20 - 8); b = 38 + k * (120 - 38);
        } else if (t < 0.67) {
          const k = (t - 0.34) / 0.33;
          r = 96 + k * (255 - 96); g = 20 + k * (52 - 20); b = 120 + k * (140 - 120);
        } else {
          const k = (t - 0.67) / 0.33;
          r = 255; g = 52 + k * (226 - 52); b = 140 + k * (170 - 140);
        }
        r = r * 0.82 + d[o] * 0.18;
        g = g * 0.82 + d[o + 1] * 0.18;
        b = b * 0.82 + d[o + 2] * 0.18;

        const ink = Math.min(1, Math.max(0, (mag - threshold) / softness));
        // cyan rim light on softer edges
        const rim = Math.min(1, Math.max(0, (mag - threshold * 0.4) / (threshold * 0.3))) * (1 - ink);
        g += 90 * rim; b += 115 * rim; r *= 1 - 0.35 * rim;

        // film grain
        const nz = (noiseTab[(i + noiseSeed) & 65535] - 128) * 0.1;
        d[o] = r * (1 - ink) + 10 * ink + nz;
        d[o + 1] = g * (1 - ink) + 6 * ink + nz;
        d[o + 2] = b * (1 - ink) + 30 * ink + nz;
        continue;
      }

      if (style === "cyber") {
        // neon duotone: shadows -> deep indigo, mids -> magenta, highlights -> cyan
        const t = Math.round((lum / 255) * (levels - 1)) / (levels - 1);
        let r, g, b;
        if (t < 0.55) {
          const k = t / 0.55;
          r = 12 + k * (255 - 12);
          g = 6 + k * (30 - 6);
          b = 40 + k * (200 - 40);
        } else {
          const k = (t - 0.55) / 0.45;
          r = 255 + k * (0 - 255);
          g = 30 + k * (240 - 30);
          b = 200 + k * (255 - 200);
        }
        // edges glow bright cyan-white
        const glow = Math.min(1, Math.max(0, (mag - threshold * 0.7) / (threshold * 0.7)));
        r = r * (1 - glow) + 150 * glow;
        g = g * (1 - glow) + 255 * glow;
        b = b * (1 - glow) + 255 * glow;
        d[o] = r * rowScan; d[o + 1] = g * rowScan; d[o + 2] = b * rowScan;
        continue;
      }

      // soft ink line: 0 = none, 1 = full black
      const ink = Math.min(1, Math.max(0, (mag - threshold) / softness));

      // cel shading: quantise brightness only, keep the original hue (skin stays skin)
      const band = Math.min(levels - 1, Math.floor((lum / 256) * levels));
      const qLum = 30 + (band / (levels - 1)) * 215;
      const scale = qLum / Math.max(lum, 8);
      let r = d[o] * scale, g = d[o + 1] * scale, b = d[o + 2] * scale;
      const avg = (r + g + b) / 3;
      r = avg + (r - avg) * 1.2;
      g = avg + (g - avg) * 1.2;
      b = avg + (b - avg) * 1.2;

      // comic halftone dots in the shadows
      if (style === "comic" && band <= 1) {
        const dot = (x % dotP < half && y % dotP < half) ? (band === 0 ? 0.45 : 0.75) : 1;
        r *= dot; g *= dot; b *= dot;
      }

      d[o] = r * (1 - ink) + 18 * ink;
      d[o + 1] = g * (1 - ink) + 18 * ink;
      d[o + 2] = b * (1 - ink) + 18 * ink;
    }
  }

  if (style === "cyber" || style === "anime") applyCyberFx(d, sw, sh, now);
  sctx.putImageData(img, 0, 0);
}

// ---------- Original cyber-character overlays (face tracked) ----------
function fp(i) {
  const w = canvas.width, h = canvas.height;
  const x = faceSm[i * 2], y = faceSm[i * 2 + 1];
  return { x: (ui.mirror.checked ? 1 - x : x) * w, y: y * h };
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b, t = 0.5) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

// natural eye: glossy anime highlights + magenta glow ring
function drawAnimeEye(e) {
  const r = dist(e.c1, e.c2) * 0.5;
  const c = e.iris;
  ctx.save();
  ctx.shadowColor = "#ff2bd6";
  ctx.shadowBlur = r * 0.9;
  ctx.strokeStyle = "rgba(255,43,214,0.85)";
  ctx.lineWidth = r * 0.16;
  ctx.beginPath();
  ctx.arc(c.x, c.y, r * 0.62, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.beginPath();
  ctx.arc(c.x - r * 0.2, c.y - r * 0.22, r * 0.17, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(c.x + r * 0.2, c.y + r * 0.2, r * 0.08, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// implant eye: dark socket, glowing iris, slit pupil, rotating rings, crosshair ticks
function drawCyberEye(e, now, dir, fw) {
  const r = dist(e.c1, e.c2) * 0.5;
  const c = e.iris;
  const R = r * 0.95;
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = R * 1.1;
  ctx.fillStyle = "rgba(6,4,18,0.92)";
  ctx.beginPath();
  ctx.arc(0, 0, R, 0, Math.PI * 2);
  ctx.fill();

  const g = ctx.createRadialGradient(0, 0, R * 0.05, 0, 0, R * 0.8);
  g.addColorStop(0, "#ffffff");
  g.addColorStop(0.3, "#fcee0a");
  g.addColorStop(1, "#ff2bd6");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.8, 0, Math.PI * 2);
  ctx.fill();

  ctx.shadowBlur = 0;
  const pulse = 0.35 + 0.1 * Math.sin(now / 300);
  ctx.fillStyle = "#07040f";
  ctx.beginPath();
  ctx.ellipse(0, 0, R * 0.12, R * pulse * 1.4, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.shadowBlur = R * 0.8;
  ctx.shadowColor = "#00f0ff";
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = R * 0.1;
  ctx.setLineDash([R * 0.5, R * 0.3]);
  ctx.rotate(now / 700);
  ctx.beginPath();
  ctx.arc(0, 0, R * 1.12, 0, Math.PI * 2);
  ctx.stroke();

  ctx.rotate(-now / 350);
  ctx.setLineDash([R * 0.2, R * 0.25]);
  ctx.strokeStyle = "#ff2bd6";
  ctx.shadowColor = "#ff2bd6";
  ctx.lineWidth = R * 0.07;
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.88, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // crosshair ticks
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.strokeStyle = "#fcee0a";
  ctx.shadowColor = "#fcee0a";
  ctx.shadowBlur = R * 0.5;
  ctx.lineWidth = Math.max(1.5, R * 0.08);
  for (let k = 0; k < 4; k++) {
    ctx.beginPath();
    ctx.moveTo(R * 1.28, 0);
    ctx.lineTo(R * 1.5, 0);
    ctx.stroke();
    ctx.rotate(Math.PI / 2);
  }
  ctx.restore();

  // leader line + label
  const fs = Math.max(10, fw * 0.045);
  ctx.save();
  ctx.font = `${fs}px "Share Tech Mono", monospace`;
  ctx.fillStyle = "#00f0ff";
  ctx.strokeStyle = "#00f0ff";
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 6;
  ctx.lineWidth = Math.max(1, fw * 0.004);
  const sx = c.x + dir * R * 1.5, sy = c.y - R * 0.3;
  const mx = c.x + dir * R * 2.3, my = c.y - R * 1.5;
  const ex = mx + dir * fw * 0.18;
  ctx.beginPath();
  ctx.moveTo(sx, sy); ctx.lineTo(mx, my); ctx.lineTo(ex, my);
  ctx.stroke();
  ctx.textAlign = dir > 0 ? "left" : "right";
  ctx.fillText("OPTIC // 0.98", dir > 0 ? mx + 4 : mx - 4, my - fs * 0.3);
  ctx.restore();
}

// circuit traces on the cheek with pulses travelling along them
function drawCircuit(edge, nose, fw, now) {
  const cheek = mid(nose, edge, 0.6);
  cheek.y += fw * 0.1;
  const dir = Math.sign(edge.x - cheek.x) || 1;
  const u = fw * 0.01;
  const paths = [
    [[0, 0], [12, 0], [18, -7], [30, -7]],
    [[0, 6], [8, 6], [14, 12], [14, 22]],
    [[0, -6], [6, -6], [10, -12], [10, -20]],
  ];
  ctx.save();
  ctx.lineJoin = "round";
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = fw * 0.03;
  for (const path of paths) {
    const pts = path.map(([x, y]) => ({ x: cheek.x + x * dir * u, y: cheek.y + y * u }));
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k].x, pts[k].y);
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(0,240,255,0.55)";
    ctx.lineWidth = Math.max(1.5, u * 0.7);
    ctx.stroke();
    // travelling pulse
    ctx.setLineDash([u * 3, u * 2.5]);
    ctx.lineDashOffset = -now / 40;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(1.5, u * 0.9);
    ctx.stroke();
    // end node
    const last = pts[pts.length - 1];
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff2bd6";
    ctx.shadowColor = "#ff2bd6";
    ctx.beginPath();
    ctx.arc(last.x, last.y, u * 1.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowColor = "#00f0ff";
  }
  ctx.restore();
}

// temple port with a blinking LED
function drawTemplePort(edge, nose, fw, now) {
  const dir = Math.sign(edge.x - nose.x) || 1;
  const x = edge.x - dir * fw * 0.02, y = edge.y - fw * 0.14;
  const w = fw * 0.055, h = fw * 0.1;
  ctx.save();
  ctx.fillStyle = "rgba(6,4,18,0.92)";
  ctx.strokeStyle = "#00f0ff";
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = fw * 0.03;
  ctx.lineWidth = Math.max(1.5, fw * 0.006);
  ctx.fillRect(x - w / 2, y - h / 2, w, h);
  ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  ctx.beginPath();
  ctx.moveTo(x - w * 0.25, y - h * 0.1); ctx.lineTo(x + w * 0.25, y - h * 0.1);
  ctx.moveTo(x - w * 0.25, y + h * 0.15); ctx.lineTo(x + w * 0.25, y + h * 0.15);
  ctx.stroke();
  const on = Math.floor(now / 400) % 2 === 0;
  ctx.fillStyle = on ? "#fcee0a" : "#ff2bd6";
  ctx.shadowColor = ctx.fillStyle;
  ctx.beginPath();
  ctx.arc(x, y - h * 0.32, fw * 0.011, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// face-tracking brackets, scan sweep and readout text
function drawFaceHud(edgeL, edgeR, fw, now) {
  const top = fp(10), chin = fp(152);
  const x1 = Math.min(edgeL.x, edgeR.x) - fw * 0.12;
  const x2 = Math.max(edgeL.x, edgeR.x) + fw * 0.12;
  const y1 = top.y - fw * 0.15;
  const y2 = chin.y + fw * 0.1;
  const cl = fw * 0.12;
  ctx.save();
  ctx.strokeStyle = "#fcee0a";
  ctx.shadowColor = "#fcee0a";
  ctx.shadowBlur = 8;
  ctx.lineWidth = Math.max(2, fw * 0.008);
  ctx.beginPath();
  ctx.moveTo(x1, y1 + cl); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cl, y1);
  ctx.moveTo(x2 - cl, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cl);
  ctx.moveTo(x2, y2 - cl); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cl, y2);
  ctx.moveTo(x1 + cl, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - cl);
  ctx.stroke();

  // scan sweep
  const sy = y1 + ((now / 2200) % 1) * (y2 - y1);
  const gr = ctx.createLinearGradient(0, sy - fw * 0.12, 0, sy);
  gr.addColorStop(0, "rgba(0,240,255,0)");
  gr.addColorStop(1, "rgba(0,240,255,0.28)");
  ctx.shadowBlur = 0;
  ctx.fillStyle = gr;
  ctx.fillRect(x1, sy - fw * 0.12, x2 - x1, fw * 0.12);
  ctx.fillStyle = "#00f0ff";
  ctx.fillRect(x1, sy, x2 - x1, Math.max(1.5, fw * 0.005));

  // readout text (flickers)
  const fs = Math.max(10, fw * 0.05);
  ctx.font = `${fs}px "Share Tech Mono", monospace`;
  ctx.globalAlpha = Math.random() < 0.05 ? 0.3 : 0.95;
  ctx.fillStyle = "#fcee0a";
  const pct = 82 + Math.floor(8 * (0.5 + 0.5 * Math.sin(now / 900)));
  ctx.fillText(`NEURAL LINK  ${pct}%`, x1, y1 - fs * 0.4);
  ctx.fillStyle = "#ff2bd6";
  ctx.fillText("SYNC // STABLE", x1, y2 + fs * 1.2);
  ctx.restore();
}

function drawCyberFace(now) {
  if (!faceHas || now - lastFaceTime > 400) return;
  const eyeA = { iris: fp(468), c1: fp(33), c2: fp(133) };
  const eyeB = { iris: fp(473), c1: fp(362), c2: fp(263) };
  const edgeL = fp(234), edgeR = fp(454);
  const nose = fp(1);
  const fw = dist(edgeL, edgeR);
  if (fw < 20) return;

  // implant on the eye that appears on the right of the screen
  const [natural, implant] = eyeA.iris.x < eyeB.iris.x ? [eyeA, eyeB] : [eyeB, eyeA];
  const edgeImp = edgeL.x > edgeR.x ? edgeL : edgeR;
  const edgeNat = edgeImp === edgeL ? edgeR : edgeL;
  const dir = Math.sign(implant.iris.x - nose.x) || 1;

  drawCircuit(edgeNat, nose, fw, now);
  drawTemplePort(edgeImp, nose, fw, now);
  drawAnimeEye(natural);
  drawCyberEye(implant, now, dir, fw);
  drawFaceHud(edgeL, edgeR, fw, now);
}

// ---------- Draw loop ----------
function drawBase() {
  const w = canvas.width, h = canvas.height;
  ctx.save();
  if (ui.mirror.checked) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
}

function drawInside(px, py, pw, ph, now) {
  const w = canvas.width, h = canvas.height;
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, pw, ph);
  ctx.clip();

  const useAi = ui.mode.value === "ai" && hasAi && aiVideo.readyState >= 2;
  if (useAi) {
    // keep AI clip in sync with the real one
    if (!usingWebcam && Math.abs(aiVideo.currentTime - video.currentTime) > 0.15) {
      aiVideo.currentTime = video.currentTime;
    }
  } else {
    comicFrame(now);
  }

  const paint = () => {
    if (useAi) {
      ctx.save();
      if (ui.mirror.checked) { ctx.translate(w, 0); ctx.scale(-1, 1); }
      ctx.drawImage(aiVideo, 0, 0, w, h);
      ctx.restore();
    } else {
      ctx.drawImage(small, 0, 0, w, h);
    }
  };

  ctx.imageSmoothingQuality = "high";
  paint();

  // bloom: blurred copy added on top makes bright areas glow
  const glow = ui.glow.value / 100;
  if (glow > 0 && !(ui.style.value === "sketch" && !useAi)) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.8 * glow;
    ctx.filter = `blur(${Math.max(4, w / 160)}px)`;
    paint();
    ctx.restore();
  }

  if (!useAi && ui.style.value === "anime") drawCyberFace(now);

  // flash when the window opens
  if (openFlash > 0.03) {
    ctx.fillStyle = `rgba(255,255,255,${(openFlash * 0.55).toFixed(3)})`;
    ctx.fillRect(px, py, pw, ph);
  }
  ctx.restore();

  if (ui.border.checked) {
    if (ui.style.value === "cyber") {
      // glowing, pulsing neon frame with corner brackets
      const pulse = 0.75 + 0.25 * Math.sin(now / 200);
      ctx.save();
      ctx.shadowColor = "#00f0ff";
      ctx.shadowBlur = Math.max(12, w / 60) * (0.6 + pulse);
      ctx.lineWidth = Math.max(3, w / 320);
      ctx.strokeStyle = "#00f0ff";
      ctx.strokeRect(px, py, pw, ph);
      ctx.shadowColor = "#ff2bd6";
      ctx.strokeStyle = "#ff2bd6";
      ctx.lineWidth = Math.max(1.5, w / 640);
      ctx.strokeRect(px + 6, py + 6, pw - 12, ph - 12);

      const L = Math.min(pw, ph) * 0.12;
      ctx.shadowColor = "#fcee0a";
      ctx.strokeStyle = "#fcee0a";
      ctx.lineWidth = Math.max(4, w / 240);
      ctx.beginPath();
      ctx.moveTo(px, py + L); ctx.lineTo(px, py); ctx.lineTo(px + L, py);
      ctx.moveTo(px + pw - L, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + L);
      ctx.moveTo(px + pw, py + ph - L); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw - L, py + ph);
      ctx.moveTo(px + L, py + ph); ctx.lineTo(px, py + ph); ctx.lineTo(px, py + ph - L);
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.lineWidth = Math.max(4, w / 240);
      ctx.strokeStyle = "#111";
      ctx.strokeRect(px, py, pw, ph);
      ctx.lineWidth = Math.max(2, w / 480);
      ctx.strokeStyle = "#fff";
      ctx.strokeRect(px + 4, py + 4, pw - 8, ph - 8);
    }
  }
}

function frame() {
  if (video.readyState < 2) { requestAnimationFrame(frame); return; }
  sizeCanvases();
  const now = performance.now();

  const newFrame = usingWebcam || video.currentTime !== lastVideoTime;
  if (newFrame) lastVideoTime = video.currentTime;
  if (landmarker && newFrame) {
    lastResult = landmarker.detectForVideo(video, now);
  }
  // face tracking only runs for the character style, every other frame
  if (faceLandmarker && newFrame && ui.style.value === "anime" && faceTick++ % 2 === 0) {
    const fr = faceLandmarker.detectForVideo(video, now);
    const lm = fr.faceLandmarks && fr.faceLandmarks[0];
    if (lm) {
      const n = Math.min(lm.length, 478);
      for (let i = 0; i < n; i++) {
        faceSm[i * 2] = faceHas ? faceSm[i * 2] + (lm[i].x - faceSm[i * 2]) * 0.6 : lm[i].x;
        faceSm[i * 2 + 1] = faceHas ? faceSm[i * 2 + 1] + (lm[i].y - faceSm[i * 2 + 1]) * 0.6 : lm[i].y;
      }
      faceHas = true;
      lastFaceTime = now;
    }
  }
  updateRect(computeTarget(lastResult?.landmarks));

  const open = !!rect;
  if (open && !wasOpen) openFlash = 1;
  wasOpen = open;
  openFlash *= 0.9;

  drawBase();
  if (rect) {
    const w = canvas.width, h = canvas.height;
    const px = rect.x1 * w, py = rect.y1 * h;
    drawInside(px, py, (rect.x2 - rect.x1) * w, (rect.y2 - rect.y1) * h, now);
  }
  requestAnimationFrame(frame);
}

// ---------- Recording ----------
$("btnRec").addEventListener("click", () => {
  const btn = $("btnRec");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  const stream = canvas.captureStream(30);
  recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "finger-window.webm";
    a.click();
    btn.textContent = "● Record";
    btn.classList.remove("rec");
    setStatus("Saved finger-window.webm");
  };
  recorder.start();
  btn.textContent = "■ Stop & save";
  btn.classList.add("rec");
  setStatus("Recording…");
});