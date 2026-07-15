/**
 * Structure-tensor flowmap extraction (Unity-style RG encode).
 * Image space: +x right, +y down. Output invertY matches Unity UV +y up.
 */

function gaussianKernel(sigma) {
  if (sigma <= 0) return { k: new Float32Array([1]), r: 0 };
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(r * 2 + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { k, r };
}

function blurSeparable(src, w, h, sigma) {
  if (sigma <= 0) return src.slice();
  const { k, r } = gaussianKernel(sigma);
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xx = (x + i + w) % w;
        acc += src[y * w + xx] * k[i + r];
      }
      tmp[y * w + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yy = (y + i + h) % h;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

function percentile(arr, p) {
  const a = Array.from(arr);
  a.sort((x, y) => x - y);
  const i = Math.min(a.length - 1, Math.max(0, Math.floor((p / 100) * (a.length - 1))));
  return a[i];
}

/**
 * @param {Float32Array} gray 0..1
 * @param {number} w
 * @param {number} h
 * @param {{gradSigma?: number, tensorSigma?: number, invertY?: boolean}} opts
 * @returns {{rgba: Uint8ClampedArray, w: number, h: number}}
 */
export function extractFlowmap(gray, w, h, opts = {}) {
  const gradSigma = opts.gradSigma ?? 1.0;
  const tensorSigma = opts.tensorSigma ?? 6.0;
  const invertY = opts.invertY ?? true;

  let g = gray;
  let gmin = Infinity;
  let gmax = -Infinity;
  for (let i = 0; i < g.length; i++) {
    if (g[i] < gmin) gmin = g[i];
    if (g[i] > gmax) gmax = g[i];
  }
  const span = Math.max(gmax - gmin, 1e-8);
  const norm = new Float32Array(g.length);
  for (let i = 0; i < g.length; i++) norm[i] = (g[i] - gmin) / span;

  const gs = blurSeparable(norm, w, h, gradSigma);
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const xl = (x - 1 + w) % w;
      const xr = (x + 1) % w;
      const yu = (y - 1 + h) % h;
      const yd = (y + 1) % h;
      gx[i] = 0.5 * (gs[y * w + xr] - gs[y * w + xl]);
      gy[i] = 0.5 * (gs[yd * w + x] - gs[yu * w + x]);
    }
  }

  const jxxSrc = new Float32Array(w * h);
  const jyySrc = new Float32Array(w * h);
  const jxySrc = new Float32Array(w * h);
  const gradMag = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    jxxSrc[i] = gx[i] * gx[i];
    jyySrc[i] = gy[i] * gy[i];
    jxySrc[i] = gx[i] * gy[i];
    gradMag[i] = Math.hypot(gx[i], gy[i]);
  }

  const jxx = blurSeparable(jxxSrc, w, h, tensorSigma);
  const jyy = blurSeparable(jyySrc, w, h, tensorSigma);
  const jxy = blurSeparable(jxySrc, w, h, tensorSigma);

  const vx = new Float32Array(w * h);
  const vy = new Float32Array(w * h);
  const coherence = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const trace = jxx[i] + jyy[i];
    const det = jxx[i] * jyy[i] - jxy[i] * jxy[i];
    const disc = Math.sqrt(Math.max(trace * trace - 4 * det, 0));
    const l2 = 0.5 * (trace - disc);
    const l1 = 0.5 * (trace + disc);

    let ax = jxy[i];
    let ay = l2 - jxx[i];
    if (Math.abs(jxy[i]) < 1e-12) {
      ax = l2 - jyy[i];
      ay = jxy[i];
    }
    const n = Math.hypot(ax, ay) + 1e-12;
    vx[i] = ax / n;
    vy[i] = ay / n;
    coherence[i] = (l1 - l2) / (l1 + l2 + 1e-12);
  }

  // Resolve 180° flip via double-angle smoothing
  let c2 = new Float32Array(w * h);
  let s2 = new Float32Array(w * h);
  const weight = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const ang = Math.atan2(vy[i], vx[i]);
    const wgt = Math.min(1, Math.max(0.05, coherence[i]));
    weight[i] = wgt;
    c2[i] = Math.cos(2 * ang) * wgt;
    s2[i] = Math.sin(2 * ang) * wgt;
  }
  for (let pass = 0; pass < 8; pass++) {
    const c2s = blurSeparable(c2, w, h, 2);
    const s2s = blurSeparable(s2, w, h, 2);
    const ws = blurSeparable(weight, w, h, 2);
    for (let i = 0; i < w * h; i++) {
      const den = ws[i] + 1e-8;
      c2[i] = c2s[i] / den;
      s2[i] = s2s[i] / den;
    }
  }
  for (let i = 0; i < w * h; i++) {
    const ang2 = 0.5 * Math.atan2(s2[i], c2[i]);
    const a0 = Math.atan2(vy[i], vx[i]);
    const d0 = Math.abs(Math.atan2(Math.sin(a0 - ang2), Math.cos(a0 - ang2)));
    const d1 = Math.abs(Math.atan2(Math.sin(a0 + Math.PI - ang2), Math.cos(a0 + Math.PI - ang2)));
    if (d1 < d0) {
      vx[i] = -vx[i];
      vy[i] = -vy[i];
    }
  }

  const gRef = percentile(gradMag, 90) + 1e-8;
  const mag = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const ridge = norm[i];
    const gm = gradMag[i] / gRef;
    mag[i] = Math.min(1, Math.max(0, coherence[i] * (0.35 + 0.65 * ridge) * (0.25 + 0.75 * Math.min(1, gm))));
  }

  const fx = new Float32Array(w * h);
  const fy = new Float32Array(w * h);
  const speeds = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    fx[i] = vx[i] * mag[i];
    fy[i] = (invertY ? -vy[i] : vy[i]) * mag[i];
    speeds[i] = Math.hypot(fx[i], fy[i]);
  }
  const peak = percentile(speeds, 99) || 1e-6;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    let x = fx[i] / peak;
    let y = fy[i] / peak;
    x = Math.min(1, Math.max(-1, x));
    y = Math.min(1, Math.max(-1, y));
    rgba[i * 4] = Math.round((x * 0.5 + 0.5) * 255);
    rgba[i * 4 + 1] = Math.round((y * 0.5 + 0.5) * 255);
    rgba[i * 4 + 2] = Math.round(mag[i] * 255);
    rgba[i * 4 + 3] = 255;
  }
  return { rgba, w, h };
}

export function imageToGray(img, maxSize = 1024) {
  let w = img.width;
  let h = img.height;
  if (w > maxSize || h > maxSize) {
    const s = maxSize / Math.max(w, h);
    w = Math.round(w * s);
    h = Math.round(h * s);
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  }
  return { gray, w, h, canvas: c };
}

export function rgbaToImageBitmap(rgba, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  return c;
}
