/**
 * Structure-tensor flowmap extraction (Unity-style RG encode).
 * Image space: +x right, +y down. Output invertY matches Unity UV +y up.
 *
 * Continuity: local orientation is unsigned (180° ambiguous). We align every
 * pixel's direction to a continuous "guide" field = 90°-rotated gradient of a
 * heavily blurred source, so magenta/green domain walls collapse.
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

function gradient(src, w, h) {
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const xl = (x - 1 + w) % w;
      const xr = (x + 1) % w;
      const yu = (y - 1 + h) % h;
      const yd = (y + 1) % h;
      gx[i] = 0.5 * (src[y * w + xr] - src[y * w + xl]);
      gy[i] = 0.5 * (src[yd * w + x] - src[yu * w + x]);
    }
  }
  return { gx, gy };
}

function percentile(arr, p) {
  const a = Array.from(arr);
  a.sort((x, y) => x - y);
  const i = Math.min(a.length - 1, Math.max(0, Math.floor((p / 100) * (a.length - 1))));
  return a[i];
}

/** Soft-blend opposite borders so a non-tileable source becomes more periodic. */
function makeScalarSeamless(src, w, h, band) {
  const out = src.slice();
  const b = Math.max(1, Math.min(band, (Math.min(w, h) / 2) | 0));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < b; x++) {
      const t = x / b;
      const a = 0.5 * (1 - t) * (1 - t);
      const iL = y * w + x;
      const iR = y * w + (w - 1 - x);
      const avg = 0.5 * (src[iL] + src[iR]);
      out[iL] = src[iL] * (1 - a) + avg * a;
      out[iR] = src[iR] * (1 - a) + avg * a;
    }
  }
  const tmp = out.slice();
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < b; y++) {
      const t = y / b;
      const a = 0.5 * (1 - t) * (1 - t);
      const iT = y * w + x;
      const iB = (h - 1 - y) * w + x;
      const avg = 0.5 * (tmp[iT] + tmp[iB]);
      out[iT] = tmp[iT] * (1 - a) + avg * a;
      out[iB] = tmp[iB] * (1 - a) + avg * a;
    }
  }
  return out;
}

/** Align & feather vector field across wrap so tile edges don't hard-cut. */
function makeVectorSeamless(vx, vy, w, h, band) {
  const b = Math.max(1, Math.min(band, (Math.min(w, h) / 2) | 0));

  // Flip signs so opposite borders agree, then average
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < b; x++) {
      const iL = y * w + x;
      const iR = y * w + (w - 1 - x);
      if (vx[iL] * vx[iR] + vy[iL] * vy[iR] < 0) {
        vx[iR] = -vx[iR];
        vy[iR] = -vy[iR];
      }
      const t = x / b;
      const a = 0.5 * (1 - t);
      const ax = 0.5 * (vx[iL] + vx[iR]);
      const ay = 0.5 * (vy[iL] + vy[iR]);
      const n = Math.hypot(ax, ay) + 1e-12;
      const mx = ax / n;
      const my = ay / n;
      vx[iL] = vx[iL] * (1 - a) + mx * a;
      vy[iL] = vy[iL] * (1 - a) + my * a;
      vx[iR] = vx[iR] * (1 - a) + mx * a;
      vy[iR] = vy[iR] * (1 - a) + my * a;
      const nL = Math.hypot(vx[iL], vy[iL]) + 1e-12;
      const nR = Math.hypot(vx[iR], vy[iR]) + 1e-12;
      vx[iL] /= nL; vy[iL] /= nL;
      vx[iR] /= nR; vy[iR] /= nR;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < b; y++) {
      const iT = y * w + x;
      const iB = (h - 1 - y) * w + x;
      if (vx[iT] * vx[iB] + vy[iT] * vy[iB] < 0) {
        vx[iB] = -vx[iB];
        vy[iB] = -vy[iB];
      }
      const t = y / b;
      const a = 0.5 * (1 - t);
      const ax = 0.5 * (vx[iT] + vx[iB]);
      const ay = 0.5 * (vy[iT] + vy[iB]);
      const n = Math.hypot(ax, ay) + 1e-12;
      const mx = ax / n;
      const my = ay / n;
      vx[iT] = vx[iT] * (1 - a) + mx * a;
      vy[iT] = vy[iT] * (1 - a) + my * a;
      vx[iB] = vx[iB] * (1 - a) + mx * a;
      vy[iB] = vy[iB] * (1 - a) + my * a;
      const nT = Math.hypot(vx[iT], vy[iT]) + 1e-12;
      const nB = Math.hypot(vx[iB], vy[iB]) + 1e-12;
      vx[iT] /= nT; vy[iT] /= nT;
      vx[iB] /= nB; vy[iB] /= nB;
    }
  }
}

/**
 * @param {Float32Array} gray 0..1
 * @param {number} w
 * @param {number} h
 * @param {{gradSigma?: number, tensorSigma?: number, guideSigma?: number, vectorSigma?: number, invertY?: boolean, seamless?: boolean, seamBand?: number}} opts
 */
export function extractFlowmap(gray, w, h, opts = {}) {
  const gradSigma = opts.gradSigma ?? 1.0;
  const tensorSigma = opts.tensorSigma ?? 4.0;
  // Large blur → continuous isocontour guide (kills 180° domain walls)
  const guideSigma = opts.guideSigma ?? 48.0;
  // Soften directed vectors after sign alignment
  const vectorSigma = opts.vectorSigma ?? Math.max(2, tensorSigma * 0.35);
  const invertY = opts.invertY ?? true;
  const seamless = opts.seamless ?? true;
  const seamBand = opts.seamBand ?? 8;

  let gmin = Infinity;
  let gmax = -Infinity;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] < gmin) gmin = gray[i];
    if (gray[i] > gmax) gmax = gray[i];
  }
  const span = Math.max(gmax - gmin, 1e-8);
  let norm = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) norm[i] = (gray[i] - gmin) / span;
  if (seamless) {
    norm = makeScalarSeamless(norm, w, h, seamBand);
    // Extra wrap blur so derivatives at borders match
    norm = blurSeparable(norm, w, h, Math.max(1, seamBand * 0.15));
  }

  // --- Local orientation via structure tensor (unsigned) ---
  const gs = blurSeparable(norm, w, h, gradSigma);
  const { gx, gy } = gradient(gs, w, h);

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

  const ox = new Float32Array(w * h);
  const oy = new Float32Array(w * h);
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
    ox[i] = ax / n;
    oy[i] = ay / n;
    coherence[i] = (l1 - l2) / (l1 + l2 + 1e-12);
  }

  // Smooth orientation in double-angle space (still unsigned)
  let c2 = new Float32Array(w * h);
  let s2 = new Float32Array(w * h);
  const weight = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const ang = Math.atan2(oy[i], ox[i]);
    const wgt = Math.min(1, Math.max(0.05, coherence[i]));
    weight[i] = wgt;
    c2[i] = Math.cos(2 * ang) * wgt;
    s2[i] = Math.sin(2 * ang) * wgt;
  }
  const orientBlur = Math.max(3, tensorSigma * 0.75);
  for (let pass = 0; pass < 4; pass++) {
    const c2s = blurSeparable(c2, w, h, orientBlur);
    const s2s = blurSeparable(s2, w, h, orientBlur);
    const ws = blurSeparable(weight, w, h, orientBlur);
    for (let i = 0; i < w * h; i++) {
      const den = ws[i] + 1e-8;
      c2[i] = c2s[i] / den;
      s2[i] = s2s[i] / den;
      const n = Math.hypot(c2[i], s2[i]) + 1e-12;
      c2[i] /= n;
      s2[i] /= n;
      weight[i] = 1;
    }
  }
  for (let i = 0; i < w * h; i++) {
    const ang = 0.5 * Math.atan2(s2[i], c2[i]);
    ox[i] = Math.cos(ang);
    oy[i] = Math.sin(ang);
  }

  // --- Continuous guide: isocontour direction of heavily blurred source ---
  const guide = blurSeparable(norm, w, h, guideSigma);
  const ggrad = gradient(guide, w, h);
  const gxG = ggrad.gx;
  const gyG = ggrad.gy;

  const vx = new Float32Array(w * h);
  const vy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    // Rotate gradient 90° → flow along isolines (continuous except at extrema)
    let gdx = -gyG[i];
    let gdy = gxG[i];
    const gn = Math.hypot(gdx, gdy);
    if (gn < 1e-8) {
      vx[i] = ox[i];
      vy[i] = oy[i];
      continue;
    }
    gdx /= gn;
    gdy /= gn;

    // Pick orientation sign so it agrees with the continuous guide
    const dot = ox[i] * gdx + oy[i] * gdy;
    if (dot < 0) {
      vx[i] = -ox[i];
      vy[i] = -oy[i];
    } else {
      vx[i] = ox[i];
      vy[i] = oy[i];
    }
  }

  // Blend toward guide in low-coherence regions, then blur directed field
  for (let i = 0; i < w * h; i++) {
    let gdx = -gyG[i];
    let gdy = gxG[i];
    const gn = Math.hypot(gdx, gdy) + 1e-12;
    gdx /= gn;
    gdy /= gn;
    const a = Math.min(1, Math.max(0.15, coherence[i]));
    let x = vx[i] * a + gdx * (1 - a);
    let y = vy[i] * a + gdy * (1 - a);
    const n = Math.hypot(x, y) + 1e-12;
    vx[i] = x / n;
    vy[i] = y / n;
  }

  const vxB = blurSeparable(vx, w, h, vectorSigma);
  const vyB = blurSeparable(vy, w, h, vectorSigma);
  for (let i = 0; i < w * h; i++) {
    const n = Math.hypot(vxB[i], vyB[i]) + 1e-12;
    vx[i] = vxB[i] / n;
    vy[i] = vyB[i] / n;
  }

  // One more neighbor sign polish against local average (removes residual flips)
  for (let pass = 0; pass < 3; pass++) {
    const ax = blurSeparable(vx, w, h, 1.5);
    const ay = blurSeparable(vy, w, h, 1.5);
    for (let i = 0; i < w * h; i++) {
      if (vx[i] * ax[i] + vy[i] * ay[i] < 0) {
        vx[i] = -vx[i];
        vy[i] = -vy[i];
      }
    }
  }

  if (seamless) {
    makeVectorSeamless(vx, vy, w, h, seamBand);
    const vxS = blurSeparable(vx, w, h, Math.max(1.5, seamBand * 0.12));
    const vyS = blurSeparable(vy, w, h, Math.max(1.5, seamBand * 0.12));
    for (let i = 0; i < w * h; i++) {
      const n = Math.hypot(vxS[i], vyS[i]) + 1e-12;
      vx[i] = vxS[i] / n;
      vy[i] = vyS[i] / n;
    }
  }

  const gRef = percentile(gradMag, 90) + 1e-8;
  const mag = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const ridge = norm[i];
    const gm = Math.min(1, gradMag[i] / gRef);
    const guideM = Math.hypot(gxG[i], gyG[i]);
    mag[i] = Math.min(
      1,
      Math.max(0, (0.35 + 0.65 * coherence[i]) * (0.4 + 0.6 * ridge) * (0.35 + 0.65 * gm) * (0.5 + 0.5 * Math.min(1, guideM * 8)))
    );
  }
  const magB = blurSeparable(mag, w, h, Math.max(1, vectorSigma));

  const fx = new Float32Array(w * h);
  const fy = new Float32Array(w * h);
  const speeds = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    fx[i] = vx[i] * magB[i];
    fy[i] = (invertY ? -vy[i] : vy[i]) * magB[i];
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
    rgba[i * 4 + 2] = Math.round(magB[i] * 255);
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
  w = Math.max(1, w | 0);
  h = Math.max(1, h | 0);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  // Composite over mid-gray so transparent / black-RGB+alpha cutouts keep detail
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, w, h);
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
