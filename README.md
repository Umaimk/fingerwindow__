# Finger Window

A browser app that recreates the "dimension between your fingers" edit: two hands frame a
rectangular window, and inside it you see a stylised version of yourself while the rest of
the frame stays real.

Built with plain HTML/CSS/JS and MediaPipe Hand Landmarker (loaded from a CDN). No build step.

## Run it

Camera access needs `localhost` or HTTPS, so serve the folder:

```bash
cd finger-window
python -m http.server 8000
```

Open http://localhost:8000 in Chrome or Edge.

## Use it

1. **Start camera** (or **Load video** to use a pre-recorded clip).
2. Raise both hands and move them apart. A window opens between them; bring them together to close it.
3. Pick what shows inside the window:
   - **Live comic filter**: choose a Style: Cyberpunk (neon magenta/cyan duotone, scanlines, glowing frame; the default), Cartoon (cel-shaded), Comic (halftone shadows) or Pencil sketch. Computed in real time.
   - **My AI clip**: load a clip generated with Kling Motion Control (Higgsfield) from the *same* source video. It stays in sync with the real clip. Use this with **Load video**, not the webcam.
4. **Record** saves a `.webm` of the result.

## Tips

- Locked-off camera, good light, hands clearly separated and visible.
- Left hand and right hand are detected automatically; the window spans from the inner edge of one hand to the inner edge of the other.
- If the window jitters, raise **Smoothing**. If it lags behind your hands, lower it.
- For the AI-clip mode, the generated clip must have the same framing and length as the source.

## How it works

- `HandLandmarker` returns 21 landmarks per hand each frame.
- The window = inner edges of both hands (x) and the vertical extent of both hands (y), smoothed with a lerp.
- The window is a canvas clip region. The base frame is drawn first, then the stylised layer is drawn inside the clip.
- Comic filter: downscale to 480px wide with a light blur (removes camera noise), Sobel edges become soft ink lines, brightness is quantised into bands while the original hue is kept (cel shading), and the comic style adds halftone dots in the shadows. Uses `ctx.filter`, so use Chrome or Edge.

## Ideas to extend

- Halftone dot overlay.
- Pinch gesture to switch styles.
- Export an MP4 via ffmpeg.wasm.
- Deploy to GitHub Pages.
