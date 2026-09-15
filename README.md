# S2R-Adaptation-TTS

Project website for **From Reliable Text to Real Voices: Trust-Aware Progressive Adaptation for Low-Resource TTS**.

The page includes the paper, method diagram, Burmese and Lao audio comparisons, experimental results, and citation information.

## Website

GitHub Pages serves the root of the `main` branch. The site is plain HTML, CSS, and JavaScript; no build step or backend is needed.

- `index.html`: research content, methods, results, and resource links
- `app.js`: audio comparison, result tabs, and citation controls
- `demo-data.js` and `assets/demo-data.json`: matching audio metadata
- `assets/audio/`: selected model outputs and reference recordings
- `assets/paper.pdf`: manuscript
- `assets/fonts/`: local fonts and their license files

The audio section contains one selected text per language: Burmese case13 and Lao case12. Each sample retains all available model outputs and the same reference voice. S→R audio is labeled cubic according to the authors' corrected model naming.

Open `index.html` locally or serve this directory with a static HTTP server. To update the website, edit these files and push to `main`.

Training code and model weight release links have not yet been provided.
