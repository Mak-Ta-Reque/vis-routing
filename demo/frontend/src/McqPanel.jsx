import { useState, useMemo, useEffect } from "react";

const API = "";

// Bar showing one option's probability, highlighting correct/predicted letters.
function ProbBar({ letter, name, prob, isCorrect, isPredicted }) {
  const pct = Math.round((prob ?? 0) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
      <div style={{ width: 18, fontWeight: 700, color: isPredicted ? "#4ade80" : "#ccc" }}>{letter}</div>
      <div style={{ flex: "0 0 120px", fontSize: 12, color: isCorrect ? "#4ade80" : "#ccc" }}>
        {name} {isCorrect && "✓"}
      </div>
      <div style={{ flex: 1, background: "#222", borderRadius: 4, height: 14, position: "relative", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: isPredicted ? "#4ade80" : "#555" }} />
      </div>
      <div style={{ width: 40, fontSize: 12, textAlign: "right", color: "#999" }}>{pct}%</div>
    </div>
  );
}

function McqResult({ title, letters, options, correctLetter, predictedLetter, probs }) {
  const correct = predictedLetter === correctLetter;
  return (
    <div style={{ background: "#1a1a1a", padding: 12, borderRadius: 6, flex: 1 }}>
      <div style={{ fontSize: 12, color: "#999", marginBottom: 6 }}>
        {title} &middot; predicted <b style={{ color: correct ? "#4ade80" : "#f87171" }}>{predictedLetter ?? "?"}</b>{" "}
        {predictedLetter && (correct ? "(correct)" : "(incorrect)")}
      </div>
      {letters.map((l) => (
        <ProbBar
          key={l}
          letter={l}
          name={options[l]}
          prob={probs?.[l]}
          isCorrect={l === correctLetter}
          isPredicted={l === predictedLetter}
        />
      ))}
    </div>
  );
}

function iou(a, b) {
  // a, b: [x0, y0, x1, y1]
  const ix0 = Math.max(a[0], b[0]);
  const iy0 = Math.max(a[1], b[1]);
  const ix1 = Math.min(a[2], b[2]);
  const iy1 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

// MCQ demonstration mode: only available for example images carrying known
// ground-truth region content (mcq metadata). Uses the SAME mouse-click
// selection as the shared image canvas -- a click on a region is matched
// against the image's known region bboxes by IoU to determine the
// ground-truth answer. Steering heads are the validated N=300 precomputed
// ranking, loaded automatically per model -- no noisy single-sample live
// discovery, and no head list shown; it just works.
export default function McqPanel({ mcqMeta, imageDataUrl, modelInfo, modelKey, hasPrecomputed, box, embedded = false }) {
  const [heads, setHeads] = useState([]); // [[layer, head], ...]
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!modelKey || !hasPrecomputed) {
      setHeads([]);
      return;
    }
    let cancelled = false;
    fetch(`${API}/api/precomputed_heads/${modelKey}`)
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        setHeads(data[`${data.recommended_method}_top_heads`] || data.raw_top_heads || []);
      })
      .catch(() => { if (!cancelled) setHeads([]); });
    return () => { cancelled = true; };
  }, [modelKey, hasPrecomputed]);

  if (!mcqMeta) return null;

  const { cell_names: names, cell_bboxes: bboxes } = mcqMeta;
  const letters = ["A", "B", "C", "D", "E", "F"].slice(0, names.length);

  const drawnBboxArr = box ? [box.x0, box.y0, box.x1, box.y1] : null;
  const { matchIndex, matchScore } = useMemo(() => {
    if (!drawnBboxArr) return { matchIndex: null, matchScore: 0 };
    let best = -1, bestScore = -1;
    bboxes.forEach((b, i) => {
      const score = iou(drawnBboxArr, b);
      if (score > bestScore) { bestScore = score; best = i; }
    });
    return { matchIndex: best, matchScore: bestScore };
  }, [box, mcqMeta]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeBbox = drawnBboxArr || bboxes[0];
  const activeIndex = matchIndex ?? 0;

  const mcqOptionLines = letters.map((l, i) => `${l}) ${names[i]}`).join("  ");
  const mcqPrompt = `Which of the following is shown in this image? ${mcqOptionLines}. Answer with only the letter.`;

  const runMcq = async () => {
    if (!drawnBboxArr) {
      setError("Click a region on the image above first.");
      return;
    }
    if (heads.length === 0) {
      setError(hasPrecomputed ? "Still loading validated heads, try again in a moment." : "No validated heads available for this model.");
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/generate_mcq`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageDataUrl,
          bbox: activeBbox,
          heads,
          options: names,
          correct_index: activeIndex,
          mode: "boost_suppress",
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "generate_mcq failed");
      setResult(await res.json());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={embedded ? {} : { marginTop: 20, borderTop: "1px solid #333", paddingTop: 16 }}>
      {!embedded && <h3 style={{ fontSize: 15, margin: "0 0 8px" }}>MCQ steering demo</h3>}
      <p style={{ fontSize: 12, color: "#999", margin: "0 0 10px" }}>
        Forces a choice among the image's real region contents (single-token answer) --
        the project's strongest, most unambiguous steering demonstration. Click a region
        on the image above to select it.
      </p>

      {drawnBboxArr ? (
        <div style={{ fontSize: 12, marginBottom: 10, padding: 8, background: "#1a1a1a", borderRadius: 4 }}>
          <div>
            Matched region: <b style={{ color: "#4ade80" }}>{letters[activeIndex]}) {names[activeIndex]}</b>
            {" "}(overlap {Math.round(matchScore * 100)}%)
          </div>
          <div style={{ marginTop: 6, color: "#999" }}>
            MCQ prompt (shown to the model at answer time): <code style={{ color: "#ccc" }}>{mcqPrompt}</code>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: 12, marginBottom: 10, padding: 8, background: "#2a1a1a", borderRadius: 4, color: "#f59e0b" }}>
          Click a region on the image above to pick it.
        </div>
      )}

      <button onClick={runMcq} disabled={running || !modelInfo || !drawnBboxArr || heads.length === 0} style={{ fontWeight: 600 }}>
        {running ? "Running..." : "Run MCQ (baseline + steered)"}
      </button>

      {error && <div style={{ color: "#f87171", fontSize: 13, marginTop: 8 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 12, color: "#999", marginBottom: 8 }}>{result.prompt}</div>
          <div style={{ display: "flex", gap: 12 }}>
            <McqResult
              title="Baseline"
              letters={letters}
              options={result.options}
              correctLetter={result.correct_letter}
              predictedLetter={result.baseline_letter}
              probs={result.baseline_probs}
            />
            <McqResult
              title="Steered"
              letters={letters}
              options={result.options}
              correctLetter={result.correct_letter}
              predictedLetter={result.steered_letter}
              probs={result.steered_probs}
            />
          </div>
        </div>
      )}
    </div>
  );
}
