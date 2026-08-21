import { useState, useEffect } from "react";
import ImageCanvas from "./ImageCanvas";
import McqPanel from "./McqPanel";
import "./App.css";

// Relative paths only -- vite.config.js proxies /api and /examples to the
// FastAPI backend server-side, so the browser only ever needs to reach the
// Vite dev server's own origin (whatever host/port that was opened at, incl.
// over an SSH-forwarded port). No second port needs to be reachable from the
// browser.
const API = "";

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function TabButton({ active, onClick, children, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 16px",
        fontSize: 13,
        fontWeight: 600,
        background: active ? "#2563eb" : "transparent",
        color: disabled ? "#666" : "#fff",
        border: "1px solid #444",
        borderBottom: active ? "1px solid #2563eb" : "1px solid #444",
        borderRadius: "6px 6px 0 0",
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [models, setModels] = useState([]);
  const [modelKey, setModelKey] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [loadingModel, setLoadingModel] = useState(false);

  const [examples, setExamples] = useState([]);
  const [imageUrl, setImageUrl] = useState(null);   // for display (may be a static /examples URL)
  const [imageDataUrl, setImageDataUrl] = useState(null); // base64, always what we send to backend
  const [mcqMeta, setMcqMeta] = useState(null); // {cell_names, cell_bboxes} for examples that have it

  const [taskMode, setTaskMode] = useState("mcq"); // "mcq" | "freeform"

  const [box, setBox] = useState(null);
  // Deliberately generic and NOT region-referencing: any localization the
  // model does should come from the attention-steering intervention (the
  // boosted/suppressed heads), not from telling it in words where to look.
  // A prompt like "describe the highlighted region" would confound the two.
  const [prompt, setPrompt] = useState("Describe the image.");
  const [heads, setHeads] = useState([]); // [[layer, head], ...], auto-loaded validated preset
  const [maxNewTokens, setMaxNewTokens] = useState(80);

  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/models`).then((r) => r.json()).then(setModels).catch(() => {});
    fetch(`${API}/api/examples`).then((r) => r.json()).then(setExamples).catch(() => {});
  }, []);

  const loadModel = async (key) => {
    setModelKey(key);
    setModelInfo(null);
    setError(null);
    if (!key) return;
    setLoadingModel(true);
    try {
      const res = await fetch(`${API}/api/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_key: key }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "load failed");
      setModelInfo(await res.json());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoadingModel(false);
    }
  };

  const onUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const dataUrl = await fileToDataUrl(file);
    setImageUrl(dataUrl);
    setImageDataUrl(dataUrl);
    setMcqMeta(null); // uploaded images have no known ground-truth region contents
    setTaskMode("freeform"); // MCQ mode is unusable without mcqMeta
    setResult(null);
  };

  const onPickExample = async (ex) => {
    const res = await fetch(`${API}${ex.url}`);
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onload = () => {
      setImageDataUrl(reader.result);
      setImageUrl(`${API}${ex.url}`);
      setMcqMeta(ex.mcq || null);
      setTaskMode(ex.mcq ? "mcq" : "freeform");
      setResult(null);
    };
    reader.readAsDataURL(blob);
  };

  const hasPrecomputed = models.find((m) => m.key === modelKey)?.has_precomputed_heads;

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

  const runGenerate = async () => {
    if (!imageDataUrl || !box) {
      setError("Select a model, image, and draw a region first.");
      return;
    }
    if (heads.length === 0) {
      setError(hasPrecomputed ? "Still loading validated heads, try again in a moment." : "No validated heads available for this model.");
      return;
    }
    setGenerating(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageDataUrl,
          prompt,
          bbox: [box.x0, box.y0, box.x1, box.y1],
          heads,
          max_new_tokens: maxNewTokens,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "generate failed");
      setResult(await res.json());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif", color: "#eee" }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Vis-Head Interactive Steering</h1>
      <p style={{ color: "#999", marginTop: 0, fontSize: 14 }}>
        Pick a model, load or upload an image, drag a box around a region, then steer the model's attention there.
      </p>

      <div style={{ display: "flex", gap: 24, marginTop: 20 }}>
        <div style={{ flex: "0 0 auto" }}>
          <ImageCanvas
            imageUrl={imageUrl}
            onBoxChange={setBox}
            cellBboxes={taskMode === "mcq" && mcqMeta ? mcqMeta.cell_bboxes : null}
          />

          <div style={{ marginTop: 12 }}>
            <input type="file" accept="image/*" onChange={onUpload} style={{ fontSize: 13 }} />
          </div>
          {examples.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>
                Example images (blue border = MCQ mode available):
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {examples.map((ex) => (
                  <img
                    key={ex.name}
                    src={`${API}${ex.url}`}
                    alt={ex.name}
                    title={ex.mcq ? "Has known region contents (MCQ mode available)" : ""}
                    onClick={() => onPickExample(ex)}
                    style={{
                      width: 56, height: 56, objectFit: "cover", borderRadius: 4, cursor: "pointer",
                      border: ex.mcq ? "2px solid #2563eb" : "1px solid #333",
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          <label style={{ fontSize: 13, display: "block", marginTop: 16 }}>
            Model
            <select
              value={modelKey}
              onChange={(e) => loadModel(e.target.value)}
              style={{ display: "block", width: "100%", marginTop: 4, padding: 6 }}
            >
              <option value="">-- select --</option>
              {models.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
            {loadingModel && <span style={{ fontSize: 12, color: "#f59e0b" }}>Loading model...</span>}
            {modelInfo && (
              <span style={{ fontSize: 12, color: "#4ade80" }}>
                {" "}Loaded &middot; {modelInfo.n_layers} layers x {modelInfo.n_heads} heads ({modelInfo.family})
              </span>
            )}
          </label>
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", gap: 4 }}>
            <TabButton active={taskMode === "mcq"} disabled={!mcqMeta} onClick={() => setTaskMode("mcq")}>
              MCQ
            </TabButton>
            <TabButton active={taskMode === "freeform"} onClick={() => setTaskMode("freeform")}>
              Free-form
            </TabButton>
          </div>
          <div style={{ border: "1px solid #444", borderTop: "none", borderRadius: "0 6px 6px 6px", padding: 16, flex: 1 }}>
            {taskMode === "mcq" ? (
              mcqMeta ? (
                <McqPanel
                  mcqMeta={mcqMeta}
                  imageDataUrl={imageDataUrl}
                  modelInfo={modelInfo}
                  modelKey={modelKey}
                  hasPrecomputed={hasPrecomputed}
                  box={box}
                  embedded
                />
              ) : (
                <div style={{ fontSize: 13, color: "#999" }}>
                  Pick one of the blue-bordered example images to use MCQ mode &mdash;
                  it needs known ground-truth region contents, which only the
                  example composite-grid images carry.
                </div>
              )
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <label style={{ fontSize: 13 }}>
                  Prompt
                  <div style={{ fontSize: 11, color: "#999", fontWeight: 400, marginTop: 2 }}>
                    Keep this generic (e.g. "Describe the image.") &mdash; don't mention the
                    region in words. Localization comes purely from the attention
                    steering below, not from prompt hints.
                  </div>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                    style={{ display: "block", width: "100%", marginTop: 4, padding: 6, fontFamily: "inherit" }}
                  />
                </label>

                <label style={{ fontSize: 13 }}>
                  Max new tokens
                  <input
                    type="number"
                    value={maxNewTokens}
                    onChange={(e) => setMaxNewTokens(parseInt(e.target.value, 10) || 40)}
                    style={{ display: "block", width: 100, marginTop: 4, padding: 6 }}
                  />
                </label>

                <button
                  onClick={runGenerate}
                  disabled={generating || !modelInfo}
                  style={{ padding: "10px 16px", fontSize: 14, fontWeight: 600, cursor: "pointer" }}
                >
                  {generating ? "Generating..." : "Generate (baseline + steered)"}
                </button>

                {error && <div style={{ color: "#f87171", fontSize: 13 }}>{error}</div>}

                {result && (
                  <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={{ fontSize: 12, color: "#999" }}>
                      {result.n_target_tokens} target image tokens boosted, {result.n_other_tokens} suppressed.
                    </div>
                    <div style={{ background: "#1a1a1a", padding: 12, borderRadius: 6 }}>
                      <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Baseline (no steering)</div>
                      <div style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>{result.baseline}</div>
                    </div>
                    <div style={{ background: "#0f2818", padding: 12, borderRadius: 6, border: "1px solid #4ade8055" }}>
                      <div style={{ fontSize: 12, color: "#4ade80", marginBottom: 4 }}>Steered (attention forced to selected region)</div>
                      <div style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>{result.steered}</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
