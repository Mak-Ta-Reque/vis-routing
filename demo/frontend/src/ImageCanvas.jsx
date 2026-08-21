import { useRef, useState, useCallback, useEffect } from "react";

// Interactive image canvas: wheel to zoom, shift+drag (or middle-mouse) to
// pan. Two selection modes:
//  - cellBboxes given (MCQ mode, grid geometry known): a single click selects
//    whichever known cell contains the click point -- no drag needed.
//  - cellBboxes absent (Free-form mode, arbitrary image): drag draws an
//    arbitrary bounding box, same as before.
// Emits the box in [0,1] coordinates relative to the ORIGINAL (un-zoomed)
// image, which is what the backend's family adapters expect.
export default function ImageCanvas({ imageUrl, onBoxChange, cellBboxes = null }) {
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [box, setBox] = useState(null); // {x0,y0,x1,y1} in [0,1] image-space
  const [dragStart, setDragStart] = useState(null);
  const [dragMode, setDragMode] = useState(null); // "box" | "pan" | "click"
  const [panStart, setPanStart] = useState(null);
  const [mouseDownClient, setMouseDownClient] = useState(null);
  const CONTAINER_SIZE = 560;
  const CLICK_THRESHOLD_PX = 5;

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setBox(null);
    onBoxChange(null);
  }, [imageUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const clientToImageFrac = useCallback(
    (clientX, clientY) => {
      const rect = containerRef.current.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      const imgX = (localX - pan.x) / zoom;
      const imgY = (localY - pan.y) / zoom;
      return { x: imgX / CONTAINER_SIZE, y: imgY / CONTAINER_SIZE };
    },
    [pan, zoom]
  );

  const cellAt = useCallback(
    (fracX, fracY) => {
      if (!cellBboxes) return null;
      for (const b of cellBboxes) {
        if (fracX >= b[0] && fracX <= b[2] && fracY >= b[1] && fracY <= b[3]) return b;
      }
      return null;
    },
    [cellBboxes]
  );

  const onWheel = (e) => {
    e.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.min(8, Math.max(1, zoom * factor));
    // keep the point under the cursor fixed while zooming
    const imgX = (cursorX - pan.x) / zoom;
    const imgY = (cursorY - pan.y) / zoom;
    setPan({ x: cursorX - imgX * newZoom, y: cursorY - imgY * newZoom });
    setZoom(newZoom);
  };

  const onMouseDown = (e) => {
    e.preventDefault();
    if (e.shiftKey || e.button === 1) {
      setDragMode("pan");
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    } else if (cellBboxes) {
      setDragMode("click");
      setMouseDownClient({ x: e.clientX, y: e.clientY });
    } else {
      setDragMode("box");
      setDragStart(clientToImageFrac(e.clientX, e.clientY));
    }
  };

  const onMouseMove = (e) => {
    if (dragMode === "pan" && panStart) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    } else if (dragMode === "box" && dragStart) {
      const cur = clientToImageFrac(e.clientX, e.clientY);
      const x0 = Math.max(0, Math.min(dragStart.x, cur.x));
      const x1 = Math.min(1, Math.max(dragStart.x, cur.x));
      const y0 = Math.max(0, Math.min(dragStart.y, cur.y));
      const y1 = Math.min(1, Math.max(dragStart.y, cur.y));
      setBox({ x0, y0, x1, y1 });
    }
  };

  const onMouseUp = (e) => {
    if (dragMode === "box" && box) {
      onBoxChange(box);
    } else if (dragMode === "click" && mouseDownClient) {
      const moved = Math.hypot(e.clientX - mouseDownClient.x, e.clientY - mouseDownClient.y);
      if (moved <= CLICK_THRESHOLD_PX) {
        const { x, y } = clientToImageFrac(e.clientX, e.clientY);
        const cell = cellAt(x, y);
        if (cell) {
          const newBox = { x0: cell[0], y0: cell[1], x1: cell[2], y1: cell[3] };
          setBox(newBox);
          onBoxChange(newBox);
        }
      }
    }
    setDragMode(null);
    setDragStart(null);
    setPanStart(null);
    setMouseDownClient(null);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div>
      <div
        ref={containerRef}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        style={{
          width: CONTAINER_SIZE,
          height: CONTAINER_SIZE,
          overflow: "hidden",
          position: "relative",
          border: "1px solid #444",
          borderRadius: 8,
          background: "#111",
          cursor: dragMode === "pan" ? "grabbing" : "crosshair",
          userSelect: "none",
        }}
      >
        {imageUrl && (
          <img
            src={imageUrl}
            alt="input"
            draggable={false}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: CONTAINER_SIZE,
              height: CONTAINER_SIZE,
              objectFit: "contain",
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "0 0",
              pointerEvents: "none",
            }}
          />
        )}
        {cellBboxes &&
          cellBboxes.map((b, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: pan.x + b[0] * CONTAINER_SIZE * zoom,
                top: pan.y + b[1] * CONTAINER_SIZE * zoom,
                width: (b[2] - b[0]) * CONTAINER_SIZE * zoom,
                height: (b[3] - b[1]) * CONTAINER_SIZE * zoom,
                border: "1px dashed rgba(255,255,255,0.35)",
                pointerEvents: "none",
              }}
            />
          ))}
        {box && (
          <div
            style={{
              position: "absolute",
              left: pan.x + box.x0 * CONTAINER_SIZE * zoom,
              top: pan.y + box.y0 * CONTAINER_SIZE * zoom,
              width: (box.x1 - box.x0) * CONTAINER_SIZE * zoom,
              height: (box.y1 - box.y0) * CONTAINER_SIZE * zoom,
              border: "2px solid #4ade80",
              background: "rgba(74, 222, 128, 0.15)",
              pointerEvents: "none",
            }}
          />
        )}
      </div>
      <div style={{ fontSize: 12, color: "#999", marginTop: 6, display: "flex", gap: 12, alignItems: "center" }}>
        <span>
          {cellBboxes
            ? "scroll to zoom · click a region to select it · shift+drag to pan"
            : "scroll to zoom · drag to select region · shift+drag to pan"}
        </span>
        <button onClick={resetView} style={{ fontSize: 12 }}>reset view</button>
        {box && (
          <span>
            box: [{box.x0.toFixed(2)}, {box.y0.toFixed(2)}, {box.x1.toFixed(2)}, {box.y1.toFixed(2)}]
          </span>
        )}
      </div>
    </div>
  );
}
