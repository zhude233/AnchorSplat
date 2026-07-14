const sceneData = [
  {
    id: "benchmark",
    label: "3DGS-SR Benchmark",
    description: "Fine texture recovery with structure-consistent edges on the held-out 3DGS-SR benchmark.",
  },
  {
    id: "trellis",
    label: "TRELLIS Outputs",
    description: "Sharper mechanical boundaries and clearer thin structures on generated 3D assets.",
  },
  {
    id: "lgm",
    label: "LGM Outputs",
    description: "Richer local geometry and appearance details across complex generated objects.",
  },
  {
    id: "realworld",
    label: "Real-world Captures",
    description: "Robust refinement under noisy capture, imperfect reconstruction, and sensor artifacts.",
  },
];

function initializeIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function initializeComparison() {
  const comparison = document.querySelector("[data-comparison]");
  if (!comparison) return;

  const range = comparison.querySelector(".comparison-range");
  const output = comparison.querySelector("[data-output-layer]");
  const divider = comparison.querySelector("[data-divider]");

  const update = () => {
    const value = Number(range.value);
    output.style.clipPath = `inset(0 0 0 ${value}%)`;
    divider.style.left = `${value}%`;
  };

  range.addEventListener("input", update);
  update();
}

function initializeResults() {
  const tabs = [...document.querySelectorAll("[data-scene]")];
  const stage = document.querySelector("[data-result-stage]");
  const label = document.querySelector("[data-scene-label]");
  const description = document.querySelector("[data-scene-description]");
  const index = document.querySelector("[data-scene-index]");
  const controlButtons = [...document.querySelectorAll(".result-controls .icon-button")];
  let activeIndex = 0;

  const selectScene = (nextIndex) => {
    activeIndex = (nextIndex + sceneData.length) % sceneData.length;
    const scene = sceneData[activeIndex];

    tabs.forEach((tab) => {
      const active = tab.dataset.scene === scene.id;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });

    stage.dataset.sceneName = scene.label;
    label.textContent = scene.label;
    description.textContent = scene.description;
    index.textContent = `${String(activeIndex + 1).padStart(2, "0")} / ${String(sceneData.length).padStart(2, "0")}`;
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      selectScene(sceneData.findIndex((scene) => scene.id === tab.dataset.scene));
    });
  });

  controlButtons[0]?.addEventListener("click", () => selectScene(activeIndex - 1));
  controlButtons[1]?.addEventListener("click", () => selectScene(activeIndex + 1));
  selectScene(0);
}

function initializeCitationCopy() {
  const button = document.querySelector("[data-copy-citation]");
  const citation = document.querySelector("[data-citation]");
  if (!button || !citation) return;

  button.addEventListener("click", async () => {
    const label = button.querySelector("span");
    try {
      await navigator.clipboard.writeText(citation.textContent.trim());
      label.textContent = "Copied";
      setTimeout(() => {
        label.textContent = "Copy BibTeX";
      }, 1600);
    } catch {
      label.textContent = "Select BibTeX";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initializeIcons();
  initializeComparison();
  initializeResults();
  initializeCitationCopy();
});
