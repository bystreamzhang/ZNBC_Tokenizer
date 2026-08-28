import { initializeBpeView } from "./views/bpe.js";
import { initializeDecoderView } from "./views/decoder.js";
import { initializeEncoderView } from "./views/encoder.js";

function initializeViewTabs() {
  const tabs = [...document.querySelectorAll("[data-view-target]")];
  const views = [...document.querySelectorAll("[data-tokenizer-view]")];

  function activate(tab) {
    const targetId = tab.dataset.viewTarget;
    for (const candidate of tabs) {
      const active = candidate === tab;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-selected", String(active));
      candidate.tabIndex = active ? 0 : -1;
    }
    for (const view of views) {
      view.hidden = view.id !== targetId;
    }
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + direction + tabs.length) % tabs.length];
      activate(next);
      next.focus();
    });
  });
}

const decoderView = initializeDecoderView();
const encoderView = initializeEncoderView({
  onResult: decoderView.useEncoderResult,
});
initializeBpeView({ onResult: encoderView.useTokenizerResult });
initializeViewTabs();
