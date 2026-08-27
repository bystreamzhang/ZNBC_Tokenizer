const strictDecoder = new TextDecoder("utf-8", { fatal: true });
const MAX_TOKENS_PER_SAMPLE = 240;

export function formatRatio(value) {
  return `${Number(value).toFixed(2)}×`;
}

export function formatPercentage(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export function describeBytes(bytes) {
  if (!bytes || bytes.length === 0) return "∅";
  if (bytes.length === 1) {
    if (bytes[0] === 32) return "SPACE";
    if (bytes[0] === 9) return "\\t";
    if (bytes[0] === 10) return "\\n";
    if (bytes[0] === 13) return "\\r";
  }

  try {
    const text = strictDecoder.decode(Uint8Array.from(bytes));
    if (text && !/[\u0000-\u001f\u007f]/u.test(text)) {
      return JSON.stringify(text).replaceAll(" ", "␠");
    }
  } catch (_error) {
    // 单个 UTF-8 byte 可能不是完整字符；这种情况改为显示十六进制。
  }

  return bytes
    .map((value) => value.toString(16).padStart(2, "0").toUpperCase())
    .join(" ");
}

function createToken(tokenId, vocabulary, { selected = false, isNew = false } = {}) {
  const token = document.createElement("span");
  token.className = "token-card";
  if (selected) token.classList.add("is-selected");
  if (isNew) token.classList.add("is-new");

  const marker = document.createElement("span");
  marker.className = "token-marker";
  marker.textContent = isNew ? "新 token" : selected ? "待合并" : "token";

  const id = document.createElement("span");
  id.className = "token-id";
  id.textContent = String(tokenId);

  const bytes = document.createElement("span");
  bytes.className = "token-bytes";
  bytes.textContent = describeBytes(vocabulary[String(tokenId)]);

  token.append(marker, id, bytes);
  return token;
}

function visibleTokenIndexes(length) {
  if (length <= MAX_TOKENS_PER_SAMPLE) {
    return Array.from({ length }, (_value, index) => index);
  }
  const half = MAX_TOKENS_PER_SAMPLE / 2;
  return [
    ...Array.from({ length: half }, (_value, index) => index),
    null,
    ...Array.from({ length: half }, (_value, index) => length - half + index),
  ];
}

export function renderSequences(
  container,
  sequences,
  vocabulary,
  { mergedPairStarts = null, newTokenId = null, labels = null } = {},
) {
  container.replaceChildren();

  sequences.forEach((tokens, sampleIndex) => {
    const sample = document.createElement("div");
    sample.className = "sample-sequence";

    const label = document.createElement("span");
    label.className = "sample-label";
    label.textContent = labels?.[sampleIndex] || `样本 ${sampleIndex + 1}`;

    const stream = document.createElement("div");
    stream.className = "token-stream";
    const selectedPositions = new Set();
    for (const start of mergedPairStarts?.[sampleIndex] || []) {
      selectedPositions.add(start);
      selectedPositions.add(start + 1);
    }

    if (tokens.length === 0) {
      const empty = document.createElement("span");
      empty.className = "empty-state";
      empty.textContent = "空 sequence";
      stream.append(empty);
    }

    for (const tokenIndex of visibleTokenIndexes(tokens.length)) {
      if (tokenIndex === null) {
        const omitted = document.createElement("span");
        omitted.className = "token-omitted";
        omitted.textContent = `… 中间 ${tokens.length - MAX_TOKENS_PER_SAMPLE} 个 token 已折叠 …`;
        stream.append(omitted);
        continue;
      }
      const tokenId = tokens[tokenIndex];
      stream.append(
        createToken(tokenId, vocabulary, {
          selected: selectedPositions.has(tokenIndex),
          isNew: tokenId === newTokenId,
        }),
      );
    }

    sample.append(label, stream);
    container.append(sample);
  });
}

export function renderPairCounts(container, step, vocabulary) {
  container.replaceChildren();
  if (!step || step.pair_counts.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "没有可展示的相邻 pair。";
    container.append(empty);
    return;
  }

  const maximum = step.pair_counts[0].frequency;
  for (const entry of step.pair_counts) {
    const chosen = entry.pair[0] === step.pair[0] && entry.pair[1] === step.pair[1];
    const row = document.createElement("div");
    row.className = "pair-row";
    if (chosen) row.classList.add("is-chosen");

    const name = document.createElement("span");
    name.className = "pair-name";
    const left = describeBytes(vocabulary[String(entry.pair[0])]);
    const right = describeBytes(vocabulary[String(entry.pair[1])]);
    name.textContent = `(${entry.pair[0]}, ${entry.pair[1]})  ${left} + ${right}`;

    const track = document.createElement("span");
    track.className = "bar-track";
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-label", `pair ${entry.pair.join(", ")} 出现次数`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", String(maximum));
    track.setAttribute("aria-valuenow", String(entry.frequency));

    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${(entry.frequency / maximum) * 100}%`;
    track.append(fill);

    const frequency = document.createElement("span");
    frequency.className = "pair-frequency";
    frequency.textContent = `${entry.frequency} 次`;
    row.append(name, track, frequency);
    container.append(row);
  }
}

export function makeInvariant(label, passed) {
  const item = document.createElement("span");
  item.className = "invariant";
  if (!passed) item.classList.add("is-failed");
  item.textContent = `${passed ? "✓" : "✗"} ${label}`;
  return item;
}

