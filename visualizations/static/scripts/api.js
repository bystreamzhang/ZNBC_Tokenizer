export async function requestBpeTrace(payload) {
  const response = await fetch("/api/bpe/trace", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  let result;
  try {
    result = await response.json();
  } catch (_error) {
    throw new Error(`server 返回了无法解析的响应（HTTP ${response.status}）。`);
  }

  if (!response.ok) {
    throw new Error(result.error || `请求失败（HTTP ${response.status}）。`);
  }
  return result;
}

