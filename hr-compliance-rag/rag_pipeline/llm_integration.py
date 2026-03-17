import os
from typing import Dict, Any

try:
    import openai
except Exception:
    openai = None


class LLMClient:
    """Lightweight LLM client supporting OpenAI (if available) or a placeholder.

    Behavior:
    - If `openai` package is installed and `LLM_API_KEY` is provided, uses OpenAI
      ChatCompletion API (gpt-4o-mini / gpt-3.5-turbo-compatible interface).
    - Otherwise falls back to a deterministic placeholder response.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        if openai and self.api_key:
            openai.api_key = self.api_key

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Generate text from the prompt.

        Returns a dict with keys: `text`, `usage`, `raw`.
        """
        # If a mock LLM is requested via env, return a deterministic mock response.
        mock_flag = os.getenv("MOCK_LLM", "0").lower()
        if mock_flag in ("1", "true", "yes"):
            snippet = (prompt[:200] + "...") if len(prompt) > 200 else prompt
            return {"text": f"MOCK_RESPONSE: This is a deterministic mock answer based on the prompt: {snippet}", "usage": {"mock": True}, "raw": {}}
        if openai and self.api_key:
            try:
                # Support both the old `openai.ChatCompletion` interface and the
                # newer `openai.OpenAI().chat.completions.create(...)` interface
                # used by openai-python >= 1.0.0. Try the most-compatible
                # approaches and normalize the response.
                text = ""
                usage = {}
                raw = {}

                # Prefer the new interface (openai>=1.0) if available
                if hasattr(openai, "OpenAI"):
                    client = openai.OpenAI(api_key=self.api_key)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    raw = resp
                    # resp.choices[0].message.content
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        choice0 = resp.choices[0]
                        msg = getattr(choice0, "message", None) or choice0.get("message", {})
                        text = getattr(msg, "get", None) and msg.get("content") or getattr(msg, "content", None) or ""
                    else:
                        text = str(resp)
                    usage = getattr(resp, "usage", {}) or resp.get("usage", {})

                # Old interface (pre-1.0)
                elif hasattr(openai, "ChatCompletion"):
                    resp = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    raw = resp
                    # streaming/delta format
                    text = "".join([c.get("delta", {}).get("content", "") for c in resp.get("choices", [{}])])
                    if not text:
                        choice = resp.get("choices", [{}])[0]
                        text = choice.get("message", {}).get("content") or choice.get("text", "")
                    usage = resp.get("usage", {})

                # New interface (openai>=1.0)
                elif hasattr(openai, "OpenAI"):
                    client = openai.OpenAI(api_key=self.api_key)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    raw = resp
                    # resp.choices[0].message.content
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        choice0 = resp.choices[0]
                        # choice0 may be a mapping or an object
                        msg = getattr(choice0, "message", None) or choice0.get("message", {})
                        text = getattr(msg, "get", None) and msg.get("content") or getattr(msg, "content", None) or ""
                    else:
                        # Fallback to stringifying
                        text = str(resp)
                    usage = getattr(resp, "usage", {}) or resp.get("usage", {})

                else:
                    raise RuntimeError("Installed openai package has an unsupported interface")

                return {"text": text, "usage": usage or {}, "raw": raw}
            except Exception as e:
                return {"text": f"LLM call failed: {e}", "usage": {}, "raw": {}}

        # Fallback placeholder
        return {"text": "PLACEHOLDER_RESPONSE: OpenAI not configured.", "usage": {}, "raw": {}}
