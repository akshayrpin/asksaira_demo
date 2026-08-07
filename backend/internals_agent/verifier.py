"""internals_agent/verifier.py — numeric self-check (generator -> verifier, one extra pass).

After the agent produces a final answer that asserts counts, this re-reads the exact tool
results the agent received and confirms every figure in the answer matches them (a stated total
equals the relevant count; any 'sum of breakdown' matches). If not, it returns a corrected answer.
Cheap insurance against transcription/summation slips in a staff-facing analytics answer.
Only runs when the transcript actually contains count() calls.
"""

import json
import logging

_SYS = (
    "You verify a data assistant's answer against the EXACT tool results it received. Check ONLY "
    "the numbers/counts in the answer. If every figure in the answer matches a value present in the "
    "tool results (including that a stated total equals the relevant count, and any sum of a "
    "breakdown matches its parts), reply with the single token: OK. Otherwise reply with a corrected "
    "version of the answer that uses the correct figures from the tool results, and nothing else."
)


async def verify(answer, transcript, question, client, model):
    """transcript: list of (tool_name, args, result). Returns the (possibly corrected) answer."""
    if not answer or not any(t[0] == "count" for t in transcript):
        return answer
    tool_dump = "\n".join(
        f"{name}({json.dumps(args, default=str)}) -> {json.dumps(res, default=str)}"
        for name, args, res in transcript
    )
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content":
                    f"Question: {question}\n\nTool results:\n{tool_dump}\n\nAnswer:\n{answer}"},
            ],
        )
        verdict = (resp.choices[0].message.content or "").strip()
        if verdict and verdict.strip().upper() != "OK":
            return verdict
    except Exception:
        logging.exception("internals verifier failed; returning original answer")
    return answer
