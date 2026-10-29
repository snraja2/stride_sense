"""Append-only run events with in-memory streaming and timed replay."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator


class EventRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.events: list[dict] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False

    async def emit(self, event: dict) -> None:
        record = {"seq": len(self.events), "t_rel": time.monotonic() - self.started, "event": event}
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"t_rel": record["t_rel"], "event": event}, ensure_ascii=False) + "\n")
        await self.queue.put(record)

    async def finish(self) -> None:
        self.done = True
        await self.queue.put(None)


def read_events(path: str | Path) -> list[dict]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("event"), dict):
            events.append(record)
    return events


async def replay_events(path: str | Path, speed: float = 1.0) -> AsyncIterator[dict]:
    if speed <= 0:
        raise ValueError("replay speed must be positive")
    previous = 0.0
    for record in read_events(path):
        current = max(previous, float(record.get("t_rel", previous)))
        delay = (current - previous) / speed
        if delay:
            await asyncio.sleep(delay)
        previous = current
        yield record["event"]
