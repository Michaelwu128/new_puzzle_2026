#!/usr/bin/env python3
"""Render the v1/v2 piece sets and known solutions as a standalone SVG."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "encoding"))

from puzzle_defs import PUZZLES  # noqa: E402


COLORS = [
    "#4472C4",
    "#ED7D31",
    "#70AD47",
    "#FFC000",
    "#5B9BD5",
    "#A5A5A5",
    "#264478",
    "#9E480E",
    "#43682B",
    "#997300",
    "#636363",
]


def raw_shape_cells(rows: list[str]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
        if value == "1"
    ]


def rect(x: float, y: float, size: float, fill: str, stroke: str = "#ffffff") -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{size:g}" height="{size:g}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )


def text(x: float, y: float, value: str, size: int = 15, weight: int = 400) -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
        f'fill="#222222">{escape(value)}</text>'
    )


def render_puzzle(key: str, top: int) -> list[str]:
    puzzle = PUZZLES[key]
    board_x, board_y, board_cell = 42, top + 52, 25
    pieces_x, pieces_y, piece_cell = 390, top + 57, 17
    elements = [
        text(40, top + 25, f"{key}：11 塊六格拼圖與我提供的已知解", 22, 600),
        text(board_x, board_y - 12, "12×12 已知解", 15, 600),
        text(pieces_x, board_y - 12, "拼圖塊（原始方向）", 15, 600),
    ]

    piece_at: dict[tuple[int, int], int] = {}
    assert puzzle.known_solution is not None
    for piece_id, cells in enumerate(puzzle.known_solution):
        for cell in cells:
            piece_at[cell] = piece_id

    for r in range(12):
        for c in range(12):
            piece_id = piece_at.get((r, c))
            fill = COLORS[piece_id] if piece_id is not None else "#F3F4F6"
            stroke = "#ffffff" if piece_id is not None else "#D1D5DB"
            elements.append(
                rect(
                    board_x + c * board_cell,
                    board_y + r * board_cell,
                    board_cell,
                    fill,
                    stroke,
                )
            )
            if piece_id is not None:
                label = str(piece_id + 1)
                elements.append(
                    f'<text x="{board_x + (c + 0.5) * board_cell:g}" '
                    f'y="{board_y + (r + 0.68) * board_cell:g}" '
                    'text-anchor="middle" font-size="10" fill="#ffffff">'
                    f"{label}</text>"
                )

    for piece_id, (rows, name) in enumerate(zip(puzzle.pieces_raw, puzzle.piece_names)):
        col, row = piece_id % 4, piece_id // 4
        origin_x = pieces_x + col * 190
        origin_y = pieces_y + row * 145
        elements.append(text(origin_x, origin_y, f"{piece_id + 1}. {name}", 14, 600))
        for r, c in raw_shape_cells(rows):
            elements.append(
                rect(
                    origin_x + c * piece_cell,
                    origin_y + 12 + r * piece_cell,
                    piece_cell,
                    COLORS[piece_id],
                )
            )
    return elements


def main() -> None:
    out = ROOT / "docs" / "images" / "v1-v2-puzzles.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1080" '
        'width="1200" height="1080" role="img" '
        'aria-label="v1 and v2 puzzle pieces with known solutions">',
        '<rect width="1200" height="1080" fill="#ffffff"/>',
        text(40, 38, "12×12 No-Touch 六格拼圖：v1 / v2", 26, 700),
        *render_puzzle("v1", 55),
        '<line x1="40" y1="555" x2="1160" y2="555" stroke="#D1D5DB"/>',
        *render_puzzle("v2", 575),
        "</svg>",
    ]
    out.write_text("\n".join(elements) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
