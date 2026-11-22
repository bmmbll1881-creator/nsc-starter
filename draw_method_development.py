"""Draw a standalone SVG of the method stages discussed in Figure 1."""

from pathlib import Path
from xml.sax.saxutils import escape


STAGES = (
    ("16 free layer weights", "Ridge fit and cross-validation on GPT-2 development"),
    ("Single-parameter depth tilt", "Fit the historical beta of +0.67"),
    ("GPT-2 and external checks", "Evaluate fixed scores across the language-model panels"),
    ("ResNSC comparisons", "Compare scoring context and published allocation outcomes"),
    ("Size-depth orthogonalized tilt", "Freeze beta at -1.59 and validate on the reported panels"),
)


def main():
    """Render the five stages and connecting arrows without external packages."""
    output = Path(__file__).resolve().parent / "results" / "method_development.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="480" viewBox="0 0 620 480">']
    svg.append('<title>Method development</title>')
    for i, (title, detail) in enumerate(STAGES):
        y = 12 + i * 94
        svg.append(f'<rect x="30" y="{y}" width="560" height="70" rx="10" fill="#eef3f8" stroke="#567188"/>')
        svg.append(f'<text x="50" y="{y + 28}" font-family="sans-serif" font-size="19">{escape(title)}</text>')
        svg.append(f'<text x="50" y="{y + 51}" font-family="sans-serif" font-size="15">{escape(detail)}</text>')
        if i < len(STAGES) - 1:
            svg.append(f'<path d="M310 {y + 70}v19m-6 -6 6 6 6 -6" fill="none" stroke="#567188" stroke-width="2"/>')
    output.write_text("\n".join([*svg, "</svg>"]) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
