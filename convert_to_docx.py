"""
Convert manuscript_draft.md to professional MS Word (.docx) document with embedded figures and formatted tables.
"""
import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_color="F2F2F2"):
    """Set background shading for table cells."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell margins (padding)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_manuscript_docx(md_path="manuscript_draft.md", docx_path="manuscript_Q1_final.docx"):
    doc = Document()

    # Page Margins (1 inch all sides)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0
    in_code_block = False
    code_block_lines = []

    while i < len(lines):
        line = lines[i]

        # Handle Code Block / ASCII Tables
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_block_lines = []
            else:
                in_code_block = False
                # Parse ASCII Table into a genuine Word Table
                table_text = "\n".join(code_block_lines)
                if '───' in table_text or '|' in table_text or 'Mean' in table_text or 'Model' in table_text:
                    create_table_from_text(doc, code_block_lines)
                else:
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Inches(0.5)
                    run = p.add_run(table_text)
                    run.font.name = 'Courier New'
                    run.font.size = Pt(9.5)
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Title
        if line.startswith('# '):
            title_text = line.replace('# ', '').strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(14)
            run = p.add_run(title_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
            i += 1
            continue

        # Heading 1
        if line.startswith('## '):
            h1_text = line.replace('## ', '').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h1_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

            # Insert Figures where appropriate
            if "4.2 Multi-Model Benchmark Tournament" in h1_text:
                pass
            i += 1
            continue

        # Heading 2
        if line.startswith('### '):
            h2_text = line.replace('### ', '').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h2_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12.5)
            run.font.bold = True
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            i += 1
            continue

        # Horizontal Rule
        if line.strip() == '---':
            i += 1
            continue

        # Regular Paragraph
        if line.strip():
            p = doc.add_paragraph()
            
            # Check for blockquote or bold start
            clean_line = line.strip()
            
            # Format math equations and bold runs
            add_formatted_runs(p, clean_line)

            # Insert images after relevant sections
            if "Table 2: 5-Fold Cross-Validation Performance Comparison" in clean_line:
                if os.path.exists("figures/fig1_model_benchmark_r2.png"):
                    doc.add_paragraph()
                    img_p = doc.add_paragraph()
                    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    img_p.add_run().add_picture("figures/fig1_model_benchmark_r2.png", width=Inches(6.0))
                    cap_p = doc.add_paragraph()
                    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap_run = cap_p.add_run("Figure 1: Cross-Validated R² Performance Distribution Across Model Specifications.")
                    cap_run.font.size = Pt(10)
                    cap_run.font.italic = True

            elif "Table 3: Global Tree-SHAP Feature Importance Rankings" in clean_line:
                if os.path.exists("figures/fig2_shap_beeswarm.png"):
                    doc.add_paragraph()
                    img_p = doc.add_paragraph()
                    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    img_p.add_run().add_picture("figures/fig2_shap_beeswarm.png", width=Inches(6.0))
                    cap_p = doc.add_paragraph()
                    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap_run = cap_p.add_run("Figure 2: Global Tree-SHAP Summary Beeswarm Plot (Marginal Impact on ln(Price)).")
                    cap_run.font.size = Pt(10)
                    cap_run.font.italic = True

            elif "Table 4: Dynamic Lead-Time Price Elasticity Across Booking Horizons" in clean_line:
                if os.path.exists("figures/fig3_actual_vs_predicted.png"):
                    doc.add_paragraph()
                    img_p = doc.add_paragraph()
                    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    img_p.add_run().add_picture("figures/fig3_actual_vs_predicted.png", width=Inches(5.2))
                    cap_p = doc.add_paragraph()
                    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cap_run = cap_p.add_run("Figure 3: Actual vs. Predicted Hotel Room Rates (USD) for Out-of-Sample Test Set.")
                    cap_run.font.size = Pt(10)
                    cap_run.font.italic = True

        i += 1

    doc.save(docx_path)
    print(f"✅ Microsoft Word document successfully created at: {docx_path}")

def add_formatted_runs(paragraph, text):
    """Parse inline markdown formatting (bold, italic, code) into Word text runs."""
    # Split by bold markers
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(0x88, 0x22, 0x22)
        else:
            paragraph.add_run(part)

def create_table_from_text(doc, lines):
    """Convert text/markdown table lines into a native formatted Word Table."""
    valid_rows = []
    for l in lines:
        if re.match(r'^[─\-\s+=|]+$', l.strip()):
            continue
        # Split by multiple spaces or pipe
        if '|' in l:
            cols = [c.strip() for c in l.split('|') if c.strip() != '']
        else:
            cols = re.split(r'\s{2,}', l.strip())
        if cols and len(cols) >= 2:
            valid_rows.append(cols)

    if not valid_rows:
        return

    num_cols = max(len(r) for r in valid_rows)
    table = doc.add_table(rows=len(valid_rows), cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for r_idx, row_data in enumerate(valid_rows):
        row = table.rows[r_idx]
        is_header = (r_idx == 0)
        for c_idx in range(num_cols):
            cell = row.cells[c_idx]
            val = row_data[c_idx] if c_idx < len(row_data) else ""
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            
            run = p.runs[0] if p.runs else p.add_run(val)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            
            if is_header:
                run.bold = True
                set_cell_background(cell, "EAEAEA")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F9F9F9")
                
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

    doc.add_paragraph()  # spacing after table

if __name__ == "__main__":
    create_manuscript_docx()
