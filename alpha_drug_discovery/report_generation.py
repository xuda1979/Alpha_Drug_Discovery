from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from typing import Iterable

def create_report(filename: str, title: str, content: str | Iterable[str], image_path: str | None = None) -> None:
    """
    Generate a PDF report.

    Parameters:
    filename (str): Name of the PDF file to generate.
    title (str): Title of the report.
    content (str): Main content of the report.
    image_path (str, optional): Path to an image to include in the report.

    Returns:
    None
    """
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 20)
    c.drawString(100, 750, title)

    c.setFont("Helvetica", 12)
    if isinstance(content, str):
        lines = [content]
    else:
        lines = list(content)
    y = 700
    for line in lines:
        c.drawString(100, y, line)
        y -= 20
    
    if image_path:
        c.drawImage(image_path, 100, 500, width=5*inch, height=3*inch)
    
    c.save()
    print(f"Report generated: {filename}")
