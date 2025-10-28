from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def main() -> None:
	c = canvas.Canvas('sample.pdf', pagesize=letter)
	t = c.beginText(72, 720)
	t.textLines('''Title: Transformers and Attention

Transformers use self-attention mechanisms to weigh token interactions.
Scaled dot-product attention computes weights via queries and keys.
Positional encodings inject order information into token embeddings.''')
	c.drawText(t)
	c.showPage()
	c.save()
	print('PDF created: sample.pdf')


if __name__ == "__main__":
	main()


